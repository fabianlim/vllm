import torch
from torch import nn
from torch.nn.parameter import Parameter

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_state_update)
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined)
from vllm.model_executor.models.mamba_cache import MambaCacheParams
from vllm.model_executor.utils import set_weight_attrs
from vllm.distributed import (divide, get_tensor_model_parallel_world_size,
                              get_tensor_model_parallel_rank,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.model_loader.weight_utils import (
    composed_weight_loader, default_weight_loader, sharded_weight_loader, 
    LoaderFunction)

from typing import Tuple, Union, Optional, List
from vllm.model_executor.custom_op import CustomOp

# Adapted from transformers.models.mamba2.modeling_mamba2.MambaRMSNormGated
# also referenced https://github.com/vllm-project/vllm/pull/9292
@CustomOp.register("mixer2_gated_rms_norm")
class Mixer2RMSNormGated(CustomOp):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.tp_size = get_tensor_model_parallel_world_size()
        set_weight_attrs(self.weight,
                         {"weight_loader": sharded_weight_loader(0)})

    def forward_native(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ):
        input_dtype = x.dtype
        x = x * nn.functional.silu(gate.to(torch.float32))

        if self.tp_size > 1:
            # Compute local sum and then reduce to obtain global sum
            local_sums = x.pow(2).sum(dim=-1, keepdim=True)
            global_sums = tensor_model_parallel_all_reduce(local_sums)
            # Calculate the variance
            count = self.tp_size * x.shape[-1]
            variance = (global_sums / count)

        else:
            variance = x.pow(2).mean(-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)

    def forward_cuda(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if self.tp_size > 1:
            return self.forward_native(x, gate)

        from vllm import _custom_ops as ops

        # cast x and gate to float32 before silu
        out = torch.empty_like(x)
        y = x * nn.functional.silu(gate.to(torch.float32))
        ops.rms_norm(
            out,
            y.to(x.dtype),
            self.weight.data,
            self.variance_epsilon,
        )
        return out

def extra_groups_for_head_shards(ngroups: int, tp_size: int):
    """Compute the extra (logical) groups to account for head shards"""

    # in the case ngoups % tp_size == 0, this will be zero
    if ngroups % tp_size == 0:
        return 0

    return tp_size - ngroups % tp_size

def mamba_v2_sharded_weight_loader(
    shard_spec: List[int], tp_size: int, tp_rank: int,
) -> LoaderFunction:
    """Create a weight loader for mamba v2. This ensures that the projections are
    correctly sharded so that they can be split into x, B, C. It also ensures the 
    the all the groups corresponding to a head shard is placed together with it.
    """

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:

        # - track boundary of (sharded) param, and loaded_weight, respectively
        boundary, loaded_boundary = 0, 0
        for full_dim, extra, ratio in shard_spec:
            # - full dim is the expected size of the model
            # - if extra > 0, this means there was some expansion

            # - num of dims expected to be loaded
            shard_size = full_dim // tp_size

            # - compute where to take the loaded shard from
            rank = tp_rank // ratio

            # - should start from here (determined by rank)
            loaded_skip = rank * shard_size # take these number dims from loaded
            loaded_start_idx = loaded_boundary + loaded_skip

            # - these many number dims to take from loaded_weight
            take = min(shard_size, full_dim - extra - loaded_skip)

            # - always shard on dim 0
                param.data[
                boundary:boundary+take,...
                ] = loaded_weight[
                loaded_start_idx:loaded_start_idx+take
                ]

            # move boundaries
            boundary += shard_size
            loaded_boundary += (full_dim - extra)

    return loader

# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
@CustomOp.register("mamba_mixer2") 
class MambaMixer2(CustomOp):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute
    the `contextualized_states`. A, D are input independent
    (see Mamba paper [1] Section 3.5.2 "Interpretation of A"
    for why A isn't selective) ∆, B, C are input-dependent
    (this is a key difference between Mamba and the linear time
    invariant S4, and is why Mamba is called
    **selective** state spaces)
    """

    def __init__(self,
                 hidden_size: int,
                 ssm_state_size: int,
                 conv_kernel_size: int,
                 intermediate_size: int,
                 use_conv_bias: bool,
                 use_bias: bool,
                 use_rms_norm: bool,
                 n_groups: int = 1,
                 num_heads: int = 128,
                 head_dim: int = 64,
                 rms_norm_eps: float = 1e-5,
                 activation="silu",
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()

        # For TP, the sharding plan is as follows:
        # - for the conv modules, since 
        #   conv_dim = intermediate_size * 2 * n_groups * ssm_state_size,
        #   we shard intermediate_size and n_groups
        # - since intermediate_size = n_heads * head_dim, sharding on
        #   intermediate_size is achieved by sharding on n_heads.
        # - so if world_size divides groups, then sharding 
        #   (n_groups / world_size, n_heads / world_size)
        #   also maintains the invariant n_heads % n_groups == 0
        # - HOWEVER< if world_size DOES NOT divide groups, then we need to allocate
        #   extra space in the shard, such that the WHOLE GROUP must be placed
        #   together with the HEAD SHARD.
        self.tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        self.ssm_state_size = ssm_state_size
        self.use_rms_norm = use_rms_norm
        self.activation = activation

        self.chunk_size = 256
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_heads = num_heads

        self.n_groups = n_groups
        if n_groups % self.tp_size != 0:
            # - for TP we shard conv_dim by sharding on n_groups, 
            # - but if n_groups cannot divide tp_size, we need to 
            #   extend some extra groups
            # self.n_groups = n_groups + self.tp_size - (n_groups % self.tp_size)
            self.n_groups = n_groups + extra_groups_for_head_shards(n_groups, self.tp_size)

        self.conv_dim = (
            intermediate_size + 2 * self.n_groups * ssm_state_size
        )
        self.conv1d = ColumnParallelLinear(
            input_size=conv_kernel_size,
            output_size=self.conv_dim,
            bias=use_conv_bias,
            quant_config=None,
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size + self.conv_dim + self.num_heads,
            bias=use_bias,
            quant_config=quant_config)

        # - because in_proj is a concatenation of 3 weights, we 
        #   need to interleave them before sharding
        # - use the custom weight loader mamba_v2_sharded_weight_loader
        #   for conv1d.bias, covn1d.weight and in_proj.weight
        # - need to set these settings, to assign the groups to the head shards
        group_shard_settings = (
            self.n_groups * self.ssm_state_size, # expected model size
            (self.n_groups - n_groups) * self.ssm_state_size, # extra dims assigned
            self.num_heads // n_groups, # ratio for mapping back to original group
        )
        intemediate_settings = (intermediate_size, 0, 1)
        head_setings = (self.num_heads, 0, 1)

        delattr(self.conv1d.bias, "weight_loader")
        set_weight_attrs(self.conv1d.bias, {
            "weight_loader": mamba_v2_sharded_weight_loader(
                [
                    intemediate_settings, group_shard_settings, group_shard_settings,
                ],
                self.tp_size, tp_rank,
            )
        })

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(self.conv1d.weight, {
            "weight_loader": mamba_v2_sharded_weight_loader(
                [
                    intemediate_settings, group_shard_settings, group_shard_settings,
                ],
                self.tp_size, tp_rank
            )
        })

        delattr(self.in_proj.weight, "weight_loader")
        set_weight_attrs(self.in_proj.weight, {
            "weight_loader": mamba_v2_sharded_weight_loader(
                [
                    intemediate_settings, # for gate
                    intemediate_settings, group_shard_settings, group_shard_settings,
                    head_setings,  # for dt
                ],
                self.tp_size, tp_rank
            )
        })

        # - these are TPed by heads to reduce the size of the 
        #   temporal shape
        self.A = nn.Parameter(
            torch.empty(
                divide(num_heads, self.tp_size), dtype=torch.float32,
            ))
        self.D = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.dt_bias = nn.Parameter(torch.ones(num_heads // self.tp_size))

        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})
        a_weight_loader = composed_weight_loader(
            sharded_weight_loader(0), lambda x: -torch.exp(x.float()))
        set_weight_attrs(self.A, {"weight_loader": a_weight_loader})
        set_weight_attrs(self.dt_bias,
                         {"weight_loader": sharded_weight_loader(0)})

        self.out_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=use_bias,
            input_is_parallel=True,
            quant_config=quant_config)

        self.norm = Mixer2RMSNormGated(
            intermediate_size // self.tp_size, eps=rms_norm_eps
        )

    def forward_native(self, hidden_states: torch.Tensor,
                       attn_metadata: AttentionMetadata,
                       conv_state: torch.Tensor, ssm_state: torch.Tensor):
        pass

    def forward_cuda(self, hidden_states: torch.Tensor,
                     attn_metadata: AttentionMetadata,
                     mamba_cache_params: MambaCacheParams):


        seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size

        # - doing it differently from mixer v1; little confused with its logic
        # - we need to do is to detect if there is any prefill; if there are 
        #   no prefils, then each example will be coming in one sample at a time
        # - on the other hand v1 checks for "query_start_loc" and "context_lens_tensor"
        #   however we have noticed that, even when the samples are coming in
        #   one at a time, they are still non-NO.e
        #   * "query_start_loc" = [0, 1, ..]
        #   * "context_lens_tensor" = [8, ...]
        has_prefill = attn_metadata.num_prefills > 0 

        # 1. Gated MLP's linear projection
        projected_states, _ = self.in_proj(hidden_states)
        gate, hidden_states_B_C, dt = torch.split(
            projected_states,
            [
                self.intermediate_size // self.tp_size, 
                self.conv_dim // self.tp_size, 
                self.num_heads // self.tp_size,
            ],
            dim=-1,
        )

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                               self.conv1d.weight.size(2))

        if has_prefill:
            # |---------- N-1 iteration --------|
            # |---------------- N iteration ---------------------|
            # |- tokenA -|......................|-- newTokens ---|
            # |---------- context_len ----------|
            # |-------------------- seq_len ---------------------|
            #                                   |-- query_len ---|

            # - "cache_indices" upates the conv_state cache in positions
            #   pointed to by "mamba_cache_params.state_indices_tensor"
            hidden_states_B_C = causal_conv1d_fn(
                hidden_states_B_C.transpose(0, 1),
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=mamba_cache_params.conv_state,
                has_initial_state=attn_metadata.context_lens_tensor > 0,
                cache_indices=mamba_cache_params.state_indices_tensor,
                query_start_loc=attn_metadata.query_start_loc
            ).transpose(0, 1)[:seq_len]
        else:
            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                mamba_cache_params.conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=mamba_cache_params.state_indices_tensor
            )

        # - get hidden_states, B and C after depthwise convolution.
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [
                self.intermediate_size // self.tp_size, 
                groups_time_state_size // self.tp_size,
                groups_time_state_size // self.tp_size,
            ],
            dim=-1,
        )

        # 3. State Space Model sequence transformation
        if has_prefill:
            
            # FIXME: we are having problems using mamba_chunk_scan_combined
            # with chunked prefill. This is because there is no
            # initial_states requires initial_states.shape[0] to match
            # the batch size, but cu_seqlens requires batch_size = 1.
            # Therefore as of now, initial_states and cu_seqlens are 
            # mutually exclusive.

            initial_states = None
            # if any(attn_metadata.context_lens_tensor > 0):
            #     initial_states = mamba_cache_params.ssm_state[
            #         mamba_cache_params.state_indices_tensor
            #     ]

            scan_output, varlen_state = mamba_chunk_scan_combined(
                hidden_states.view(1, seq_len, self.num_heads // self.tp_size, self.head_dim),
                dt.unsqueeze(0),
                self.A,
                B.view(1, seq_len, self.n_groups // self.tp_size, -1),
                C.view(1, seq_len, self.n_groups // self.tp_size, -1),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                dt_bias=self.dt_bias,
                seq_idx=attn_metadata.seq_idx.unsqueeze(0),
                cu_seqlens=attn_metadata.query_start_loc,
                initial_states=initial_states,
                return_varlen_states=True,
                return_final_states=False,
                dt_softplus=True,
                dt_limit=(0.0, float("inf")),
            )

            # update ssm states
            # - varlen state is a (batch, nheads, headdim, dstate) tensor
            for i, idx in enumerate(mamba_cache_params.state_indices_tensor):
                mamba_cache_params.ssm_state[idx].copy_(varlen_state[i])

            # - reshape
            hidden_states = scan_output.view(seq_len, -1)
        else:

            # NOTE: can be optimized? 
            n_groups = self.n_groups // self.tp_size
            A = self.A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.view(-1, n_groups, B.shape[1] // n_groups)
            C = C.view(-1, n_groups, C.shape[1] // n_groups)
            hidden_states_reshaped = hidden_states.view(-1, self.num_heads // self.tp_size, self.head_dim)

            # - the hidden is reshaped into number of current batches
            # - in this case there is no more prefil, so the batches gen
            #   1 token at a time
            # - thus hidden will be (bs, num_heads, head_dim)
            # - mamba_cache_params.ssm_state's slots will be selected
            #   using "mamba_cache_params.state_indices_tensor", just as
            #   above in the prefill case

            hidden_states = selective_state_update(
                mamba_cache_params.ssm_state,
                hidden_states_reshaped,
                dt,
                A, 
                B,
                C,
                D, 
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
                state_batch_indices=mamba_cache_params.state_indices_tensor,
            )
            hidden_states = hidden_states.view(
                -1, (self.num_heads // self.tp_size) * self.head_dim
            )

        # # 4. gated MLP
        hidden_states = self.norm(hidden_states, gate)

        # # 5. Final linear projection
        out, _ = self.out_proj(hidden_states)
        return out 