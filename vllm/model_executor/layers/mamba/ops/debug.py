# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_chunk_scan.py

# ruff: noqa: E501

import torch
import triton
import triton.language as tl
from packaging import version

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')


@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_M': 16,
                'BLOCK_SIZE_N': 16,
                # 'BLOCK_SIZE_K': 16
            },
            num_stages=1,
            num_warps=1),
    ],
    key=['chunk_size', 'hdim', 'dstate', 'IS_CAUSAL'],
)
@triton.jit
def _debug_kernel(
    out_ptr,
    # dA_cumsum_ptr,
    seq_idx_ptr,
    C_ptr,
    states_ptr,
    initstates_ptr,
    chunk_indices_ptr,
    chunk_offsets_ptr,
    chunk_meta_num,
    # Matrix dimensions
    chunk_size,
    hdim,
    dstate,
    batch,
    seqlen,
    nheads_ngroups_ratio,
    stride_out_batch,
    stride_out_seqlen,
    stride_out_head,
    stride_out_hdim,
    # stride_dA_cs_batch,
    # stride_dA_cs_chunk,
    # stride_dA_cs_head,
    # stride_dA_cs_csize,
    stride_seq_idx_batch,
    stride_seq_idx_seqlen,
    stride_C_batch,
    stride_C_seqlen,
    stride_C_head,
    stride_C_dstate,
    stride_states_batch,
    stride_states_chunk,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    stride_init_states_batch,
    stride_init_states_head,
    stride_init_states_hdim,
    stride_init_states_dstate,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1).to(tl.int64)
    # pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    c_idx = tl.load(chunk_indices_ptr + pid_c, mask=pid_c > -1, other=0)
    c_off = tl.load(chunk_offsets_ptr + pid_c, mask=pid_c > -1, other=0)

    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    # dA_cumsum_ptr += pid_b * stride_dA_cs_batch + c_idx * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    C_ptr += pid_b * stride_C_batch + c_idx * chunk_size * stride_C_seqlen + (
        pid_h // nheads_ngroups_ratio) * stride_C_head

    # M-block offsets and prev states
    #  - logic in next block may override these if there is an active offset
    offs_m = pid_m * BLOCK_SIZE_M + c_off + tl.arange(0, BLOCK_SIZE_M)
    prev_states_ptr = states_ptr + pid_b * stride_states_batch + c_idx * stride_states_chunk + pid_h * stride_states_head
    prev_states_hdim = stride_states_hdim 
    prev_states_dstate = stride_states_dstate

    seq_idx_ptr += pid_b * stride_seq_idx_batch + c_idx * chunk_size * stride_seq_idx_seqlen

    # if there are init states, we only need seq_idx_m to point
    # what is the current seq_idx
    # - the prev is not needed in this case

    if (c_idx == 0 and c_off == 0) or c_off > 0:

        # just need to get the current one
        # - shouldnt need any guards, since c_off points to leftmost boundary
        seq_idx_m = tl.load(
            seq_idx_ptr + (pid_m * BLOCK_SIZE_M + c_off) * stride_seq_idx_seqlen
        )

        # - replace prev_states_ptr with init_states
        prev_states_ptr = initstates_ptr + seq_idx_m * stride_init_states_batch + pid_h * stride_init_states_head
        prev_states_hdim = stride_init_states_hdim # override strides
        prev_states_dstate = stride_init_states_dstate

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # - handle chunk state limit
    chunk_size_limit = min(chunk_size, seqlen - c_idx * chunk_size)

    # have to split this if otherwise compilation will have problems
    if (c_idx == 0 and c_off == 0) or c_off > 0:
        # this is the case where the seqlen may end within the current chunk
        #  .. c_off | .... | c_off + 1
        c_idx_n = tl.load(
            chunk_offsets_ptr + (pid_c+1), 
            mask=pid_c > -1 and pid_c < chunk_meta_num, other=-1 # to trigger different chunk
        )
        c_off_n = tl.load(
            chunk_offsets_ptr + (pid_c+1), 
            mask=pid_c > -1 and pid_c < chunk_meta_num, other=chunk_size
        )
        if c_idx == c_idx_n:
            chunk_size_limit = min(c_off_n, seqlen - c_idx * chunk_size)


    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Without the if (pid_c > -1), with Triton 2.1.0, I get
    # Assertion `!(srcMmaLayout && dstMmaLayout) && "Unexpected mma -> mm a layout conversion"' failed.
    # With Triton 2.2.0, this works
    if IS_TRITON_22 or c_idx > -1:
        # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
        offs_k_dstate = tl.arange(
            0, BLOCK_SIZE_DSTATE 
        )
        C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen +
                          offs_k_dstate[None, :] * stride_C_dstate)
        
        prev_states_ptrs = prev_states_ptr + (
            offs_n[None, :] * prev_states_hdim +
            offs_k_dstate[:, None] * prev_states_dstate)

        C = tl.load(C_ptrs,
                    mask=(offs_m[:, None] < chunk_size_limit) &
                    (offs_k_dstate[None, :] < dstate),
                    other=0.0)
                    
        prev_states = tl.load(prev_states_ptrs,
                                mask=(offs_k_dstate[:, None] < dstate) &
                                (offs_n[None, :] < hdim),
                                other=0.0)
        prev_states = prev_states.to(C_ptr.dtype.element_ty)
        acc = tl.dot(C, prev_states) 
        # if pid_h == 0 and pid_bc == 1:
        #     print ("prev_states", prev_states)

    offs_out_m = pid_m * BLOCK_SIZE_M + c_off + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    out_ptr += pid_b * stride_out_batch + c_idx * chunk_size * stride_out_seqlen + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_out_seqlen * offs_out_m[:, None] +
                          offs_out_n[None, :] * stride_out_hdim)
    tl.store(out_ptrs,
             acc,
             mask=(offs_out_m[:, None] < chunk_size_limit) &
             (offs_out_n[None, :] < hdim))


def _debug(
    dA_cumsum,
    C,
    states,
    seq_idx=None,
    initial_states=None,
):
    (batch, seqlen, ngroups, dstate) = C.shape
    _, _, ngroups, dstate = C.shape
    (batch, nheads, nchunks, chunk_size) = dA_cumsum.shape
    (batch, nchunks, nheads, headdim, dstate) = states.shape 

    chunk_indices, chunk_offsets = None, None
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)

        if initial_states is not None:
            # with initial states, we need to take care of how 
            # seq_idx crosses the boundaries
            assert batch == 1, "chunk scan only supports initial states with batch 1"
            assert initial_states.shape == (seq_idx[0].max()+1, nheads, headdim, dstate)

            # extra = 0
            p = 0
            chunk_indices, chunk_offsets = [], []
            for i, idx in enumerate(seq_idx[0]):
                o = i % chunk_size
                c = idx > p
                if o == 0 or c:
                    # this means we have a change in sequence 
                    # - that does not accur on the chunk boundary
                    chunk_indices.append(i // chunk_size)
                    chunk_offsets.append(o)

                    if c:
                        p = idx # new sequence

            chunk_indices = torch.tensor(chunk_indices, dtype=torch.int, device=seq_idx.device)
            chunk_offsets = torch.tensor(chunk_offsets, dtype=torch.int, device=seq_idx.device)

    # Allocates output.
    out = torch.empty(batch,
                      seqlen,
                      nheads,
                      headdim,
                      device=C.device,
                      dtype=C.dtype)

    
    grid = lambda META: (triton.cdiv(
        chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(
            headdim, META['BLOCK_SIZE_N']), 
            batch * nchunks if chunk_offsets is None else len(chunk_offsets),
            nheads
        )
    _debug_kernel[grid](
        out,
        # dA_cumsum,
        seq_idx,
        C,
        states,
        initial_states,
        chunk_indices,
        chunk_offsets,
        len(chunk_indices) if chunk_indices is not None else 0,
        chunk_size,
        headdim,
        dstate,
        batch,
        seqlen,
        nheads // ngroups,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        # dA_cumsum.stride(0),
        # dA_cumsum.stride(2),
        # dA_cumsum.stride(1),
        # dA_cumsum.stride(3),
        *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else
          (0, 0)),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        C.stride(3),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        states.stride(4),
        *(
            (
                initial_states.stride(0), initial_states.stride(1),
                initial_states.stride(2), initial_states.stride(3)
            ) if initial_states is not None else (0, 0, 0, 0)
        ),
        True,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        IS_TRITON_22=TRITON_22,
    )
    return out
