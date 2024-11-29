VALUES = {}

def lets_go():
    VALUES['GO'] = True

def are_we_go():
    return VALUES.get('GO', False)