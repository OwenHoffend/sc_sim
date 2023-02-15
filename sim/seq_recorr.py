import numpy as np

#Seqential re-correlators as implemented in V. Lee, A Alaghi, L. Ceze, 2018. 
#Correlation Manipulating Circuits for Stochastic Computing
def fsm_reco(x1_bs, x2_bs, packed=False): #Depth of 1
    state = 1
    if packed:
        x1_bs = np.unpackbits(x1_bs)
        x2_bs = np.unpackbits(x2_bs)
    N1 = x1_bs.size
    N2 = x2_bs.size
    assert N1 == N2
    z1_bs = np.zeros(N1, dtype=np.bool_)
    z2_bs = np.zeros(N1, dtype=np.bool_)
    for i in range(N1):
        x1 = x1_bs[i]
        x2 = x2_bs[i]
        if x1 == x2:
            z1 = x1
            z2 = x2
        else:
            if state == 0:
                if not x1 and x2:
                    state = 1
                    z1 = True
                    z2 = True
                else: #x1 and not x2:
                    z1 = True
                    z2 = False
            elif state == 1:
                if not x1 and x2:
                    state = 2
                    z1 = False
                    z2 = False
                else:
                    state = 0
                    z1 = False
                    z2 = False
            else:
                if not x1 and x2:
                    z1 = False
                    z2 = True
                else:
                    state = 1
                    z1 = True
                    z2 = True
        z1_bs[i] = z1
        z2_bs[i] = z2
    if packed:
        z1_bs = np.packbits(z1_bs)
        z2_bs = np.packbits(z2_bs)
    return z1_bs, z2_bs

def fsm_reco_N(x1_bs, x2_bs, d, packed=False):
    """FSM-based recorrelation with an arbitrary depth d. When d=1, this is equivalent to fsm_reco"""
    if packed:
        x1_bs = np.unpackbits(x1_bs)
        x2_bs = np.unpackbits(x2_bs)
    N1 = x1_bs.size
    N2 = x2_bs.size
    assert N1 == N2
    assert d <= N1
    z1_bs = np.zeros(N1 + d, dtype=np.bool_)
    z2_bs = np.zeros(N1 + d, dtype=np.bool_)

    state = 0
    for i in range(N1):
        x1 = x1_bs[i]
        x2 = x2_bs[i]
        z1 = x1
        z2 = x2
        if x1 != x2: #unpaired x1
            if x1:
                if state > 0:
                    z1 = True
                    z2 = True
                else:
                    z1 = state == -d
                    z2 = False
                state -= 1
            else: #unpaired x2
                if state < 0:
                    z1 = True
                    z2 = True
                else:
                    z1 = False
                    z2 = state == d
                state += 1
            if np.abs(state) > d:
                state = d * np.sign(state)
        z1_bs[i] = z1
        z2_bs[i] = z2
    if packed:
        z1_bs = np.packbits(z1_bs)
        z2_bs = np.packbits(z2_bs)
    return z1_bs, z2_bs
