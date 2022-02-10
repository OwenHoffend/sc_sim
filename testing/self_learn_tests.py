import numpy as np
from sim.bitstreams import *
import matplotlib.pyplot as plt

def test_learn_AND_weight():
    N = 256
    Pw_ideal = np.random.uniform()
    w_cmp = N / 2
    Px = np.random.uniform()
    Pzc = Pw_ideal * Px
    rng = SC_RNG()
    Bzc = rng.bs_lfsr(N, Pzc, keep_rng=False, pack=False)
    Bx = rng.bs_lfsr(N, Px, keep_rng=False, pack=False)

    lfsr_sz = int(np.ceil(np.log2(N)))
    w_LFSR_run = rng._run_lfsr(N, lfsr_sz, keep_rng=False)
    ws = []
    lr = 1
    repeats = 4
    for _ in range(repeats):
        for i in range(N): #Sequential simulation (for now)
            Bw = w_cmp > w_LFSR_run[i]
            Z = Bw and Bx[i]
            if Z and not Bzc[i]:
                w_cmp -= lr
            elif Bzc[i] and not Z:
                w_cmp += lr
            ws.append(w_cmp / N)
    plt.plot(ws)
    plt.plot([Pw_ideal for _ in range(repeats*N)])
    plt.show()

def self_learn_tests():
    test_learn_AND_weight()