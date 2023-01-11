import numpy as np
from sim.seq_recorr import *
from sim.bitstreams import *

def test_fsm_reco():
    rng = SC_RNG()
    N = 1024
    p1 = np.random.uniform()
    p2 = np.random.uniform()
    bs1 = rng.bs_lfsr(N, p1, keep_rng=False)
    bs2 = rng.bs_lfsr(N, p2, keep_rng=False)
    print("p1 start: ", bs_mean(bs1, bs_len=N))
    print("p2 start: ", bs_mean(bs2, bs_len=N))
    print("SCC start: ", bs_scc(bs1, bs2, bs_len=N))
    bs1_r, bs2_r = fsm_reco(bs1, bs2, packed=True)
    print("p1 end: ", bs_mean(bs1_r, bs_len=N))
    print("p2 end: ", bs_mean(bs2_r, bs_len=N))
    print("SCC end: ", bs_scc(bs1_r, bs2_r, bs_len=N))