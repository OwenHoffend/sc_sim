import numpy as np
from sim.seq_recorr import *
from sim.bitstreams import *

def test_fsm_reco():
    rng = SC_RNG()
    N = 256
    p1 = np.random.uniform()
    p2 = np.random.uniform()
    bs1 = rng.bs_lfsr(N, p1, keep_rng=False)
    bs2 = rng.bs_lfsr(N, p2, keep_rng=False)
    print("p1 start: ", bs_mean(bs1, bs_len=N))
    print("p2 start: ", bs_mean(bs2, bs_len=N))
    print("SCC start: ", bs_scc(bs1, bs2, bs_len=N))
    for i in range(1, 16):
        bs1_r, bs2_r = fsm_reco_N(bs1, bs2, i, packed=True)
        print(i)
        print("p1 end: ", bs_mean(bs1_r, bs_len=N))
        print("p2 end: ", bs_mean(bs2_r, bs_len=N))
        print("SCC end: ", bs_scc(bs1_r, bs2_r, bs_len=N))
    #return bs1, bs2, bs1_r, bs2_r

def get_seq_reco_verilog_testcases():
    def bstr(bs):
        return ''.join(str(x) for x in np.unpackbits(bs))
    #Generate 25 test cases to test the functionality of the seq_reco circuit
    for _ in range(4):
        bs1, bs2, bs1_r, bs2_r = test_fsm_reco()
        print(bstr(bs1))
        print(bstr(bs2))
        print(bstr(bs1_r))
        print(bstr(bs2_r))