import numpy as np
from sim.seq_recorr import *
from sim.bitstreams import *

def test_fsm_reco(N, depth):
    rng = SC_RNG()
    p1 = np.random.uniform()
    p2 = np.random.uniform()
    bs1 = rng.bs_lfsr(N, p1, keep_rng=False)
    bs2 = rng.bs_lfsr(N, p2, keep_rng=False)
    print("p1 start: ", bs_mean(bs1, bs_len=N))
    print("p2 start: ", bs_mean(bs2, bs_len=N))
    print("SCC start: ", bs_scc(bs1, bs2, bs_len=N))
    bs1_r, bs2_r = fsm_reco_d(bs1, bs2, depth, packed=True)
    print("p1 end: ", bs_mean(bs1_r, bs_len=N))
    print("p2 end: ", bs_mean(bs2_r, bs_len=N))
    print("SCC end: ", bs_scc(bs1_r, bs2_r, bs_len=N))
    return bs1, bs2, bs1_r, bs2_r

def get_seq_reco_verilog_testcases():
    N = 256
    def bstr(bs):
        return "{}'b".format(N) + ''.join(str(x) for x in np.unpackbits(bs))
    NUM_DEPTHS = 10
    bs_strs = []
    for d in range(1, NUM_DEPTHS+1):
        print(d)
        bs1, bs2, bs1_r, bs2_r = test_fsm_reco(N, d)
        bs_strs.append(bstr(bs1))
        bs_strs.append(bstr(bs2))
        bs_strs.append(bstr(bs1_r))
        bs_strs.append(bstr(bs2_r))
    bs_str = ", ".join(reversed(bs_strs))

    verilog_code = f"""
`ifndef RECO_TEST_CASES
`define RECO_TEST_CASES
localparam NUM_DEPTHS = {NUM_DEPTHS};
localparam N = {N};
localparam [{N-1}:0] tcs [{4*NUM_DEPTHS-1}:0] = {{
{bs_str}
}};
`endif
"""
    with open(f"./verilog/recorrelators/testbench/reco_test_cases.svh", 'w') as outfile:
        outfile.write(verilog_code)