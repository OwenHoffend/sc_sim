import numpy as np
from sim.seq_recorr import *
from sim.bitstreams import *
from sim.SEC_opt_macros import ilog2

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

def test_fsm_reco_transfer():
    """Get a transfer function input-->output SCC for FSM reco"""
    num_trials = 500
    num_corrs = 100
    cs = np.linspace(0, 1, num_corrs)
    N = 256
    dmax = 1
    data = np.zeros((2, num_corrs, dmax))
    for d in range(1, dmax+1):
        for i in range(num_corrs):
            print("\nc: ", cs[i])
            c_avg = 0
            c_reco_avg = 0
            for j in range(num_trials):
                rng = SC_RNG()
                p1 = np.random.uniform()
                p2 = np.random.uniform()
                bs1, bs2 = gen_correlated(cs[i], N, p1, p2, rng.bs_lfsr)
                #bs1_r, bs2_r = fsm_reco_d(bs1, bs2, d, packed=True)
                bs1_r, bs2_r = fsm_reco_abdellatef(bs1, bs2, packed=True)
                c_avg += bs_scc(bs1, bs2)
                c_reco_avg += bs_scc(bs1_r, bs2_r)
            c_avg /= num_trials
            c_reco_avg /= num_trials
            data[0, i, d-1] = c_avg
            data[1, i, d-1] = c_reco_avg
            print("c_avg: ", c_avg)
            print("c_reco_avg: ", c_reco_avg)
    np.save("fsm_reco_transfer_abdel.npy", data)

def test_fsm_reco_transfer_3():
    """Test whether FSM recorrelators have the transitive property: 
    does recorrelating between A<-->B<-->C implies A<-->C as well"""
    N = 256
    num_trials = 100
    dmax = 256
    data = np.zeros((6, dmax+1))
    for d in range(1, dmax+1):
        print(d)
        c_avg = np.zeros((3,3))
        c_avg_r = np.zeros((3,3))
        for i in range(num_trials):
            rng = SC_RNG()
            ps = np.random.uniform(0.0, 1.0, 3)
            bs_mat = rng.bs_lfsr_mat(N, ps, keep_rng=False)
            c_avg += get_corr_mat(bs_mat, N)
            bs1_r, bs2_r = fsm_reco_d(bs_mat[0, :], bs_mat[1, :], d, packed=True)
            bs2_r, bs3_r = fsm_reco_d(bs2_r, bs_mat[2, :], d, packed=True)
            bs1_r, bs3_r = fsm_reco_d(bs1_r, bs3_r, d, packed=True)
            bs_mat_r = np.array([bs1_r, bs2_r, bs3_r]) 
            c_avg_r += get_corr_mat(bs_mat_r, N)
        c_avg /= num_trials
        c_avg_r /= num_trials
        data[0, d] = c_avg[1, 0]
        data[1, d] = c_avg[2, 0]
        data[2, d] = c_avg[2, 1]
        data[3, d] = c_avg_r[1, 0]
        data[4, d] = c_avg_r[2, 0]
        data[5, d] = c_avg_r[2, 1]
        print("c_avg: ", c_avg)
        print("c_reco_avg: ", c_avg_r)
    np.save("fsm_reco_transfer_3.npy", data)

def test_fsm_reco_transfer_4():
    """Test whether FSM recorrelators have the transitive property: 
    does recorrelating between A<-->B<-->C implies A<-->C as well"""
    N = 256
    num_trials = 100
    dmax = 256
    for d in range(1, dmax+1):
        print(d)
        c_avg = np.zeros((4,4))
        c_avg_r = np.zeros((4,4))
        for i in range(num_trials):
            rng = SC_RNG()
            ps = np.random.uniform(0.0, 1.0, 4)
            bs_mat = rng.bs_lfsr_mat(N, ps, keep_rng=False)
            c_avg += get_corr_mat(bs_mat, N)
            bs1_r, bs2_r = fsm_reco_d(bs_mat[0, :], bs_mat[1, :], d, packed=True)
            bs3_r, bs4_r = fsm_reco_d(bs_mat[2, :], bs_mat[3, :], d, packed=True)
            bs1_r, bs3_r = fsm_reco_d(bs1_r, bs3_r, d, packed=True)
            bs2_r, bs4_r = fsm_reco_d(bs2_r, bs4_r, d, packed=True)
            bs1_r, bs4_r = fsm_reco_d(bs1_r, bs4_r, d, packed=True)
            bs2_r, bs3_r = fsm_reco_d(bs2_r, bs3_r, d, packed=True)
            bs_mat_r = np.array([bs1_r, bs2_r, bs3_r, bs4_r]) 
            c_avg_r += get_corr_mat(bs_mat_r, N)
        c_avg /= num_trials
        c_avg_r /= num_trials
        print("c_avg: ", c_avg)
        print("c_reco_avg: ", c_avg_r)

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