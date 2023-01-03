import numpy as np
from sim.verilog_gen import *
from sim.circuits import mux_1
from sim.PTM import get_func_mat, B_mat
from sim.SEC import Ks_to_Mf, opt_K_max

def FA(a, b, cin):
    sum = np.bitwise_xor(np.bitwise_xor(a, b), cin)
    cout = np.bitwise_or(np.bitwise_or(
        np.bitwise_and(a, b),
        np.bitwise_and(a, cin)),
        np.bitwise_and(b, cin)
    )
    return sum, cout

def test_ptm_to_verilog():
    mux_ptm = get_func_mat(mux_1, 3, 1)
    ptm_to_verilog(mux_ptm, "mux")

    FA_ptm = get_func_mat(FA, 3, 2)
    ptm_to_verilog(FA_ptm, "FA")

def test_ptm_to_tb():
    FA_ptm = get_func_mat(FA, 3, 2)
    ptm_to_verilog_tb(FA_ptm, "FA")

def test_larger_ptm_to_verilog():
    gb4_ptm = np.load("gb4_ptm.npy")
    ptm_to_verilog_tb(gb4_ptm, "gb4") #Already did this one

    A = gb4_ptm @ B_mat(4)
    Ks = []
    Ks_opt = []
    for i in range(4):
        K = A[:, i].reshape(2**4, 2**16).T
        K_opt = opt_K_max(K)
        Ks.append(K)
        Ks_opt.append(K_opt)
    gb4_ptm_opt = Ks_to_Mf(Ks_opt)
    np.save("gb4_opt_ptm.npy", gb4_ptm_opt)
    ptm_to_verilog_tb(gb4_ptm_opt, "gb4_opt")

def test_espresso_out_to_verilog():
    espresso_out_to_verilog("gb4.out", "gb4")
    espresso_out_to_verilog("gb4_opt.out", "gb4_opt")