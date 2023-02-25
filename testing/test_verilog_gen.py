import numpy as np
from sim.verilog_gen import *
from sim.circuits import mux_1
from sim.PTM import get_func_mat, B_mat
from sim.SEC import Ks_to_Mf, opt_K_max
from sim.espresso import espresso_get_opt_file

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
    #gb4_ptm = np.load("gb4_ptm.npy")
    #ptm_to_verilog_tb(gb4_ptm, "gb4") #Already did this one
    #gb4_opt_ptm = np.load("gb4_opt_ptm.npy")
    #ptm_to_verilog_tb(gb4_opt_ptm, "gb4_opt")

    gb4_opt_a_ptm = np.load("gb4_opt_a_ptm.npy")
    ptm_to_verilog_tb(gb4_opt_a_ptm, "gb4_opt_a")

def test_espresso_out_to_verilog():
    gb2_opt = np.load("gb2.npy")
    espresso_get_opt_file(gb2_opt, "gb2_opt.in", "gb2_opt.out")
    espresso_out_to_verilog("gb2_opt.out", "gb2_opt")

def test_ptm_to_canonical_opt_verilog():
    gb4_ptm = np.load("gb4_ptm.npy")
    ptm_to_canonical_opt_verilog(gb4_ptm, 4, 16, "gb4")