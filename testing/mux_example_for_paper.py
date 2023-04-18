import numpy as np
from sim.circuits import mux_1
from sim.PTM import *
from sim.SEC import *
from sim.espresso import *
from sim.SEC_opt_macros import IO_Params

def weird_mux(x1, x2, c1, c2, c3):
    return mux_1(c2, c1, x1), mux_1(c3, c1, x2)

def get_SE_mats():
    ptm = get_func_mat(weird_mux, 5, 2)
    io = IO_Params(3, 2, 2)
    Ks = get_Ks_from_ptm(ptm, io)
    print(Ks[0]*1)
    print(Ks[1]*1)
    K1_opt, K2_opt = opt_K_max(Ks[0]), opt_K_max(Ks[1])
    print(K1_opt*1)
    print(K2_opt*1)

    ptm_opt = Ks_to_Mf([K1_opt, K2_opt])

    cost = espresso_get_SOP_area(ptm_opt, "weird_mux.in", do_print=True)