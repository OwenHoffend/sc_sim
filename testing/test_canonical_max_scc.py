import numpy as np
from sim.PTM import *
from sim.SEC import *
from sim.SEC_opt_macros import *
from sim.canonical_max_scc import *
from sim.espresso import espresso_get_SOP_area

def test_canonical_max_scc():
    K1 = np.array([
        [False, False, False, True], #1
        [False, True, False, False], #1
        [True, False, True, False], #2
        [True, True, False, True]  #3
    ])
    K2 = np.array([
        [False, True, True, True], #3
        [True, True, False, True], #3
        [False, True, False, True], #2
        [False, False, False, True]  #1
    ])
    ptm_orig = Ks_to_Mf([K1, K2])
    ptm_opt_old = Ks_to_Mf([opt_K_max(K1), opt_K_max(K2)])
    ptm_opt_new = Ks_to_Mf(opt_K_max_area_aware_multi([K1, K2]))
    f = lambda *x: (canonical_max_scc(*x, K=K1, nv=2), canonical_max_scc(*x, K=K2, nv=2))
    ptm_canonical = get_func_mat(f, 4, 2)

    io = IO_Params(2, 2, 2)
    print(get_Ks_from_ptm(ptm_canonical, io))
    print(test_avg_corr(ptm_orig, ptm_opt_old, xfunc_uniform(2), 10000, io))
    print(test_avg_corr(ptm_orig, ptm_opt_new, xfunc_uniform(2), 10000, io))
    print(test_avg_corr(ptm_orig, ptm_canonical, xfunc_uniform(2), 10000, io))

    print(espresso_get_SOP_area(ptm_orig, "canonical.in"))
    print(espresso_get_SOP_area(ptm_opt_old, "canonical.in"))
    print(espresso_get_SOP_area(ptm_opt_new, "canonical.in"))
    print(espresso_get_SOP_area(ptm_canonical, "canonical.in"))