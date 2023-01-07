import numpy as np
from sim.espresso import *
from sim.SEC import *
from sim.HMT import *
from sim.SEC_opt_macros import *

def test_espresso():
    cir = PARALLEL_MAC_2([0.125, 0.875, 0.875, 0.125], 4, bipolar=False)
    #cir = PARALLEL_ADD(2)
    K1, K2 = get_K_2outputs_old(cir)

    inames = ['x1', 'x2', 'x3', 'x4', 'c1', 'c2', 'c3', 's']
    onames = ['z1', 'z2']
    original_cost = espresso_get_SOP_area(cir.ptm(), "mux.in", do_print=True)
    print("Original cost " + str(original_cost) + "\n\n")
    costs = []
    min_cost = None
    K1_opt, K2_opt = opt_K_max(K1), opt_K_max(K2)

    #Area optimization by simulated annealing
    best_ptm = opt_area_SECO(K1_opt, K2_opt, cache_file="2x2_kernel.json", print_final_espresso=True, simulated_annealing=True, sort=False)

    #Area optimization by rolling
    for (K_opt1, K_opt2) in zip(get_all_rolled(K1_opt), get_all_rolled(K2_opt)):
        ptm_opt = Ks_to_Mf([K_opt1, K_opt2])
        cost = espresso_get_SOP_area(ptm_opt, "mux_opt.in", do_print=True)
        if min_cost is None or cost < min_cost:
            min_cost = cost
        print("Cost " + str(cost) + "\n\n")
        costs.append(cost)
    xnames = [str(x) for x in range(len(costs))]
    plt.bar("Orig. \n cost", original_cost, color='orange')
    plt.bar(xnames, costs, color='blue')
    plt.show()

    print(original_cost)
    print(min_cost)

def test_espresso_2output():
    #Test of the function f1=x2+x1, f2=!x2+x1. Espresso should share logic between these
    ptm = np.array([
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [0,0,0,1]
    ]
    )
    cost = espresso_get_SOP_area(ptm, "2output_test.in", do_print=True)
    print(cost)

def test_espresso_get_opt_file():
    gb4_opt_a_ptm = np.load("gb4_opt_a_ptm.npy")
    espresso_get_opt_file(gb4_opt_a_ptm, "gb4_opt_a.in", "gb4_opt_a.out")


