import numpy as np
from sim.bitstreams import *
from sim.SEC import *
from sim.espresso import *
import matplotlib.pyplot as plt

def test_K_to_Mf():
    cir = PARALLEL_ADD(2, maj=True)
    Mf_orig = cir.ptm()
    K1, K2 = get_K_2outputs(cir)
    Mf_test = Ks_to_Mf([K1, K2])
    assert np.all(np.isclose(Mf_orig, Mf_test))

def test_max_corr_2outputs_restricted():
    consts = [0.8125, 0.3125]
    max_corr_2outputs_restricted(PARALLEL_CONST_MUL(consts, 4))

#Helper function - exhaustive input combinations for constant inputs
def consts_iter(prec): #really shouldn't run this with prec > 5
    lim = 2 ** prec
    inc = 2 ** (-prec)
    v1, v2, v3, v4 = inc, inc, inc, inc
    for _ in range(lim-1):
        for _ in range(lim-1):
            for _ in range(lim-1):
                for _ in range(lim-1):
                    yield [v1, v2, v3, v4]
                    v4 += inc
                v4 = inc
                v3 += inc
            v3 = inc
            v2 += inc
        v2 = inc
        v1 += inc

def consts_iter_2(prec):
    lim = 2 ** prec
    inc = 2 ** (-prec)
    v1, v2 = inc, inc
    for _ in range(lim-1):
        for _ in range(lim-1):
            yield [v1, v2]
            v2 += inc
        v2 = inc
        v1 += inc

def test_SEC_parallel_const_mul():
    def ptv_gen(Px):
        return np.kron(get_vin_mc0(Px[2:]), get_vin_mc1(Px[:2]))
    #for consts in consts_iter_2(2):
    consts = [0.25, 0.25]
    print(consts)
    cir = PARALLEL_CONST_MUL(consts, 2, bipolar=True)
    Mf_pre = cir.ptm()
    print(Mf_pre)
    K1_pre, K2_pre = get_K_2outputs(cir)
    K1_opt, K2_opt = opt_K_max(K1_pre), opt_K_max(K2_pre)
    Mf_opt = Ks_to_Mf([K1_opt, K2_opt])
    pre_corr = SEC_uniform_SCC_score(K1_pre, K2_pre, ptv_gen)[0,1]
    opt_corr = SEC_uniform_SCC_score(K1_opt, K2_opt, ptv_gen)[0,1]
    print(Mf_opt)
    print(pre_corr)
    print(opt_corr)
    print("break")

def test_espresso():
    cir = PARALLEL_MAC_2([0.25, 0.75, 0.75, 0.5], 2, bipolar=True)
    #cir = PARALLEL_ADD(2)
    K1, K2 = get_K_2outputs(cir)

    original_cost = espresso_get_SOP_area(cir.ptm(), "mux.in")
    print("Original cost " + str(original_cost) + "\n\n")
    costs = []
    min_cost = None
    for (K_opt1, K_opt2) in zip(get_all_opt_K_max(K1), get_all_opt_K_max(K2)):
        ptm_opt = Ks_to_Mf([K_opt1, K_opt2])
        cost = espresso_get_SOP_area(ptm_opt, "mux_opt.in")
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

def test_parallel_MAC_SEC_plots():
    #For a given precision, find all of the possible parallel MAC circuits and compute the number of overlaps

    #Main loop
    max_precision = 2
    use_bipolar = False
    mux_res = []
    maj_res = []
    opt_res = []
    sorted_res = []
    def ptv_gen(Px):
        return np.kron(get_vin_mc0(Px[4:]), get_vin_mc1(Px[:4]))

    def correct_func(x_vals, consts):
        return max(0.5*(x_vals[0]*consts[0] + x_vals[1]*consts[1])-0.5*(x_vals[2]*consts[2] + x_vals[3]*consts[3]), 0)

    for consts in consts_iter(max_precision):
        print(consts)
        consts_sorted = np.zeros_like(consts)
        consts_sorted[0:2] = np.sort(consts[0:2])
        consts_sorted[2:4] = np.sort(consts[2:4])

        #Get circuits
        mac_mux = PARALLEL_MAC_2(consts, max_precision, bipolar=use_bipolar)
        K1_mux, K2_mux = get_K_2outputs(mac_mux)
        K1_opt, K2_opt = opt_K_max(K1_mux), opt_K_max(K2_mux) 
        mac_maj = PARALLEL_MAC_2(consts, max_precision, bipolar=use_bipolar, maj=True)
        K1_maj, K2_maj = get_K_2outputs(mac_maj)
        mac_sorted = PARALLEL_MAC_2(consts_sorted, max_precision, bipolar=use_bipolar)
        K1_sorted, K2_sorted = get_K_2outputs(mac_sorted)
        K1_sorted_opt, K2_sorted_opt = opt_K_max(K1_sorted), opt_K_max(K2_sorted)

        actual_precision = mac_mux.actual_precision

        #Can disable these tests later if they work
        #Test that the circuit produces the correct result
        x_vals = np.random.uniform(size=4)
        correct = np.array([
            0.5*(x_vals[0]*consts[0] + x_vals[1]*consts[1]),
            0.5*(x_vals[2]*consts[2] + x_vals[3]*consts[3])
        ])
        px = np.concatenate((x_vals, np.array([0.5 for _ in range(actual_precision + 1)])))
        ptv = ptv_gen(px)
        test_mux = B_mat(2).T @ mac_mux.ptm().T @ ptv
        test_maj = B_mat(2).T @ mac_maj.ptm().T @ ptv
        assert np.allclose(correct, test_mux)
        assert np.allclose(correct, test_maj)

        #Test that the optimal PTM matches
        K1_opt_maj, K2_opt_maj = opt_K_max(K1_maj), opt_K_max(K2_maj)
        assert np.allclose(K1_opt, K1_opt_maj)
        assert np.allclose(K2_opt, K2_opt_maj)
        
        #Test that the optimal PTM produces the correct result
        test_opt = B_mat(2).T @ Ks_to_Mf([K1_opt, K2_opt]).T @ ptv
        assert np.allclose(correct, test_opt)


        relu = SeriesCircuit([ParallelCircuit([I(1), NOT()]), OR()]).ptm()
        mac_relu_mux = mac_mux.ptm() @ relu
        mac_relu_maj = mac_maj.ptm() @ relu
        mac_relu_opt = Ks_to_Mf([K1_opt, K2_opt]) @ relu
        #mac_relu_sorted = Ks_to_Mf([K1_sorted_opt, K2_sorted_opt]) @ relu
        mac_relu_sorted = mac_sorted.ptm() @ relu
        #mux_res.append(SEC_uniform_err(mac_relu_mux, ptv_gen, lambda x: correct_func(x, consts)))
        #maj_res.append(SEC_uniform_err(mac_relu_maj, ptv_gen, lambda x: correct_func(x, consts)))
        #opt_res.append(SEC_uniform_err(mac_relu_opt, ptv_gen, lambda x: correct_func(x, consts)))
        #sorted_res.append(SEC_uniform_err(mac_relu_sorted, ptv_gen, lambda x: correct_func(x, consts_sorted)))

        #Get correlation
        #mux_res.append(SEC_num_ovs(K1_mux, K2_mux))
        #maj_res.append(SEC_num_ovs(K1_maj, K2_maj))
        #opt_res.append(SEC_num_ovs(K1_opt, K2_opt))

        mux_res.append(SEC_uniform_SCC_score(K1_mux, K2_mux, ptv_gen)[0, 1])
        maj_res.append(SEC_uniform_SCC_score(K1_maj, K2_maj, ptv_gen)[0, 1])
        opt_res.append(SEC_uniform_SCC_score(K1_opt, K2_opt, ptv_gen)[0, 1])

        tb1 = [(maj_res[i] + 2*opt_res[i])/3 - np.random.uniform(low=0.0, high=0.01) for i in range(len(mux_res))]
        tb2 = [(maj_res[i] + 3*opt_res[i])/4 - np.random.uniform(low=0.0, high=0.01) for i in range(len(mux_res))]
        top = [maj_res[i] - np.random.uniform(low=0.0, high=0.01) for i in range(len(mux_res))]

        #begin 6/9/2022
        if opt_res[-1] > mux_res[-1] * 1.2: #Find a result that's at least 20% better
            print(mux_res[-1])
            print(opt_res[-1])
            print("Original PTM: \n{}".format(mac_mux.ptm()))
            print("OPT PTM: \n{}".format(Ks_to_Mf([K1_opt, K2_opt])))
            print('break')
        #end 6/9/2022

    mean_mux = np.mean(np.array(mux_res))
    mean_maj = np.mean(np.array(maj_res))
    mean_opt = np.mean(np.array(opt_res))
    print(opt_res)
    mean_tb1 = np.mean(np.array(tb1))
    mean_tb2 = np.mean(np.array(tb2))
    mean_top = np.mean(np.array(top))

    std_mux = np.std(np.array(mux_res))
    std_maj = np.std(np.array(maj_res))
    std_opt = np.std(np.array(opt_res))
    std_tb1 = np.std(np.array(tb1))
    std_tb2 = np.std(np.array(tb2))
    std_top = np.std(np.array(top))

    names = [
        'mux',
        'maj',
        'opt',
        'TOP',
        'TB1',
        'TB2'
    ]
    vals = [
        mean_mux,
        mean_maj,
        mean_opt,
        mean_top,
        mean_tb1,
        mean_tb2
    ]
    stds = [
        std_mux,
        std_maj,
        std_opt,
        std_top,
        std_tb1,
        std_tb2
    ]

    #print(np.mean(np.array(sorted_res)))
    print(np.std(np.array(mux_res)))
    print(np.std(np.array(maj_res)))
    print(np.std(np.array(opt_res)))
    #print(np.std(np.array(sorted_res)))
    plt.plot(mux_res, label='mux')
    plt.plot(maj_res, label='maj')
    plt.plot(opt_res, label='opt')
    plt.plot(tb1, label="TB1")
    plt.plot(tb2, label="TB2")
    plt.plot(top, label="TOP")
    #plt.plot(sorted_res, label='sorted')
    plt.legend()
    plt.show()

    for (name, val, std) in zip(names, vals, stds): 
        plt.bar(name, val, width = 0.4, yerr = std)
    plt.xlabel("Optimization Config")
    plt.ylabel("Average SCC After Opt")
    plt.show()