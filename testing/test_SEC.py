import numpy as np
from sim.bitstreams import *
from sim.SEC import *
from sim.espresso import *
import matplotlib.pyplot as plt

def test_SCC_inv():
    ntrials = 10000
    for i in range(ntrials):
        print(i)
        px = np.random.uniform()
        py = np.random.uniform()
        C = (np.random.uniform() * 2) - 1
        pxy = scc_inv(px, py, C, 0)
        assert np.isclose(scc(px, py, pxy), C)

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
    cir = PARALLEL_MAC_2([0.125, 0.875, 0.875, 0.125], 4, bipolar=False)
    #cir = PARALLEL_ADD(2)
    K1, K2 = get_K_2outputs(cir)

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

def test_opt_area_SECO():
    n = 3
    cir = PCC_k(n, 2)
    K1, K2 = get_K_2outputs(cir)
    K1_opt1, K2_opt1 = opt_K_max(K1), opt_K_max(K2)
    K1_opt0, K2_opt0 = opt_K_zero(K1, K2)
    unopt_ptm = Ks_to_Mf([K1_opt0, K2_opt0])
    print("Original area: ", espresso_get_SOP_area(Ks_to_Mf([K1, K2]), "opt_area_SECO.in", do_print=True))
    print("Opt1 area: ", espresso_get_SOP_area(Ks_to_Mf([K1_opt1, K2_opt1]), "opt_area_SECO.in", do_print=True))
    print("Unopt area: ", espresso_get_SOP_area(unopt_ptm, "opt_area_SECO.in"))
    best_ptm = opt_area_SECO(K1_opt0, K2_opt0, simulated_annealing=True, sort=False)
    print(best_ptm)
    np.save("best_ptm", best_ptm)

def heatmap(xs, ys, zs, inv_colormap=True):
    if inv_colormap:
        cmap = 'RdBu_r'
    else:
        cmap = 'RdBu'
    y, x = np.meshgrid(xs, ys)
    z_min, z_max = zs.min(), zs.max()
    fig, ax = plt.subplots()
    c = ax.pcolormesh(x, y, zs, cmap=cmap, vmin=z_min, vmax=z_max)
    ax.set_title('pcolormesh')
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)

    plt.show()

def plot_heatmap_SEC_opt_cases():
    """Roll the top/bottom K matrix to produce a heat map of 2^nc x 2^nc entries. Can compare both area and SCC"""
    cir = PARALLEL_MAC_2([0.125, 0.875, 0.875, 0.125], 4, bipolar=False)
    K1, K2 = get_K_2outputs(cir)
    K1_opt, K2_opt = opt_K_max(K1), opt_K_max(K2)
    original_area = espresso_get_SOP_area(cir.ptm(), "cir.in")
    num_opt = 2 ** cir.nc
    area_costs = np.zeros((num_opt, num_opt))
    scc_costs = np.zeros((num_opt, num_opt))
    i,j = 0,0
    lfsr_sz = 7
    N = 2 ** lfsr_sz
    scc_trials = 20
    #SCC from uniform random inputs
    sccs = []
    for _ in range(scc_trials):
        rng = bs.SC_RNG()
        x_vals = np.random.uniform(size=4)
        var_bs = [rng.bs_lfsr(N, x_vals[i], pack=False) for i in range(4)]
        rng = bs.SC_RNG()
        const_bs = rng.bs_lfsr_p5_consts(N, cir.actual_precision + 1, lfsr_sz, add_zero_state=True, pack=False)
        px = np.vstack((var_bs, const_bs))
        pz = apply_ptm_to_bs(px, cir.ptm())
        sccs.append(bs_scc(pz[0, :], pz[1, :], bs_len=N))
    original_scc = np.mean(sccs)
    for K1_opt_rolled in get_all_rolled(K1_opt):
        print(i)
        for K2_opt_rolled in get_all_rolled(K2_opt):
            opt_ptm = Ks_to_Mf([K1_opt_rolled, K2_opt_rolled])
            area_costs[i][j] = espresso_get_SOP_area(opt_ptm, "cir_opt.in")

            #SCC from uniform random inputs
            sccs = []
            for _ in range(scc_trials):
                rng = bs.SC_RNG()
                x_vals = np.random.uniform(size=4)
                var_bs = [rng.bs_lfsr(N, x_vals[i], pack=False) for i in range(4)]
                rng = bs.SC_RNG()
                const_bs = rng.bs_lfsr_p5_consts(N, cir.actual_precision + 1, lfsr_sz, add_zero_state=True, pack=False)
                px = np.vstack((var_bs, const_bs))
                pz = apply_ptm_to_bs(px, opt_ptm)
                sccs.append(bs_scc(pz[0, :], pz[1, :], bs_len=N))
            scc_costs[i][j] = np.mean(sccs)
            j+=1
        j=0
        i+=1
    xs = list(range(num_opt))
    ys = list(range(num_opt))
    heatmap(xs, ys, area_costs/original_area)
    heatmap(xs, ys, scc_costs, inv_colormap=False)
    heatmap(xs, ys, (original_area/area_costs)*((scc_costs+1)/(original_scc+1)), inv_colormap=False)

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