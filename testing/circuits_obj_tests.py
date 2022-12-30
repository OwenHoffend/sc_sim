from sim.circuits_obj import *
from sim.SEC import *
from sim.PTM import B_mat
import sim.bitstreams as bs
from sim.espresso import espresso_get_SOP_area

import matplotlib.pyplot as plt

all_4_precision_consts = [
    0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625, 0.5625, 0.3125, 0.8125, 0.1875, 0.6875, 0.4375, 0.9375
]

all_3_precision_consts = [
    0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875
]

def test_parallel_func():
    AND = Circuit(np.bitwise_and, 2, 1)
    OR = Circuit(np.bitwise_or, 2, 1)
    test_parallel = ParallelCircuit([AND, OR])
    print(test_parallel.eval(True, False, True, False))
    print(test_parallel.ptm() @ B_mat(2))

def test_series_func():
    AND = Circuit(np.bitwise_and, 2, 1)
    NOT = Circuit(np.bitwise_not, 1, 1)
    test_series = SeriesCircuit([AND, NOT, NOT, NOT])
    print(test_series.ptm())

def test_parallel_and_series():
    mux = MUX()
    print(mux.ptm())
    print(mux.n)
    print(mux.k)
    print(mux.eval(False, False, False)) #S, x2, x1
    print(mux.eval(True, False, False))
    print(mux.eval(False, True, False))
    print(mux.eval(True, True, False))
    print(mux.eval(False, False, True))
    print(mux.eval(True, False, True))
    print(mux.eval(False, True, True))
    print(mux.eval(True, True, True))

def test_sim_AND():
    cir = AND()
    rng = bs.SC_RNG()
    px = 0.5
    py = 0.75
    N = 8192
    X = rng.bs_lfsr(N, px, keep_rng=False)
    Y = rng.bs_lfsr(N, py, keep_rng=False)
    print(bs.bs_mean(cir.eval(X, Y), bs_len=N))

def test_parallel_AND():
    #Share one const 0.5 input between to AND gates
    mapping = [0, 2, 1, 2]
    cir = SeriesCircuit([BUS(3, 4, mapping, nc=1), ParallelCircuit([AND(), AND()])])
    K1, K2 = get_K_2outputs_old(cir)
    K1_opt, K2_opt = opt_K_zero(K1, K2)
    print("Area before: ", espresso_get_SOP_area(Ks_to_Mf([K1_opt, K2_opt]), "test.in", do_print=True))
    best_ptm = opt_area_SECO(K1_opt, K2_opt, cache_file="test_parallel_AND.json", simulated_annealing=True)
    print("Area before: ", espresso_get_SOP_area(best_ptm, "test.in", do_print=True))

def test_PCC():
    n = 4
    cir = PCC(n)
    ptm = cir.ptm()
    ptm_opt = Ks_to_Mf([opt_K_max(get_K(cir)), ])
    rs = np.array([1.0/(2**n) for _ in range(2 ** n)]) #ptv for lfsr (constant) inputs to pcc

    for x in range(2 ** n):
        xs = np.array([1.0 if i == x else 0.0 for i in range(2 ** n)])
        vin = np.kron(rs, xs)
        vout = ptm.T @ vin
        vout_opt = ptm_opt.T @ vin
        print(vout)
        print(vout_opt)

def test_parallel_PCC():
    """Tests relating to SCC before/after correlation opt on a pair of PCCs"""
    n = 5
    cir = PCC_k(n, 2)
    ptm = cir.ptm()
    K1, K2 = get_K_2outputs_old(cir)
    K1_opt1, K2_opt1 = opt_K_max(K1), opt_K_max(K2)
    K2_opt_n1 = opt_K_min(K2)
    #K1_opt0 = K1_opt1
    #K2_opt0 = np.roll(K2_opt1, 8)
    K1_opt0, K2_opt0 = opt_K_zero(K1, K2)
    nv2, nc2 = K1.shape
    row_sccs0 = np.zeros(nv2)
    row_sccs = np.zeros(nv2)
    for i in range(nv2):
        row_sccs[i] = np.abs(bs.bs_scc(K1[i, :], K2[i, :], bs_len=nc2))
        row_sccs0[i] = np.abs(bs.bs_scc(K1_opt0[i, :], K2_opt0[i, :], bs_len=nc2))
    print(row_sccs)
    print(row_sccs0)
    print(np.mean(row_sccs))
    print(np.mean(row_sccs0))

    #min_area = espresso_get_SOP_area(Ks_to_Mf([K1_opt0, K2_opt0]), 'test.in')
    #r1 = list(get_all_rolled(K1_opt0))
    #r2 = list(get_all_rolled(K2_opt0))
    #for i in range(len(r1)):
    #    k10 = r1[i]
    #    k20 = r2[i]
    #    new_area = espresso_get_SOP_area(Ks_to_Mf([k10, k20]), 'test.in')
    #    if new_area < min_area:
    #        min_area = new_area
    #        K1_opt0 = k10
    #        K2_opt0 = k20
    #print(espresso_get_SOP_area(Ks_to_Mf([K1_opt0, K2_opt0]), 'test.in', do_print=True))

    ptm_opt1 = Ks_to_Mf([K1_opt1, K2_opt1])
    ptm_opt_n1 = Ks_to_Mf([K1_opt1, K2_opt_n1])
    ptm_opt0 = Ks_to_Mf([K1_opt0, K2_opt0])
    B2 = B_mat(2)
    rs = np.array([1.0/(2**n) for _ in range(2 ** n)]) #ptv for lfsr (constant) inputs to pcc
    unopt_sccs = []
    opt1_sccs = []
    opt_n1_sccs = []
    opt0_sccs = []
    for x in range(2 ** n):
        xs = np.array([1.0 if i == x else 0.0 for i in range(2 ** n)])
        for y in range(2 ** n):
            ys = np.array([1.0 if i == y else 0.0 for i in range(2 ** n)])
            vin = reduce(np.kron, [rs, ys, xs])
            vout = ptm.T @ vin
            vout_opt1 = ptm_opt1.T @ vin
            pout = B2.T @ vout
            pout_opt1 = B2.T @ vout_opt1
            vout_opt_n1 = ptm_opt_n1.T @ vin
            pout_opt_n1 = B2.T @ vout_opt_n1
            vout_opt0 = ptm_opt0.T @ vin
            pout_opt0 = B2.T @ vout_opt0
            #print(pout)
            #print(pout_opt1)
            assert np.all(pout == pout_opt1)
            assert np.all(pout == pout_opt_n1)
            assert np.all(pout == pout_opt0)
            unopt_sccs.append(np.abs(get_corr_mat_paper(vout)[0,1]))
            opt1_sccs.append(get_corr_mat_paper(vout_opt1)[0,1])
            opt_n1_sccs.append(get_corr_mat_paper(vout_opt_n1)[0,1])
            opt0scc = np.abs(get_corr_mat_paper(vout_opt0)[0,1])
            opt0_sccs.append(opt0scc)
            if opt0scc == 0:
                print("----found----")
                print(pout)
                print(xs)
                print(ys)

    print(np.mean(unopt_sccs))
    print(np.mean(opt1_sccs))
    print(np.mean(opt_n1_sccs))
    print(np.mean(opt0_sccs))


    print(espresso_get_SOP_area(ptm, 'test.in'))
    print(espresso_get_SOP_area(ptm_opt0, 'test.in'))

def test_logic_reduce():
    cir = LOGIC_REDUCE(2, AND)
    print(cir.ptm())

def test_img_seg_circ():
    """Very specific image segmentation circuit from wk_7_7_22 slides"""
    n = 3 #LFSR precision
    k = 3 #Number of "image pixels" being used

    ##CREATAE THE CIRCUIT
    #Comparators
    comps = PCC_k(n, k, use_maj=True)

    #Minmax
    center = np.floor(k / 2).astype(np.int)
    mappings = 2*[x for x in range(k)] + [center]
    mm_bus = BUS(k, 2*k+1, mappings) 
    mm = ParallelCircuit([I(1), LOGIC_REDUCE(k, OR), LOGIC_REDUCE(k, AND)])
    mm_xor = ParallelCircuit([I(1), XOR()])
    mm_out = SeriesCircuit([mm_bus, mm, mm_xor])

    #NOT gate
    nt = ParallelCircuit([NOT(), I(1)])

    #Output
    cir = SeriesCircuit([comps, mm_out, nt])
    ptm = cir.ptm()
    print(espresso_get_SOP_area(ptm, 'test.in'))

    #0.5 ish correlation, bad area
    K1, K2 = get_K_2outputs_old(cir)
    #print("Number in SEC K1: ", get_num_in_SEC(K1))
    #print("Number in SEC K2: ", get_num_in_SEC(K2))
    K1_opt, K2_opt = opt_K_zero(K1, K2)
    ptm_opt = Ks_to_Mf([K1_opt, K2_opt])
    print(espresso_get_SOP_area(ptm_opt, 'test.in'))
    #opt_area_SECO(K1_opt, K2_opt, cache_file="image_seg.json", print_final_espresso=True, simulated_annealing=True, sort=False)

    #0.6 correlation, better area
    K1_opt, K2_opt = opt_K_max(K1), opt_K_max(K2)
    K2_opt_rolled = np.roll(K2_opt, 1, axis=1)
    ptm_opt2 = Ks_to_Mf([K1_opt, K2_opt_rolled])
    print(espresso_get_SOP_area(ptm_opt2, 'test.in'))
    #opt_area_SECO(K1_opt, K2_opt, cache_file="image_seg.json", print_final_espresso=True, simulated_annealing=True, sort=False)
    
    B2 = B_mat(2)

    ##PERFORM TESTING ON THE CIRCUIT
    assert k == 3 #This section assumes k = 3, need to rewrite it otherwise
    rs = np.array([1.0/(2**n) for _ in range(2 ** n)]) #ptv for lfsr (constant) inputs to pcc
    vals = [x*(2.0 ** -n) for x in range(2 ** n)]
    unopt_sccs = [] 
    opt_sccs = []
    opt_sccs2 = []

    unopt_errs = []
    opt_errs = []
    opt2_errs = []
    for ii, x in enumerate(range(1, 2 ** n)):
        print(vals[ii+1])
        xs = np.array([1.0 if i == x else 0.0 for i in range(2 ** n)])
        for jj, y in enumerate(range(1, 2 ** n)):
            ys = np.array([1.0 if i == y else 0.0 for i in range(2 ** n)])
            for kk, z in enumerate(range(1, 2 ** n)):
                zs = np.array([1.0 if i == z else 0.0 for i in range(2 ** n)])
                vin = reduce(np.kron, [rs, zs, ys, xs])
                vout = ptm.T @ vin
                vout_opt = ptm_opt.T @ vin
                vout_opt2 = ptm_opt2.T @ vin
                pout = B2.T @ vout
                pout_opt = B2.T @ vout_opt
                pout_opt2 = B2.T @ vout_opt2
                px = vals[ii+1]
                py = vals[jj+1] #center
                pz = vals[kk+1]
                ps = [px, py, pz]
                correct = [max(ps) - min(ps), 1-py]
                assert np.all(np.isclose(pout, correct))
                assert np.all(np.isclose(pout, pout_opt))
                assert np.all(np.isclose(pout, pout_opt2))
                unopt_sccs.append(get_corr_mat_paper(vout)[0,1])
                opt_sccs.append(get_corr_mat_paper(vout_opt)[0,1])
                opt_sccs2.append(get_corr_mat_paper(vout_opt2)[0,1])

                #Correlation error
                and_ptm = AND().ptm().T
                pout_and = (and_ptm @ vout)[1]
                pout_opt_and = (and_ptm @ vout_opt)[1]
                pout_opt2_and = (and_ptm @ vout_opt2)[1]
                correct_and = correct[0] * correct[1]
                unopt_errs.append(np.abs(pout_and - correct_and))
                opt_errs.append(np.abs(pout_opt_and - correct_and))
                opt2_errs.append(np.abs(pout_opt2_and - correct_and))

    ##RESULTS
    #SCCs
    print(np.mean(np.abs(unopt_sccs)))
    print(np.mean(np.abs(opt_sccs)))
    print(np.mean(np.abs(opt_sccs2)))
    print(np.std(np.abs(unopt_sccs)))
    print(np.std(np.abs(opt_sccs)))
    print(np.std(np.abs(opt_sccs2)))

    plt.bar("unopt - avg scc", 0.8478, width=0.4, yerr=0.308)
    plt.bar("opt - avg scc", 0.5386, width=0.4, yerr=0.419)
    plt.bar("balanced - avg scc", 0.674, width=0.4, yerr=0.387)
    plt.ylabel("Avg SCC")
    plt.xlabel("Variant")
    plt.title("Avg SCC vs. Variant")
    plt.show()

    #Errs
    print(np.mean(np.abs(unopt_errs)))
    print(np.mean(np.abs(opt_errs)))
    print(np.mean(np.abs(opt2_errs)))
    print(np.std(np.abs(unopt_errs)))
    print(np.std(np.abs(opt_errs)))
    print(np.std(np.abs(opt2_errs)))

    plt.bar("unopt - avg err", 0.091, width=0.4, yerr=0.052)
    plt.bar("opt - avg err", 0.0355, width=0.4, yerr=0.021)
    plt.bar("balanced - avg err", 0.067, width=0.4, yerr=0.047)
    plt.ylabel("Avg Abs Err")
    plt.xlabel("Variant")
    plt.title("Avg Abs Err vs. Variant")
    plt.show()

    #Area
    plt.bar("unopt - area", 421, width=0.4)
    plt.bar("opt - area", 2423, width=0.4)
    plt.bar("balanced - area", 884, width=0.4)
    plt.bar("opt - SA area", 1526, width=0.4)
    plt.bar("balanced - SA area", 884, width=0.4)
    plt.ylabel("Area Cost")
    plt.xlabel("Variant")
    plt.title("Area vs. Variant")
    plt.show()

    #SCC Scatterplots
    vals, counts = np.unique(unopt_sccs, return_counts=True)
    vals_opt, counts_opt = np.unique(opt_sccs, return_counts=True)
    vals_opt2, counts_opt2 = np.unique(opt_sccs2, return_counts=True)

    plt.plot(vals, counts, label="unopt", marker='o')
    plt.plot(vals_opt, counts_opt, label="opt", marker='o')
    plt.plot(vals_opt2, counts_opt2, label="balanced", marker='o')
    plt.legend()
    plt.ylabel("# of input patterns generating SCC")
    plt.xlabel("SCC")
    plt.title("Distribution of output SCCs")
    plt.show()

def test_all_const_VALs():
    for v in CONST_VAL.all_const_vals(4):
        test_sim_CONST_VAL(v, 4, 8)

def test_sim_CONST_VAL(const, precision, lfsr_sz):
    N = 2 ** lfsr_sz
    cir = CONST_VAL(const, precision, bipolar=False)
    rng = bs.SC_RNG()
    consts = rng.bs_lfsr_p5_consts(N, cir.actual_precision, lfsr_sz)
    print(bs.bs_mean(cir.eval(*consts), bs_len=N))

def test_CONST_VAL(): #Needs updating for actual_precision
    precision = 4
    ptv = np.array([1.0/(2**precision) for _ in range(2 ** precision)])
    c1 = CONST_VAL(0.8125, precision, bipolar=False)
    mf1 = c1.ptm()
    print(mf1.T @ ptv)
    c2 = CONST_VAL(0.4875, precision, bipolar=False)
    mf2 = c2.ptm()
    print(mf2.T @ ptv)
    c3 = CONST_VAL(0.5, precision, bipolar=False)
    mf3 = c3.ptm()
    print(mf3)

def test_PARALLEL_CONST():
    precision = 4
    cir = PARALLEL_CONST([0.8125, 0.3125, 0.9375, 0.6875], precision, bipolar=False)
    actual_precision = cir.actual_precision
    ptv = np.array([1.0/(2**actual_precision) for _ in range(2 ** actual_precision)])
    print(B_mat(4).T @ cir.ptm().T @ ptv)

def test_PARALLEL_ADD():
    mux2 = PARALLEL_ADD(2, maj=True)
    px = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) #0.15, 0.35
    ptv = get_vin_mc0(px)
    Mf = mux2.ptm()
    print(B_mat(2).T @ Mf.T @ ptv) #Works

def test_CONST_VAL_sim():
    rng = bs.SC_RNG()
    precision = 4
    lfsr_sz = precision
    N = 2 ** lfsr_sz
    for val in all_4_precision_consts:
        cir = CONST_VAL(val, precision, bipolar=False)
        const_bs = rng.bs_lfsr_p5_consts(N, cir.actual_precision, lfsr_sz, add_zero_state=True)
        const_bs_unpacked = rng.bs_lfsr_p5_consts(N, cir.actual_precision, lfsr_sz, add_zero_state=True, pack=False)
        ptm = cir.ptm()
        z = cir.eval(*list(const_bs))
        z2 = apply_ptm_to_bs(const_bs_unpacked, ptm)
        result = bs.bs_mean(z, bs_len=N)
        result2 = bs.bs_mean(z2, bs_len=N)
        print("Result: {}, Correct: {}".format(result, result == val))
        print("Result2: {}, Correct: {}".format(result2, result2 == val))

def test_PARALLEL_CONST_sim():
    rng = bs.SC_RNG()
    precision = 4
    lfsr_sz = precision
    N = 2 ** lfsr_sz
    for i in range(1000):
        consts = [all_4_precision_consts[i] for i in np.random.randint(len(all_4_precision_consts), size=4)]
        cir = PARALLEL_CONST(consts, precision, bipolar=False)
        const_bs = rng.bs_lfsr_p5_consts(N, cir.actual_precision, lfsr_sz, add_zero_state=True)
        zs = cir.eval(*list(const_bs))
        #print(bs.get_corr_mat_np(np.unpackbits(np.array(zs), axis=1)))
        results = np.array([bs.bs_mean(z, bs_len=N) for z in reversed(zs)])
        print("Result: {}, Correct: {}, Same? {}".format(results, consts, np.all(results == np.array(consts))))
        if not np.all(results == np.array(consts)):
            print("err")
            return

def test_MAC_RELU_small_sim():
    rng = bs.SC_RNG()
    lfsr_sz = 12
    N = 2 ** lfsr_sz
    mac_relu = MAC_RELU([0.3125, 0.375, 0.0], [0.3125, 0.375], 4, relu=False)
    consts = rng.bs_lfsr_p5_consts(N, mac_relu.actual_precision + 2, lfsr_sz)
    X = [rng.bs_lfsr(N, 0.5, keep_rng=False) for _ in range(5)]

    inputs = list(consts) + X
    results = mac_relu.eval(*inputs)
    print(bs.bs_mean(results[0], bs_len=N))
    print(bs.bs_mean(results[1], bs_len=N))

def test_MAC_RELU_sim(consts_pos, consts_neg, precision, lfsr_sz=14, relu=False):
    rng = bs.SC_RNG()
    N = 2 ** lfsr_sz
    mac_relu = MAC_RELU(consts_pos, consts_neg, precision, relu=relu)
    consts = rng.bs_lfsr_p5_consts(N, mac_relu.actual_precision + mac_relu.depth, lfsr_sz, add_zero_state=True)
    #px = np.random.uniform(size=mac_relu.width) #Random input probabilities
    px = np.array([0.125, 0.8125, 0.9875, 0.8125, 0.0625]) #Pre-specified input probabilties
    #px = np.array([0.0625, 0.8125, 0.0625, 0.8125]) #Pre-specified input probabilties
    bsx = [rng.bs_lfsr(N, p, keep_rng=False, add_zero_state=True) for p in px]
    print(bs.get_corr_mat_np(np.unpackbits(np.vstack((np.array(bsx), consts)), axis=1))) #Test correlation of bsx with consts
    inputs = list(consts) + bsx
    #results_mul = mac_relu.mul_circuit.eval(*inputs)
    #results_pre_add = mac_relu.add_input_test.eval(*inputs)
    results = mac_relu.eval(*inputs)
    correct_pos = (2.0 ** (-mac_relu.depth)) * px[:len(consts_pos)] @ np.array(list(consts_pos))
    correct_neg = (2.0 ** (-mac_relu.depth)) * px[len(consts_pos):] @ np.array(list(consts_neg))

    if relu: #Evaluate based on ReLU
        pass
    else:
        #actual_mul = [bs.bs_mean(m, bs_len=N) for m in results_mul]
        #print(actual_mul)
        #actual_pre_add = [bs.bs_mean(m, bs_len=N) for m in results_pre_add]
        #print(actual_pre_add)
        actual_pos = bs.bs_mean(results[0], bs_len=N)
        actual_neg = bs.bs_mean(results[1], bs_len=N)
        err_p = np.abs(correct_pos - actual_pos)
        err_n = np.abs(correct_neg - actual_neg)
        print("Correct pos: {}, Actual pos: {}, Error: {}, Good? {}".format(correct_pos, actual_pos, 
            err_p, err_p <= (2.0 ** -precision), 3))
        print("Correct neg: {}, Actual neg: {}, Error: {}, Good? {}".format(correct_neg, actual_neg, 
            err_n, err_n <= (2.0 ** -precision), 3))