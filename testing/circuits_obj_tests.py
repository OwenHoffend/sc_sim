from sim.circuits_obj import *
from sim.SEC import *
from sim.PTM import B_mat
import sim.bitstreams as bs
from sim.espresso import espresso_get_SOP_area

all_4_precision_consts = [
    0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625, 0.5625, 0.3125, 0.8125, 0.1875, 0.6875, 0.4375, 0.9375
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
    n = 4
    cir = PCC_2(n)
    ptm = cir.ptm()
    K1, K2 = get_K_2outputs(cir)
    K1_opt1, K2_opt1 = opt_K_max(K1), opt_K_max(K2)
    K2_opt_n1 = opt_K_min(K2)
    K1_opt0, K2_opt0 = opt_K_zero(K1, K2)
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
    for x in range(1, 2 ** n):
        xs = np.array([1.0 if i == x else 0.0 for i in range(2 ** n)])
        for y in range(1, 2 ** n):
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
            print(pout)
            #print(pout_opt1)
            assert np.all(pout == pout_opt1)
            assert np.all(pout == pout_opt_n1)
            assert np.all(pout == pout_opt0)
            unopt_sccs.append(get_corr_mat_paper(vout)[0,1])
            opt1_sccs.append(get_corr_mat_paper(vout_opt1)[0,1])
            opt_n1_sccs.append(get_corr_mat_paper(vout_opt_n1)[0,1])
            opt0_sccs.append(get_corr_mat_paper(vout_opt0)[0,1])

    print(np.mean(unopt_sccs))
    print(np.mean(opt1_sccs))
    print(np.mean(opt_n1_sccs))
    print(np.mean(opt0_sccs))

    print(espresso_get_SOP_area(Ks_to_Mf([K1]), 'test.in', do_print=True))
    print(espresso_get_SOP_area(Ks_to_Mf([K1_opt1]), 'test.in', do_print=True))
    print(espresso_get_SOP_area(Ks_to_Mf([K2_opt1]), 'test.in', do_print=True))
    print(espresso_get_SOP_area(Ks_to_Mf([K2_opt_n1]), 'test.in', do_print=True))
    print(espresso_get_SOP_area(Ks_to_Mf([K2_opt0]), 'test.in', do_print=True))
    print('hi')

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