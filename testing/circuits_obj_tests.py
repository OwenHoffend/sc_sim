from sim.circuits_obj import *
from sim.SEC import *
from sim.PTM import B_mat
import sim.bitstreams as bs

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

def test_MAC_RELU_old():
    mac_relu = MAC_RELU([0.8125 for _ in range(4)], [0.3125 for _ in range(7)], 4)
    #ptm = mac_relu.ptm()
    print('hi')

def test_MAC_RELU_small():
    rng = bs.SC_RNG()
    lfsr_sz = 12
    N = 2 ** lfsr_sz
    mac_relu = MAC_RELU([0.3125, 0.375], [0.3125, 0.375, 0.0], 4, relu=False)
    consts = rng.bs_lfsr_p5_consts(N, mac_relu.actual_precision + 2, lfsr_sz)
    X = [rng.bs_lfsr(N, 0.5, keep_rng=False) for _ in range(5)]

    inputs = list(consts) + X
    results = mac_relu.eval(*inputs)
    print(bs.bs_mean(results[0], bs_len=N))
    print(bs.bs_mean(results[1], bs_len=N))
