from sim.circuits_obj import *
from sim.SEC import *
from sim.PTM import B_mat

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

def test_CONST_VAL(): #Needs updating for actual_precision
    precision = 4
    ptv = np.array([1.0/(2**precision) for _ in range(2 ** precision)])
    c1 = CONST_VAL(0.8125, precision, bipolar=False)
    mf1 = c1.ptm()
    print(mf1.T @ ptv)
    c2 = CONST_VAL(0.3125, precision, bipolar=False)
    mf2 = c2.ptm()
    print(mf2.T @ ptv)
    c3 = CONST_VAL(0.5, precision, bipolar=False)
    mf3 = c3.ptm()
    print(mf3)

def test_PARALLEL_CONST():
    precision = 4
    cir = PARALLEL_CONST([0.8125, 0.3125, 0.5, 0.75], precision, bipolar=False)
    actual_precision = cir.actual_precision
    ptv = np.array([1.0/(2**actual_precision) for _ in range(2 ** actual_precision)])
    print(B_mat(4).T @ cir.ptm().T @ ptv)

def test_MAC():
    px = np.array([0.2, 0.4, 0.5, 0.5, 0.5]) #Consts are ALWAYS on the right side
    ptv = get_vin_mc0(px)
    mac = MAC([0.75, 0.75], 2, bipolar=False)
    Mf = mac.ptm()
    print(Mf.T @ ptv)

def test_PARALLEL_ADD():
    mux2 = PARALLEL_ADD(2, maj=True)
    px = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) #0.15, 0.35
    ptv = get_vin_mc0(px)
    Mf = mux2.ptm()
    print(B_mat(2).T @ Mf.T @ ptv) #Works

def test_PARALLEL_CONST_MUL():
    mac2 = PARALLEL_CONST_MUL([0.25, 0.25, 0.75, 0.75], 2, bipolar=False)
    px = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.5])
    ptv = get_vin_mc0(px)
    Mf = mac2.ptm()
    print(B_mat(4).T @ Mf.T @ ptv)

def test_PARALLEL_MAC_2():
    mac2 = PARALLEL_MAC_2([0.25, 0.25, 0.75, 0.75], 2, bipolar=False)
    px = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5])
    ptv = get_vin_mc0(px)
    Mf = mac2.ptm()
    print(B_mat(2).T @ Mf.T @ ptv)
