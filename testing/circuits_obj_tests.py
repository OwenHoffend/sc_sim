from sim.circuits_obj import *
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
