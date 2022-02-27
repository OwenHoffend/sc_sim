import numpy as np
from functools import reduce
from sim.PTM import get_func_mat

#Object-oriented version of circuits.py
class Circuit:
    def __init__(self, func, n, k):
        self.func = func
        self.n = n
        self.k = k

    def eval(self, *args):
        assert len(args) == self.n
        results = self.func(*args)
        assert len(results) == self.k
        return results

    def ptm(self):
        return get_func_mat(self.func, self.n, self.k)

class ParallelCircuit(Circuit):
    def _parallel_func(func1, func2, n1, n2):
        def new_func(*args):
            assert len(args) == n1 + n2
            func1_results = func1(*args[:n1])
            func2_results = func2(*args[n1:])
            if isinstance(func1_results, tuple):
                result_tuple = list(func1_results)
            else:
                result_tuple = [func1_results, ]
            if isinstance(func2_results, tuple):
                result_tuple += list(func2_results)
            else:
                result_tuple.append(func2_results)
            return tuple(result_tuple) 
        return new_func

    def _merge_funcs(funcs, ns):
        ntot = 0
        func = funcs[0]
        ntot = ns[0]
        for f, n in zip(funcs[1:], ns[1:]):
            func = ParallelCircuit._parallel_func(func, f, ntot, n)
            ntot += n
        return func, ntot

    def __init__(self, circuits):
        self.subcircuits = circuits
        funcs = [c.func for c in circuits]
        ns = [c.n for c in circuits]
        func, n = ParallelCircuit._merge_funcs(funcs, ns)
        k = sum([c.k for c in circuits])
        super().__init__(func, n, k)

    def ptm(self): #override
        sub_ptms = [c.ptm() for c in self.subcircuits]
        return reduce(lambda a, b: np.kron(a, b), sub_ptms)

class SeriesCircuit(Circuit):
    def _series_func(func1, func2):
        #assert k1 == n2
        def new_func(*args):
            return func2(func1(*args))
        return new_func

    def _merge_funcs(funcs):
        func = funcs[0]
        for f in funcs[1:]:
            func = SeriesCircuit._series_func(func, f)
        return func

    def __init__(self, circuits):
        self.subcircuits = circuits
        funcs = [c.func for c in circuits]
        func = SeriesCircuit._merge_funcs(funcs)
        super().__init__(func, circuits[0].n, circuits[-1].k)

    def ptm(self): #override
        sub_ptms = [c.ptm() for c in self.subcircuits]
        return reduce(lambda a, b: a @ b, sub_ptms)