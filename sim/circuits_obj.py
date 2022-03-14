import numpy as np
from functools import reduce
from sim.PTM import get_func_mat

#Object-oriented version of circuits.py
class Circuit:
    def __init__(self, func, n, k, nc=0):
        self.func = func
        self.n = n
        self.k = k
        self.nc = nc
        self.nv = n - nc

    def eval(self, *args):
        assert len(args) == self.n
        results = self.func(*args)
        if isinstance(results, tuple):
            assert len(results) == self.k
        else:
            assert self.k == 1
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
        nc = sum([c.nc for c in circuits])
        super().__init__(func, n, k, nc=nc)

    def ptm(self): #override
        sub_ptms = [c.ptm() for c in self.subcircuits]
        return reduce(lambda a, b: np.kron(a, b), sub_ptms)

class SeriesCircuit(Circuit):
    def _series_func(func1, func2):
        #assert k1 == n2
        def new_func(*args):
            return func2(*func1(*args))
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
        super().__init__(func, circuits[0].n, circuits[-1].k, nc=circuits[0].nc)

    def ptm(self): #override
        sub_ptms = [c.ptm() for c in self.subcircuits]
        return reduce(lambda a, b: a @ b, sub_ptms)

#CIRCUIT LIBRARY
class AND(Circuit):
    def __init__(self, nc=0):
        super().__init__(np.bitwise_and, 2, 1, nc=nc)

class OR(Circuit):
    def __init__(self, nc=0):
        super().__init__(np.bitwise_or, 2, 1, nc=nc)

class XOR(Circuit):
    def __init__(self, nc=0):
        super().__init__(np.bitwise_xor, 2, 1, nc=nc)

class NOT(Circuit): #Propagates constants
    def __init__(self, nc=0):
        super().__init__(np.bitwise_not, 1, 1, nc=nc)

class BUS(Circuit):
    def __init__(self, n, k, mappings, nc=0):
        def func(*args):
            assert len(args) == n
            assert len(mappings) == k
            out = []
            for m in mappings:
                if m < 0: # use as shorthand for NOT
                    out.append(np.bitwise_not(args[-m]))
                else:
                    out.append(args[m])
            if len(out) == 1:
                return out[0]
            return tuple(out)
        super().__init__(func, n, k, nc=nc)

class I(BUS): #Special bus that just passes the input to the output
    def __init__(self, width, nc=0):
        super().__init__(width, width, [x for x in range(width)], nc=nc)

class FANOUT(BUS):
    def __init__(self, n, num_dup, nc=0):
        super().__init__(n, n * num_dup, [i % n for i in range(n * num_dup)], nc=nc)

class MUX(SeriesCircuit):
    def __init__(self):
        layers = [
            BUS(3, 4, [0, 2, 1, -2], nc=1),
            ParallelCircuit([AND(), AND()]),
            OR()
        ]
        super().__init__(layers)

class PARALLEL_MUX(SeriesCircuit):
    def __init__(self, iters):
        #bus s1 s2 x1 x2 x3 x4 --> x1 x2 s1 x3 x4 s2, etc
        bus_mapping = []
        idx = 0
        for i in range(3*iters):
            if i % 3 != 2:
                bus_mapping.append(idx)
                idx += 1
            else:
                bus_mapping.append(2*iters)
        layers = [
            BUS(2 * iters + 1, 3 * iters, bus_mapping, nc=1),
            ParallelCircuit([MUX() for _ in range(iters)])
        ]
        super().__init__(layers)

class MUX_TREE(SeriesCircuit):
    def __init__(self, n):
        #For now, this only accepts powers of 2
        num_layers = np.log2(n)
        #Check that n is a power of 2

class CONST_VAL(SeriesCircuit):
    #Circuit generates a constant from a number of 0.5 const inputs
    def __init__(self, radix_bits):
        #radix_bits is an array such as [0, 1, 0, 1] corresponding to binary radix 0.0101, etc.
        precision = len(radix_bits)
        assert radix_bits[-1] #Last entry needs to be True, no trailing zeros
        #A couple special cases
        if precision == 1:
            super().__init__([I(1, nc=1)])
            return
        #Main cases
        layers = []
        for bit in radix_bits[:-1]:
            if bit: #Add an OR gate
                if precision == 2:
                    layers.append(OR(nc=2))
                else:
                    layers.append(ParallelCircuit([OR(nc=2), I(precision-2, nc=precision-2)]))
            else: #Add an AND gate
                if precision == 2:
                    layers.append(AND(nc=2))
                else:
                    layers.append(ParallelCircuit([AND(nc=2), I(precision-2, nc=precision-2)]))
            precision -= 1
        super().__init__(layers)

class PARALLEL_CONST(SeriesCircuit):
    def __init__(self, radix_bit_mat):
        width, precision = radix_bit_mat.shape
        layers = [
            FANOUT(precision, width, nc=precision),
            ParallelCircuit([CONST_VAL(row) for row in radix_bit_mat])
        ]
        super().__init__(layers)

class PARALLEL_CONST_MUL(SeriesCircuit):
    def __init__(self, radix_bit_mat):
        width, _ = radix_bit_mat.shape
        mappings = [x for x in range(0, 2*width, 2)] + [x for x in range(1, 2*width, 2)]
        layers = [
            ParallelCircuit([PARALLEL_CONST(radix_bit_mat), I(width)]),
            BUS(2*width, 2*width, mappings),
            ParallelCircuit([XOR(), XOR()])
        ] 
        super().__init__(layers)

class MAC_CONSTS(SeriesCircuit):
    def __init__(self, radix_bit_mat):
        pass