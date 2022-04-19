from audioop import mul
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
        self.ptm_cache = None

    def eval(self, *args):
        assert len(args) == self.n
        results = self.func(*args)
        if isinstance(results, tuple):
            assert len(results) == self.k
        else:
            assert self.k == 1
        return results

    def ptm(self):
        if self.ptm_cache is None:
            self.ptm_cache = get_func_mat(self.func, self.n, self.k)
        return self.ptm_cache

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

class MAJ(SeriesCircuit):
    def __init__(self):
        layers = [
            BUS(3, 6, [0, 1, 0, 2, 1, 2], nc=1),
            ParallelCircuit([AND(), AND(), AND()]),
            ParallelCircuit([OR(), I(1)]),
            OR()
        ]
        super().__init__(layers)

class PARALLEL_ADD(SeriesCircuit):
    def __init__(self, iters, maj=False):
        #input scheme: x, x, x, x, s
        #bus x0, x1, s, x2, x3, s, etc
        bus_mapping = []
        idx = 0
        if maj:
            adds = [MAJ() for _ in range(iters)]
        else:
            adds = [MUX() for _ in range(iters)]
        for i in range(3*iters):
            if i % 3 != 2:
                bus_mapping.append(idx)
                idx += 1
            else:
                bus_mapping.append(2*iters)
        layers = [
            BUS(2 * iters + 1, 3 * iters, bus_mapping, nc=1),
            ParallelCircuit(adds)
        ]
        super().__init__(layers)

class CONST_VAL(SeriesCircuit):
    #Circuit generates a constant from a number of 0.5 const inputs
    def __init__(self, val, precision, bipolar=True):
        assert val != 0
        assert val != 1

        #One bit of precision is trivial
        if precision == 1:
            super().__init__([I(1, nc=1)])
            return
            
        #Correct bipolar encoding
        #if bipolar:
        #    val = (val + 1) / 2

        #Convert val into radix bits
        #radix_bits is an array such as [0, 1, 0, 1] corresponding to binary radix 0.0101, etc.
        radix_bits = np.zeros(precision, dtype=np.bool_)
        cmp = 0.5
        for i in range(precision):
            if val >= cmp:
                radix_bits[i] = 1
                val -= cmp
            else:
                radix_bits[i] = 0
            cmp /= 2
        while radix_bits[-1] == 0:
            radix_bits = radix_bits[:-1]
        precision = radix_bits.size
        self.actual_precision = precision

        #Construct the circuit
        layers = []
        if precision == 1:
            layers.append(I(1, nc=1))
            super().__init__(layers)
            return
        radix_bits = radix_bits[:-1]
        for bit in radix_bits[::-1]:
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
    def __init__(self, consts, precision, bipolar=True):
        const_vals = [CONST_VAL(const, precision, bipolar=bipolar) for const in reversed(consts)]
        precisions = [x.actual_precision for x in reversed(const_vals)]
        self.actual_precision = max(precisions)
        sp = sum(precisions)
        mappings = []
        for p in precisions:
            mappings += list(range(p))
        layers = [
            BUS(self.actual_precision, sp, mappings, nc=self.actual_precision),
            ParallelCircuit(const_vals)
        ]
        super().__init__(layers)

class PARALLEL_CONST_MUL(SeriesCircuit):
    def __init__(self, consts, precision, bipolar=True):
        width = len(consts)
        mappings = []
        for i in range(width):
            mappings.append(i)
            mappings.append(i+width)
        if bipolar:
            mul_layer = ParallelCircuit([SeriesCircuit([XOR(), NOT()]) for _ in range(width)])
        else:
            mul_layer = ParallelCircuit([AND() for _ in range(width)])
        parallel_const = PARALLEL_CONST(consts, precision, bipolar=bipolar)
        layers = [
            ParallelCircuit([parallel_const, I(width)]),
            BUS(2*width, 2*width, mappings),
            mul_layer
        ]
        self.actual_precision = parallel_const.actual_precision
        super().__init__(layers)

class MAC(SeriesCircuit):
    def __init__(self, consts, precision, bipolar=True):
        const_mul = PARALLEL_CONST_MUL(consts, precision, bipolar=bipolar)
        layers = [
            ParallelCircuit([const_mul, I(1, nc=1)]),
            MUX()
        ]
        self.actual_precision = const_mul.actual_precision
        super().__init__(layers)

class PARALLEL_MAC_2(SeriesCircuit):
    def __init__(self, consts, precision, bipolar=True, maj=False):
        assert len(consts) == 4
        const_mul = PARALLEL_CONST_MUL(consts, precision, bipolar=bipolar)
        layers = [
            ParallelCircuit([I(1, nc=1), const_mul]),
            PARALLEL_ADD(2, maj=maj)
        ]
        self.actual_precision = const_mul.actual_precision
        super().__init__(layers)