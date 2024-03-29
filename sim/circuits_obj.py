import numpy as np
from functools import reduce
from sim.PTM import get_func_mat

#Object-oriented version of circuits.py
class Circuit:
    def __init__(self, func, n, k, nc=0):
        self._ptm_cache = None
        self.func = func
        self.n = n
        self.k = k
        self.nc = nc
        self.nv = n - nc

    def eval(self, *args): #<-- Really shouldn't use, just use the PTM instead
        assert len(args) == self.n
        results = self.func(*args)
        if isinstance(results, tuple):
            assert len(results) == self.k
        else:
            assert self.k == 1
        return results

    def ptm(self):
        if self._ptm_cache is None:
            self._ptm_cache = get_func_mat(self.func, self.n, self.k)
        return self._ptm_cache

    def update_ptm(self, new_ptm): #Allows for dyanmic optimizations
        self._ptm_cache = new_ptm

class ParallelCircuit(Circuit):
    def __init__(self, circuits):
        self.subcircuits = circuits
        funcs = [c.func for c in circuits]
        ns = [c.n for c in circuits]
        func, n = ParallelCircuit._merge_funcs(funcs, ns)
        k = sum([c.k for c in circuits])
        nc = sum([c.nc for c in circuits])
        super().__init__(func, n, k, nc=nc)

    def ptm(self): #override
        if self._ptm_cache is None:
            sub_ptms = [c.ptm() for c in self.subcircuits]
            self._ptm_cache = reduce(lambda a, b: np.kron(a, b), sub_ptms)
        return self._ptm_cache

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

class SeriesCircuit(Circuit):
    def __init__(self, circuits):
        self.subcircuits = circuits
        funcs = [c.func for c in circuits]
        func = SeriesCircuit._merge_funcs(funcs)
        super().__init__(func, circuits[0].n, circuits[-1].k, nc=circuits[0].nc)

    def ptm(self): #override
        if self._ptm_cache is None:
            sub_ptms = [c.ptm() for c in self.subcircuits]
            self._ptm_cache = reduce(lambda a, b: a @ b, sub_ptms)
        return self._ptm_cache
        
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


#CIRCUIT LIBRARY
class CONST_1(Circuit):
    def __init__(self, n=1): #n doesn't affect the output, so allow this module to absorb any number of inputs
        super().__init__(lambda *x: np.array([True]), n, 1, nc=0)

class CONST_0(Circuit):
    def __init__(self, n=1): #n doesn't affect the output, so allow this module to absorb any number of inputs
        super().__init__(lambda *x: np.array([False]), n, 1, nc=0)

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

class LOGIC_REDUCE(SeriesCircuit):
    def __init__(self, n, f):
        assert n >= 2
        layers = []
        for i in range(n-1):
            if n-i > 2:
                layers.append(ParallelCircuit([f(), I(n-i-2)]))
            else:
                layers.append(f())
        super().__init__(layers)

class BUS(Circuit):
    """Takes a list of length k called "mappings", which specifies the source input index for each output of the bus"""
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
        self.actual_precision = precision
        if val == 1.0:
            super().__init__([CONST_1(n=precision)])
            return
        if val == 0.0:
            super().__init__([CONST_0(n=precision)])
            return

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

    @staticmethod
    def all_const_vals(precision):
        inc = 2 ** -precision
        val = 0.0
        while val <= 1:
            yield val
            val += inc

class PCC(SeriesCircuit):
    """Generate an n-bit PCC, where the variable inputs are either 0 or 1, and the constant inputs come from LFSR"""
    def __init__(self, n, use_maj=False):
        mappings = []
        for i in range(n):
            mappings.append(i) #Variable inputs
            mappings.append(i+n) #Constant inputs
        layers = [
            BUS(2*n, 2*n, mappings, nc = n), 
            ParallelCircuit([I(2*n-2), AND()])
        ]
        if use_maj:
            adder = MAJ
        else:
            adder = MUX
        remaining = 2*n-2
        for i in range(n-2):
            remaining -= 2
            layers.append(ParallelCircuit([I(remaining), adder()]))
        layers.append(adder())
        super().__init__(layers)

class PCC_k(SeriesCircuit):
    def __init__(self, n, k, use_maj=False):
        if n == 1:
            self = PCC(n, use_maj=use_maj)
        mappings = []
        for k_i in range(k):
            for i in range(n):
                mappings.append(i+k_i*n)
            for i in range(n):
                mappings.append(i+k*n)
        layers = [
            BUS((k+1)*n, 2*k*n, mappings, nc = n),
            ParallelCircuit([PCC(n, use_maj=use_maj) for _ in range(k)])
        ]
        super().__init__(layers)

class PARALLEL_CONST(SeriesCircuit):
    """Generate a set of parallel constant generators
        reuse: When True, the class will only generate one SNG for each unique constant
             an appropriate BUS will be added to connect the SNG to duplicate instances

        Bitstreams that this generates don't have any correlation assumption
        For un-correlated, simply use ParallelCircuit([CONST_VAL(), CONST_VAL(), ...])
    """
    def __init__(self, consts, precision, bipolar=True, reuse=False):
        if reuse:
            unique_consts = np.unique(consts)
            sngs = self._parallel_consts(unique_consts, precision, bipolar=bipolar)
            m = {x : i for i, x in enumerate(unique_consts)}
            const_mappings = [m[x] for x in consts]
            layers = sngs + [BUS(unique_consts.size, len(consts), const_mappings), ]
        else:
            layers = self._parallel_consts(consts, precision, bipolar=bipolar)
        super().__init__(layers)

    def _parallel_consts(self, consts, precision, bipolar=True):
        """Old init function, from before the ability to reuse SNGs was added"""
        const_vals = [CONST_VAL(const, precision, bipolar=bipolar) for const in reversed(consts)]
        precisions = [x.actual_precision for x in reversed(const_vals)]
        #precisions = [x.actual_precision for x in const_vals] #?
        self.actual_precision = max(precisions)
        sp = sum(precisions)
        mappings = []
        for p in precisions:
            mappings += list(range(p))
        layers = [
            BUS(self.actual_precision, sp, mappings, nc=self.actual_precision),
            ParallelCircuit(const_vals)
        ]
        return layers

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

class PARALLEL_MAC_TREE_2(SeriesCircuit):
    def __init__(self, consts, precision, bipolar=True, maj=False):
        assert len(consts) == 4
        pmac = PARALLEL_MAC_2(consts, precision, bipolar=bipolar, maj=maj)
        layers = [
            ParallelCircuit([I(1, nc=1), pmac]),
            MUX()
        ]
        self.actual_precision = pmac.actual_precision
        super().__init__(layers)

class MAC_RELU(SeriesCircuit):
    """Positive/negative MAC + RELU
        This circuit uses balanced addition trees for the positive and negative components, so all SELs can be 0.5
        If the number of constants is not a power of 2, the extras will be ANDed with the SEL at the correct locations
        in order to ensure proper scaling. The number of select inputs necessary is the height of the larger of the pos/neg
        addition trees.
    """

    def __init__(self, consts_pos, consts_neg, precision, bipolar=False, reuse=False, relu=True):
        wp = len(consts_pos)
        wn = len(consts_neg)
        p_depth = np.ceil(np.log2(wp)).astype(np.int)
        n_depth = np.ceil(np.log2(wn)).astype(np.int)
        depth = max(p_depth, n_depth)
        width = wp + wn

        self.depth = depth
        self.width = width

        #Input signal layout:
        #depth x sel consts
        #precision x sng consts
        #width x X_inputs

        #SNGs
        sngs = PARALLEL_CONST(list(reversed(consts_pos + consts_neg)), precision, bipolar=bipolar, reuse=reuse)
        self.actual_precision = sngs.actual_precision
        
        #Multiplication layers
        mul_mappings = []
        for i in range(width):
            mul_mappings.append(i)
            mul_mappings.append(i+width)
        if bipolar:
            muls = ParallelCircuit([SeriesCircuit([XOR(), NOT()]) for _ in range(width)])
        else:
            muls = ParallelCircuit([AND() for _ in range(width)])

        mul_layers = [
            ParallelCircuit([I(depth, nc=depth), sngs, I(width)]),
            ParallelCircuit([I(depth, nc=depth), BUS(2*width, 2*width, mul_mappings)]),
            ParallelCircuit([I(depth, nc=depth), muls])
        ]

        #self.mul_circuit = SeriesCircuit(mul_layers) #DEBUG ONLY

        #Signal layout after mul layer:
        #depth x sel consts
        #width x mul results

        #Addition tree layers
        def add_tree_balanced(full_depth, w):
            layers = []
            current_w = w
            for d in range(full_depth):
                adds = []
                remaining_sels = full_depth - d
                passed_sels = remaining_sels - 1 #Number of sels passed onto the next layer
                if passed_sels > 0:
                    bus_mappings = [x for x in range(passed_sels)]
                else:
                    bus_mappings = []
                for w_i in range(current_w):
                    bus_mappings.append(remaining_sels + w_i) #data input
                    if w_i % 2 == 1:
                        adds.append(MUX())
                        bus_mappings.append(passed_sels) #select input
                if current_w % 2 == 1:
                    adds.append(AND())
                    next_w = int((current_w - 1)/2) + 1
                    bus_mappings.append(passed_sels)
                else:
                    next_w = int(current_w/2)
                if passed_sels > 0:
                    adds = [I(full_depth - d - 1),] + adds
                layers.append(BUS(current_w + remaining_sels, len(bus_mappings), bus_mappings))
                layers.append(ParallelCircuit(adds))
                current_w = next_w
            return SeriesCircuit(layers)

        sel_map = [x for x in range(depth)]
        add_sel_mappings = sel_map + [depth + x for x in range(wn)] + sel_map + [depth + wn + x for x in range(wp)]
        add_layers = [
            BUS(depth + width, 2*depth + width, add_sel_mappings, nc=depth),
            ParallelCircuit([add_tree_balanced(depth, wn), add_tree_balanced(depth, wp)])
        ]

        #self.add_input_test = SeriesCircuit(mul_layers + [add_layers[0], ]) #DEBUG ONLY

        layers = mul_layers + add_layers

        #ReLU layer
        if relu:
            relu_layers = [
                ParallelCircuit([I(1), NOT()]),
                AND()
            ]
            layers += relu_layers

        super().__init__(layers)