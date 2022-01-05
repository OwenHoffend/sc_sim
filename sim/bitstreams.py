import numpy as np
import torch
from pylfsr import LFSR

class SC_RNG:
    """Wrapper class to allow rngs to have persistent state"""
    def __init__(self):
        self.reset()

    def reset(self):
        #Uniform related values
        self.u = None

        #LFSR related values
        self.lfsr_sz = None
        self.lfsr_init = None
        self.lfsr = None

    def _lfsr_get_fpoly(self, lfsr_sz):
        """If the lfsr size is changed, then get a polynomial for the new size, otherwise use the old one"""
        if self.lfsr_sz is None or lfsr_sz != self.lfsr_sz:
            L = LFSR()
            self.fpoly = L.get_fpolyList(m=lfsr_sz)[0] #The 0th index holds the lfsr poly with the fewest xor gates
        self.lfsr_sz = lfsr_sz
        return self.fpoly

    def _lfsr_init_nonzero(self, fpoly, lfsr_sz, init_state):
        if init_state is None:
            while True:
                L = LFSR(fpoly=fpoly, initstate='random')
                if not np.all(L.state == np.zeros(lfsr_sz)):
                    break
            return L
        return LFSR(fpoly=fpoly, initstate=init_state)

    def _run_lfsr(self, n, lfsr_sz, keep_rng=True, inv=False, save_init=False, init_state=None):
        if not keep_rng or self.lfsr is None:
            fpoly = self._lfsr_get_fpoly(lfsr_sz)
            if save_init:
                if self.lfsr_init is None:
                    L = self._lfsr_init_nonzero(fpoly, lfsr_sz, init_state)
                    self.lfsr_init = L.state
                else:
                    L = LFSR(fpoly=fpoly, initstate=self.lfsr_init)
            else:
                L = self._lfsr_init_nonzero(fpoly, lfsr_sz, init_state)
            self.lfsr = np.zeros(n, dtype=np.uint32)
            for i in range(n):
                L.runKCycle(1)
                self.lfsr[i] = bit_vec_to_int(L.state)
        if inv:
            mask = (1 << lfsr_sz) - 1
            return np.bitwise_not(self.lfsr) & mask
        return self.lfsr

    def bs_uniform(self, n, p, keep_rng=True, inv=False, pack=True):
        """Return a numpy array containing a unipolar stochastic bistream
            with value p based on n numbers drawn from the uniform random distribution on [0, 1).
            A mathematically equivalent alternative to bs_bernoulli. 
            The random sample u is kept between invocations if keep_rng=True."""
        if not keep_rng or self.u is None:
            self.u = np.random.rand(1, n) #Generate random vector
        if inv:
            result = (1 - self.u) <= p
        else:
            result = self.u <= p #Apply thresholding, return an efficient bit-packed data structure
        if pack:
            return np.packbits(result)
        return result

    def bs_lfsr(self, n, p, keep_rng=True, inv=False, save_init=False, init_state=None, pack=True): #Warning: when keep_rng=False, runtime is very slow 
        """Generate a stochastic bitstream using an appropriately-sized simulated LFSR"""
        lfsr_sz = int(np.ceil(np.log2(n)))
        if lfsr_sz < 4: 
            raise ValueError("LFSR Size is too small")
        lfsr_run = self._run_lfsr(n, lfsr_sz, keep_rng=keep_rng, inv=inv, save_init=save_init, init_state=init_state)
        bs = lfsr_run / ((2**lfsr_sz)-1) <= p
        #assert bs_mean(bs) == p
        if pack:
            return np.packbits(bs)
        return bs

    def up_to_bp_lfsr(self, n, up, keep_rng=True, inv=False, save_init=False):
        """Map a unipolar SN in the range [0, 1] onto a bipolar one on [0.5, 1]"""
        lfsr_sz = int(np.ceil(np.log2(2*n))) #Generate an LFSR that is slightly too large
        lfsr_run = self._run_lfsr(n, lfsr_sz, keep_rng=keep_rng, inv=inv, save_init=save_init)
        msb_mask = 1 << lfsr_sz - 1
        bp = (lfsr_run & np.full((1, n), ~msb_mask)) / (2**(lfsr_sz-1)) <= up
        return np.packbits(bp | (lfsr_run & np.full((1, n), msb_mask) == msb_mask))

    def bs_bp_lfsr(self, n, bp, keep_rng=True, inv=False, save_init=False):
        """Generate a bipolar stochastic bitstream via the bs_lfsr method. Can be correlated"""
        up = (bp + 1.0) / 2.0
        return self.bs_lfsr(n, up, keep_rng=keep_rng, inv=inv, save_init=save_init)

    def bs_bp_uniform(self, n, bp, keep_rng=True, inv=False):
        """Generate a bipolar stochastic bitstream via the bs_uniform method."""
        up = (bp + 1.0) / 2.0
        return self.bs_uniform(n, up, keep_rng=keep_rng, inv=inv)

bv_int_cache = {}
def bit_vec_to_int(vec):
    """Utility function for converting a np array bit vector to an integer"""
    str_vec = "".join([str(x) for x in vec])
    if str_vec in bv_int_cache.keys():
        return bv_int_cache[str_vec]
    result = vec.dot(2**np.arange(vec.size)[::-1])
    bv_int_cache[str_vec] = result 
    return result

def bs_bernoulli(n, p):
    """Return a numpy array containing a unipolar stochastic bitstream
        with value p based on the results of n bernoulli trials.
        Faster than bs_uniform if persistent state is not required."""
    return np.packbits(np.random.binomial(1, p, n))

def bs_unpack(bs):
    axis = len(bs.shape) - 1
    try:
        return np.unpackbits(bs, axis=axis)
    except TypeError:
        return bs

def bs_mean(bs, bs_len=None):
    """Evaluate the probability value of a bitstream, taken as the mean value of the bitstream.
        For bitstreams that don't align to byte boundaries, use bs_len to supply the exact bitstream length."""
    axis = len(bs.shape) - 1
    unp = bs_unpack(bs)
    if bs_len != None:
        return np.sum(unp, axis=axis) / bs_len 
    else:
        return np.mean(unp, axis=axis)

def bs_var(bs, bs_len=None):
    mean = bs_mean(bs, bs_len=bs_len)
    return np.sum((bs_unpack(bs) - mean) ** 2) / (bs_len ** 2)

def bs_count_cuda(bs):
    """Using a well-known population count algorithm"""
    global m1, m2, m4
    m1 = torch.cuda.ByteTensor([0x55])
    m2 = torch.cuda.ByteTensor([0x33])
    m4 = torch.cuda.ByteTensor([0x0F])
    b = (bs & m1) + (bs >> 1 & m1)
    b = (b & m2) + (b >> 2 & m2)
    b = (b & m4) + (b >> 4 & m4)
    dim = 0 if len(b.shape) == 1 else 1
    return torch.sum(b, dim)

def bs_mean_bp(bs, bs_len=None):
    """Evaluate the bipolar probability value of a bitstream"""
    m = bs_mean(bs, bs_len=bs_len) #Unipolar mean
    return 2.0 * m - 1.0

def bs_mean_bp_abs(bs, bs_len=None):
    return abs(bs_mean_bp(bs, bs_len=bs_len))

def bs_scc(bsx, bsy, bs_len=None):
    """Compute the stochastic cross-correlation between two bitstreams according to Eq. (1)
    in [A. Alaghi and J. P. Hayes, Exploiting correlation in stochastic circuit design]"""
    px = bs_mean(bsx, bs_len=bs_len)
    py = bs_mean(bsy, bs_len=bs_len)
    if px in (0, 1) or py in (0, 1):
        #raise ValueError("SCC is undefined for bitstreams with value 0 or 1") 
        return None
    p_uncorr  = px * py
    p_actual  = bs_mean(np.bitwise_and(bsx, bsy), bs_len=bs_len)
    if p_actual > p_uncorr:
        return (p_actual - p_uncorr) / (np.minimum(px, py) - p_uncorr)
    else:
        return (p_actual - p_uncorr) / (p_uncorr - np.maximum(px + py - 1, 0))

def bs_scc_ovs(pi, pj, No, N):
    if pi in (0, 1) or pj in (0, 1):
        #raise ValueError("SCC is undefined for bitstreams with value 0 or 1") 
        return 1 

    p_uncorr  = pi * pj
    p_actual  = No / N
    if p_actual > p_uncorr:
        return (p_actual - p_uncorr) / (np.minimum(pi, pj) - p_uncorr)
    else:
        return (p_actual - p_uncorr) / (p_uncorr - np.maximum(pi + pj - 1, 0))

def bs_zce(bsx, bsy, bs_len):
    px = bs_mean(bsx, bs_len=bs_len)
    py = bs_mean(bsy, bs_len=bs_len)
    if px in (0, 1) or py in (0, 1):
        #raise ValueError("SCC is undefined for bitstreams with value 0 or 1") 
        return None

    p_uncorr  = px * py
    p_actual  = bs_mean(np.bitwise_and(bsx, bsy), bs_len=bs_len)

    delta0 = np.floor(p_uncorr * bs_len + 0.5)/bs_len - p_uncorr
    delta  = p_actual - p_uncorr
    return delta/np.abs(delta) * (np.abs(delta) - np.abs(delta0))

def bs_zscc(bsx, bsy, bs_len):
    """Zeroed SCC"""
    px = bs_mean(bsx, bs_len=bs_len)
    py = bs_mean(bsy, bs_len=bs_len)
    if px in (0, 1) or py in (0, 1):
        #raise ValueError("SCC is undefined for bitstreams with value 0 or 1") 
        return 1 

    p_uncorr  = px * py
    p_actual  = bs_mean(np.bitwise_and(bsx, bsy), bs_len=bs_len)
    delta0 = np.floor(p_uncorr * bs_len + 0.5)/bs_len - p_uncorr
    delta  = p_actual - p_uncorr
    numer = (delta - delta0)
    if numer == 0:
        return 0
    if p_actual > p_uncorr + delta0:
        denom = (np.minimum(px, py) - p_uncorr - delta0)
        assert denom != 0
        return  numer / denom
    denom = (p_uncorr - np.maximum(px + py - 1, 0) + delta0)
    assert denom != 0
    return numer / denom

def bs_zscc_ovs(pi, pj, No, N):
    """Compute zeroed SCC using an overlap count"""
    if pi in (0, 1) or pj in (0, 1):
        #raise ValueError("SCC is undefined for bitstreams with value 0 or 1") 
        return 1 

    p_uncorr  = pi * pj
    p_actual  = No / N
    delta0 = np.floor(p_uncorr * N + 0.5)/N - p_uncorr
    delta  = p_actual - p_uncorr
    numer = (delta - delta0)
    if numer == 0:
        return 0
    if p_actual > p_uncorr + delta0:
        denom = (np.minimum(pi, pj) - p_uncorr - delta0)
        assert denom != 0
        return  numer / denom
    denom = (p_uncorr - np.maximum(pi + pj - 1, 0) + delta0)
    assert denom != 0
    return numer / denom

def bs_zscc_cuda(bsx, bsy, N):
    """Single bitstream bsx is being compared to multiple bitstreams in bsy"""
    px = bs_count_cuda(bsx) / N
    py = bs_count_cuda(bsy) / N
    if px in (0, 1) or 1 in py or 0 in py:
        return 1 
    p_uncorr = px * py
    p_actual = bs_count_cuda(torch.bitwise_and(bsx, bsy)) / N
    p_max = torch.min(px, py)
    p_min = torch.max(px + py - 1, 0).values
    delta0 = torch.floor(p_uncorr * N + 0.5)/N - p_uncorr
    delta  = p_actual - p_uncorr
    numer = (delta - delta0)
    result = torch.cuda.FloatTensor(len(py)).fill_(0)
    gt_denom = p_max - p_uncorr - delta0
    gt_mask = torch.bitwise_and(numer > 0, gt_denom != 0)
    lt_denom = p_uncorr - p_min + delta0 
    lt_mask = torch.bitwise_and(numer < 0, lt_denom != 0)
    result += (numer / (gt_denom + 1e-15)) * gt_mask
    result += (numer / (lt_denom + 1e-15)) * lt_mask
    return result

def get_corr_mat(bs_arr, bs_len=None, use_zscc=False, use_cov=False):
    """Returns a correlation matrix representing the measured scc values of the given bitstream array"""
    n = len(bs_arr)
    Cij = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            if use_zscc:
                Cij[i][j] = bs_zscc(bs_arr[i], bs_arr[j], bs_len=bs_len)
            elif use_cov:
                pi = bs_mean(bs_arr[i], bs_len=bs_len)
                pj = bs_mean(bs_arr[j], bs_len=bs_len)
                pij = bs_mean(np.bitwise_and(bs_arr[i], bs_arr[j]), bs_len=bs_len)
                Cij[i][j] = pij - pi * pj
            else:
                Cij[i][j] = bs_scc(bs_arr[i], bs_arr[j], bs_len=bs_len)
    return Cij

def get_corr_mat_np(bs_mat, use_zscc=False):
    """Same as get_corr_mat, but accepts np array as input"""
    n, N = bs_mat.shape
    Cij = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                if use_zscc:
                    Cij[i][j] = bs_zscc(np.packbits(bs_mat[i, :]), np.packbits(bs_mat[j, :]), bs_len=N)
                else:
                    Cij[i][j] = bs_scc(np.packbits(bs_mat[i, :]), np.packbits(bs_mat[j, :]), bs_len=N)
            else:
                Cij[i][j] = 1
    return Cij

def get_corr_mat_cuda(bs_arr):
    """
    Use pytorch cuda to get zscc matrix.
    Only supports bitstreams of length divisible by 8 for now
    """
    n, N = bs_arr.shape
    Cij = torch.cuda.FloatTensor(n, n).fill_(0)
    for i in range(1, n):
            Cij[i][:i] = bs_zscc_cuda(bs_arr[i][:], bs_arr[:i][:], N<<3)
    return Cij

def mc_mat(c, n):
    """Returns a correlation matrix representing mutual correlation at the desired c value"""
    m = np.ones((n, n)) * c
    return np.tril(m, -1)

def mc_scc(bs_arr, bs_len=None, use_zscc=False):
    """Test if an array of bitstreams are mutually correlated"""
    Cij = get_corr_mat(bs_arr, bs_len=bs_len, use_zscc=use_zscc)
    c = Cij[0][1]
    return np.all((Cij == c) | (Cij == 0))
    
def hamming_dist(bv1, bv2):
    """Return the Hamming Distance between two bitstreams"""
    return np.sum(np.abs(bv1 - bv2))

def gen_correlated(scc, n, p1, p2, bs_gen_func):
    """Using the method in [A. Alaghi and J. P. Hayes, Exploiting correlation in stochastic circuit design],
    generate two bitstreams with a specified SCC value"""
    bs1 = bs_gen_func(n, p1, keep_rng=True)
    if scc < 0:
        bs2_scc1 = bs_gen_func(n, p2, keep_rng=True, inv=True) #scc: -1
    else:
        bs2_scc1 = bs_gen_func(n, p2, keep_rng=True) #scc: 1
    bs2_scc0 = bs_gen_func(n, p2, keep_rng=False)
    bs_mag = bs_gen_func(n, np.abs(scc), keep_rng=False)
    scc0_weighted = np.bitwise_and(np.bitwise_not(bs_mag), bs2_scc0)
    scc1_weighted = np.bitwise_and(bs_mag, bs2_scc1)
    return bs1, np.bitwise_or(scc0_weighted, scc1_weighted)

def mut_corr_err(mc, c_mat):
    n, _ = c_mat.shape
    mc_mat = np.tril(np.ones_like(c_mat) * mc, -1)
    c_mat_tril = np.tril(c_mat, -1)
    return np.sum(np.abs(mc_mat - c_mat_tril)) / (n * (n-1) / 2)

if __name__ == "__main__":
    """Test bs_bp_lfsr"""
    #rng = SC_RNG()
    #print(bs_mean_bp(rng.up_to_bp_lfsr(512, 0))) #Play with the values here

    """Test gen_correlated"""
    #rng = SC_RNG()
    #bs1, bs2 = gen_correlated(-0.33, 1024, 0.33, 0.33, rng.bs_lfsr)
    #print(bs_scc(bs1, bs2))

    """Test forcing SCC to various values"""
    #bs1 = np.packbits(np.array([0,0,1,1,1,0]))
    #bs2 = np.packbits(np.array([1,1,1,0,0,0]))
    #bs3 = np.packbits(np.array([0,1,0,1,0,1]))
    #bs4 = np.packbits(np.array([1,0,0,0,1,1]))
    #print(bs_zscc(bs1, bs2, bs_len=8))
    #print(bs_zscc(bs1, bs3, bs_len=8))
    #print(bs_zscc(bs2, bs3, bs_len=8))
    #print(bs_zscc(bs1, bs4, bs_len=8))
    #print(bs_zscc(bs2, bs4, bs_len=8))
    #print(bs_zscc(bs3, bs4, bs_len=8))

    #bs1 = np.packbits(np.array([1,1,1,1,1,0]))
    #bs2 = np.packbits(np.array([0,0,0,0,0,1]))
    #print(bs_zscc(bs1, bs2, bs_len=6))

    #bs5 = np.packbits(np.array([0,1,1,0,0,0,0,1,1,0]))
    #bs6 = np.packbits(np.array([1,0,0,1,0,0,1,0,0,1]))
    #print(bs_scc(bs5, bs6, bs_len=10))

    """Test mutually mc_scc"""
    bs1 = np.packbits(np.array([1,1,1,0,0,0]))
    bs2 = np.packbits(np.array([1,1,0,1,1,0]))
    bs_arr = [bs1, bs2]
    print(get_corr_mat(bs_arr, bs_len=6, use_zscc=True))
    #print(mc_scc(bs_arr, bs_len=6))

    """Test mutual correlation generation"""
    #print(mc_mat(-1, 3))
    #rng = SC_RNG()
    #n = 5
    #bs_arr = [rng.bs_lfsr(16, 0.5, keep_rng=False) for _ in range(n)]
    #Cij = get_corr_mat(bs_arr)
    #print(Cij)

    """Test mut_corr_err"""
    #c_mat = np.array([
    #    [0, 0, 0],
    #    [-1, 0, 0],
    #    [-1, -1, 0]
    #])
    #print(mut_corr_err(1, c_mat))