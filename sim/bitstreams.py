import numpy as np
from pylfsr import LFSR

class SC_RNG:
    """Wrapper class to allow rngs to have persistent state"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.u = None
        self.lfsr = None

    def _run_lfsr(self, n, lfsr_sz, keep_rng=True, inv=False):
        if not keep_rng or self.lfsr is None:
            L = LFSR()
            fpoly = L.get_fpolyList(m=lfsr_sz)[0] #The 0th index holds the lfsr poly with the fewest xor gates
            L = LFSR(fpoly=fpoly, initstate='random')
            self.lfsr = np.zeros(n, dtype=np.uint32)
            for i in range(n):
                L.runKCycle(1)
                self.lfsr[i] = bit_vec_to_int(L.state)
        if inv:
            mask = (1 << lfsr_sz) - 1
            return np.bitwise_not(self.lfsr) & mask
        return self.lfsr

    def bs_uniform(self, n, p, keep_rng=True, inv=False):
        """Return a numpy array containing a unipolar stochastic bistream
            with value p based on n numbers drawn from the uniform random distribution on [0, 1).
            A mathematically equivalent alternative to bs_bernoulli. 
            The random sample u is kept between invocations if keep_rng=True."""
        if not keep_rng or self.u is None:
            self.u = np.random.rand(1, n) #Generate random vector
        if inv:
            return np.packbits((1 - self.u) <= p)
        else:
            return np.packbits(self.u <= p) #Apply thresholding, return an efficient bit-packed data structure

    def bs_lfsr(self, n, p, keep_rng=True, inv=False): #Warning: when keep_rng=False, runtime is very slow 
        """Generate a stochastic bitstream using an appropriately-sized simulated LFSR"""
        lfsr_sz = int(np.ceil(np.log2(n)))
        lfsr_run = self._run_lfsr(n, lfsr_sz, keep_rng=keep_rng, inv=inv)
        return np.packbits(lfsr_run / ((2**lfsr_sz)-1) <= p)

    def up_to_bp_lfsr(self, n, up, keep_rng=True, inv=False):
        """Map a unipolar SN in the range [0, 1] onto a bipolar one on [0.5, 1]"""
        lfsr_sz = int(np.ceil(np.log2(2*n))) #Generate an LFSR that is slightly too large
        lfsr_run = self._run_lfsr(n, lfsr_sz, keep_rng=keep_rng, inv=inv)
        msb_mask = 1 << lfsr_sz - 1
        bp = (lfsr_run & np.full((1, n), ~msb_mask)) / (2**(lfsr_sz-1)) <= up
        return np.packbits(bp | (lfsr_run & np.full((1, n), msb_mask) == msb_mask))

    def bs_bp_lfsr(self, n, bp, keep_rng=True, inv=False):
        """Generate a bipolar stochastic bitstream via the bs_lfsr or bs_uniform method. Can be correlated"""
        up = (bp + 1.0) / 2.0
        return self.bs_lfsr(n, up, keep_rng=keep_rng, inv=inv)

def bit_vec_to_int(vec):
    """Utility function for converting a np array bit vector to an integer"""
    return vec.dot(2**np.arange(vec.size)[::-1])

def bs_bernoulli(n, p):
    """Return a numpy array containing a unipolar stochastic bitstream
        with value p based on the results of n bernoulli trials.
        Faster than bs_uniform if persistent state is not required."""
    return np.packbits(np.random.binomial(1, p, n))

def bs_mean(bs, bs_len=None):
    """Evaluate the probability value of a bitstream, taken as the mean value of the bitstream.
        For bitstreams that don't align to byte boundaries, use bs_len to supply the exact bitstream length."""
    unp = np.unpackbits(bs)
    if bs_len != None:
        return np.sum(unp) / bs_len 
    else:
        return np.mean(unp)

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
    px = bs_mean(bsx, bs_len=bs_len)
    py = bs_mean(bsy, bs_len=bs_len)
    if px in (0, 1) or py in (0, 1):
        #raise ValueError("SCC is undefined for bitstreams with value 0 or 1") 
        return None

    p_uncorr  = px * py
    p_actual  = bs_mean(np.bitwise_and(bsx, bsy), bs_len=bs_len)
    delta0 = np.floor(p_uncorr * bs_len + 0.5)/bs_len - p_uncorr
    delta  = p_actual - p_uncorr
    if p_actual > p_uncorr:
        numer = (delta - np.abs(delta0))
        if numer == 0:
            return 0 
        denom = (np.abs(np.minimum(px, py) - p_uncorr) - np.abs(delta0))
        assert denom != 0
        return  numer / denom
    else:
        numer = (np.abs(delta0) + delta)
        if numer == 0:
            return 0
        denom = (np.abs(p_uncorr - np.maximum(px + py - 1, 0)) - np.abs(delta0))
        assert denom != 0
        return numer / denom

def get_corr_mat(bs_arr, bs_len=None, use_zscc=False):
    """Returns a correlation matrix representing the measured scc values of the given bitstream array"""
    n = len(bs_arr)
    Cij = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            if use_zscc:
                Cij[i][j] = bs_zscc(bs_arr[i], bs_arr[j], bs_len=bs_len)
            else:
                Cij[i][j] = bs_scc(bs_arr[i], bs_arr[j], bs_len=bs_len)
    return Cij

def mc_mat(c, n):
    """Returns a correlation matrix representing mutual correlation at the desired c value"""
    m = np.ones((n, n)) * c
    return np.tril(m, -1)

def mc_scc(bs_arr, bs_len=None):
    """Test if an array of bitstreams are mutually correlated"""
    Cij = get_corr_mat(bs_arr, bs_len=bs_len)
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

    bs1 = np.packbits(np.array([1,1,1,1,1,0]))
    bs2 = np.packbits(np.array([0,0,0,0,0,1]))
    print(bs_zscc(bs1, bs2, bs_len=6))

    #bs5 = np.packbits(np.array([0,1,1,0,0,0,0,1,1,0]))
    #bs6 = np.packbits(np.array([1,0,0,1,0,0,1,0,0,1]))
    #print(bs_scc(bs5, bs6, bs_len=10))

    """Test mutually mc_scc"""
    #bs1 = np.packbits(np.array([1,0,1,0,1,1]))
    #bs2 = np.packbits(np.array([1,1,0,0,1,1]))
    #bs3 = np.packbits(np.array([0,0,1,1,0,0]))
    #bs_arr = [bs1, bs2, bs3]
    #print(get_corr_mat(bs_arr, bs_len=6))
    #print(mc_scc(bs_arr, bs_len=6))

    """Test mutual correlation generation"""
    #print(mc_mat(-1, 3))
    #rng = SC_RNG()
    #n = 5
    #bs_arr = [rng.bs_lfsr(16, 0.5, keep_rng=False) for _ in range(n)]
    #Cij = get_corr_mat(bs_arr)
    #print(Cij)