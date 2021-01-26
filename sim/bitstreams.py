import numpy as np
from pylfsr import LFSR

class SC_RNG:
    """Wrapper class to allow rngs to have persistent state"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.u = None
        self.lfsr = None

    def bs_uniform(self, n, p, keep_rng=True):
        """Return a numpy array containing a unipolar stochastic bistream
            with value p based on n numbers drawn from the uniform random distribution on [0, 1).
            A mathematically equivalent alternative to bs_bernoulli. 
            The random sample u is kept between invocations if keep_rng=True."""
        if not keep_rng or self.u is None:
            self.u = np.random.rand(1, n) #Generate random vector
        return np.packbits(self.u <= p) #Apply thresholding, return an efficient bit-packed data structure

    def bs_lfsr(self, n, p, keep_rng=True): #Warning: when keep_rng=False, runtime is very slow 
        """Generate a stochastic bitstream using an appropriately-sized simulated LFSR"""
        if not keep_rng or self.lfsr is None:
            lfsr_sz = int(np.ceil(np.log2(n)))
            L = LFSR()
            fpoly = L.get_fpolyList(m=lfsr_sz)[0] #The 0th index holds the lfsr poly with the fewest xor gates
            L = LFSR(fpoly=fpoly, initstate='random')
            self.lfsr = np.zeros(n, dtype=np.uint32)
            for i in range(n):
                #TODO: Add a method to skip values in the LFSR spectrum that aren't used
                L.runKCycle(1)
                self.lfsr[i] = bit_vec_to_int(L.state)
        return np.packbits(self.lfsr / n <= p)

def bit_vec_to_int(vec):
    """Utility function for converting a np array bit vector to an integer"""
    return vec.dot(2**np.arange(vec.size)[::-1])

def bs_bernoulli(n, p):
    """Return a numpy array containing a unipolar stochastic bitstream
        with value p based on the results of n bernoulli trials.
        Faster than bs_uniform if persistent state is not required."""
    return np.packbits(np.random.binomial(1, p, n))

def bs_mean(bs):
    """Evaluate the probability value of a bitstream, taken as the mean value of the bitstream."""
    unp = np.unpackbits(bs)
    return np.mean(unp)

def bs_scc(bsx, bsy):
    """Compute the stochastic cross-correlation between two bitstreams according to Eq. (1)
    in [A. Alaghi and J. P. Hayes, Exploiting correlation in stochastic circuit design]"""
    px = bs_mean(bsx)
    py = bs_mean(bsy)
    p_uncorr  = px * py
    p_actual  = bs_mean(np.bitwise_and(bsx, bsy))
    if p_actual > p_uncorr:
        return (p_actual - p_uncorr) / (np.minimum(px, py) - p_uncorr)
    else:
        return (p_actual - p_uncorr) / (p_uncorr + np.maximum(px + py - 1, 0))