import numpy as np

def bs_bernoulli(n, p):
    """Return a numpy array containing a unipolar stochastic bitstream
        with value p based on the results of n bernoulli trials."""
    return np.packbits(np.random.binomial(1, p, n))

def bs_uniform(n, p):
    """Return a numpy array containing a unipolar stochastic bistream
        with value p based on n numbers drawn from the uniform random distribution on [0, 1).
        A mathematically equivalent alternative to bs_bernoulli."""
    u = np.random.rand(1, n) #Generate random vector
    thresh = np.vectorize(lambda t: 1 if t <= p else 0) #Thresholding
    return np.packbits(thresh(u)) #Apply thresholding, return an efficient bit-packed data structure

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