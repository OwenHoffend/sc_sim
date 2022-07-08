import numpy as np
import torch
from torch._C import _valgrind_supported_platform
import sim.bitstreams as bs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(bool)[::-1] #Changed to [::-1] here to enforce ordering globally (12/29/2021)

def int_array(bmat):
    "Convert a bin_array back to an int one"
    if len(bmat.shape) == 1:
        n = bmat.size
    else:
        _, n = bmat.shape
    bmap = np.array([1 << x for x in range(n)])
    return bmat @ bmap

B_mat_dict = {}
def B_mat(n, cuda=False):
    """Generate a 2^n by n matrix of all of the possible values of n bits"""
    if n in B_mat_dict.keys():
        return B_mat_dict[n]
    B = np.zeros((2 ** n, n), dtype=bool)
    for i in range(2 ** n):
        B[i][:] = bin_array(i, n)
    if cuda:
        B = torch.tensor(B.astype(np.float32)).to(device)
    B_mat_dict[n] = B
    return B

#----------------------------------------------------------------------------------#
# PTV GENERATION
#----------------------------------------------------------------------------------#
def Ov(bs_mat):
    """Overlap function
    The primary difference between this function and N_actual_overlaps from scc_sat.py
    is its ability to count multiple overlaps per bit location for arrays such as [2,1,0] and [2,1,1]"""
    N, n = bs_mat.shape
    Ov = np.zeros((n, n), dtype=np.uint32)
    for i in range(n):
        for j in range(i):
            Ov[i][j] = np.sum(np.bitwise_and(bs_mat[:, i], bs_mat[:, j]))
    return Ov

def get_actual_vin(bs_mat):
    n, N = bs_mat.shape
    Vin = np.zeros(2 ** n)
    uniques, counts = np.unique(bs_mat.T, axis=0, return_counts=True)
    for unq, cnt in zip(uniques, counts):
        Vin[bs.bit_vec_to_int(unq)] = cnt / N
    return Vin

def get_vin_iterative(p_vec, cov_mat, N):
    """Performs a conditional-subtract based iterative algorithm to find Vin"""
    n = p_vec.size
    p_vec = p_vec[np.newaxis]
    No = N*(cov_mat + np.tril(p_vec.T @ p_vec, -1))
    Bn = B_mat(n)
    O = np.zeros((2 ** n, n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            O[:, i, j] = Bn[:, i] & Bn[:, j]
    O_sum = np.sum(O, axis=(1,2))
    S = np.argsort(O_sum)[::-1]
    v_remaining = 1
    p_remaining = np.copy(p_vec).reshape((n,))
    Vin = np.zeros(2 ** n)
    for q in S:
        Oq = O[q, :, :]
        if q == 0: #All zeros, no overlaps
            Vin[q] = v_remaining
        elif not np.any(np.tril(Oq, -1)): #Single bit, no overlaps
            v = p_remaining[np.log2(q).astype(np.uint16)]
            Vin[q] = v
            v_remaining -= v
        else: #At least one overlap
            alpha = 0
            while not np.any(np.tril(No - (alpha + 1) * Oq, -1) < 0):
                alpha += 1
            No -= alpha * np.tril(Oq, -1)
            v = alpha / N
            Vin[q] = v
            v_remaining -= v
            p_rem_mask = np.sum(Oq, axis=1) > 0
            p_remaining -= v * p_rem_mask
    return np.round(Vin + 1e-15, 12) #Round off small floating point errors - makes the output look nicer

def get_vin_mc1(Pin):
    """Generates a Vin vector for bitstreams mutually correlated with ZSCC=1"""
    n = Pin.size
    Vin = np.zeros(2 ** n)
    Vin[0] = 1 - np.max(Pin)
    Vin[2 ** n - 1] = np.min(Pin)
    Pin_sorted = np.argsort(Pin)[::-1]
    i = 0
    for k in range(1, n):
        i += 2 ** Pin_sorted[k - 1]
        Vin[i] = Pin[Pin_sorted[k - 1]] - Pin[Pin_sorted[k]]
    return np.round(Vin, 12)

def get_vin_mc1_paper(Pin):
    """Generates PTV for bitstreams mutually correlated with SCC=1 according to the equation in the paper"""
    n = Pin.size
    Bn = B_mat(n)
    Vin = np.zeros(2 ** n)
    for i in range(2 ** n):
        r1 = set([1,])
        r0 = set([0,])
        for j in range(n):
            if Bn[i, j]:
                r1.add(Pin[j])
            else:
                r0.add(Pin[j])
        Vin[i] = max(0, min(r1) - max(r0))
    return np.round(Vin, 12)

def get_vin_mcn1_paper(Pin):
    n = Pin.size
    Bn = B_mat(n)
    Vin = np.zeros(2 ** n)
    if n > 2 and np.sum(Pin) > 1:
        return None
    for i in range(2 ** n):
        r1 = set()
        r0 = set()
        for j in range(n):
            if Bn[i, j]:
                r1.add(Pin[j])
            else:
                r0.add(Pin[j])
        if i == 0:
            Vin[i] = max(0, 1 - np.sum(Pin))
        elif len(r1) == 1:
            Vin[i] = sum(r1) - max(0, np.sum(Pin) - 1)
        else:
            Vin[i] = max(0, np.sum(Pin) - 1)
    if not np.isclose(np.sum(Vin), 1.0):
        print("hi")
    return np.round(Vin, 12)

def get_vin_mc1_cuda(Pin):
    b, n = Pin.shape
    Vin = torch.cuda.FloatTensor(b, 2 ** n).fill_(0)
    Vin[:, 0] = 1 - torch.max(Pin, dim=1).values
    Vin[:, 2 ** n - 1] = torch.min(Pin, dim=1).values
    Pin_sorted = torch.argsort(Pin, dim=1, descending=True)
    i = torch.cuda.LongTensor(b).fill_(0)
    lastmax = torch.gather(Pin, 1, Pin_sorted[:, 0].view(b, 1))
    for k in range(1, n):
        i += 2 ** Pin_sorted[:, k - 1]
        max2 = torch.gather(Pin, 1, Pin_sorted[:, k].view(b, 1))
        Vin.scatter_(1, i.view(b, 1), lastmax - max2)
        lastmax = max2
    return Vin.T

def get_vin_mcn1(Pin):
    """Generates a Vin vector for bitstreams mutually correlated with ZSCC=-1"""
    if np.sum(Pin) > 1:
        return None
    n = Pin.size
    Vin = np.zeros(2 ** n)
    Vin[0] = 1 - np.sum(Pin)
    Pin_sorted = np.argsort(Pin)[::-1]
    for k in range(n):
        i = 2 ** Pin_sorted[k]
        Vin[i] = Pin[Pin_sorted[k]]
    return np.round(Vin, 12)

def get_vin_mcn1_cuda(Pin):
    b, n = Pin.shape
    Vin = torch.cuda.FloatTensor(b, 2 ** n).fill_(0)
    Vin[0] = 1 - torch.sum(Pin, dim=1)
    Pin_sorted = torch.argsort(Pin, dim=1, descending=True)
    for k in range(n):
        i = 2 ** Pin_sorted[:, k]
        max_ = torch.gather(Pin, 1, Pin_sorted[:, k]).view(b, 1)
        Vin.scatter_(1, i.view(b, 1), max_)
    return Vin.T

def get_vin_mc0(Pin):
    """Generates a Vin vector for bitstreams mutually correlated with ZSCC=0"""
    n = Pin.size
    Bn = B_mat(n)
    return np.prod(Bn * Pin + (1 - Bn) * (1 - Pin), 1)

def get_vin_mc0_cuda(Pin):
    b, n = Pin.shape
    Bn = B_mat(n, cuda=True).repeat(b, 1, 1)
    Vin = torch.prod(Bn * Pin.unsqueeze(1) + (1 - Bn) * (1 - Pin.unsqueeze(1)), 2)
    return Vin.T

def get_vin_mc0_cuda_test(Pin):
    n = Pin.shape[0]
    Bn = B_mat(n, cuda=True)
    return torch.prod(Bn * Pin + (1 - Bn) * (1 - Pin), 1)

def get_vin_mc_any(Pin, c):
    if c < 0:
        v_pm1 = get_vin_mcn1(Pin)
        if v_pm1 is None:
            return None
    else:
        v_pm1 = get_vin_mc1(Pin)
    v_0 = get_vin_mc0(Pin)
    c = abs(c)
    return (1-c) * v_0 + c * v_pm1

def sample_from_ptv(ptv, N):
    """Uniformly sample N bitstream samples (each of width n) from a given PTV"""
    n = int(np.log2(ptv.shape[0]))
    bs_mat = np.zeros((n, N), dtype=np.uint8)
    for i in range(N):
        sel = np.random.choice(ptv.shape[0], p=ptv)
        bs_mat[:, i] = bin_array(sel, n)
    return bs_mat

def deterministic_sample_from_ptv(ptv, N):
    """Create N bitstream samples (each of width n) from a given PTV, with a deterministic number of each possible pattern"""
    n = int(np.log2(ptv.shape[0]))
    sample_nums = np.round(ptv * N)
    N_new = np.sum(sample_nums).astype(np.uint32)
    bs_mat = np.zeros((n, N_new), dtype=np.uint8)
    j = 0
    i = 0
    while i < N_new:
        if sample_nums[j] != 0:
            sample_nums[j] -= 1
            bs_mat[:, i] = bin_array(j, n)
            i += 1
        else:
            j += 1
    return bs_mat

def PTV_swap_cols(ptv, swap_inds):
    n_2 = ptv.size
    Bn = B_mat(np.log2(n_2).astype(np.int))
    ptv_swap_inds = int_array(Bn[:, swap_inds])
    return ptv[ptv_swap_inds]

#----------------------------------------------------------------------------------#
# PTM GENERATION & MANIPULATION
#----------------------------------------------------------------------------------#

def get_func_mat(func, n, k):
    """Compute the PTM for a boolean function with n inputs and k outputs
        Does not handle probabilistic functions, only pure boolean functions"""
    Mf = np.zeros((2 ** n, 2 ** k), dtype=bool)
    for i in range(2 ** n):
        res = func(*list(bin_array(i, n)))
        if k == 1:
            num = res.astype(np.uint8)
        else:
            num = 0
            for idx, j in enumerate(res):
                if j:
                    num += 1 << idx
        Mf[i][num] = 1
    return Mf

def apply_ptm_to_bs(bs_mat, Mf):
    """Given a set of input bitstrems, compute the output bitstreams for the circuit defined by the PTM Mf
    FOR NOW: Doesn't consider packing
    """
    n, N = bs_mat.shape
    n2, k2 = Mf.shape
    k = np.log2(k2).astype(np.int)
    ints = int_array(bs_mat.T)
    bs_out = np.zeros((k, N), dtype=np.bool_)
    bm = B_mat(k)
    for i in range(N):
        bs_out[:, i] = Mf[ints[i], :] @ bm
    return bs_out

def reduce_func_mat(Mf, idx, p):
    """Reduce a PTM matrix with a known probability value on one input"""
    n, k = np.log2(Mf.shape).astype(np.uint16)
    print("Warning, reduce_func_mat is known to cause incorrectness with Kvv calculations")
    ss1, ss2 = [], []
    for i in range(2 ** n):
        if i % (2 ** (idx + 1)) < 2 ** idx:
            ss1.append(i)
        else:
            ss2.append(i)
    Mff = Mf.astype(np.float32)
    return Mff[ss1, :] * p + Mff[ss2, :] * (1-p)

#----------------------------------------------------------------------------------#
# CORRELATION & COVARIANCE MATRICES
#----------------------------------------------------------------------------------#

def get_output_corr_mat(Vin, Mf, N, use_zscc=True):
    """Using ZSCC, compute the output correlation matrix given an input PTV,
    input probabilities, and the circuit's PTM"""
    n, k = np.log2(Mf.shape).astype(np.uint16)
    Bk = B_mat(k)

    Vout = Mf.T @ Vin
    Pout = Bk.T @ Vout

    #Get Cout
    No_out = Ov((Bk * N * np.column_stack(k*[Vout])).astype(np.uint32))
    Cout = np.zeros((k, k))
    for i in range(k):
        for j in range(i):
            if use_zscc:
                Cout[i][j] = bs.bs_zscc_ovs(Pout[i], Pout[j], No_out[i][j], N)
            else:
                Cout[i][j] = bs.bs_scc_ovs(Pout[i], Pout[j], No_out[i][j], N)
    return Vout, Cout

def get_input_corr_mat(Vin, Mf, N):
    """Using ZSCC, compute the input correlation matrix given an input PTV"""
    n, k = np.log2(Mf.shape).astype(np.uint16)
    Bn = B_mat(n)

    Pin = Bn.T @ Vin

    #Get Cin
    No_in = Ov((Bn * N * np.column_stack(n*[Vin])).astype(np.uint32))
    Cin = np.zeros((k, k))
    for i in range(k):
        for j in range(i):
            Cin[i][j] = bs.bs_zscc_ovs(Pin[i], Pin[j], No_in[i][j], N)
    return Cin

def get_corr_mat_paper(ptv, invalid_corr=1):
    n = int(np.log2(ptv.size))
    Bn = B_mat(n)
    P = Bn.T @ ptv
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            p_uncorr = P[i] * P[j]
            cov = (Bn[:, i] * Bn[:, j]) @ ptv - p_uncorr
            if cov > 0:
                norm = np.minimum(P[i], P[j]) - p_uncorr
            else:
                norm = p_uncorr - np.maximum(P[i] + P[j] - 1, 0)
            if norm == 0:
                C[i, j] = invalid_corr
            else:
                C[i, j] = cov / norm
    return C

def kvv(ptv):
    """Helper function for ptm_input_cov_mat and ptm_output_cov_mat"""
    n_2 = ptv.size
    diag_mask = 1 - np.eye(n_2)
    ptv_mat = ptv.reshape((n_2, 1))
    covs = -(ptv_mat @ ptv_mat.T) * diag_mask
    vars_ = np.diag(ptv * (1 - ptv))
    return covs + vars_

def ptm_input_cov_mat(ptv):
    """Build a covariance matrix using a given PTV (new formula)
        Uses the Bernoulli Model"""
    n_2 = ptv.size
    n = np.log2(n_2).astype(np.uint16)
    kvv_ = kvv(ptv)
    Bn = B_mat(n)
    return Bn.T @ kvv_ @ Bn

def ptm_output_cov_mat(ptv, Mf):
    """Build an output covariance matrix using a given PTV and circuit PTM (new formula)"""
    _, k = np.log2(Mf.shape).astype(np.uint16)
    kvv_ = kvv(ptv)
    Bk = B_mat(k)
    return Bk.T @ Mf.T @ kvv_ @ Mf @ Bk

def cov_to_scc(cov, ptv):
    n, _ = cov.shape
    C = np.zeros((n, n))
    P = B_mat(n).T @ ptv
    for i in range(n):
        for j in range(n):
            if cov[i, j] > 0:
                if i == j:
                    assert np.isclose(cov[i, j], (np.minimum(P[i], P[j]) - P[i] * P[j]))
                C[i, j] = cov[i, j] / (np.minimum(P[i], P[j]) - P[i] * P[j])
            else:
                C[i, j] = cov[i, j] / (P[i] * P[j] - np.maximum(P[i] + P[j] - 1, 0))
    return C

def ptm_input_corr_mat(ptv):
    cov = ptm_input_cov_mat(ptv)
    return cov_to_scc(cov, ptv)

def ptm_output_corr_mat(ptv, Mf):
    cov_o = ptm_output_cov_mat(ptv, Mf)
    ptv_o = Mf.T @ ptv
    return cov_to_scc(cov_o, ptv_o)