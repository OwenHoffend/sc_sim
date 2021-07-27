import typing_extensions
import numpy as np
import torch
from numpy.lib import unpackbits
import sim.bitstreams as bs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

B_mat_dict = {}
def B_mat(n, cuda=False):
    """Generate a 2^n by n matrix of all of the possible values of n bits"""
    if n in B_mat_dict.keys():
        return B_mat_dict[n]
    B = np.zeros((2 ** n, n), dtype=bool)
    for i in range(2 ** n):
        #B[i][:] = bin_array(i, n)[::-1] #Might cause issues with endianness... right now it's 1 --> [True, False, False]
        B[i][:] = bin_array(i, n) #The old one is the line above
    if cuda:
        B = torch.tensor(B.astype(np.float32)).to(device)
    B_mat_dict[n] = B
    return B

def ptm_based_cov_mat(V, p_vec, N):
    """Build a covariance matrix using a ptm vector V"""
    n = np.log2(V.size).astype(np.uint32)
    C = np.zeros((n, n))
    Bn = B_mat(n)
    for q in range(2 ** n):
        c = np.zeros((n, n))
        for i in range(n):
            for j in range(i):
                pij = Bn[q, i] & Bn[q, j]
                c[i, j] = pij - p_vec[i] * p_vec[j]
        C += c * V[q]
    return C

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
    N, n = bs_mat.shape
    Vin = np.zeros(2 ** n)
    uniques, counts = np.unique(bs_mat, axis=0, return_counts=True)
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

def get_output_corr_mat(Vin, Mf, N, use_zscc=True):
    """Using ZSCC, compute the output correlation matrix given an input correlation matrix,
    input probabilities, and the circuit's truth table"""
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

def corr_err(Px, Mf1, Mf2, c1, c2, N=128):
    """Get the correlation error produced by a two-layer circuit"""
    def get_vin(Pin, c):
        if c == -1:
            return get_vin_mcn1(Pin)
        elif c == 1:
            return get_vin_mc1(Pin)
        elif c == 0:
            return get_vin_mc0(Pin)
        else:
            return get_vin_iterative(Pin, c, N) #If c is not a scalar constant, interpret as a covariance matrix
    n1, k1 = np.log2(Mf1.shape).astype(np.uint16)
    k_check, k2 = np.log2(Mf2.shape).astype(np.uint16)
    assert k_check == k1
    Vc1 = get_vin(Px, c1)
    Bk1 = B_mat(k1)
    Bk2 = B_mat(k2)
    Vz1_actual = Mf1.T @ Vc1
    Vz1_ideal = get_vin(Bk1.T @ Vz1_actual, c2)
    return np.abs(Bk2.T @ Mf2.T @ (Vz1_actual - Vz1_ideal))

def corr_err_cuda(Px, Mf1, Mf2, c1, c2, N=128):
    """Get the correlation error produced by a two-layer circuit"""
    def get_vin_cuda(Pin, c):
        if c == -1:
            return get_vin_mcn1_cuda(Pin)
        elif c == 1:
            return get_vin_mc1_cuda(Pin)
        elif c == 0:
            return get_vin_mc0_cuda(Pin)
        else:
            raise NotImplementedError
    n1, k1 = np.log2(Mf1.shape).astype(np.uint16)
    k_check, k2 = np.log2(Mf2.shape).astype(np.uint16)
    assert k_check == k1
    Vc1 = get_vin_cuda(Px, c1) #Vc1 old: 2 ** n1 x 1, new: 2 ** n1 x b
    Bk1 = B_mat(k1, cuda=True)
    Bk2 = B_mat(k2, cuda=True)
    Vz1_actual = Mf1.T @ Vc1 #2 ** k1 x b
    Vz1_ideal = get_vin_cuda((Bk1.T @ Vz1_actual).T, c2)
    Vz1_ideal_test = torch.zeros_like(Vz1_ideal)
    for i in range(Px.shape[0]):
        Vz1_ideal_test[:, i] = get_vin_mc0_cuda_test((Bk1.T @ Vz1_actual).T[i, :])
    assert torch.all(Vz1_ideal == Vz1_ideal_test)
    return torch.abs(Bk2.T @ Mf2.T @ (Vz1_actual - Vz1_ideal))

def err_sweep(N, Mf, vin_type, err_type='e', Mf2=None):
    n, k = np.log2(Mf.shape).astype(np.uint16)
    if Mf2 is not None:
        _, k = np.log2(Mf2.shape).astype(np.uint16)
    p_arr = np.zeros(n)
    err = np.zeros((k, k))
    max_err = np.zeros((k, k))
    for i in range((N - 1) ** n):
        for j in range(n):
            p_arr[j] = (np.floor(i / ((N-1) ** j)) % (N - 1) + 1) / N
        if err_type == 'e':
            if vin_type == 1:
                vin = get_vin_mc1(p_arr)
            elif vin_type == -1:
                if np.sum(p_arr) > 1:
                    continue
                vin = get_vin_mcn1(p_arr)
            elif vin_type == 0:
                vin = get_vin_mc0(p_arr)
            else:
                raise ValueError("Not valid vin type")
            new_err = e_err(vin, Mf, N, vin_type)
        elif err_type == 'c':
            new_err = corr_err(p_arr, Mf, Mf2, vin_type, 0, N=N)
        else:
            raise ValueError("Not a valid error type")
        #with open('err_sweep_test.txt', 'a') as outfile:
        #    outfile.write(str(new_err) + '\n')
        err += new_err
        max_err = np.maximum(max_err, new_err)
    print(test)
    return max_err, err / (N - 1) ** n

def err_sweep_cuda(N, Mf, vin_type, err_type='e', Mf2=None):
    """CUDA-accelerated version of err_sweep"""
    n, k = np.log2(Mf.shape).astype(np.uint16)
    if Mf2 is not None:
        _, k = np.log2(Mf2.shape).astype(np.uint16)
    batch_size = N-1
    nprobs = (N - 1) ** n
    p_arr = torch.cuda.FloatTensor(nprobs, n).fill_(0)
    err = torch.cuda.FloatTensor(k, k).fill_(0)
    max_err = torch.cuda.FloatTensor(k, k).fill_(0)
    for i in range(nprobs):
        for j in range(n):
            p_arr[i, j] = (np.floor(i / ((N-1) ** j)) % (N - 1) + 1) / N

    for b in range((nprobs / batch_size).astype(np.uint32)):
        p_range = p_arr[b:(b+batch_size), :]
        if err_type == 'e':
            if vin_type == 1:
                vin = get_vin_mc1_cuda(p_range)
            elif vin_type == -1:
                if np.sum(p_range) > 1:
                    continue
                vin = get_vin_mcn1_cuda(p_range)
            elif vin_type == 0:
                vin = get_vin_mc0_cuda(p_range)
            else:
                raise ValueError("Not valid vin type")
            new_err = e_err_cuda(vin, Mf, N, vin_type)
        elif err_type == 'c':
            new_err = corr_err_cuda(p_range, Mf, Mf2, vin_type, 0, N=N)
        else:
            raise ValueError("Not a valid error type")
        #with open('err_sweep_test_2.txt', 'a') as outfile:
        #    for i in range(batch_size):
        #        outfile.write(str(new_err[i]) + '\n')
        err += torch.sum(new_err)
        max_err = torch.maximum(max_err, torch.max(new_err))
    return max_err, err / nprobs

def e_err(Vin, Mf, N, vin_type):
    _, k = np.log2(Mf.shape).astype(np.uint16)
    Bk = B_mat(k)

    Vout = Mf.T @ Vin
    Pout = Bk.T @ Vout

    err = np.zeros((k, k))
    for i in range(k):
        for j in range(i):
            s = np.sum(np.multiply(np.bitwise_and(Bk[:, i], Bk[:, j]), Vout))
            if vin_type == 1:
                correct = np.minimum(Pout[i], Pout[j])
            elif vin_type == -1:
                correct = np.maximum(Pout[i] + Pout[j] - 1, 0)
            else:
                correct = np.floor(Pout[i] * Pout[j] * N + 0.5)/N
            err[i, j] = 2 * np.abs(s - correct) / (s + correct + 1e-15)
    return err

def e_err_cuda(Vin, Mf, N, vin_type):
    raise NotImplementedError

def get_func_mat(func, n, k):
    """Compute the truth table matrix for a boolean function with n inputs and k outputs
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

def reduce_func_mat(Mf, idx, p):
    """Reduce a PTM matrix with a known probability value on one input"""
    n, k = np.log2(Mf.shape).astype(np.uint16)
    ss1, ss2 = [], []
    for i in range(2 ** n):
        if i % (2 ** (idx + 1)) < 2 ** idx:
            ss1.append(i)
        else:
            ss2.append(i)
    Mff = Mf.astype(np.float32)
    return Mff[ss1, :] * p + Mff[ss2, :] * (1-p)

def circular_shift_compare(shifts, k, comp, *x):
    """Simulation of the circular shift alg"""
    vals = []
    x_np = np.array(x)
    for i in range(k):
        x_np_int = np.sum([v * (2 ** ind) for ind, v in enumerate(x_np[::-1])])
        vals.append(comp[i] > x_np_int)
        x_np = np.roll(x_np, -shifts)
    return (*vals,)

def circular_shift_corr_sweep(n, k, shifts):
    Vin = np.array([0, ] + [1 / (2 ** n - 1) for _ in range(2 ** n - 1)])
    scc_mat = np.zeros((k, k))
    for i in range(2 ** n):
        for j in range(2 ** n):
            func = lambda *x: circular_shift_compare(shifts, k, [i, j], *x)
            Mf = get_func_mat(func, n, k)
            scc_mat += np.abs(get_output_corr_mat(Vin, Mf, 2 ** n, use_zscc=False)[1])
        #print(i)
    return scc_mat / ((2 ** n) ** 2)