import numpy as np
from numpy.lib import unpackbits
import scc_sat as sat
import bitstreams as bs

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

def B_mat(n):
    """Generate a 2^n by n matrix of all of the possible values of n bits"""
    B = np.zeros((2 ** n, n), dtype=bool)
    for i in range(2 ** n):
        B[i][:] = bin_array(i, n)
    return B

def ptm_based_cov_mat(V, p_vec, N):
    """Build a covariance matrix using a ptm vector V"""
    n = np.log2(V.size).astype(np.uint32)
    C = np.zeros((n, n))
    Bn = B_mat(n)
    for q in range(2 ** n):
        c = np.zeros((n, n))
        for i in range(n):
            for j in range(j):
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
    Pin_sorted = np.append(np.append(1, np.sort(Pin)[::-1]), 0)
    for i in range(n+1):
        Vin[2 ** i - 1] = Pin_sorted[i] - Pin_sorted[i + 1]
    return Vin

def get_output_corr_mat(Vin, Mf, N):
    """Using ZSCC, compute the output correlation matrix given an input correlation matrix,
    input probabilities, and the circuit's truth table"""
    n, k = np.log2(Mf.shape).astype(np.uint16)
    Bk = B_mat(k)

    Vout = Mf.T @ Vin
    Pout = Bk.T @ Vout

    #Get Cout
    No_out = Ov((Bk * N * np.column_stack(n*[Vout])).astype(np.uint32))
    Cout = np.zeros((k, k))
    for i in range(k):
        for j in range(i):
            Cout[i][j] = bs.bs_zscc_ovs(Pout[i], Pout[j], No_out[i][j], N)
    return Vout, Cout

def get_func_mat(func, n, k):
    """Compute the truth table matrix for a boolean function with n inputs and k outputs
        Does not handle probabilistic functions, only pure boolean functions"""
    Mf = np.zeros((2 ** n, 2 ** k), dtype=bool)
    for i in range(2 ** n):
        res = func(*list(bin_array(i, n)))
        num = 0
        for idx, j in enumerate(res):
            if j:
                num += 1 << idx
        Mf[i][num] = 1
    return Mf

def test_num_overlaps(max_n, max_N, iters, use_zscc=True):
    """Quick random test to be sure that Mij through ZSCC returns the same overlap matrix as count_overlaps"""
    for _ in range(iters):
        n = max_n
        N = max_N
        rng = bs.SC_RNG()
        bs_arr = [rng.bs_uniform(N, np.random.rand(), keep_rng=False) for _ in range(n)]
        p_arr = np.array([bs.bs_mean(s, bs_len=N) for s in bs_arr])
        if np.any(p_arr == 1.0) or np.any(p_arr == 0.0): #Filter out streams with undefined sccs wrt the others
            continue
        c_mat = bs.get_corr_mat(bs_arr, bs_len=N, use_zscc=use_zscc)
        No_correct = np.zeros((n, n))
        No_predicted = Ov(np.array([np.unpackbits(x) for x in bs_arr]).T) #np.zeros((n, n))
        for i in range(n):
            for j in range(i):
                #Ni = N * p_arr[i]
                #Nj = N * p_arr[j]
                #c = c_mat[i][j]
                #No_predicted[i][j] = sat.Mij(Ni, Nj, c, N, use_zscc=use_zscc)
                No_correct[i][j] = sat.N_actual_overlaps(bs_arr[i], bs_arr[j])
        if not np.allclose(No_predicted, No_correct):
            print("FAIL: \n{} != \n{}".format(No_predicted, No_correct))
            print("bs_arr: {}".format([np.unpackbits(x) for x in bs_arr]))
            print("P_arr: {}".format(p_arr))
            print("c_mat: {}".format(c_mat))
            return False
        print("Case PASS: {}".format(No_predicted))
    print("OVERALL PASS")
    return True

def and_3(a, b, c):
    o1 = np.bitwise_and(a, b)
    o2 = np.bitwise_and(b, c)
    o3 = np.bitwise_and(a, c)
    return o1, o2, o3 

if __name__ == "__main__":
    """Test B_mat"""
    #Vin = np.array([1/6, 0, 0, 1/6, 1/6, 1/6, 1/6, 1/6])
    #print(B_mat(3).T @ Vin))

    """Test get_func_mat"""
    #test_func = lambda a, b: np.array([a & b])
    #print(get_func_mat(test_func, 2, 1))

    """Test Ov"""
    #test_num_overlaps(6, 24, 1000)
    #a = np.array([
    #    [0, 0, 0],
    #    [0, 1, 1],
    #    [1, 0, 0],
    #    [2, 2, 0]
    #])
    #print(Ov(a))

    """Test get_output_corr_mat"""
    #Cin = None
    #Pin = None
    #Mf = get_func_mat(and_3, 3, 3)
    #N = 6
    #get_output_corr_mat(Cin, Pin, Mf, N)

    """test ptm_based_corr_mat"""
    bs1 = np.packbits(np.array([1,0,0,0,0,0]))
    bs2 = np.packbits(np.array([0,1,1,1,0,0]))
    bs3 = np.packbits(np.array([0,0,0,0,1,1]))
    bs_arr = [bs1, bs2, bs3]
    p_vec = np.array([bs.bs_mean(b, bs_len=6) for b in bs_arr])
    V = get_vin_mc1(p_vec)

    cov_mat = bs.get_corr_mat(bs_arr, bs_len=6, use_cov=True)

    print(get_vin_iterative(p_vec, cov_mat, 6))
    #print(cov_mat)
    #print(ptm_based_cov_mat(V, p_vec, 6))

    """Test get_vin_mc1"""
    #print(get_vin_mc1(np.array([3/6, 4/6, 5/6])))