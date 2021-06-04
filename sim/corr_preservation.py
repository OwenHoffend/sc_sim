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
        No_predicted = np.zeros((n, n))
        for i in range(n):
            for j in range(i):
                Ni = N * p_arr[i]
                Nj = N * p_arr[j]
                c = c_mat[i][j]
                No_predicted[i][j] = sat.Mij(Ni, Nj, c, N, use_zscc=use_zscc)
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

if __name__ == "__main__":
    """Test B_mat"""
    #Vin = np.array([1/6, 0, 0, 1/6, 1/6, 1/6, 1/6, 1/6])
    #print(np.matmul(B_mat(3).T, Vin))

    """Test get_func_mat"""
    #test_func = lambda a, b: np.array([a & b])
    #print(get_func_mat(test_func, 2, 1))

    """test_num_overlaps"""
    test_num_overlaps(3, 24, 10000)