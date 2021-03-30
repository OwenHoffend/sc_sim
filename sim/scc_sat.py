#Simulations for the SCC satisfiability problem

import numpy as np
import bitstreams as bs
from scipy import special

def min_add_matmul(A, B):
    """Perform a matrix multiplication of two same-shaped square matrices, but map the operands
        + --> max, * --> +. This is inspired by the Floyd-Warshall Algorithm"""
    ma, na = A.shape
    mb, nb = B.shape
    if not (ma == na == mb == nb):
        raise ValueError("A and B must be same-shaped square matrices")

    ans = np.zeros((ma, ma))
    for i in range(ma):
        for j in range(ma):
            ans[i][j] = np.max(A[i, :] + B[:, j])
    return ans

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

def num_possible_sccs_eq(N, n, p_arr):
    """Compute the number of possible sccs via the candidate equation"""
    res = 1
    N_arr = p_arr * N
    for Ni in N_arr:
        res *= special.binom(N, Ni)
    return res / np.math.factorial(N)

def Mij(Ni, Nj, c, N):
    No_max = np.minimum(Ni, Nj)
    No_min = np.maximum(Ni + Nj - N, 0)

    PiNj = (Ni / N) * Nj
    cond = c * (No_max - PiNj)
    if cond > 0:
        return cond + PiNj
    else:
        return c * (PiNj - No_min) + PiNj

def possible_sccs_bf(N, p_arr):
    """Compute the number of possible sccs via a brute force count"""
    n = p_arr.size
    def get_sccs(Pi, Pj):
        sccs = set()
        Ni = np.round(Pi * N).astype(np.uint32)
        Nj = np.round(Pj * N).astype(np.uint32)
        PiNj = Pi * Nj
        No_max = np.minimum(Ni, Nj)
        No_min = np.maximum(Ni + Nj - N, 0)
        for No in range(No_min, No_max + 1):
            if No > PiNj:
                sccs.add((No - PiNj) / (No_max - PiNj))
            else:
                sccs.add((No - PiNj) / (PiNj - No_min))
        return sccs
    
    final_sccs = get_sccs(p_arr[0], p_arr[1])
    for i in range(1, n-1):
        for j in range(i+1, n):
            c = get_sccs(p_arr[i], p_arr[j])
            final_sccs = final_sccs.intersection(c)
    return final_sccs

def scc_sat(N, n, c_mat, p_arr, q_err_thresh=0.01, m_err_thresh=0.01, for_gen=False):
    """This is the primary SCC satisfaction function"""
    
    #Quantization error check (O(n))
    N_arr = np.round(N * p_arr).astype(np.uint32)
    if np.any(np.abs(N_arr - N * p_arr) > q_err_thresh):
        print("SCC SAT FAIL: Quantization error check failed.")
        return False

    #n=2 SCC satisfiability check (O(n^2))
    Dij = np.zeros((n,n), dtype=np.uint32)
    if for_gen: #Save the Mij matrix if this satisfaction check is being used for correlated bitstream generation
        Mij = np.zeros((n,n), dtype=np.uint32)
    for i in range(n):
        for j in range(i): #upper triangle only
            Ni, Nj = N_arr[i], N_arr[j]
            No_max = np.minimum(Ni, Nj)
            No_min = np.maximum(Ni + Nj - N, 0)
            PiNj = (Ni / N) * Nj
            cond = c_mat[i][j] * (No_max - PiNj)
            if cond > 0:
                m = cond + PiNj
            else:
                m =  c_mat[i][j] * (PiNj - No_min) + PiNj
            rm = np.round(m)
            if (not (No_min <= rm <= No_max)):
                print("SCC SAT FAIL: n=2 bounds check failed")
                return False
            if (np.abs(rm - m) > m_err_thresh): #Non-integer overlap count
                print("SCC SAT FAIL: n=2 integer check failed")
                return False
            if for_gen:
                Mij[i][j] = rm
            Dij[i][j] = Ni + Nj - 2*rm

    #n>2 SCC satisfiability check - could perhaps use a matrix multiplication method here too
    for i in range(n):
        for j in range(i): # i > j
            for k in range(j): # j > k 
                if k != i and k != j: 
                    Ds = Dij[i][j] + Dij[j][k] + Dij[i][k]
                    if Ds % 2 == 1: 
                        print("SCC SAT FAIL: n>2 evenness check failed")
                        return False
                    if Ds > 2 * N:
                        print("SCC SAT FAIL: n>2 magnitude check failed")
                        return False

    print("SCC SAT PASS")
    if for_gen:
        return True, Mij, N_arr
    return True

def pset(s):
    """Utility function for gen_multi_correlated"""
    if len(s) <= 0:
        yield []
    else:
        for item in pset(s[1:]):
            yield [s[0]]+item
            yield item

def gen_multi_correlated(N, n, c_mat, p_arr, verify=False):
    """Generate a set of bitstreams that are correlated according to the supplied correlation matrix"""

    #Test if the desired parameters are satisfiable
    sat_result = scc_sat(N, n, c_mat, p_arr, for_gen=True)
    if not sat_result:
        print("GEN_MULTI_CORRELATED FAILED: SCC matrix not satisfiable")
        return

    #Perform the generation
    Mij = sat_result[1]
    N_arr = sat_result[2]
    bs_arr = np.zeros((n,N), dtype=np.uint8)
    idx_set = pset(range(n))
    nxt = True

    b = 0
    while b < N:
        if nxt:
            idxs = next(idx_set)
            nxt = False
        if idxs == []:
            break
        isz = len(idxs)
        if np.all(N_arr[idxs] > 0) and \
            (isz == 1 or np.all(Mij[idxs, :][:, idxs] + np.triu(np.ones((isz, isz), dtype=np.uint8)) > 0)):
            bs_arr[idxs, b] = 1
            N_arr[idxs] -= 1
            Mij_temp = Mij[idxs, :]
            Mij_temp[:, idxs] -= np.tril(np.ones((isz, isz), dtype=np.uint8), -1)
            Mij[idxs, :] = Mij_temp
            b += 1
        else:
            nxt = True

    #Verify the generation
    print(bs_arr)
    if verify:
        cmat_actual = bs.get_corr_mat(bs_arr, bs_len=N)
        if np.any(np.abs(cmat_actual - c_mat) > 1e-6):
            print("GEN_MULTI_CORRELATED FAILED: Resulting SCC Matrix doesn't match: \n {} \n should be \n {}"
            .format(cmat_actual, c_mat))
            return False
        for idx, bs_i in enumerate(bs_arr):
            p_actual = bs.bs_mean(np.packbits(bs_i), bs_len=N)
            if p_actual != p_arr[idx]:
                print("GEN_MULTI_CORRELATED FAILED: Resulting probability is incorrect: {} (should be {})".format(p_actual, p_arr[idx]))
                return False
        print("GEN_MULTI_CORRELATED PASS")
    return True, np.packbits(bs_arr, axis=1)

def N_actual_overlaps(bs1, bs2):
    unp = np.unpackbits(np.bitwise_and(bs1, bs2))
    return np.sum(unp)

def N_overlaps_sweep_test(max_N):
    """Sweep through a range of possible bitstream configurations and compare the overlap to the correct value"""
    print("Total overlap sweep iters will be: {}".format(2 ** (2*max_N)))
    for bs1 in range(2 ** max_N):
        print("{} out of {} outer loops complete".format(bs1, 2 ** max_N))
        for bs2 in range(2 ** max_N):
            bs1_bin = bin_array(bs1, max_N)
            bs2_bin = bin_array(bs2, max_N)
            Ni = np.sum(bs1_bin)
            Nj = np.sum(bs2_bin)
            bs1_p = np.packbits(bs1_bin)
            bs2_p = np.packbits(bs2_bin)
            try:
                scc = bs.bs_scc(bs1_p, bs2_p, bs_len=max_N)
            except ValueError: #For configs where p1 or p2 are 0 or 1
                continue
            n_a_ov = N_actual_overlaps(bs1_p, bs2_p)
            n_r_ov = Mij(Ni, Nj, scc, max_N)
            #Verbose mode:
            #if n_a_ov == n_r_ov:
            #    print("PASSED: bs1={}, bs2={}, scc={}, ov={}".format(bs1_bin, bs2_bin, scc, n_a_ov))
            #else:
            #    print("FAILED: bs1={}, bs2={}, scc={}, a_ov={}, r_ov={}".format(bs1_bin, bs2_bin, scc, n_a_ov, n_r_ov))
            if n_a_ov != np.round(n_r_ov):
                print("FAILED: bs1={}, bs2={}, scc={}, a_ov={}, r_ov={}".format(bs1_bin, bs2_bin, scc, n_a_ov, n_r_ov))
                return
    print("PASSED")

def scc_sat_rand_test(max_n, max_N, iters):
    """Sweep through a set of random valid bit configurations, and verify that scc_sat reports true for all of them"""
    print("Total scc_sat random iters will be {}".format(iters))
    for i in range(iters):
        n = np.random.randint(1, max_n+1)
        N = np.random.randint(1, max_N+1)
        rng = bs.SC_RNG()
        bs_arr = [rng.bs_uniform(N, np.random.rand(), keep_rng=False) for _ in range(n)]
        p_arr = np.array([bs.bs_mean(s, bs_len=N) for s in bs_arr])
        if np.any(p_arr == 1.0) or np.any(p_arr == 0.0): #Filter out streams with undefined sccs wrt the others
            continue
        c_mat = bs.get_corr_mat(bs_arr, bs_len=N)
        if not scc_sat(N, n, c_mat, p_arr):
            print("FAILED with: N={}, n={}, c_mat=\n{}, p_arr={}".format(N, n, c_mat, p_arr))
            return
        print("Iter {} with N={}, n={} PASSED".format(i, N, n))
    print("OVERALL PASSED")

def gen_multi_corr_rand_test(max_n, max_N, iters):
    """Sweep through a set of random valid bit configurations, and verify that the reconstruction scc matrix matches the original"""
    print("Total scc_sat random iters will be {}".format(iters))
    for i in range(iters):
        n = np.random.randint(1, max_n+1)
        N = np.random.randint(1, max_N+1)
        rng = bs.SC_RNG()
        bs_arr = [rng.bs_uniform(N, np.random.rand(), keep_rng=False) for _ in range(n)]
        p_arr = np.array([bs.bs_mean(s, bs_len=N) for s in bs_arr])
        if np.any(p_arr == 1.0) or np.any(p_arr == 0.0): #Filter out streams with undefined sccs wrt the others
            continue
        c_mat = bs.get_corr_mat(bs_arr, bs_len=N)
        if not gen_multi_correlated(N, n, c_mat, p_arr, verify=True):
            return
        print("Iter {} with N={}, n={} PASSED".format(i, N, n))
    print("OVERALL PASSED")   

def N_overlaps_rand_test(max_N, iters):
    """Test a large number of random bitstream configurations and compare the overlap to the correct value"""
    print("Total overlap sweep iters will be: {}".format(iters))
    for _ in range(iters):
        bs1 = np.random.randint(0, 2 ** max_N)
        bs2 = np.random.randint(0, 2 ** max_N)
        bs1_bin = bin_array(bs1, max_N)
        bs2_bin = bin_array(bs2, max_N)
        Ni = np.sum(bs1_bin)
        Nj = np.sum(bs2_bin)
        bs1_p = np.packbits(bs1_bin)
        bs2_p = np.packbits(bs2_bin)
        try:
            scc = bs.bs_scc(bs1_p, bs2_p, bs_len=max_N)
        except ValueError: #For configs where p1 or p2 are 0 or 1
            continue
        n_a_ov = N_actual_overlaps(bs1_p, bs2_p)
        n_r_ov = Mij(Ni, Nj, scc, max_N)
        if n_a_ov != np.round(n_r_ov):
            print("FAILED: bs1={}, bs2={}, scc={}, a_ov={}, r_ov={}".format(bs1_bin, bs2_bin, scc, n_a_ov, n_r_ov))
            return
    print("PASSED")

if __name__ == "__main__":
    """Test min_add_matmul"""
    #A = np.array([
    #    [0, 2, 1, 4],
    #    [2, 0, 3, 6],
    #    [1, 3, 0, 3],
    #    [4, 6, 3, 0],
    #])

    #A_sq = min_add_matmul(A, A)
    #print(A_sq)
    #A_cu = min_add_matmul(A_sq, A)
    #print(A_cu)

    """Test N_actual_overlaps"""
    #bs1 = np.packbits(np.array([1,1,1,0,0,0]))
    #bs2 = np.packbits(np.array([1,0,1,0,1,0]))
    #print(N_actual_overlaps(bs1, bs2))

    """N overlaps sweep test"""
    #N_overlaps_sweep_test(10)

    """N overlaps random test"""
    #N_overlaps_rand_test(31, 1000000)

    """possible_sccs_bf test"""
    #p_arr = np.array([0.544343, 0.5, 0.9, 0.9])
    #print(possible_sccs_bf(4, p_arr))

    """Test SCC sat"""
    #scc_sat(6, 3, bs.mc_mat(-1, 3), np.array([0.333333, 0.333333, 0.33333])) #Example of a test case that passes
    #scc_sat(6, 3, bs.mc_mat(-1, 3), np.array([0.5, 0.5, 0.5])) #Example of a test case that passes n=2 but fails n>2
    #c_mat = np.array([
    #    [0, 0, 0],
    #    [0.25, 0, 0],
    #    [-0.25, -1, 0]
    #])
    #scc_sat(6, 3, c_mat, np.array([0.66666667, 0.66666667, 0.3333333])) #Example of a condition that passes using a correlation matrix
    #scc_sat_rand_test(3, 8, 1000000)

    """Test gen_multi_correlated"""
    #c_mat = np.array([
    #    [0, 0, 0],
    #    [0.25, 0, 0],
    #    [-0.25, -1, 0]
    #])
    #gen_multi_correlated(6, 3, c_mat, np.array([0.66666667, 0.66666667, 0.3333333]), verify=True)

    c_mat = np.array([
        [0,0,0],
        [-1,0,0],
        [-1,0.1111111111,0]
    ])
    p_arr = np.array([0.875, 0.625, 0.375])
    gen_multi_correlated(8, 3, c_mat, p_arr, verify=True)
    #gen_multi_corr_rand_test(3, 8, 10000)