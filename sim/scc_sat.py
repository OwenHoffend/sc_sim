#Simulations for the SCC satisfiability problem

import numpy as np
import bitstreams as bs
import itertools
import random
from scipy import special

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

def num_possible_sccs(N, p1, p2):
    """Compute the number of possible sccs via the overlap-shift equation"""
    nmax = N * np.maximum(p1, p2)
    nmin = N * np.minimum(p1, p2)
    return (N + 1) - nmax - np.maximum(N - nmax - nmin, 0)

def Mij(Ni, Nj, c, N):
    No_max = np.minimum(Ni, Nj)
    No_min = np.maximum(Ni + Nj - N, 0)

    PiNj = (Ni / N) * Nj
    cond = c * (No_max - PiNj)
    if cond > 0:
        return cond + PiNj
    else:
        return c * (PiNj - No_min) + PiNj

def corr_sat(N, n, c_mat, p_arr, q_err_thresh=1e-4, m_err_thresh=1e-4, for_gen=False, print_stat=True, is_mc=False, use_zscc=False):
    """This is the primary SCC satisfaction function"""
    
    #Quantization error check (O(n))
    N_arr = np.round(N * p_arr).astype(np.uint32)
    if np.any(np.abs(N_arr - N * p_arr) > q_err_thresh):
        if print_stat:
            print("SCC SAT FAIL: Quantization error check failed.")
        return False

    #n=2 SCC satisfiability check (O(n^2))
    Dij = np.zeros((n,n), dtype=np.uint32)
    for i in range(n):
        for j in range(i): #upper triangle only
            Ni, Nj = N_arr[i], N_arr[j]
            No_max = np.minimum(Ni, Nj)
            No_min = np.maximum(Ni + Nj - N, 0)
            PiNj = (Ni / N) * Nj
            c = c_mat[i][j]
            if use_zscc:
                delta0 = (np.floor(PiNj + 0.5) - PiNj)/N
                if c > 0:
                    m = c * np.abs(No_max - PiNj) - (N * (c-1) * np.abs(delta0)) + PiNj
                else:
                    m = c * np.abs(No_min - PiNj) - (N * (c+1) * np.abs(delta0)) + PiNj
            else:
                if c > 0:
                    m = c * (No_max - PiNj) + PiNj
                else:
                    m =  c * (PiNj - No_min) + PiNj
            rm = np.round(m)
            if (not (No_min <= rm <= No_max)):
                if print_stat:
                    print("SCC SAT FAIL: n=2 bounds check failed")
                return False
            if (np.abs(rm - m) > m_err_thresh): #Non-integer overlap counter
                if print_stat:
                    print("SCC SAT FAIL: n=2 integer check failed")
                return False
            Dij[i][j] = Ni + Nj - 2*rm

    #n>2 SCC satisfiability check
    #Magnitude check
    dsum = np.sum(Dij)
    if n % 2 == 0:
        dmax = N*((n/2) ** 2)
    else:
        dmax = N*((n**2) - 1)/4
    if dsum > dmax:
        if print_stat:
            print("SCC SAT FAIL: n>2 magnitude check failed")
        return False

    #Evenness check
    for i in range(n):
        for j in range(i): # i > j
            for k in range(j): # j > k 
                if k != i and k != j: 
                    Ds = Dij[i][j] + Dij[j][k] + Dij[i][k]
                    if Ds % 2 == 1:
                        if print_stat: 
                            print("SCC SAT FAIL: n>2 evenness check failed")
                        return False

    if is_mc: #FIXME
        if n * Dij[1][0] > 2 * N:
            if print_stat:
                print("SCC SAT FAIL: n>2 MC check failed")
            return False

    if print_stat:
        print("SCC SAT PASS @ N={}, n={}".format(N, n))
    if for_gen:
        return True, Dij, N_arr
    return True

def get_combs(N, N1):
    idxs = {s for s in range(N)}
    return itertools.combinations(idxs, N1)

def next_cand(N, N1, Dij, bs_arr, i):
    g = get_combs(N, N1)
    for c in g:
        bin_arr = np.zeros(N)
        bin_arr[list(c)] = 1
        valid = True
        for j in range(i):
            if bs.hamming_dist(bin_arr, bs_arr[j, :]) != Dij[i][j]:
                valid = False
        if valid:
            yield bin_arr

def gen_multi_correlated(N, n, c_mat, p_arr, verify=False, is_mc=False, pack_output=True, print_stat=True):
    """Generate a set of bitstreams that are correlated according to the supplied correlation matrix"""

    #Test if the desired parameters are satisfiable
    sat_result = corr_sat(N, n, c_mat, p_arr, for_gen=True, is_mc=is_mc, print_stat=print_stat)
    if not sat_result:
        if print_stat:
            print("GEN_MULTI_CORRELATED FAILED: SCC matrix not satisfiable")
        return False

    if print_stat:
        print(c_mat)
        print(p_arr)

    #Perform the generation
    Dij = sat_result[1]
    N_arr = sat_result[2]
    bs_arr = np.zeros((n,N), dtype=np.uint8)

    def gmc_rec(i):
        """Recursive portion of gen_multi_correlated"""
        nonlocal N, n, N_arr, Dij, bs_arr
        if i == n-1:
            sentinel = 's'
            last_cand = next(next_cand(N, N_arr[i], Dij, bs_arr, i), sentinel)
            if last_cand is not sentinel:
                bs_arr[i, :] = last_cand
                return True
            else:
                return False
        else:
            for cand in next_cand(N, N_arr[i], Dij, bs_arr, i):
                bs_arr[i, :] = cand
                if gmc_rec(i+1):
                    return True
            return False

    if not gmc_rec(0):
        if print_stat:
            print("GEN_MULTI_CORRELATED FAILED: Couldn't find a valid solution")
        return False

    #Verify the generation
    if print_stat:
        print(bs_arr)
    if verify:
        cmat_actual = bs.get_corr_mat(bs_arr, bs_len=N)
        if np.any(np.abs(cmat_actual - c_mat) > 1e-6):
            if print_stat:
                print("GEN_MULTI_CORRELATED FAILED: Resulting SCC Matrix doesn't match: \n {} \n should be \n {}"
                .format(cmat_actual, c_mat))
            return False
        for idx, bs_i in enumerate(bs_arr):
            p_actual = bs.bs_mean(np.packbits(bs_i), bs_len=N)
            if np.any(np.abs(p_actual - p_arr[idx]) > 1e-6):
                if print_stat:
                    print("GEN_MULTI_CORRELATED FAILED: Resulting probability is incorrect: {} (should be {})".format(p_actual, p_arr[idx]))
                return False
        if print_stat:
            print("GEN_MULTI_CORRELATED PASS")

    if pack_output:
        return True, np.packbits(bs_arr, axis=1)
    else:
        return True, bs_arr

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

def corr_sat_rand_test(max_n, max_N, iters, use_zscc=False):
    """Sweep through a set of random valid bit configurations, and verify that corr_sat reports true for all of them"""
    print("Total corr_sat random iters will be {}".format(iters))
    for i in range(iters):
        n = max_n #np.random.randint(1, max_n+1)
        N = max_N #np.random.randint(1, max_N+1)
        rng = bs.SC_RNG()
        bs_arr = [rng.bs_uniform(N, np.random.rand(), keep_rng=False) for _ in range(n)]
        p_arr = np.array([bs.bs_mean(s, bs_len=N) for s in bs_arr])
        if np.any(p_arr == 1.0) or np.any(p_arr == 0.0): #Filter out streams with undefined sccs wrt the others
            continue
        c_mat = bs.get_corr_mat(bs_arr, bs_len=N, use_zscc=use_zscc)
        print(c_mat)
        if not corr_sat(N, n, c_mat, p_arr, use_zscc=use_zscc):
            print("FAILED with: N={}, n={}, c_mat=\n{}, p_arr={}".format(N, n, c_mat, p_arr))
            return
        print("Iter {} with N={}, n={}, p_arr={} PASSED".format(i, N, n, p_arr))
    print("OVERALL PASSED")

def gen_multi_corr_rand_test(max_n, max_N, iters):
    """Sweep through a set of random valid bit configurations, and verify that the reconstruction scc matrix matches the original"""
    print("Total corr_sat random iters will be {}".format(iters))
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
    """Test N_actual_overlaps"""
    #bs1 = np.packbits(np.array([1,1,1,0,0,0]))
    #bs2 = np.packbits(np.array([1,0,1,0,1,0]))
    #print(N_actual_overlaps(bs1, bs2))

    """N overlaps sweep test"""
    #N_overlaps_sweep_test(10)

    """N overlaps random test"""
    #N_overlaps_rand_test(31, 1000000)

    """Test SCC sat"""
    #corr_sat(6, 3, bs.mc_mat(-1, 3), np.array([0.333333, 0.333333, 0.333333]), use_zscc=True) #Example of a test case that passes
    #corr_sat(6, 3, bs.mc_mat(-1, 3), np.array([0.5, 0.5, 0.5])) #Example of a test case that passes n=2 but fails n>2
    #c_mat = np.array([
    #    [0, 0, 0],
    #    [0.25, 0, 0],
    #    [-0.25, -1, 0]
    #])
    #corr_sat(6, 3, c_mat, np.array([0.66666667, 0.66666667, 0.3333333])) #Example of a condition that passes using a correlation matrix
    corr_sat_rand_test(10, 128, 1000000, use_zscc=True)

    """A ZSCC test"""
    bs2 = np.packbits(np.array([0,0,0,0,1,0]))
    bs1 = np.packbits(np.array([1,1,1,1,1,0]))
    bs3 = np.packbits(np.array([0,0,1,1,1,1]))
    c_mat = bs.get_corr_mat([bs1, bs2, bs3], bs_len=6, use_zscc=True)
    corr_sat(6, 3, c_mat, np.array([5/6, 1/6, 4/6]), use_zscc=True)

    """Test gen_multi_correlated"""
    #c_mat = np.array([
    #    [0, 0, 0],
    #    [0.25, 0, 0],
    #    [-0.25, -1, 0]
    #])
    #gen_multi_correlated(24, 3, c_mat, np.array([0.66666667, 0.66666667, 0.3333333]), verify=True)

    #c_mat = np.array([
    #    [0,0,0],
    #    [-1,0,0],
    #    [-1,0.1111111111,0]
    #])
    #p_arr = np.array([0.875, 0.625, 0.375])
    #gen_multi_correlated(16, 3, c_mat, p_arr, verify=True)
    #gen_multi_corr_rand_test(8, 24, 10000)
    #plot_mcc_change()

