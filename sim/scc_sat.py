#Simulations for the SCC satisfiability problem

import numpy as np
import bitstreams as bs

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

def N_required_overlaps(Ni, Nj, c, N):
    No_max = np.minimum(Ni, Nj)
    No_min = np.maximum(Ni + Nj - N, 0)
    
    #Old (wrong) method
    #return 0.5 * (c-1) * (No_max - No_min) + No_max

    PiNj = (Ni / N) * Nj
    cond = c * (No_max - PiNj)
    if cond > 0:
        return cond + PiNj
    else:
        return c * (PiNj - No_min) + PiNj 

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
            n_r_ov = N_required_overlaps(Ni, Nj, scc, max_N)
            #Verbose mode:
            #if n_a_ov == n_r_ov:
            #    print("PASSED: bs1={}, bs2={}, scc={}, ov={}".format(bs1_bin, bs2_bin, scc, n_a_ov))
            #else:
            #    print("FAILED: bs1={}, bs2={}, scc={}, a_ov={}, r_ov={}".format(bs1_bin, bs2_bin, scc, n_a_ov, n_r_ov))
            if n_a_ov != np.round(n_r_ov):
                print("FAILED: bs1={}, bs2={}, scc={}, a_ov={}, r_ov={}".format(bs1_bin, bs2_bin, scc, n_a_ov, n_r_ov))
                return
    print("PASSED")

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
        n_r_ov = N_required_overlaps(Ni, Nj, scc, max_N)
        if n_a_ov != np.round(n_r_ov):
            print("FAILED: bs1={}, bs2={}, scc={}, a_ov={}, r_ov={}".format(bs1_bin, bs2_bin, scc, n_a_ov, n_r_ov))
            return
    print("PASSED")

if __name__ == "__main__":
    """Test min_add_matmul"""
    A = np.array([
        [0, 2, 1, 4],
        [2, 0, 3, 6],
        [1, 3, 0, 3],
        [4, 6, 3, 0],
    ])

    A_sq = min_add_matmul(A, A)
    print(A_sq)
    A_cu = min_add_matmul(A_sq, A)
    print(A_cu)

    """Test N_actual_overlaps"""
    #bs1 = np.packbits(np.array([1,1,1,0,0,0]))
    #bs2 = np.packbits(np.array([1,0,1,0,1,0]))
    #print(N_actual_overlaps(bs1, bs2))

    """N overlaps sweep test"""
    #N_overlaps_sweep_test(10)

    """N overlaps random test"""
    #N_overlaps_rand_test(31, 1000000)