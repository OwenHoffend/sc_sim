import numpy as np
import sim.bitstreams as bs
from sim.PTM import *
from sim.bitstreams import *

def uniform_ptv_test(v_c, m, n, N):
    Bn = B_mat(n)
    q = 1.0 / N
    c_mat_avg = np.zeros((n, n))
    c_pred_avg = np.zeros((n, n))
    m_actual = 0
    for i in range(m):
        Px = np.random.uniform(size=n)
        #Px /= np.linalg.norm(Px, 1) #Scale by L1-norm to ensure probabilities sum to 1

        ptv = v_c(Px)
        if ptv is None:
            continue
        if not np.isclose(np.sum(ptv), 1.0):
            print("FAILED: ptv sum is wrong: sum: {}, ptv: {}".format(np.sum(ptv), ptv))
            return False

        #Test that the ptv reduction probabilities match
        Px_test = Bn.T @ ptv
        if not np.all(np.isclose(Px, Px_test)):
            print("Px FAILED: \n Px: {}, \n Px_test: {}, \n ptv: {}".format(Px, Px_test, ptv))
            return False

        #Generate bitstreams from the ptv
        bs_mat = sample_from_ptv(ptv, N)
        _, N_new = bs_mat.shape

        #Skip if any of the bitstreams are 0 or 1
        sums = np.sum(bs_mat, 1)
        if 0 in sums or N_new in sums:
            continue
        m_actual += 1

        #Test that the correlation matches

        #Predicted correlation matrix
        c_pred_avg += get_corr_mat_paper(ptv)

        #Actual correlation matrix
        c_mat_avg += get_corr_mat_np(bs_mat)

    c_pred_avg /= m_actual
    c_mat_avg /= m_actual
    print("Avg predicted corr mat: \n", c_pred_avg)
    print("Avg actual corr mat: \n", c_mat_avg)
    #print("PASSED")
    return True

def test_ptv_gen():
    n = 5
    m = 1000
    N = 1000
    #print("Testing +1 PTV generation")
    #assert uniform_ptv_test(get_vin_mc1_paper, m, n, N)
    #(Works always)

    #print("Testing 0 PTV generation")
    #assert uniform_ptv_test(get_vin_mc0, m, n, N)
    #(Works on average, probably if satisfiable)

    #print("Test -1 PTV generation")
    #assert uniform_ptv_test(get_vin_mcn1, m, n, N)
    #(Works always, if satisfiable)

    #print("Test any c PTV generation")
    #c = -0.3
    #func = lambda Px: get_vin_mc_any(Px, c)
    #assert uniform_ptv_test(func, m, n, N)
    #(Works on average, probably if satisfiable)

    #print("Test hybrid (contiguous) PTV generation")
    #def hybrid1(Px):
    #    S1 = get_vin_mc1(Px[0:3])
    #    S2 = get_vin_mc1(Px[3:5])
    #    return np.kron(S2, S1)
    #assert uniform_ptv_test(hybrid1, m, 5, N)
    #(Works, but it appears that the kron is backwards from what I would expect)

    print("Testing hybrid (non-contiguous) PTV generation")
    def hybrid2(Px):
        S1 = get_vin_mc1(np.array([Px[0], Px[2]]))
        S2 = get_vin_mc1(np.array([Px[1], Px[3]]))
        pre_swap = np.kron(S2, S1)
        swap_inds = np.array([0, 2, 1, 3])
        return PTV_swap_cols(pre_swap, swap_inds)
    assert uniform_ptv_test(hybrid2, m, 4, N)
    #(Works when the generated ptv is propery reordered)

def testing_for_paper():
    test_ptv_gen()