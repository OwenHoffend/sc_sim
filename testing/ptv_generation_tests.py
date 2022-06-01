from numpy.lib.shape_base import kron
from sim.PTM import *
from sim.bitstreams import get_corr_mat_np
np.set_printoptions(linewidth=np.inf)

def test_ptv_genfunc(v_c, n=4, N=500000):
    p_arr = np.random.uniform(size=n)
    p_arr /= np.linalg.norm(p_arr, 1) #Scale by L1-norm to ensure probabilities sum to 1
    ptv = v_c(p_arr)
    bs_mat = sample_from_ptv(ptv, N)
    actual_ptv = get_actual_vin(bs_mat)

    c_mat = get_corr_mat_np(bs_mat)
    print(np.round(c_mat, 3))

def test_ptv_to_corr(v_c, n=4, N=100):
    p_arr = np.random.uniform(size=n)
    ptv = v_c(p_arr)
    print(ptm_input_corr_mat(ptv))

def test_0_ptv():
    test_ptv_genfunc(get_vin_mc0)

def test_1_ptv():
    test_ptv_genfunc(get_vin_mc1)

def test_n1_ptv():
    test_ptv_genfunc(get_vin_mcn1)

def test_0r5_ptv():
    vin_mc0r5 = lambda pin: (get_vin_mc0(pin) + get_vin_mc1(pin)) / 2
    test_ptv_genfunc(vin_mc0r5)

def test_kron_ptv(n):
    def kron_ptv(pin):
        #Half sets - for now
        n2 = int(n/2)
        vin1 = get_vin_mc1(pin[0:n2])
        vin2 = get_vin_mc1(pin[n2:n])
        return np.kron(vin1, vin2)
    test_ptv_genfunc(kron_ptv, n=n)

def ptv_generation_tests_main():
    test_kron_ptv(8)

def test_ptv_swap():
    """Generate the following correlation matrix:
    x_1 1 1 0 0 0
    x_3 1 1 0 0 0
    x_2 0 0 1 0 0
    x_4 0 0 0 1 0
    x_5 0 0 0 0 1

    Then do a symmetric permutation to get:
    x_1 1 0 1 0 0
    x_2 0 1 0 0 0
    x_3 1 0 1 0 0
    x_4 0 0 0 1 0
    x_5 0 0 0 0 1
    """
    N = 10000
    p_arr = np.random.uniform(size=5)
    print(p_arr)
    top_ptv = get_vin_mc0(np.array([p_arr[1], p_arr[3], p_arr[4]]))
    bot_ptv = get_vin_mc1(np.array([p_arr[0], p_arr[2]]))

    ptv_pre_perm = np.kron(top_ptv, bot_ptv)
    b5 = B_mat(5)
    bs_mat = sample_from_ptv(ptv_pre_perm, N)
    actual_ptv_pre_perm = get_actual_vin(bs_mat)
    print(get_corr_mat_paper(actual_ptv_pre_perm))
    print(list(reversed(b5.T @ actual_ptv_pre_perm)))

    ptv = PTV_swap_cols(ptv_pre_perm, [0, 2, 1, 3, 4])
    bs_mat = sample_from_ptv(ptv, N)
    actual_ptv = get_actual_vin(bs_mat)
    print(get_corr_mat_paper(actual_ptv))
    print(list(reversed(b5.T @ actual_ptv)))

def test_ptv_swap_2():
    """Generate the following correlation matrix:
    x_1 1 1 0 0 0
    x_3 1 1 0 0 0
    x_2 0 0 1 0 0
    x_4 0 0 0 1 0
    x_5 0 0 0 0 1

    Then do a symmetric permutation to get:
    x_1 1 0 0 0 0
    x_2 0 1 0 1 0
    x_3 0 0 1 0 0
    x_4 0 1 0 1 0
    x_5 0 0 0 0 1
    """
    N = 10000
    p_arr = np.random.uniform(size=5)
    print(p_arr)
    top_ptv = get_vin_mc0(np.array([p_arr[1], p_arr[2], p_arr[4]]))
    bot_ptv = get_vin_mc1(np.array([p_arr[0], p_arr[3]]))

    ptv_pre_perm = np.kron(bot_ptv, top_ptv)
    b5 = B_mat(5)
    bs_mat = sample_from_ptv(ptv_pre_perm, N)
    actual_ptv_pre_perm = get_actual_vin(bs_mat)
    print(get_corr_mat_paper(actual_ptv_pre_perm))
    print(list(reversed(b5.T @ actual_ptv_pre_perm)))

    ptv = PTV_swap_cols(ptv_pre_perm, [3, 1, 2, 0, 4])
    bs_mat = sample_from_ptv(ptv, N)
    actual_ptv = get_actual_vin(bs_mat)
    print(get_corr_mat_paper(actual_ptv))
    print(list(reversed(b5.T @ actual_ptv)))

def test_corr_mat_perturbation():
    delta = 0.5
    N = 100000
    p_arr = np.random.uniform(size=4)

    #zero correlation to start
    base_ptv = get_vin_mc0(p_arr)
    ptv_0 = get_vin_mc0(p_arr)
    base_corr = np.kron(get_vin_mc1(p_arr[2:]), get_vin_mc1(p_arr[:2])) 
    result = base_ptv + delta * (base_corr - ptv_0) 
    print(np.sum(result))
    bs_mat = sample_from_ptv(result, N)
    print(get_corr_mat_paper(get_actual_vin(bs_mat)))
