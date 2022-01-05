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