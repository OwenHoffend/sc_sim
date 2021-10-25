from sim.corr_preservation import *
from sim.PTM import *
from sim.circuits import *

def test_circular_shift_compare():
    shifts = 2
    n = 4
    k = 2
    val = 14
    func = lambda *x: circular_shift_compare(shifts, k, val, *x)
    Mf = get_func_mat(func, n, k)
    print(Mf)
    for i in range(1, 5):
        print(circular_shift_corr_sweep(5, 2, i))

def test_B_mat():
    Vin = np.array([1/6, 0, 0, 1/6, 1/6, 1/6, 1/6, 1/6])
    print(B_mat(3).T @ Vin)

def test_get_func_mat():
    test_func = lambda a, b: np.array([a & b])
    print(get_func_mat(test_func, 2, 1))
    print(get_func_mat(xor_4_to_2, 4, 2).T)

def test_get_output_corr_mat():
    Mf = get_func_mat(and_3, 3, 3)
    N = 6
    Vin = get_vin_mc1(np.array([3/6, 4/6, 5/6]))
    Vin = get_vin_mcn1(np.array([1/6, 3/6, 2/6]))
    print(get_output_corr_mat(Vin, Mf, N)[1])

def test_get_vin_iterative():
    bs1 = np.array([1,1,1,1,0,0])
    bs2 = np.array([1,1,1,0,0,0])
    bs3 = np.array([1,1,1,1,1,0])
    bs_arr = [bs1, bs2, bs3]
    bs_arr_p = [np.packbits(x) for x in bs_arr]
    p_vec = np.array([bs.bs_mean(b, bs_len=6) for b in bs_arr_p])
    V = get_vin_mc1(p_vec)
    print(V)
    cov_mat = bs.get_corr_mat(bs_arr_p, bs_len=6, use_cov=True)
    cov_mat2 = np.cov(np.array(bs_arr), bias=True) #Bias=True sets this to be a population variance
    print(get_vin_iterative(p_vec, cov_mat, 6))
    print(cov_mat)
    print(cov_mat2)
    print(ptm_input_cov_mat(V, 6))

def test_get_vin_mc1():
    print(get_vin_mc1(np.array([5/6, 3/6, 4/6])))

def test_get_vin_mcn1():
    print(get_vin_mcn1(np.array([1/6, 2/6, 2/6])))

def test_get_vin_mc0():
    print(get_vin_mc0(np.array([1/3, 1/4])))

def test_e_err():
    Mf = get_func_mat(and_3, 3, 3)
    Mf = get_func_mat(xor_3, 3, 3)
    print("{}\n{}".format(*err_sweep(16, Mf, -1)))

def test_reduce_func_mat():
    Mf = get_func_mat(mux_2, 5, 2)
    print(Mf.astype(np.uint16))
    print(reduce_func_mat(Mf, 4, 0.7))

def test_mux_corr_pres():
    Mf = get_func_mat(mux_2, 5, 2)
    Mf = get_func_mat(unbalanced_mux_2, 4, 2)
    N = 20
    err_tot = np.zeros((2, 2))
    for i in range(1, N):
        #Mfr = reduce_func_mat(Mf, 4, i/N)
        Mfr = reduce_func_mat(Mf, 3, i/N)
        max_err, err = err_sweep(N, Mfr, 0)
        err_tot += err
        print(i)
    err_tot /= N
    print(err_tot)

def test_xor_corr_pres():
    Mf = get_func_mat(xor_4_to_2, 4, 2)
    print(corr_uniform_rand_1layer(1, Mf, get_vin_mc1, np.ones((2, 2)), 10000))

def test_unbalanced_mux_corr_pres():
    Mf1 = reduce_func_mat(get_func_mat(unbalanced_mux_2, 4, 2), 3, 0.5)
    Mf2 = reduce_func_mat(get_func_mat(mux_1, 3, 1), 2, 0.5)
    Px = np.array([0.4, 0.1, 0.9])
    print(corr_err(Px, Mf1, Mf2, 1, 1))
    print(err_sweep(32, Mf1, -1, err_type='c', Mf2=Mf2))

def test_balanced_mux_corr_pres():
    def layer1(w1, w2, x1, x2, x3, x4):
        return mux_1(w1, x1, x2), mux_1(w2, x3, x4)
    Mf1 = reduce_func_mat(get_func_mat(layer1, 6, 2), 4, 0.5)
    Mf1 = reduce_func_mat(Mf1, 4, 0.5)
    Mf2 = reduce_func_mat(get_func_mat(mux_1, 3, 1), 2, 0.5)
    print(err_sweep(20, Mf1, 1, err_type='c', Mf2=Mf2))

def test_roberts_cross_corr_pres():
    Mf1 = get_func_mat(xor_4_to_2, 4, 2)
    Mf2 = reduce_func_mat(get_func_mat(mux_1, 3, 1), 2, 0.5)
    print(err_sweep(20, Mf1, 1, err_type='c', Mf2=Mf2))

def test_xor_and_corr_pres():
    Mf1 = get_func_mat(xor_4_to_2, 4, 2)
    Mf2 = get_func_mat(np.bitwise_and, 2, 1)
    Mf1c = torch.tensor(Mf1.astype(np.float32)).to(device)
    Mf2c = torch.tensor(Mf2.astype(np.float32)).to(device)

    #print(err_sweep(4, Mf1, 1, err_type='c', Mf2=Mf2))
    print(err_sweep_cuda(4, Mf1c, 1, err_type='c', Mf2=Mf2c))