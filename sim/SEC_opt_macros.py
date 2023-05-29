import numpy as np
from sim.PTM import *
from sim.SEC import *
from sim.bitstreams import hamming_dist
from symbolic_manip import mat_to_latex

class IO_Params:
    def __init__(self, nc, nv, k):
        self.nc = nc
        self.nv = nv
        self.k = k
        self.n = nc + nv
        self.nc2 = 2**nc
        self.nv2 = 2**nv
        self.n2 = 2**self.n
        self.k2 = 2**k

def ilog2(a):
    return int(np.log2(a).astype(int))

def compute_vout(ptm, ptm_opt, x, io, x_corr=True):
    if x_corr:
        vx = get_vin_mc1(x)
    else:
        vx = get_vin_mc0(x)
    v0 = get_vin_mc0(np.array([0.5 for _ in range(io.nc)]))
    vin = np.kron(v0, vx)
    return ptm.T @ vin, ptm_opt.T @ vin

def compute_pout_sim(ptm, ptm_opt, xvals, io, N):
    rng = bs.SC_RNG()
    var_bs = rng.bs_lfsr_mat(N, xvals)
    const_bs = rng.bs_lfsr_p5_consts(N, io.nc, 9, add_zero_state=True)
    bs_mat = np.vstack((var_bs, const_bs)) #might need to change this order
    #bs_mat = np.vstack((const_bs, var_bs)) #this line is *probably the wrong order
    bs_out = apply_ptm_to_bs(bs_mat, ptm, packed=True)
    bs_out_opt = apply_ptm_to_bs(bs_mat, ptm_opt, packed=True)
    return bs_out, bs_out_opt

def test_Kmat_hamming_dist(Ks):
    k = len(Ks)
    dij = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            dij[i,j] = hamming_dist(1*Ks[i], 1*Ks[j])
    return dij

def compare_Kmat_hamming_dist(Ks1, Ks2):
    k = len(Ks1)
    nv2, _ = Ks1[0].shape
    hdist = 0
    for i in range(k):
        for j in range(nv2):
            hdist += hamming_dist(1*Ks1[i][j, :], 1*Ks2[i][j, :])
    return hdist

#Library of functions for generating input patterns
def xfunc_uniform(num): #Basic xfunc
    return lambda: np.random.uniform(size=num)

def xfunc_3x3_img_windows(imgs=None): #For modeling image kernels
    if imgs is None: #Default is to load the 10 MATLAB test images
        imgs = list(np.load("../tim_pcc/test_images.npy", allow_pickle=True))

    num_imgs = len(imgs)
    def get_window():
        img = imgs[np.random.randint(num_imgs)]
        h, w = img.shape
        cy = np.random.randint(1, h-2)
        cx = np.random.randint(1, w-2)
        return img[cy-1:cy+2, cx-1:cx+2].reshape(9)
    return get_window

def xfunc_2x2_img_windows(imgs=None): #For modeling image kernels
    if imgs is None: #Default is to load the 10 MATLAB test images
        imgs = list(np.load("../tim_pcc/test_images.npy", allow_pickle=True))

    num_imgs = len(imgs)
    def get_window():
        img = imgs[np.random.randint(num_imgs)]
        h, w = img.shape
        cy = np.random.randint(0, h-2) #center is upper left corner
        cx = np.random.randint(0, w-2)
        return img[cy:cy+2, cx:cx+2].reshape(4)
    return get_window

def opt_multi(funcs, io, optfunc):
    Ks = []
    for i in range(io.k):
        Ks.append(get_func_mat(funcs[i], io.n, 1)[:, 1].reshape(io.nc2, io.nv2).T)
    ptm = Ks_to_Mf(Ks)
    ptm_opt = Ks_to_Mf(optfunc(Ks))
    return ptm, ptm_opt

def test_avg_corr(ptm, ptm_opt, xfunc, num_tests, io,  correct_func=None, print_=True, use_ptm=True, N=256):
    def test_correct(vout, correct_vals, io, thresh=1e-6):
        pout = (vout.T @ B_mat(io.k)).T
        assert np.allclose(pout, correct_vals, thresh)
    
    cout_avg = np.zeros((io.k, io.k))
    cout_opt_avg = np.zeros((io.k, io.k))
    for _ in range(num_tests):
        xvals = xfunc()
        if use_ptm:
            vout, vout_opt = compute_vout(ptm, ptm_opt, xvals, io)
            if correct_func is not None:
                correct = correct_func(xvals)
                test_correct(vout, correct, io)
                test_correct(vout_opt, correct, io)
            cout_avg += get_corr_mat_paper(vout)
            cout_opt_avg += get_corr_mat_paper(vout_opt)
        else:
            bs_out, bs_out_opt = compute_pout_sim(ptm, ptm_opt, xvals, io, N)
            #if correct_func is not None:
            #    pout = np.array([bs.bs_mean(s, bs_len=N) for s in bs_out])
            #    pout_opt = np.array([bs.bs_mean(s, bs_len=N) for s in bs_out_opt])
            #    correct = correct_func(xvals)
            #    print(rel_err(pout, correct))
            #    print(rel_err(pout_opt, correct))

            cout_avg += bs.get_corr_mat(bs_out, bs_len=N)
            cout_opt_avg += bs.get_corr_mat(bs_out_opt, bs_len=N)

    cout_avg /= num_tests
    cout_opt_avg /= num_tests
    if print_:
        print(cout_avg)
        print(cout_opt_avg)
    return cout_avg, cout_opt_avg

def rel_err(a, b):
    #return 2 * np.nan_to_num(np.abs(a - b)/(np.abs(a)+np.abs(b)), posinf=0.0)
    return (a-b)**2

def test_avg_err(ptm, ptm_opt, xfunc, correct_func, num_tests, io, print_=True, use_ptm=True, N=256):
    #Average error is reported in terms of RMSE
    c_err = np.zeros(num_tests)
    c_err_opt = np.zeros(num_tests)
    for test_idx in range(num_tests):
        xvals = xfunc()
        correct = correct_func(xvals)
        if use_ptm:
            vout, vout_opt = compute_vout(ptm, ptm_opt, xvals, io)
            pout = (vout.T @ B_mat(1)).T
            pout_opt = (vout_opt.T @ B_mat(1)).T
        else:
            bs_out, bs_out_opt = compute_pout_sim(ptm, ptm_opt, xvals, io, N)
            pout = np.array([bs.bs_mean(s, bs_len=N) for s in bs_out])
            pout_opt = np.array([bs.bs_mean(s, bs_len=N) for s in bs_out_opt])
        c_err[test_idx] = rel_err(pout, correct)[0]
        c_err_opt[test_idx] = rel_err(pout_opt, correct)[0]
    m_c_err = np.sqrt(np.mean(c_err))
    m_c_err_opt = np.sqrt(np.mean(c_err_opt))
    if print_:
        print(m_c_err)
        print(m_c_err_opt)
    return m_c_err, m_c_err_opt