import numpy as np
import os
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
    return np.log2(a).astype(int)

def compute_vout(ptm, ptm_opt, x, io, x_corr=True):
    if x_corr:
        vx = get_vin_mc1(x)
    else:
        vx = get_vin_mc0(x)
    v0 = get_vin_mc0(np.array([0.5 for _ in range(io.nc)]))
    vin = np.kron(v0, vx)
    return ptm.T @ vin, ptm_opt.T @ vin

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

def test_correct(vout, correct_vals, io, thresh=1e-6):
    pout = (vout.T @ B_mat(io.k)).T
    assert np.allclose(pout, correct_vals, thresh)

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
        cy = np.random.randint(1, h-1)
        cx = np.random.randint(1, w-1)
        return img[cy-1:cy+2, cx-1:cx+2].reshape(9)
    return get_window

#Top-level functions that you'd probably actually want to call
#def opt_two(func1, func2, io, optfunc):
#    assert io.k == 2
#    A = np.zeros((io.n2, 2), dtype=np.bool_)
#    A[:, 0] = get_func_mat(func1, io.n, 1)[:, 1]
#    A[:, 1] = get_func_mat(func2, io.n, 1)[:, 1]
#    ptm = A_to_Mf(A, io.n, 2)
#    K1 = A[:, 0].reshape(io.nc2, io.nv2).T
#    K2 = A[:, 1].reshape(io.nc2, io.nv2).T
#    K1_opt, K2_opt = optfunc(K1, K2)
#    #print("Novs: ", np.sum(np.bitwise_and(K1_opt, K2_opt)))
#    ptm_opt = Ks_to_Mf([K1_opt, K2_opt])
#    return ptm, ptm_opt

def opt_multi(funcs, io, optfunc):
    Ks = []
    for i in range(io.k):
        Ks.append(get_func_mat(funcs[i], io.n, 1)[:, 1].reshape(io.nc2, io.nv2).T)
    ptm = Ks_to_Mf(Ks)
    ptm_opt = Ks_to_Mf(optfunc(Ks))
    return ptm, ptm_opt

def test_avg_corr(ptm, ptm_opt, xfunc, num_tests, io,  correct_func=None):
    cout_avg = np.zeros((io.k, io.k))
    cout_opt_avg = np.zeros((io.k, io.k))
    for _ in range(num_tests):
        xvals = xfunc()
        vout, vout_opt = compute_vout(ptm, ptm_opt, xvals, io)
        if correct_func is not None:
            correct = correct_func(xvals)
            test_correct(vout, correct, io)
            test_correct(vout_opt, correct, io)
        cout_avg += get_corr_mat_paper(vout)
        cout_opt_avg += get_corr_mat_paper(vout_opt)
    cout_avg /= num_tests
    cout_opt_avg /= num_tests
    print(cout_avg)
    print(cout_opt_avg)
    return cout_avg, cout_opt_avg

def test_avg_err(ptm, ptm_opt, xfunc, correct_func, num_tests, io):
    c_err = 0.0
    c_err_opt = 0.0
    for _ in range(num_tests):
        xvals = xfunc()
        correct = correct_func(xvals)
        vout, vout_opt = compute_vout(ptm, ptm_opt, xvals, io)
        pout = (vout.T @ B_mat(1)).T
        pout_opt = (vout_opt.T @ B_mat(1)).T
        c_err += np.abs(pout - correct)[0]
        c_err_opt += np.abs(pout_opt - correct)[0]
    c_err /= num_tests
    c_err_opt /= num_tests
    #print(c_err)
    #print(c_err_opt)
    return c_err, c_err_opt