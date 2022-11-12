import numpy as np
import os
from sim.PTM import *
from sim.SEC import *

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
        cy = np.random.randint(1, h-2)
        cx = np.random.randint(1, w-2)
        return img[cy-1:cy+2, cx-1:cx+2].reshape(9)
    return get_window

#Top-level functions that you'd probably actually want to call
def opt_max_multi(funcs, io):
    """funcs: list of functions, each with 1 output"""
    if type(funcs) == list:
        A = np.zeros((io.n2, io.k))
        for i in range(io.k):
            A[:, i] = get_func_mat(funcs[i], io.n, 1)[:, 1]
        ptm = A_to_Mf(A, io.n, io.k)
    else:
        ptm = get_func_mat(funcs, io.n, io.k)
        A = ptm @ B_mat(io.k)
    Ks_opt = []
    for i in range(io.k):
        K = A[:, i].reshape(io.nc2, io.nv2).T
        Ks_opt.append(opt_K_max(K.astype(np.bool_)))
    ptm_opt = Ks_to_Mf(Ks_opt)
    return ptm, ptm_opt

def test_avg_corr(ptm, ptm_opt, xfunc, correct_func, num_tests, io):
    cout_avg = np.zeros((io.k, io.k))
    cout_opt_avg = np.zeros((io.k, io.k))
    for _ in range(num_tests):
        xvals = xfunc()
        correct = correct_func(xvals)
        vout, vout_opt = compute_vout(ptm, ptm_opt, xvals, io)
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
        c_err += np.abs(pout - correct)
        c_err_opt += np.abs(pout_opt - correct)
    c_err /= num_tests
    c_err_opt /= num_tests
    print(c_err)
    print(c_err_opt)
    return c_err, c_err_opt