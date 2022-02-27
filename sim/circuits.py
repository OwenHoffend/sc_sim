#from sim.corr_preservation import bin_array
import numpy as np
from sys import path
import torch

from os.path import dirname as dir
path.append(dir(path[0]))
import sim.bitstreams as bs
import cv.img_io as img_io
from cv.conv_filters import ConvFilters as cf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

def and_3(a, b, c):
    o1 = np.bitwise_and(a, b)
    o2 = np.bitwise_and(b, c)
    o3 = np.bitwise_and(a, c)
    return o1, o2, o3

def and_3_to_1(a, b, c):
    return np.bitwise_and(np.bitwise_and(a, b), c)

def or_3(a, b, c):
    o1 = np.bitwise_or(a, b)
    o2 = np.bitwise_or(b, c)
    o3 = np.bitwise_or(a, c)
    return o1, o2, o3

def xor_3(a, b, c):
    o1 = np.bitwise_xor(a, b)
    o2 = np.bitwise_xor(b, c)
    o3 = np.bitwise_xor(a, c)
    return o1, o2, o3

def passthrough_3(a, b, c):
    return a, b, c

def xor_4_to_2(x1, x2, x3, x4):
    o1 = np.bitwise_xor(x1, x2)
    o2 = np.bitwise_xor(x3, x4)
    return o1, o2

#def xor_8_to_4(x1, x2, x3, x4, y1, y2, y3, y4):
#    o1, o2 = xor_4_to_2(x1, x2, x3, x4)
#    o3, o4 = xor_4_to_2(y1, y2, y3, y4)
#    return o3, o4, o1, o2

def xnor_4_to_2(x4, x3, x2, x1):
    o1 = np.bitwise_not(np.bitwise_xor(x1, x2))
    o2 = np.bitwise_not(np.bitwise_xor(x3, x4))
    return o1, o2

def and_4_to_2(x1, x3, x2, x4):
    o1 = np.bitwise_and(x1, x2)
    o2 = np.bitwise_and(x3, x4)
    return o1, o2

def and_6_to_2(x2, x1, x4, x3, x6, x5):
    o2, o1 = and_4_to_2(x4, x3, x2, x1)
    o3 = np.bitwise_and(x6, x5)
    res1 = np.bitwise_and(o3, o2)
    res2 = np.bitwise_and(o2, o1)
    res3 = np.bitwise_and(o1, o3)
    return res3, res2, res1

def or_4_to_2(x1, x3, x2, x4):
    o1 = np.bitwise_or(x1, x2)
    o2 = np.bitwise_or(x3, x4)
    return o1, o2

def or_4(x1, x3, x2, x4):
    return np.bitwise_or(*or_4_to_2(x1, x3, x2, x4))

def and_3_to_2(x1, x2, x3):
    o1 = np.bitwise_and(x1, x2)
    o2 = np.bitwise_and(x2, x3)
    return o1, o2

def and_3_to_2_const(x2, x1, c0):
    o1 = np.bitwise_and(c0, x1)
    o2 = np.bitwise_and(c0, x2)
    return o1, o2

def mux_2_joint_const(x4, x3, x2, x1, c0):
    return mux_2(c0, c0, x1, x2, x3, x4)

def mux_1(s, x1, x2):
    t1 = np.bitwise_and(x1, np.bitwise_not(s))
    t2 = np.bitwise_and(x2, s)
    return np.bitwise_or(t1, t2)

def maj_1(s, x1, x2):
    a1 = np.bitwise_and(s, x2)
    a2 = np.bitwise_and(s, x1)
    a3 = np.bitwise_and(x1, x2)
    o1 = np.bitwise_or(a1, a2)
    return np.bitwise_or(o1, a3)

def maj_abs_sub(s, x11, x22, x12, x21, x11_n, x22_n, x12_n, x21_n):
    top = maj_1(s, x11, x22_n)
    top_inv = maj_1(s, x11_n, x22)
    bot = maj_1(s, x12, x21_n)
    bot_inv = maj_1(s, x12_n, x21)
    return np.bitwise_or(top, top_inv), np.bitwise_or(bot, bot_inv)

#s0 is the smallest index of the arguments
def mux_2(s0, s1, x1, x2, x3, x4):
    m1 = mux_1(s0, x2, x1)
    m2 = mux_1(s1, x4, x3)
    return m1, m2

def maj_2(s0, s1, x1, x2, x3, x4):
    m1 = maj_1(s0, x2, x1)
    m2 = maj_1(s1, x4, x3)
    return m1, m2

def mux_2_joint(s0, x1, x2, x3, x4):
    return mux_2(s0, s0, x1, x2, x3, x4)

def maj_2_joint(s0, x1, x2, x3, x4):
    return maj_2(s0, s0, x1, x2, x3, x4)

def sorter_2(x2, x1):
    top = np.bitwise_and(x2, x1)
    bot = np.bitwise_or(x2, x1)
    return top, bot

def even_odd_sorter_4(x4, x3, x2, x1):
    l2_4, l2_3 = sorter_2(x4, x3)
    l2_2, l2_1 = sorter_2(x2, x1)
    l3_4, l3_2 = sorter_2(l2_4, l2_2)
    l3_3, l3_1 = sorter_2(l2_3, l2_1)
    l4_3, l4_2 = sorter_2(l3_3, l3_2)
    return l3_4, l4_3, l4_2, l3_1

def bitonic_sorter_4(x4, x3, x2, x1):
    l2_4, l2_3 = sorter_2(x4, x3)
    l2_2, l2_1 = sorter_2(x2, x1)
    l3_4, l3_1 = sorter_2(l2_4, l2_1)
    l3_3, l3_2 = sorter_2(l2_3, l2_2)
    l4_4, l4_3 = sorter_2(l3_4, l3_3)
    l4_2, l4_1 = sorter_2(l3_2, l3_1)
    return l4_4, l4_3, l4_2, l4_1

def insertion_sorter_4(x4, x3, x2, x1):
    l2_4, l2_3 = sorter_2(x4, x3)
    l3_3, l3_2 = sorter_2(l2_3, x2)
    l4_4, l4_3 = sorter_2(l2_4, l3_3)
    l4_2, l4_1 = sorter_2(l3_2, x1)
    l5_3, l5_2 = sorter_2(l4_3, l4_2)
    l6_4, l6_3 = sorter_2(l4_4, l5_3)
    return l6_4, l6_3, l5_2, l4_1

def mux_4_to_1(s0, s1, x4, x3, x2, x1):
    m1, m2 = mux_2(s0, x4, x3, x2, x1)
    return mux_1(s1, m1, m2)

def maj_4_to_1(s0, s1, x4, x3, x2, x1):
    m1, m2 = maj_2(s0, x4, x3, x2, x1)
    return maj_1(s1, m1, m2)

def unbalanced_mux_2(s, x3, x2, x1):
    return mux_1(s, x1, x2), x3

def robert_cross(s, x11, x22, x12, x21, maj=False):
    xor1, xor2 = xor_4_to_2(x11, x22, x12, x21)
    if maj:
        return maj_1(s, xor1, xor2)
    else:
        return mux_1(s, xor1, xor2)

def robert_cross_mp_l1(s,
    x11, x12, x13, 
    x21, x22, x23, 
    x31, x32, x33
):
    xor1, xor2 = xor_4_to_2(x11, x22, x12, x21)
    xor3, xor4 = xor_4_to_2(x12, x23, x13, x22)
    xor5, xor6 = xor_4_to_2(x21, x32, x22, x31)
    xor7, xor8 = xor_4_to_2(x22, x33, x23, x32)
    return s, xor1, xor2, xor3, xor4, xor5, xor6, xor7, xor8

def robert_cross_mp_l2(s, xor1, xor2, xor3, xor4, xor5, xor6, xor7, xor8, maj=False):
    if maj:
        func = maj_1
    else:
        func = mux_1
    rc1 = func(s, xor1, xor2)
    rc2 = func(s, xor3, xor4)
    rc3 = func(s, xor5, xor6)
    rc4 = func(s, xor7, xor8)
    return rc1, rc2, rc3, rc4

def robert_cross_mp_l3(rc1, rc2, rc3, rc4):
    return or_4(rc1, rc2, rc3, rc4)

def robert_cross_mp(s,
    x11, x12, x13, 
    x21, x22, x23, 
    x31, x32, x33,
    maj=False
):
    rc1 = robert_cross(s, x11, x22, x12, x21, maj=maj)
    rc2 = robert_cross(s, x12, x23, x13, x22, maj=maj)
    rc3 = robert_cross(s, x21, x32, x22, x31, maj=maj)
    rc4 = robert_cross(s, x22, x33, x23, x32, maj=maj)
    return or_4(rc1, rc2, rc3, rc4)

def mac_relu_l1(s, x1, x2, wp1, wp2, wn1, wn2):
    a1p = np.bitwise_and(x1, wp1)
    a2p = np.bitwise_and(x2, wp2)
    a1n = np.bitwise_and(x1, wn1)
    a2n = np.bitwise_and(x2, wn2)
    return s, a1p, a2p, a1n, a2n

def mac_relu_l2(s, a1p, a2p, a1n, a2n, maj=False):
    if maj:
        pos = maj_1(s, a1p, a2p)
        neg = maj_1(s, a1n, a2n)
    else:
        pos = mux_1(s, a1p, a2p)
        neg = mux_1(s, a1n, a2n)
    return pos, neg

def mac_relu_l3(pos, neg):
    return np.bitwise_and(pos, np.bitwise_not(neg))

def mac_relu_ideal(x1, x2, wp1, wp2, wn1, wn2):
    return 0.5 * max(0.0, x1 * (wp1 - wn1) + x2 * (wp2 - wn2))

def robert_cross_ideal(x11, x22, x12, x21):
    return 0.5 * (np.abs(x11 - x22) + np.abs(x12 -x21))

def robert_cross_mp_ideal(
    x11, x12, x13, 
    x21, x22, x23, 
    x31, x32, x33
):
    rc1 = robert_cross_ideal(x11, x22, x12, x21)
    rc2 = robert_cross_ideal(x12, x23, x13, x22)
    rc3 = robert_cross_ideal(x21, x32, x22, x31)
    rc4 = robert_cross_ideal(x22, x33, x23, x32)
    return np.max(np.array([rc1, rc2, rc3, rc4]))

def mux_3(s, x6, x5, x4, x3, x2, x1):
    m1 = mux_1(s, x2, x1)
    m2 = mux_1(s, x4, x3)
    m3 = mux_1(s, x6, x5)
    return m1, m2, m3

def mux_p(x, y, p):
    x_un = np.unpackbits(x)
    y_un = np.unpackbits(y)
    z = np.zeros_like(x_un)
    for i in range(x_un.size):
        r = np.random.rand()
        z[i] = x_un[i] if r > p else y_un[i]
    return np.packbits(z)

def mux_p_cuda(x, y, rands):
    #global mask 
    #mask = torch.cuda.ByteTensor([2 ** x for x in range(8)])

    xs = x.shape[0]
    ys = y.shape[0]
    assert xs == ys
    #rands = torch.cuda.FloatTensor(xs << 3).uniform_() > p
    #rands = torch.sum(rands.view(xs, 8) * mask, 1)
    top = torch.bitwise_and(x, rands)
    bot = torch.bitwise_and(y, torch.bitwise_not(rands))
    return torch.bitwise_or(top, bot)

def maj_p_cuda(x, y, rands):
    #FIX THIS ASAP
    #global mask 
    #mask = torch.cuda.ByteTensor([2 ** x for x in range(8)])

    xs = x.shape[0]
    ys = y.shape[0]
    assert xs == ys
    and_ = torch.bitwise_and(x, y)
    or_ = torch.bitwise_or(x, y)
    top = torch.bitwise_and(and_, torch.bitwise_not(rands))
    bot = torch.bitwise_and(or_, rands)
    return torch.bitwise_or(top, bot)

def robert_cross_mux_cuda(rands, x11, x12, x21, x22):
    xor1 = torch.bitwise_xor(x11, x22)
    xor2 = torch.bitwise_xor(x12, x21)
    return mux_p_cuda(xor1, xor2, rands)

def robert_cross_maj_cuda(rands, x11, x12, x21, x22):
    xor1 = torch.bitwise_xor(x11, x22)
    xor2 = torch.bitwise_xor(x12, x21)
    return maj_p_cuda(xor1, xor2, rands)

def robert_cross_mux_cuda_nx(rands, rands2, x11, x12, x21, x22, x11_n, x12_n, x21_n, x22_n):
    """No-xor version of rober_cross_mux_cuda"""
    top = maj_p_cuda(x11, x22_n, rands)
    top_inv = maj_p_cuda(x11_n, x22, rands)
    top_max = torch.bitwise_or(top, top_inv)
    bot = maj_p_cuda(x12, x21_n, rands)
    bot_inv = maj_p_cuda(x12_n, x21, rands)
    bot_max = torch.bitwise_or(bot, bot_inv)
    return mux_p_cuda(top_max, bot_max, rands2)

def robert_cross_maj_cuda_nx(rands, rands2, x11, x12, x21, x22, x11_n, x12_n, x21_n, x22_n):
    """No-xor version of rober_cross_maj_cuda"""
    top = maj_p_cuda(x11, x22_n, rands)
    top_inv = maj_p_cuda(x11_n, x22, rands)
    top_max = torch.bitwise_or(top, top_inv)
    bot = maj_p_cuda(x12, x21_n, rands)
    bot_inv = maj_p_cuda(x12_n, x21, rands)
    bot_max = torch.bitwise_or(bot, bot_inv)
    return maj_p_cuda(top_max, bot_max, rands2)

def max_pool_cuda(x11, x12, x21, x22):
    or1 = torch.bitwise_or(x11, x22)
    or2 = torch.bitwise_or(x12, x21)
    return torch.bitwise_or(or1, or2)

def robert_cross_img(img_bs, N, no_xor=False, img_bs_inv=None, use_maj=False): #Img is greyscale
    global mask
    mask = torch.cuda.ByteTensor([2 ** x for x in range(8)])

    h, w, _ = img_bs.shape
    #Enable this code to load the image straight from this function instead of externally
    #h, w = img.shape
    #rng = bs.SC_RNG()
    #img_bs = img_io.img_to_bs(img, rng.bs_uniform, bs_len=N) #Correlated at +1
    #img_bs = torch.from_numpy(img_bs).to(device)
    #img_io.disp_img(img_io.bs_to_img(img_bs, bs.bs_mean))

    nb = N>>3
    rc_mat = torch.cuda.ByteTensor(h-1, w-1, nb).fill_(0)
    xs = img_bs[0][0].shape[0] #EXTREMELY HACKY PLEASE REMOVE THIS
    rands = torch.cuda.FloatTensor(xs << 3).uniform_() > 0.5 #Use the same set of random numbers for everything
    rands = torch.sum(rands.view(xs, 8) * mask, 1)

    if no_xor:
        rands2 = torch.cuda.FloatTensor(xs << 3).uniform_() > 0.5 #Use the same set of random numbers for everything
        rands2 = torch.sum(rands2.view(xs, 8) * mask, 1)
    if use_maj:
        if no_xor:
            rcfunc = robert_cross_maj_cuda_nx
        else:
            rcfunc = robert_cross_maj_cuda
    else:
        if no_xor:
            rcfunc = robert_cross_mux_cuda_nx
        else:
            rcfunc = robert_cross_mux_cuda
    for i in range(h-1):
        for j in range(w-1):
            if no_xor:
                rc_mat[i][j] = rcfunc(rands, rands2, img_bs[i][j], img_bs[i][j+1], img_bs[i+1][j], img_bs[i+1][j+1], \
                img_bs_inv[i][j], img_bs_inv[i][j+1], img_bs_inv[i+1][j], img_bs_inv[i+1][j+1])
            else:
                rc_mat[i][j] = rcfunc(rands, img_bs[i][j], img_bs[i][j+1], img_bs[i+1][j], img_bs[i+1][j+1])

    #img_io.disp_img(img_io.bs_to_img(rc_mat.cpu().detach().numpy(), bs.bs_mean))
    #return bs.get_corr_mat_cuda(rc_mat.view((h-1)*(w-1), nb)).to(cpu).numpy()
    return rc_mat

def max_pool_img(img_bs, N):
    h, w, _ = img_bs.shape
    nb = N>>3
    mp_mat = torch.cuda.ByteTensor(h-2, w-2, nb).fill_(0)
    for i in range(h-2):
        for j in range(w-2):
            mp_mat[i][j] = max_pool_cuda(img_bs[i][j], img_bs[i][j+1], img_bs[i+1][j], img_bs[i+1][j+1])
    return mp_mat

def cnn_kernel_3x3(img, kernel, N):
    """Unipolar kernel convolve -> AND gates
       Bipolar kernel convolve -> XNOR gates"""
    is_bp = np.any(kernel < 0)

    h, w = img.shape
    rng = bs.SC_RNG()
    rng_type = rng.bs_bp_uniform if is_bp else rng.bs_uniform
    img_bs = img_io.img_to_bs(img, rng_type, bs_len=N) #Correlated at +1
    img_bs = torch.from_numpy(img_bs).to(device)
    rng.reset()

    kernel_bs = img_io.img_to_bs(kernel, rng_type, bs_len=N, scale=False)
    kernel_bs = torch.from_numpy(kernel_bs).to(device)

    nb = N>>3
    rc_mat = torch.cuda.ByteTensor(h-2, w-2, nb).fill_(0)
    m = torch.cuda.ByteTensor(3, 3, nb).fill_(0)
    z = torch.cuda.ByteTensor(nb).fill_(0)
    for i in range(h-2):
        for j in range(w-2):
            for k in range(3):
                for l in range(3):
                    if is_bp:
                        m[k][l] = torch.bitwise_not(torch.bitwise_xor(img_bs[i + k][j + l], kernel_bs[k][l]))
                    else:
                        m[k][l] = torch.bitwise_and(img_bs[i + k][j + l], kernel_bs[k][l])
            #mux sum tree
            l1_1 = mux_p_cuda(m[0][0], m[0][1], 0.5)
            l1_2 = mux_p_cuda(m[0][2], m[1][0], 0.5)
            l1_3 = mux_p_cuda(m[1][1], m[1][2], 0.5)
            l1_4 = mux_p_cuda(m[2][0], m[2][1], 0.5)
            l2_1 = mux_p_cuda(l1_1, l1_2, 0.5)
            l2_2 = mux_p_cuda(l1_3, l1_4, 0.5)
            l3 = mux_p_cuda(l2_1, l2_2, 0.5)
            rc_mat[i][j] = mux_p_cuda(l3, mux_p_cuda(m[2][2], z, 1.0/8.0), 1.0/9.0)
    
    #mean_type = bs.bs_mean_bp if is_bp else bs.bs_mean
    #rc_mat = rc_mat.to(cpu)
    #img_io.disp_img(img_io.bs_to_img(rc_mat, mean_type, scaling=9))
    return bs.get_corr_mat_cuda(rc_mat.view((h-2)*(w-2), nb)).to(cpu).numpy()

if __name__ == "__main__":
    """Test robert_cross_img"""
    #img = img_io.load_img("./img/lena_s2.jpg", gs=True)
    #out_c_mat = robert_cross_img(img, 16)
    #err = bs.mut_corr_err(1, out_c_mat)
    #print(err)
    #print(out_c_mat)

    """Test cnn_kernel_3x3_up"""
    #img = img_io.load_img("./img/lena_s2.jpg", gs=True)
    #kernel = np.array(cf.BOX_BLUR)
#
    #import time
#
    #t0 = time.time()
    #with torch.no_grad():
    #    out_c_mat = cnn_kernel_3x3(img, kernel, 1024)
    #t1 = time.time()
    #total = t1-t0
    #print(total)
    #print(out_c_mat)

    #Test sorting networks
    #for i in range(16):
    #    print(even_odd_sorter_4(*bin_array(i, 4)))