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

def mux_1(s, x2, x1):
    t1 = np.bitwise_and(x1, np.bitwise_not(s))
    t2 = np.bitwise_and(x2, s)
    return np.bitwise_or(t1, t2)

def mux_2(s, x4, x3, x2, x1):
    m1 = mux_1(s, x2, x1)
    m2 = mux_1(s, x4, x3)
    return m1, m2

def unbalanced_mux_2(s, x3, x2, x1):
    return mux_1(s, x1, x2), x3

def robert_cross(s, x4, x3, x2, x1):
    x1, x2 = xor_4_to_2(x4, x3, x2, x1)
    return mux_1(s, x1, x2)

def robert_cross_2(s, x8, x7, x6, x5, x4, x3, x2, x1):
    r1 = robert_cross(s, x8, x7, x6, x5)
    r2 = robert_cross(s, x4, x3, x2, x1)
    return r1, r2

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

def mux_p_cuda(x, y, p):
    global mask 
    mask = torch.cuda.ByteTensor([2 ** x for x in range(8)])

    xs = x.shape[0]
    ys = y.shape[0]
    assert xs == ys
    rands = torch.cuda.FloatTensor(xs << 3).uniform_() > p
    rands = torch.sum(rands.view(xs, 8) * mask, 1)
    top = torch.bitwise_and(x, rands)
    bot = torch.bitwise_and(y, torch.bitwise_not(rands))
    return torch.bitwise_or(top, bot)

def robert_cross(x11, x12, x21, x22):
    xor1 = torch.bitwise_xor(x11, x22)
    xor2 = torch.bitwise_xor(x12, x21)
    return mux_p_cuda(xor1, xor2, 0.5)

def robert_cross_3x3_to_2x2(p_arr, N):
    #Generate input bitstreams with ZSCC = 1 at p_arr
    rng = bs.SC_RNG()
    #Generate bitstreams (uniform random for now)
    bs_arr = [torch.from_numpy(rng.bs_uniform(N, p)).to(device) for p in p_arr]
    #print(bs.mc_scc(bs_arr, use_zscc=True, bs_len=N)) #Verify that it is indeed mutually correlated at 1

    rc1 = robert_cross(bs_arr[0], bs_arr[1], bs_arr[3], bs_arr[4])
    rc2 = robert_cross(bs_arr[1], bs_arr[2], bs_arr[4], bs_arr[5])
    rc3 = robert_cross(bs_arr[3], bs_arr[4], bs_arr[6], bs_arr[7])
    rc4 = robert_cross(bs_arr[4], bs_arr[5], bs_arr[7], bs_arr[8])
    rc_torch = torch.tensor([rc1, rc2, rc3, rc4]).to(device)
    return bs.get_corr_mat_cuda(rc_torch, bs_len=N, use_zscc=True)

def robert_cross_img(img, N): #Img is greyscale
    h, w = img.shape
    rng = bs.SC_RNG()
    img_bs = img_io.img_to_bs(img, rng.bs_uniform, bs_len=N) #Correlated at +1
    img_bs = torch.from_numpy(img_bs).to(device)
    #img_io.disp_img(img_io.bs_to_img(img_bs, bs.bs_mean))

    nb = N>>3
    rc_mat = torch.cuda.ByteTensor(h-1, w-1, nb).fill_(0)
    for i in range(h-1):
        for j in range(w-1):
            rc_mat[i][j] = robert_cross(img_bs[i][j], img_bs[i][j+1], img_bs[i+1][j], img_bs[i+1][j+1])
    #img_io.disp_img(img_io.bs_to_img(rc_mat, bs.bs_mean))
    return bs.get_corr_mat_cuda(rc_mat.view((h-1)*(w-1), nb)).to(cpu).numpy()

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
    """Test robert_cross_3x3_to_2x2"""
    #p_arr = [np.random.random() for _ in range(9)]
    #print(robert_cross_3x3_to_2x2(p_arr, 32))

    """Test robert_cross_img"""
    img = img_io.load_img("./img/lena_s2.jpg", gs=True)
    out_c_mat = robert_cross_img(img, 16)
    err = bs.mut_corr_err(1, out_c_mat)
    print(err)
    print(out_c_mat)

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