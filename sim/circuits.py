import numpy as np
import bitstreams as bs
from sys import path
from os.path import dirname as dir
path.append(dir(path[0]))
import cv.img_io as img_io

def mux_p5(x, y):
    x_un = np.unpackbits(x)
    y_un = np.unpackbits(y)
    z = np.zeros_like(x_un)
    for i in range(x_un.size):
        r = np.random.rand()
        z[i] = x_un[i] if r > 0.5 else y_un[i]
    return np.packbits(z)

def robert_cross(x11, x12, x21, x22):
    xor1 = np.bitwise_xor(x11, x22)
    xor2 = np.bitwise_xor(x12, x21)
    return mux_p5(xor1, xor2)

def robert_cross_3x3_to_2x2(p_arr, N):
    #Generate input bitstreams with ZSCC = 1 at p_arr
    rng = bs.SC_RNG()
    #Generate bitstreams (uniform random for now)
    bs_arr = [rng.bs_uniform(N, p) for p in p_arr]
    #print(bs.mc_scc(bs_arr, use_zscc=True, bs_len=N)) #Verify that it is indeed mutually correlated at 1

    rc1 = robert_cross(bs_arr[0], bs_arr[1], bs_arr[3], bs_arr[4])
    rc2 = robert_cross(bs_arr[1], bs_arr[2], bs_arr[4], bs_arr[5])
    rc3 = robert_cross(bs_arr[3], bs_arr[4], bs_arr[6], bs_arr[7])
    rc4 = robert_cross(bs_arr[4], bs_arr[5], bs_arr[7], bs_arr[8])

    return bs.get_corr_mat([rc1, rc2, rc3, rc4], bs_len=N, use_zscc=True)

def robert_cross_img(img, N): #Img is greyscale
    h, w = img.shape
    rng = bs.SC_RNG()
    img_bs = img_io.img_to_bs(img, rng.bs_uniform, bs_len=N) #Correlated at +1
    img_io.disp_img(img_io.bs_to_img(img_bs, bs.bs_mean))

    rc_mat = np.zeros((h-1, w-1, N>>3), dtype=np.uint8)
    for i in range(h-1):
        for j in range(w-1):
            rc_mat[i][j] = robert_cross(img_bs[i][j], img_bs[i][j+1], img_bs[i+1][j], img_bs[i+1][j+1])
    img_io.disp_img(img_io.bs_to_img(rc_mat, bs.bs_mean))
    #return bs.get_corr_mat(, bs_len=N, use_zscc=True)

if __name__ == "__main__":
    """Test robert_cross_3x3_to_2x2"""
    #p_arr = [np.random.random() for _ in range(9)]
    #print(robert_cross_3x3_to_2x2(p_arr, 32))

    """Test robert_cross_img"""
    img = img_io.load_img("./img/lena_s3.jpg", gs=True)
    print(robert_cross_img(img, 128))