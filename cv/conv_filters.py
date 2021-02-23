import numpy as np
import img_io
from sys import path
from os.path import dirname as dir
path.append(dir(path[0]))
import sim.bitstreams as bitstreams
from scipy import signal

"""Some common filter kernels
   See https://en.wikipedia.org/wiki/Kernel_(image_processing)
"""
EDGE_DETECT_0 = [
    [ 1, 0, -1],
    [ 0, 0,  0],
    [-1, 0,  1]
]

EDGE_DETECT_4 = [
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0]
]

EDGE_DETECT_8 = [
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
]

SHARPEN = [
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
]

BOX_BLUR = [
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1]
]

IDENTITY = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0]
]

GAUSS_BLUR_3x3 = [
    [0.0625, 0.125, 0.0625],
    [0.125,  0.25,  0.125 ],
    [0.0625, 0.125, 0.0625]
]

def convolve(img, kernel):
    """Convolve a single image channel with a given filter"""
    npk = np.array(kernel).astype(np.float64) #Convert the kernel to a np array of float64s
    return signal.convolve2d(img, npk) #Scipy discrete convolution. boundary=fill, fillvalue=0

def sc_convolve(img, kernel, bs_len=256):
    """Implementation of a generic convolution kernel with SC."""
    #Convert the kernel to a stochastic representation
    rng = bitstreams.SC_RNG()
    npk = np.array(kernel).astype(np.float64)
    kernel = npk / np.max(np.absolute(npk))  #Scale kernel to be within [-1, 1] for bipolar representation
    kernel_bs = img_io.img_to_bs(kernel, rng.bs_uniform, bs_len=bs_len, scale=False) #Generate the kernel bitstreams
    rng.reset()

    #Convert the image to a stochastic representation
    i_h, i_w = img.shape
    k_h, k_w = npk.shape
    pad_width = int(np.floor(k_h / 2))
    img = np.pad(img, pad_width, mode='constant') #Pad image with zeros - the actual SC would likely have more efficient HW for accomplishing padding
    img_bs = img_io.img_to_bs(img, rng.bs_uniform, bs_len=bs_len) #Correlated

    #Perform the convolution
    ihr = list(range(pad_width, i_h + pad_width))
    ihw = list(range(pad_width, i_h + pad_width))
    khr = list(range(-pad_width, pad_width + 1))
    khw = list(range(-pad_width, pad_width + 1))
    npb = np.ceil(bs_len / 8.0).astype(int) #Compute the number of packed bytes necessary to represent this bitstream
    m = np.zeros((k_h, k_w, npb), dtype=np.uint8) #Intermediate variable
    result = np.zeros((i_h, i_w), dtype=np.uint8)
    for i, ih in enumerate(ihr):
        for j, iw in enumerate(ihw):
            #First get the XNOR values
            for k, kh in enumerate(khr):
                for l, kw in enumerate(khw):
                    m[k][l] = ~(img_bs[ih + kh][iw + kw] ^ kernel_bs[k][l])
                    #m[k][l] = img_bs[ih + kh][iw + kw] & kernel_bs[k][l]

            #Ideal sum, just to test that it's working
            s = 0.0
            for k in range(k_h):
                for l in range(k_w):
                    s += bitstreams.bs_mean(m[k][l])
            result[i][j] = np.rint(s * 255).astype(np.uint8)
    return result

if __name__ == "__main__":
    """Perform basic edge detection on a test image"""
    lena = img_io.load_img("./img/lena256.png", gs=True) #Load test image as greyscale
    lena_ed = convolve(lena, GAUSS_BLUR_3x3)
    img_io.disp_img(lena_ed)

    """Test of sc_convolve"""
    lena_sc_ed = sc_convolve(lena, GAUSS_BLUR_3x3, bs_len=4096)
    img_io.disp_img(lena_sc_ed)
