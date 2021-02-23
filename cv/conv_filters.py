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

GAUSS_BLUR_3x3 = [
    [0.0625, 0.125, 0.0625],
    [0.125,  0.25,  0.125 ],
    [0.0625, 0.125, 0.0625]
]

def convolve(img, kernel):
    """Convolve a single image channel with a given filter"""
    npk = np.array(kernel).astype(np.float64) #Convert the kernel to a np array of float64s
    return signal.convolve2d(img, npk) #Scipy discrete convolution. boundary=fill, fillvalue=0

def sc_convolve(img, kernel, bs_len=255):
    """Implementation of a generic convolution kernel with SC."""

    #Convert the image to a stochastic representation
    rng = bitstreams.SC_RNG()
    img_bs = img_io.img_to_bs(img, rng.bs_lfsr, bs_len=bs_len) #Correlated
    rng.reset()

    #Convert the kernel to a stochastic representation
    npk = np.array(kernel).astype(np.float64)
    abs_npk = np.absolute(npk) #Sign is handled seperately
    kern_img = 255.0 * abs_npk / np.max(abs_npk) #Format kernel as an image with values in the range of 0 to 255
    kern_bs = img_io.img_to_bs(kern_img, rng.bs_lfsr, bs_len=bs_len)

    #Perform the convolution
    i_h, i_w = img.shape
    k_h, k_w = npk.shape
    for i in range(i_h):
        for j in range(i_w):
            pass #TODO: Finish

if __name__ == "__main__":
    """Perform basic edge detection on a test image"""
    lena = img_io.load_img("../img/lena256.png", gs=True) #Load test image as greyscale
    lena_ed = convolve(lena, EDGE_DETECT_8)
    img_io.disp_img(lena_ed)

    """Test of sc_convolve"""
    sc_convolve(lena, EDGE_DETECT_8)