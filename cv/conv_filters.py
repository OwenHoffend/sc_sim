import numpy as np
import img_io
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

if __name__ == "__main__":
    """Perform basic edge detection on a test image"""
    lena = img_io.load_img("../img/lena256.png", gs=True) #Load test image as greyscale
    lena_ed = convolve(lena, EDGE_DETECT_8)
    img_io.disp_img(lena_ed)