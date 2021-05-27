import numpy as np
from sys import path
from scipy import signal

from os.path import dirname as dir
path.append(dir(path[0]))
import cv.img_io
import sim.bitstreams as bitstreams

"""Some common filter kernels
   See https://en.wikipedia.org/wiki/Kernel_(image_processing)
"""
class ConvFilters:
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
