import numpy as np
from sim.corr_preservation import *
from sim.PTM import *
from sim.circuits import *
from cv.img_io import *

import os

def max_pool_l1(x4, x3, x2, x1):
    return np.bitwise_or(x4, x3), np.bitwise_or(x2, x1)

def get_pins(idx):
    if idx > 1: #c10
        imgs = cifar_unpickle("./img/cifar-10-batches-py/data_batch_1")
        for i in range(100):
            print("CIFAR: {}".format(i))
            img = imgs[i, 0:1024] / 255
            l = img.shape[0]
            for window in range(l-3):
                yield img[window:window+4]
    else: #mnist
        for i in range(10):
            directory = './img/mnist_png/testing/{}/'.format(i)
            iters = 0
            print("MNIST DIGIT: {}".format(i))
            for filename in os.listdir(directory):
                path = os.path.join(directory, filename)
                img = load_img(path, gs=True) / 255
                h, w = img.shape
                for window_h in range(h-1):
                    for window_w in range(w-1):
                        yield img[window_h:window_h+2, window_w:window_w+2].reshape(4)
                iters += 1
                if iters == 10:
                    break

#cifar-10, mux:
def max_pooling_test():
    c10_mux = np.array([[1        , 0.5577475 , 0.5009727, 0.4026691],
    [0.5577475,  1        , 0.4065749, 0.5003257],
    [0.5009727,  0.4065749, 1        , 0.5579081],
    [0.4026691,  0.5003257, 0.5579081, 1       ]])

    #cifar-10, maj:
    c10_maj = np.array([[1.        , 0.67944996, 0.62174328, 0.48456234],
    [0.67944996, 1.        , 0.48993862, 0.62174656],
    [0.62174328, 0.48993862, 1.        , 0.67977668],
    [0.48456234, 0.62174656, 0.67977668, 1.        ]])

    #mnist, mux:
    mn_mux = np.array([[1.        , 0.97971839, 0.97992029, 0.97461759],
    [0.97971839, 1.        , 0.97835297, 0.97989071],
    [0.97992029, 0.97835297, 1.        , 0.9797036 ],
    [0.97461759, 0.97989071, 0.9797036 , 1.        ]])

    mn_maj = np.array([[1.        , 0.98499708, 0.98510411, 0.97760448] ,
    [0.98499708, 1.        , 0.98132139, 0.98508932], 
    [0.98510411, 0.98132139, 1.        , 0.98496749],
    [0.97760448, 0.98508932, 0.98496749, 1.        ]])

    cmats = [c10_mux, c10_maj, mn_mux, mn_maj]

    mf1 = get_func_mat(max_pool_l1, 4, 2)
    mf2 = get_func_mat(np.bitwise_or, 2, 1)
    mf = mf1 @ mf2
    for idx, c in enumerate(cmats):
        cavg = np.sum(np.tril(c, -1)) / 6
        
        err0, err1, err2, err3 = 0,0,0,0
        iter_cnt = 0
        b1 = B_mat(1)
        b2 = B_mat(2)
        c1_a = b1.T @ mf.T
        c2_a = b2.T @ mf1.T
        c2_a2 = b1.T @ mf2.T
        gen = get_pins(idx)
        for pin in gen:

            vin = (1-cavg) * get_vin_mc0(pin) + cavg * get_vin_mc1(pin)
            vin_reco = (1-cavg) * np.kron(get_vin_mc1(pin[0:2]), get_vin_mc1(pin[2:4])) + cavg * get_vin_mc1(pin)
            p_ideal = np.max(pin)

            #case 0
            err0 += np.abs(p_ideal - c1_a @ vin)

            #case 1
            c1_p_reco = c1_a @ vin_reco
            err1 += np.abs(p_ideal - c1_p_reco)

            #case 2
            c2_p1 = c2_a @ vin
            c2_vin_reco = get_vin_mc1(c2_p1)
            c2_p_reco = c2_a2 @ c2_vin_reco
            err2 += np.abs(p_ideal - c2_p_reco)

            #case 3
            c3_p_reco = c2_a @ vin_reco
            c3_vin_reco = get_vin_mc1(c3_p_reco)
            c3_p_reco2 = c2_a2 @ c3_vin_reco
            err3 += np.abs(p_ideal - c3_p_reco2)
            
            iter_cnt += 1

        print("case: 0, iter: {}: {}".format(idx, err0 / iter_cnt))
        print("case: 1, iter: {}: {}".format(idx, err1 / iter_cnt))
        print("case: 2, iter: {}: {}".format(idx, err2 / iter_cnt))
        print("case: 3, iter: {}: {}".format(idx, err3 / iter_cnt))