import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sim.bitstreams import *
from sim.circuits import *
from sim.PTM import *
from cv.img_io import *

def test_roberts_cross_mux_maj():
    #Load the image
    N=256
    mux_errs = []
    maj_errs = []
    num_iters = 30
    for i in range(10):
        directory = './img/mnist_png/testing/{}/'.format(i)
        mux_avg_err = 0
        maj_avg_err = 0
        iters = 0
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            img = load_img(path)
            rng = bs.SC_RNG()
            img_bs = img_to_bs(img, rng.bs_uniform, bs_len=N) #Correlated at +1
            img_bs = torch.from_numpy(img_bs).to(device)

            #Ideal Roberts-cross
            h, w = img.shape
            fl_img = img.astype(np.float32)
            rc_ideal = np.zeros((h-1, w-1))
            for i in range(h-1):
                for j in range(w-1):
                    rc_ideal[i][j] = 0.5 * np.abs(fl_img[i][j] - fl_img[i+1][j+1]) + 0.5 * np.abs(fl_img[i+1][j] - fl_img[i][j+1])

            #Ideal Max-pool
            mp_ideal = np.zeros((h-2, w-2))
            for i in range(h-2):
                for j in range(w-2):
                    mp_ideal[i][j] = max(rc_ideal[i][j], rc_ideal[i][j+1], rc_ideal[i+1][j], rc_ideal[i+1][j+1])
            mp_ideal = mp_ideal.astype(np.uint8)
            #save_img(mp_ideal, "./img/126_ideal.png")

            #Apply Robert's Cross
            rc_bs_mux = robert_cross_img(img_bs, N)
            #rc_img_mux = bs_to_img(rc_bs_mux.cpu().detach().numpy(), bs.bs_mean)
            #save_img(rc_img_mux, "./img/126_rc_mux.png")

            rc_bs_maj = robert_cross_img(img_bs, N, use_maj=True)
            #rc_img_maj = bs_to_img(rc_bs_maj.cpu().detach().numpy(), bs.bs_mean)
            #save_img(rc_img_maj, "./img/126_rc_maj.png")

            #Apply Max Pooling to each
            mp_bs_mux = max_pool_img(rc_bs_mux, N)
            mp_img_mux = bs_to_img(mp_bs_mux.cpu().detach().numpy(), bs.bs_mean)
            #save_img(mp_img_mux, "./img/126_mp_mux.png")

            mp_bs_maj = max_pool_img(rc_bs_maj, N)
            mp_img_maj = bs_to_img(mp_bs_maj.cpu().detach().numpy(), bs.bs_mean)
            #save_img(mp_img_maj, "./img/126_mp_maj.png")

            mux_err = np.sum(np.abs(mp_ideal - mp_img_mux)) / (h * w)
            maj_err = np.sum(np.abs(mp_ideal - mp_img_maj)) / (h * w)
            
            mux_avg_err += mux_err
            maj_avg_err += maj_err
            iters += 1
            if iters == num_iters:
                break
        mux_avg_err /= num_iters
        maj_avg_err /= num_iters
        print("Mux err: {}".format(mux_avg_err))
        print("Maj err: {}".format(maj_avg_err))
        mux_errs.append(mux_avg_err)
        maj_errs.append(maj_avg_err)

    ax = plt.subplot(111)
    ax.bar(np.array(range(10))-0.2, mux_errs, width=0.4, color='orange', align='center', label='mux')
    ax.bar(np.array(range(10))+0.2, maj_errs, width=0.4, color='b', align='center', label='maj')
    plt.xticks(list(range(10)))
    plt.title("MNIST Edge Detect --> Max Pool SC Error Comparison")
    plt.xlabel("MNIST Digit Number")
    plt.ylabel("Avg Error with respect to ideal")
    plt.legend()
    plt.show()