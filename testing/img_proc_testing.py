import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sim.bitstreams import *
from sim.circuits import *
from sim.PTM import *
from cv.img_io import *

NUM_POOLS = 2
def corr_mat_rand_window(rc_bs):
    h, w, nb = rc_bs.shape
    avg_cmat = np.zeros((4, 4))
    for window_h in range(h-1):
        for window_w in range(w-1):
            window_bs_mat = rc_bs[window_h:window_h+2, window_w:window_w+2, :].reshape((4, nb))
            avg_cmat += get_corr_mat_np(window_bs_mat.to(cpu))
    return avg_cmat / ((h-1)*(w-1))

def test_roberts_cross_mux_maj_cifar10():
    N=256
    imgs = cifar_unpickle("./img/cifar-10-batches-py/data_batch_1")
    cmat = np.zeros((4, 4))
    for i in range(100):
        print(i)
        img = imgs[i, 0:1024]
        rng = bs.SC_RNG()
        img_bs = img_to_bs(img.reshape((32, 32)), rng.bs_uniform, bs_len=N)
        img_bs = torch.from_numpy(img_bs).to(device)

        #Apply Robert's Cross
        rc_bs_mux = robert_cross_img(img_bs, N)
        rc_bs_maj = robert_cross_img(img_bs, N, use_maj=True)

        #Apply max pooling
        mp_bs_mux = max_pool_img(rc_bs_mux, N).to(cpu)
        mp_bs_maj = max_pool_img(rc_bs_maj, N).to(cpu)
        mp_img_mux = bs_to_img(mp_bs_mux, bs.bs_mean, scaling=1)
        mp_img_maj = bs_to_img(mp_bs_maj, bs.bs_mean, scaling=1)
        disp_img(mp_img_mux)
        disp_img(mp_img_maj)

        #Get correlation matrix for a random 2x2 window within the outputs
        cmat += corr_mat_rand_window(rc_bs_maj)
    cmat /= 100
    print(cmat)

def test_roberts_cross_mux_maj_mnist():
    #Load the image
    N=256
    mux_errs = []
    maj_errs = []
    #mux_no_xor_errs = []
    #maj_no_xor_errs = []
    num_iters = 10
    cmat = np.zeros((4, 4))
    for i in range(10):
        print("Digit: {}".format(i))
        directory = './img/mnist_png/testing/{}/'.format(i)
        #mux_avg_err = 0
        #maj_avg_err = 0
        #mux_no_xor_avg_err = 0
        #maj_no_xor_avg_err = 0
        iters = 0
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            img = load_img(path, gs=True)
            rng = bs.SC_RNG()
            img_bs = img_to_bs(img, rng.bs_uniform, bs_len=N) #Correlated at +1
            img_bs = torch.from_numpy(img_bs).to(device)
            #img_bs_inv = img_to_bs(img, rng.bs_uniform, bs_len=N, inv=True) #Inverted bitstreams
            #img_bs_inv = torch.from_numpy(img_bs_inv).to(device)

            #Ideal Roberts-cross
            #h, w = img.shape
            #fl_img = img.astype(np.float32)
            #rc_ideal = np.zeros((h-1, w-1))
            #for i in range(h-1):
            #    for j in range(w-1):
            #        rc_ideal[i][j] = 0.5 * np.abs(fl_img[i][j] - fl_img[i+1][j+1]) + 0.5 * np.abs(fl_img[i+1][j] - fl_img[i][j+1])

            #Ideal Max-pool
            #mp_ideal = rc_ideal.astype(np.uint8)
            #for k in range(NUM_POOLS):
            #    hm, wm = mp_ideal.shape
            #    mp_ideal_next = np.zeros((hm-2, wm-2))
            #    for i in range(hm-2):
            #        for j in range(wm-2):
            #            mp_ideal_next[i][j] = max(mp_ideal[i][j], mp_ideal[i][j+1], mp_ideal[i+1][j], mp_ideal[i+1][j+1])
            #    mp_ideal = mp_ideal_next

            #Apply Robert's Cross
            rc_bs_mux = robert_cross_img(img_bs, N)
            rc_bs_maj = robert_cross_img(img_bs, N, use_maj=True)

            #Get correlation matrix for a random 2x2 window within the outputs
            cmat += corr_mat_rand_window(rc_bs_maj)

            #rc_bs_no_xor_mux = robert_cross_img(img_bs, N, no_xor=True, img_bs_inv=img_bs_inv)
            #rc_bs_no_xor_maj = robert_cross_img(img_bs, N, use_maj=True, no_xor=True, img_bs_inv=img_bs_inv)

            #def apply_mp(rc_bs, scaling=1):
            #    return max_pool_img(rc_bs, N)

            #Apply max pooling and img conversion to each
            #mp_bs_mux = rc_bs_mux
            #mp_bs_maj = rc_bs_maj
            #for i in range(NUM_POOLS):
            #    mp_bs_mux = apply_mp(mp_bs_mux)
            #    mp_bs_maj = apply_mp(mp_bs_maj)
            #mp_img_mux = bs_to_img(mp_bs_mux.cpu().detach().numpy(), bs.bs_mean)
            #mp_img_maj = bs_to_img(mp_bs_maj.cpu().detach().numpy(), bs.bs_mean)
            #mp_img_no_xor_mux = apply_mp(rc_bs_no_xor_mux)
            #mp_img_no_xor_maj = apply_mp(rc_bs_no_xor_maj)

            #Apply max pooling again

            #disp_img(mp_ideal.astype(np.int32))
            #print("breakpoint")

            #Compute error
            #mux_err = np.sum(np.abs(mp_ideal.astype(np.int32) - mp_img_mux.astype(np.int32))) / (h * w)
            #maj_err = np.sum(np.abs(mp_ideal.astype(np.int32) - mp_img_maj.astype(np.int32))) / (h * w)
            #mux_avg_err += mux_err
            #maj_avg_err += maj_err

            #mux_no_xor_err = np.sum(np.abs(mp_ideal.astype(np.int32) - mp_img_no_xor_mux.astype(np.int32))) / (h * w)
            #maj_no_xor_err = np.sum(np.abs(mp_ideal.astype(np.int32) - mp_img_no_xor_maj.astype(np.int32))) / (h * w)
            #mux_no_xor_avg_err += mux_no_xor_err
            #maj_no_xor_avg_err += maj_no_xor_err

            iters += 1
            if iters == num_iters:
                break
        #mux_avg_err /= num_iters
        #maj_avg_err /= num_iters
        #print("Mux err: {}".format(mux_avg_err))
        #print("Maj err: {}".format(maj_avg_err))
        #mux_errs.append(mux_avg_err)
        #maj_errs.append(maj_avg_err)

        #mux_no_xor_avg_err /= num_iters
        #maj_no_xor_avg_err /= num_iters
        #print("Mux no_xor err: {}".format(mux_no_xor_avg_err))
        #print("Maj no_xor err: {}".format(maj_no_xor_avg_err))
        #mux_no_xor_errs.append(mux_no_xor_avg_err)
        #maj_no_xor_errs.append(maj_no_xor_avg_err)

    cmat /= num_iters * 10
    print(cmat)

    #Save some of the data to prevent requiring reruns
    #with open('mux_errs.npy', 'wb') as f:
    #    np.save(f, mux_errs)
    #with open('maj_errs.npy', 'wb') as f:
    #    np.save(f, maj_errs)
    ##with open('mux_no_xor_errs.npy', 'wb') as f:
    ##    np.save(f, mux_no_xor_errs)
    ##with open('maj_no_xor_errs.npy', 'wb') as f:
    ##    np.save(f, maj_no_xor_errs)
#
    #ax = plt.subplot(111)
    #ax.bar(np.array(range(10))-0.3, mux_errs, width=0.2, align='center', label='mux')
    #ax.bar(np.array(range(10))-0.1, maj_errs, width=0.2, align='center', label='maj')
    ##ax.bar(np.array(range(10))+0.1, mux_no_xor_errs, width=0.2, align='center', label='no_xor_mux')
    ##ax.bar(np.array(range(10))+0.3, maj_no_xor_errs, width=0.2, align='center', label='no_xor_maj')
    #plt.xticks(list(range(10)))
    #plt.title("MNIST Edge Detect --> Max Pool SC Error Comparison")
    #plt.xlabel("MNIST Digit Number")
    #plt.ylabel("Avg Error with respect to ideal")
    #plt.legend()
    #plt.show()