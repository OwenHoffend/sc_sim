from multiprocessing import Pool
import numpy as np
from sim.circuits import mux_1, maj_1
from sim.PTM import *
from cv.img_io import disp_img
import matplotlib.pyplot as plt

def mmc8(*c, k=0, v=[False for _ in range(8)]):
    z = False
    for i in range(8):
        if i < 8-k: #Add a mux
            z = mux_1(c[i], z, v[i])
        else: #Add a maj
            z = maj_1(c[i], z, v[i])
    return z

def get_mmc(k, v):
    return lambda *c: mmc8(*c, k=k, v=v)

def mmc_pair(*c, mmc_x=None, mmc_y=None):
    return mmc_x(*c), mmc_y(*c)

def mmc_many(*c, mmcs=None):
    return tuple([mmc(*c) for mmc in mmcs])

def s(x, y):
    return np.bitwise_and(x, y), np.bitwise_or(x, y) #idx 0: min, idx 1: max

def med3x3(
    a0, a1, a2, 
    a3, a4, a5, 
    a6, a7, a8
):
    b0, b1 = s(a0, a1) #b0: idx 0, b1: idx 1
    b2, b3 = s(a2, a3)
    b4, b5 = s(a4, a5)
    b6, b7 = s(a6, a7)
    c0, c2 = s(b0, b2)
    c1, c3 = s(b1, b3)
    c4, c6 = s(b4, b6)
    c5, c7 = s(b5, b7)
    d0, d4 = s(c0, c4)
    d1, d2 = s(c1, c2)
    d3, d7 = s(c3, c7)
    d5, d6 = s(c5, c6)
    e1, e5 = s(d1, d5)
    e2, e6 = s(d2, d6)
    f2, f4 = s(e2, d4)
    f3, f5 = s(d3, e5)
    g3, g4 = s(f3, f4)
    h3, h8 = s(g3, a8)
    return s(g4, h8)[0]

def check_sccs(k):
    SCCs = np.load("../tim_pcc_run2/SCC_data_n8.npy")
    vin = get_vin_mc0(np.array([0.5 for _ in range(8)]))
    bin_arrs = np.zeros((256, 8), dtype=bool) #cache int-->bit array conversion
    B2 = B_mat(2)
    for i in range(256):
        bin_arrs[i, :] = bin_array(i, 8)
    for i in range(256):
        print("k: {}, i: {}".format(k, i))
        px_bits = list(bin_arrs[i, :])
        mmc_x = lambda *c: mmc8(*c, k=k, v=px_bits)
        for j in range(256):
            py_bits = list(bin_arrs[j, :])
            mmc_y = lambda *c: mmc8(*c, k=k, v=py_bits)
            both = lambda *c: mmc_pair(*c, mmc_x=mmc_x, mmc_y=mmc_y)
            both_ptm = get_func_mat(both, 8, 2)
            vout = both_ptm.T @ vin
            scc = get_corr_mat_paper(vout)[0,1]

            #print("SCC, my method: ", scc)
            #print("SCC, tim's method", SCCs[k, i, j])

            #SCC Check
            if not np.isclose(scc, SCCs[k, i, j]):
                print("Actual: {}, Expected: {}".format(scc, SCCs[k, i, j]))

            #Value Check
            pout = B2.T @ vout
            if not np.isclose(pout[0], i/256) or not np.isclose(pout[1], j/256):
                print("Pout 0: {}, i/256: {}".format(pout[0], i/256))
                print("Pout 1: {}, j/256: {}".format(pout[1], j/256))

def check_imgs(k, min_idx, max_idx):
    import os
    all_images = []
    for fn in os.listdir("../tim_pcc_run2/"):
        all_images.append(np.load("../tim_pcc_run2/{}".format(fn), allow_pickle=True))
    vin = get_vin_mc0(np.array([0.5 for _ in range(8)]))
    median_filter_ptm = get_func_mat(med3x3, 9, 1)
    bin_arrs = np.zeros((256,), dtype=object) #cache int-->bit array conversion
    for i in range(256):
        bin_arrs[i] = list(bin_array(i, 8))
        
    #B9 = B_mat(9)
    for img_idx in range(min_idx, max_idx):  # all_images is a list of the 10 test images from MATLAB
        image = all_images[img_idx]
        h, w = image.shape

        #Override h and w to test some stuff without it taking forever
        image_int = (image*256).astype(np.int) #Convert pixels to integers
        curr_expected_out = np.empty((h-2, w-2))  # a 3x3 filter can be applied to an h x w image (h-2) x (w-2) times.
        img_MSE = 0.0
        for c_x in range(1, w-1): #Center x
            print("img: {}, c_x: {}, k: {}".format(img_idx, c_x, k))
            for c_y in range(1, h-1): #Center y
                kernel_vals = image_int[c_y-1:c_y+2, c_x-1:c_x+2].reshape(9)
                kernel_px = kernel_vals/256
                mmcs = [get_mmc(k, bin_arrs[val]) for val in kernel_vals]
                all_mmcs = lambda *c: mmc_many(*c, mmcs=mmcs)
                all_mmcs_ptm = get_func_mat(all_mmcs, 8, 9)
                v_pcc = all_mmcs_ptm.T @ vin

                #Sanity check
                #print(get_corr_mat_paper(v_pcc))
                #if not np.all(np.isclose(B9.T @ v_pcc, kernel_px)):
                #    print("BAD PCC OUTPUT")

                pout = (median_filter_ptm.T @ v_pcc)[1] #Expected output
                curr_expected_out[c_y-1, c_x-1] = pout
                img_MSE += (pout - np.median(kernel_px)) ** 2
        img_MSE /= 256 ** 2
        RMSE = np.sqrt(img_MSE)
        print("DONE img: {}, k: {}, RMSE: {}".format(img_idx, k, RMSE))
        if k==8 and not np.isclose(RMSE, 0):
            print("TOO MUCH ERROR FOR COMPARATOR CASE")
        np.save("../tim_pcc_run2/temp_all_es_img{}_k{}.npy".format(img_idx, k), curr_expected_out)
        np.save("../tim_pcc_run2/temp_all_rmses_img{}_k{}.npy".format(img_idx, k), RMSE)

def check_imgs_mp(p):
    if p >= 8:
        return check_imgs(p-8, 5, 10)
    else:
        return check_imgs(p, 0, 5)

def correct_median_filter(img):
    h, w = img.shape
    output = np.empty((h-2, w-2))  # a 3x3 filter can be applied to an h x w image (h-2) x (w-2) times.
    for c_x in range(1, w-1): #Center x
        for c_y in range(1, h-1): #Center y
            kernel_vals = img[c_y-1:c_y+2, c_x-1:c_x+2].reshape(9)
            output[c_y-1, c_x-1] = np.median(kernel_vals)
    return output

def tims_analysis():
    #First step would be to reproduce Tim's array using my own technique to ensure that it works
    #with Pool(8) as p:
    #    p.map(check_sccs, list(range(8)))

    check_imgs(0, 0, 5)
    #with Pool(16) as p:
    #    p.map(check_imgs_mp, list(range(16)))

    #open results - original data
    #results_k = []
    #results_rmse_k = []
    #for k in range(8):
    #    #results_0_4 = np.load("../tim_pcc_run2/temp_all_es_img4_k{}.npy".format(k), allow_pickle=True)
    #    results_rmse_0_4 = np.load("../tim_pcc_run2/temp_all_rmses_img4_k{}.npy".format(k), allow_pickle=True)
    #    #results_5_9 = np.load("../tim_pcc_run2/temp_all_es_img9_k{}.npy".format(k), allow_pickle=True)
    #    results_rmse_5_9 = np.load("../tim_pcc_run2/temp_all_rmses_img9_k{}.npy".format(k), allow_pickle=True)
    #    #results_k.append(np.concatenate((results_0_4, results_5_9)))
    #    results_rmse_k.append(np.concatenate((results_rmse_0_4, results_rmse_5_9)))
#

    #Open results - new data
    #all_noisy_images = [
    #    np.load("../tim_pcc_run2/cameraman.npy", allow_pickle=True),
    #    np.load("../tim_pcc_run2/circuit.npy", allow_pickle=True)
    #]

    #for img_idx in range(2):
    #    image = all_noisy_images[img_idx]
    #    h, w = image.shape
    #    curr_expected_output = np.empty((8, h-2, w-2))
    #    curr_rmses = np.empty((8, ))
    #    for k in range(8):
    #        curr_expected_output[k, :, :] = np.load("../tim_pcc_run2/temp_all_es_img{}_k{}.npy".format(img_idx, k), allow_pickle=True)
    #        curr_rmses[k] = np.load("../tim_pcc_run2/temp_all_rmses_img{}_k{}.npy".format(img_idx, k), allow_pickle=True)
        #disp_img(curr_expected_output[0, :, :]*256)
        #disp_img(curr_expected_output[3, :, :]*256)
        #disp_img(curr_expected_output[7, :, :]*256)
        #np.save("../tim_pcc_run2/results/img{}_expected.npy".format(img_idx), curr_expected_output)
        #np.save("../tim_pcc_run2/results/img{}_rmse.npy".format(img_idx), curr_rmses)

    #Compute SSIM
    all_images = np.load("../tim_pcc/test_images.npy", allow_pickle=True)
    from skimage.metrics import structural_similarity as ssim
    cameraman_img = correct_median_filter(all_images[7])
    circuit_img = correct_median_filter(all_images[8]) 
    ssims_cameraman = np.empty((8, ))
    ssims_circuit = np.empty((8, ))
    for k in range(8):
        cameraman_noisy = np.load("../tim_pcc_run2/temp_all_es_img0_k{}.npy".format(k), allow_pickle=True)
        circuit_noisy = np.load("../tim_pcc_run2/temp_all_es_img1_k{}.npy".format(k), allow_pickle=True)
        ssims_cameraman[k] = ssim(cameraman_img, cameraman_noisy, data_range=cameraman_noisy.max() - cameraman_noisy.min())
        ssims_circuit[k] = ssim(circuit_img, circuit_noisy, data_range=circuit_noisy.max() - circuit_noisy.min())
    np.save("../tim_pcc_run2/results/img0_SSIM.npy", ssims_cameraman)
    np.save("../tim_pcc_run2/results/img1_SSIM.npy", ssims_circuit)
    plt.plot(ssims_cameraman, marker='o', label="cameraman")
    plt.plot(ssims_circuit, marker='o', label="circuit")
    plt.title("Median Filter SSIM vs. MMC k-value")
    plt.ylabel("SSIM")
    plt.xlabel("k")
    plt.legend()
    plt.grid(True)
    plt.show()

    #labels = ["rice", "coins", "pillsetc", "coloredChips", "tape", "lighthouse", "hands1", "cameraman", "circuit", "tire"]
    #for img_idx in range(10):
    #    arr = np.empty((8,))
    #    for k in range(8):
    #        arr[k] = results_rmse_k[k][img_idx]
    #    np.save("../tim_pcc_run2/results/img{}_rmse.npy".format(img_idx), arr)
    #    plt.plot(arr, label=labels[img_idx])
    #plt.title("3x3 Median Filter - Error from Ideal")
    #plt.xlabel("k (number of MAJ gates)")
    #plt.ylabel("RMSE")
    #plt.legend()
    #plt.grid(True)
    #plt.show()

    #for img_idx in range(10):
    #    image = all_images[img_idx]
    #    h, w = image.shape
    #    curr_expected_output = np.empty((8, h-2, w-2))
    #    for k in range(8):
    #        curr_expected_output[k, :, :] = results_k[k][img_idx]
        #disp_img(curr_expected_output[0]*256)
        #disp_img(curr_expected_output[3]*256)
        #disp_img(curr_expected_output[7]*256)
        #np.save("../tim_pcc_run2/results/img{}_expected.npy".format(img_idx), curr_expected_output)