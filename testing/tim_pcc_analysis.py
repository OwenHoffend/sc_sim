from multiprocessing import Pool
import numpy as np
from sim.circuits import mux_1, maj_1
from sim.PTM import *

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
    SCCs = np.load("../tim_pcc/SCC_data_n8.npy")
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

def check_imgs(k):
    all_images = np.load("../tim_pcc/test_images.npy", allow_pickle=True)
    vin = get_vin_mc0(np.array([0.5 for _ in range(8)]))
    median_filter_ptm = get_func_mat(med3x3, 9, 1)
    bin_arrs = np.zeros((256,), dtype=object) #cache int-->bit array conversion
    for i in range(256):
        bin_arrs[i] = list(bin_array(i, 8))
    all_rmses = []
    all_expected_outputs = []
    B9 = B_mat(9)
    for img_idx, image in enumerate(all_images):  # all_images is a list of the 10 test images from MATLAB
        h, w = image.shape

        #Override h and w to test some stuff without it taking forever
        image_int = (image*256).astype(np.int) #Convert pixels to integers
        curr_expected_out = np.empty((h-2, w-2))  # a 3x3 filter can be applied to an h x w image (h-2) x (w-2) times.
        img_MSE = 0.0
        for c_x in range(1, w-1): #Center x
            print("img: {}, c_x: {}, k: {}".format(img_idx, c_x, k))
            for c_y in range(1, h-1): #Center y
                kernel_vals = image_int[c_x-1:c_x+2, c_y-1:c_y+2].reshape(9)
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
                curr_expected_out[c_x-1, c_y-1] = pout
                img_MSE += (pout - np.median(kernel_px)) ** 2
        img_MSE /= 256 ** 2
        RMSE = np.sqrt(img_MSE)
        print("DONE img: {}, k: {}, RMSE: {}".format(img_idx, k, RMSE))
        if k==8 and not np.isclose(RMSE, 0):
            print("TOO MUCH ERROR FOR COMPARATOR CASE")
        all_expected_outputs.append(curr_expected_out)
        all_rmses.append(RMSE)
        np.save("../tim_pcc/temp_all_es_img{}_k{}.npy".format(img_idx, k), np.array(all_expected_outputs, dtype=object))
        np.save("../tim_pcc/temp_all_rmses_img{}_k{}.npy".format(img_idx, k), np.array(all_rmses, dtype=object))
    return all_expected_outputs, all_rmses

def tims_analysis():
    #First step would be to reproduce Tim's array using my own technique to ensure that it works
    #with Pool(8) as p:
    #    p.map(check_sccs, list(range(8)))

    max_k = 8
    with Pool(max_k) as p:
        results = p.map(check_imgs, list(range(max_k)))

    #process results
    all_images = np.load("../tim_pcc/test_images.npy", allow_pickle=True)
    all_expected_outputs = []
    all_rmses = []
    for img_idx, image in enumerate(all_images):
        h, w = image.shape

        curr_expected_output = np.empty((max_k, h-2, w-2))
        curr_rmse = np.empty((max_k, 1))
        for k in range(max_k): #k=0 through k=max_k-1
            curr_expected_output[k, :, :] = results[k][0][img_idx]
            curr_rmse[k, :] = results[k][1][img_idx]
        all_expected_outputs.append(curr_expected_output)
        all_rmses.append(curr_rmse)

    # the follow lines saves the data so you can send to me
    filename = f"../tim_pcc/all_expected_outputs_n8.npy"
    all_expected_outputs = np.array(all_expected_outputs, dtype=object)
    np.save(filename, all_expected_outputs)

    filename = f"../tim_pcc/all_rmses_n8.npy"
    all_rmses = np.array(all_rmses, dtype=object)
    np.save(filename, all_rmses)
