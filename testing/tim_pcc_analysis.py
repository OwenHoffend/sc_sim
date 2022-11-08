from multiprocessing import Pool
import numpy as np
import os
from sim.circuits import mux_1, maj_1
from sim.PTM import *
from cv.img_io import disp_img
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

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
    for i in range(8):
        plt.plot(np.mean(SCCs[i], axis=0), label="k={}".format(i))
    plt.xlabel("Px value")
    plt.ylabel("Average SCC")
    plt.legend()
    plt.show()
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

def rand_in_window(img, wsz=None):
    def randint_2_diff(max):
        a = np.random.randint(max)
        b = np.random.randint(max)
        while a == b:
            b = np.random.randint(max)
        return a, b
    h, w = img.shape
    if wsz == None:
        x1, x2 = randint_2_diff(w)
        y1, y2 = randint_2_diff(h)
    else:
        x1_offset, x2_offset = randint_2_diff(wsz)
        y1_offset, y2_offset = randint_2_diff(wsz)
        y = np.random.randint(h-(wsz-1))
        x = np.random.randint(w-(wsz-1))
        x1 = x + x1_offset
        x2 = x + x2_offset
        y1 = y + y1_offset
        y2 = y + y2_offset
    p1 = img[y1, x1]
    p2 = img[y2, x2]
    return p1, p2

def value_distr_impact_mp(img):
    num_windows = 1000000
    result = np.zeros((256, 256), dtype=np.int32)
    data_pairs = np.zeros((), dtype=np.int32)
    for i in range(num_windows):
        if i % (num_windows/10) == 0:
            print("i: {}".format(i))
        p1, p2 = rand_in_window(img)
        result[p1, p2] += 1
    return result

def value_distr_impact(corrupted=False):
    if corrupted:
        all_images = []
        for fn in os.listdir("../tim_pcc_run2/img/"):
            all_images.append(np.load("../tim_pcc_run2/img/{}".format(fn), allow_pickle=True))
    else:
        all_images = list(np.load("../tim_pcc/test_images.npy", allow_pickle=True))
    all_images = [(image*256).astype(np.int) for image in all_images] #Convert pixels to integers]
    #with Pool(10) as p:
    #    results = p.map(value_distr_impact_mp, all_images)
    #for i, result in enumerate(results):
    #    np.save("../tim_pcc_run2/val_distr/img_{}_full".format(i), result)
    #plt.imshow(all_images[5], cmap=plt.get_cmap('gray'), interpolation='nearest')
    #plt.show()
    result = np.load("../tim_pcc_run2/val_distr/img_0_full.npy")
    plt.ylim(0, 256)
    plt.imshow(result, cmap='hot', interpolation='nearest')
    plt.show()

def mmc_scc_on_imgs(k, corrupted=False):
    """For a given k value, compute the average SCC between MMC pairs using values from the 10 MATLAB images
        For each SCC, choose pixel values within a 3x3 window (as done in median filtering)
    """
    num_windows = 1000
    if corrupted:
        all_images = []
        for fn in os.listdir("../tim_pcc_run2/img/"):
            all_images.append(np.load("../tim_pcc_run2/img/{}".format(fn), allow_pickle=True))
    else:
        all_images = list(np.load("../tim_pcc/test_images.npy", allow_pickle=True))

    bin_arrs = np.zeros((256,), dtype=object) #cache int-->bit array conversion
    for i in range(256):
        bin_arrs[i] = list(bin_array(i, 8))

    sccs = []
    vin = get_vin_mc0(np.array([0.5 for _ in range(8)]))
    for img_idx in range(10):
        print("k: {}, img: {}".format(k, img_idx))
        image = all_images[img_idx]
        image_int = (image*256).astype(np.int) #Convert pixels to integers
        for i in range(num_windows):
            if i % (num_windows/10) == 0:
                print("k: {}, i: {}".format(k, i))
            p1, p2 = rand_in_window(image_int)
            mmc1 = get_mmc(k, bin_arrs[p1])
            mmc2 = get_mmc(k, bin_arrs[p2])
            both = lambda *c: mmc_pair(*c, mmc_x=mmc1, mmc_y=mmc2)
            both_ptm = get_func_mat(both, 8, 2)
            vout = both_ptm.T @ vin
            sccs.append(get_corr_mat_paper(vout)[0,1])
    return np.mean(sccs), np.std(sccs)

def mmc_on_imgs_mp():
    #with Pool(8) as p:
    #    results = p.map(mmc_scc_on_imgs, list(range(8)))
    #np.save("../tim_pcc_run2/results/SCC_combs_full.npy", results)
    #return

    scc_uniform_data = np.load("../tim_pcc/SCC_data_n8.npy")
    sccs_uniform = np.mean(scc_uniform_data, axis=(1, 2))
    #sccs_uniform_std = np.std(scc_uniform_data, axis=(1, 2))

    results_img_3x3 = np.load("../tim_pcc_run2/results/SCC_combs_3x3.npy")
    sccs_img_3x3 = results_img_3x3[:, 0]

    results_img_5x5 = np.load("../tim_pcc_run2/results/SCC_combs_5x5.npy")
    sccs_img_5x5 = results_img_5x5[:, 0]

    results_img_7x7 = np.load("../tim_pcc_run2/results/SCC_combs_7x7.npy")
    sccs_img_7x7 = results_img_7x7[:, 0]
    #std_img = results_img[:, 1]

    #results_img_snp = np.load("../tim_pcc_run2/results/SCC_combs_snp.npy")
    #sccs_img_snp = results_img_snp[:, 0]
    #std_img_snp = results_img_snp[:, 1]

    results_img_full = np.load("../tim_pcc_run2/results/SCC_combs_full.npy")
    sccs_img_full = results_img_full[:, 0]
    #std_img_snp = results_img_snp[:, 1]

    column_width = 4
    height = 1.5
    plt.rcParams["figure.figsize"] = (column_width*2, height*2)
    fs1 = 15
    fs2 = 12
    fig, axs = plt.subplots()
    plt.plot(range(8), sccs_uniform, label="Uniform Distribution", marker='o', ms=8, color='#000000', ls=(3, (3, 2)))
    plt.plot(range(8), sccs_img_3x3, label="3x3 Image Windows", marker='*', ms=8, color='#1b9e77', ls=(3, (3, 2)))
    plt.plot(range(8), sccs_img_5x5, label="5x5 Image Windows", marker='^', ms=8, ls=(3, (3, 2)))
    plt.plot(range(8), sccs_img_7x7, label="7x7 Image Windows", marker='x', ms=8, ls=(3, (3, 2)))
    plt.plot(range(8), sccs_img_full, label="Full Image", marker='+', color='red', ms=8, ls=(3, (3, 2)))

    #plt.plot(range(8), sccs_img_snp, label="Img, Noisy", marker='^', ms=8, color='#7570b3')
    axs.set_xlabel("Number of maj gates, k\n(b)", fontsize=fs1)
    axs.set_ylabel("Average SCC(X, Y)", fontsize=fs1)
    axs.set_xticklabels(["", f"{0}\n(WBG)"] + [f"{k}" for k in range(1, 7)] + [f"{7}\n(CMP)"])
    plt.legend(fontsize=fs1, loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.show(block=True)

def check_imgs(k, min_idx, max_idx):
    all_images = []
    for fn in os.listdir("../tim_pcc_run2/img/"):
        all_images.append(np.load("../tim_pcc_run2/img/{}".format(fn), allow_pickle=True))
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
        return check_imgs(p-8, 6, 10)
    else:
        return check_imgs(p, 2, 6)

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

    with Pool(16) as p:
        p.map(check_imgs_mp, list(range(16)))

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

def tims_analysis_pt2():
    #Compute SSIM
    all_images_snp = []
    for fn in os.listdir("../tim_pcc_run2/img/"):
        all_images_snp.append(np.load("../tim_pcc_run2/img/{}".format(fn), allow_pickle=True))
    #all_images_orig = np.load("../tim_pcc/test_images.npy", allow_pickle=True)
    #img_correct = []
    #for img in range(10):
    #    print("med filter ", img)
    #    img_correct.append(correct_median_filter(all_images_snp[img]))
    #ssims_correct = np.empty((8, 10))
    #mses_correct = np.empty((8, 10))
    ##ssims_orig = np.empty((8, 10))
#
    #for img in range(10):
    #    print("SSIM: img=", img)
    #    for k in range(8):
    #        img_es = np.load("../tim_pcc_run2/temp_all_es_img{}_k{}.npy".format(img, k), allow_pickle=True)
    #        ssims_correct[k, img] = ssim(
    #            img_es,
    #            img_correct[img],
    #            data_range=img_es.max() - img_es.min(),
    #            gaussian_weights=True,
    #            win_size=11,
    #            K1=0.01,
    #            K2=0.03
    #        )
#
    #        mses_correct[k, img] = mse(
    #            img_es,
    #            img_correct[img]
    #        )
    #        #h, w = img_es.shape
    #        #ssims_orig[k, img] = ssim(
    #        #    img_es,
    #        #    all_images_orig[img][0:h, 0:w],
    #        #    data_range=img_es.max() - img_es.min(),
    #        #    gaussian_weights=True,
    #        #    win_size=11,
    #        #    K1=0.01,
    #        #    K2=0.03
    #        #)
    #    np.save("../tim_pcc_run2/results/img{}_SSIM_correct.npy".format(img), ssims_correct[:, img])
    #    np.save("../tim_pcc_run2/results/img{}_MSE_correct.npy".format(img), mses_correct[:, img])

    #Load SSIMS & MSEs
    ssims_correct = np.empty((8, 10))
    rmses_correct = np.empty((8, 10))
    for img in range(10):
        ssims_correct[:, img] = np.load("../tim_pcc_run2/results/img{}_SSIM_correct.npy".format(img))
        rmses_correct[:, img] = np.sqrt(np.load("../tim_pcc_run2/results/img{}_MSE_correct.npy".format(img)))

    #SSIM - Line plot for all images, including average
    mean_ssims_correct = np.mean(ssims_correct, axis=1)
    for idx, fn in enumerate(os.listdir("../tim_pcc_run2/img/")):
        plt.plot(ssims_correct[:, idx], label=fn[:-4])
    plt.plot(mean_ssims_correct, label="Average", color='k', lw=2)
    plt.title("SSIM vs. MMC k-value, Salt & Pepper Noise")
    plt.ylabel("SSIM")
    plt.xlabel("k")
    plt.legend()
    plt.grid(True)
    plt.show()

    #SSIM - Bar plot showing avg and std. deviation for all k
    plt.bar([x for x in range(8)], mean_ssims_correct, yerr=np.std(ssims_correct, axis=1), width=0.4)
    plt.title("Average SSIM vs. MMC k-value, Salt & Pepper Noise")
    plt.ylabel("SSIM")
    plt.xlabel("k")
    plt.grid(True)
    plt.show()

    #Relative similarity plot: When do we get diminishing returns.
    for i in range(1, 8):
        plt.bar(i, (mean_ssims_correct[i]/mean_ssims_correct[i-1])-1)
    plt.show()

    #MSE - Line plot for all images, including average
    mean_rmses_correct = np.mean(rmses_correct, axis=1)
    for idx, fn in enumerate(os.listdir("../tim_pcc_run2/img/")):
        plt.plot(rmses_correct[:, idx], label=fn[:-4])
    plt.plot(mean_rmses_correct, label="Average", color='k', lw=2)
    plt.title("RMSE vs. MMC k-value, Salt & Pepper Noise")
    plt.ylabel("RMSE")
    plt.xlabel("k")
    plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(1, 8):
        plt.bar(i, (1-mean_rmses_correct[i]/mean_rmses_correct[i-1]))
    plt.show()

    #MSE - Bar plot showing avg and std. deviation for all k
    plt.bar([x for x in range(8)], mean_rmses_correct, yerr=np.std(rmses_correct, axis=1), width=0.4)
    plt.title("Average RMSE vs. MMC k-value, Salt & Pepper Noise")
    plt.ylabel("RMSE")
    plt.xlabel("k")
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