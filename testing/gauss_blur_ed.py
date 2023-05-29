import numpy as np
from sim.bitstreams import *
from sim.PTM import *
from sim.SEC import *
from sim.circuits import mux_1, maj_1, robert_cross
from sim.espresso import *
from cv.img_io import *
from cv.img_quality import *
from sim.SEC_opt_macros import *
from sim.seq_recorr import *
import matplotlib.pyplot as plt

def robert_cross_r(x11, x22, x12, x21, s, exact=False):
    #Version of robert_cross with sel on the other side
    if exact:
        return 0.5*np.abs(x11-x22)+0.5*np.abs(x12-x21)
    else:
        return robert_cross(s, x11, x22, x12, x21)

def gauss_blur_3x1(x1, x2, x3, c0, c1, exact=False):
    if exact:
        return 0.25*x1 + 0.5*x2 + 0.25*x3
    else:
        return mux_1(c1, mux_1(c0, x1, x3), x2) 

def gauss_blur_3x3(
    x11, x12, x13,
    x21, x22, x23,
    x31, x32, x33,
    c0, c1, c2, c3,
    exact=False
):
    a1 = gauss_blur_3x1(x11, x12, x13, c0, c1, exact=exact)
    a2 = gauss_blur_3x1(x21, x22, x23, c0, c1, exact=exact)
    a3 = gauss_blur_3x1(x31, x32, x33, c0, c1, exact=exact)

    return gauss_blur_3x1(a1, a2, a3, c2, c3, exact=exact)

def gb2(
    x11, x12, x13,
    x21, x22, x23, x24,
    x31, x32, x33, x34,
         x42, x43, x44,
    c0, c1, c2, c3,
    exact=False
):
    top = gauss_blur_3x3(
        x11, x12, x13,
        x21, x22, x23,
        x31, x32, x33,
        c0, c1, c2, c3,
        exact=exact
    )

    bot = gauss_blur_3x3(
        x22, x23, x24,
        x32, x33, x34,
        x42, x43, x44,
        c0, c1, c2, c3,
        exact=exact
    )

    return top, bot

def gauss_blur_4x4(
    x11, x12, x13, x14,
    x21, x22, x23, x24,
    x31, x32, x33, x34,
    x41, x42, x43, x44,
    c0, c1, c2, c3,
    exact=False
):

    a11 = gauss_blur_3x3( #ul
        x11, x12, x13,
        x21, x22, x23,
        x31, x32, x33,
        c0, c1, c2, c3,
        exact=exact
    )
    a12 = gauss_blur_3x3( #ur
        x12, x13, x14,
        x22, x23, x24,
        x32, x33, x34,
        c0, c1, c2, c3,
        exact=exact
    )
    a21 = gauss_blur_3x3( #ll
        x21, x22, x23,
        x31, x32, x33,
        x41, x42, x43,
        c0, c1, c2, c3,
        exact=exact
    )
    a22 = gauss_blur_3x3( #lr
        x22, x23, x24,
        x32, x33, x34,
        x42, x43, x44,
        c0, c1, c2, c3,
        exact=exact
    )
    return a11, a12, \
           a21, a22

def gauss_blur_ED(
    x11, x12, x13, x14,
    x21, x22, x23, x24,
    x31, x32, x33, x34,
    x41, x42, x43, x44,
    c0, c1, c2, c3, c4,
    exact=False
):
    a11, a12, a21, a22 = gauss_blur_4x4(
        x11, x12, x13, x14,
        x21, x22, x23, x24,
        x31, x32, x33, x34,
        x41, x42, x43, x44,
        c0, c1, c2, c3,
        exact=exact
    )
    return robert_cross_r(a11, a22, a12, a21, c4, exact=exact)

def GBED_gb2_opt(
    x11, x12, x13, x14,
    x21, x22, x23, x24,
    x31, x32, x33, x34,
    x41, x42, x43, x44,
    c0, c1, c2, c3, **kwargs
):
    if 'gb2_opt_func' in kwargs:
        gb2_opt_func = kwargs['gb2_opt_func']
    else: #The following two lines are expensive so we need the option to do it without loading the PTM every time
        gb2_opt_ptm = np.load("gb2_ptm_opt.npy")
        gb2_opt_func = get_func_from_ptm(gb2_opt_ptm) # probably want to cache this if we're running this many times
    a11, a22 = gb2_opt_func(
        x11, x12, x13,
        x21, x22, x23, x24,
        x31, x32, x33, x34,
             x42, x43, x44,
        c0, c1, c2, c3
    )
    a12, a21 = gb2_opt_func(
        x14, x13, x12,
        x24, x23, x22, x21,
        x34, x33, x32, x31, 
             x43, x42, x41,
        c0, c1, c2, c3
    )
    return a11, a12, a21, a22

def extract_gauss_4x4_inputs(img, x, y, consts):
    imgs = [img[y, x], img[y, x+1], img[y, x+2], img[y, x+3],
    img[y+1, x], img[y+1, x+1], img[y+1, x+2], img[y+1, x+3],
    img[y+2, x], img[y+2, x+1], img[y+2, x+2], img[y+2, x+3],
    img[y+3, x], img[y+3, x+1], img[y+3, x+2], img[y+3, x+3]]
    return imgs + list(consts) 

def gen_opt_gb2_ptm():
    gb2_ptm = get_func_mat(gb2, 18, 2)
    np.save("gb2.npy", gb2_ptm)
    io = IO_Params(4, 14, 2)
    Ks = get_Ks_from_ptm(gb2_ptm, io)
    K1, K2 = opt_K_max(Ks[0]), opt_K_max(Ks[1])
    ptm_opt = Ks_to_Mf([K1, K2])
    np.save("gb2_ptm_opt.npy", ptm_opt)

def apply_gauss_4x4(img, x, y, consts, N, exact=False):
    inputs = extract_gauss_4x4_inputs(img, x, y, consts)
    gb4_out = np.array(gauss_blur_4x4(*inputs, exact=exact))
    if exact:
        return gb4_out.reshape(2, 2) 
    else:
        return gb4_out.reshape(2, 2, int(N/8))

def apply_gauss_4x4_ed(img, x, y, consts, exact=False):
    inputs = extract_gauss_4x4_inputs(img, x, y, consts)
    return gauss_blur_ED(*inputs, exact=exact)

def test_gauss_blur_3x1():
    gb31_ptm = get_func_mat(gauss_blur_3x1, 5, 1)
    A = gb31_ptm @ B_mat(1)
    K = A.reshape(2**2, 2**3).T
    K_opt = opt_K_max(K)
    gb31_ptm_opt = Ks_to_Mf([K_opt,])
    #print(espresso_get_SOP_area(gb31_ptm, "gb4.in", do_print=True))
    #print(espresso_get_SOP_area(gb31_ptm_opt, "gb4.in", do_print=True))
    #opt_area_SECO(K, K, cache_file="gb3.json", print_final_espresso=True, simulated_annealing=True)
    opt_area_SECO(K_opt, K_opt, cache_file="gb3.json", print_final_espresso=True)

def test_gauss_blur_3x3():
    num_tests = 1000
    gb3_ptm = get_func_mat(gauss_blur_3x3, 13, 1)
    for _ in range(num_tests):
        px = np.random.rand(3, 3)
        gk = np.array([0.25, 0.5, 0.25])
        correct_result = gk.T @ px @ gk
        v_in = np.kron(get_vin_mc0(np.array([0.5, 0.5, 0.5, 0.5])), get_vin_mc1(px.reshape(9, )))
        result = B_mat(1).T @ gb3_ptm.T @ v_in
        assert np.isclose(correct_result, result)

def test_gauss_blur_img_bs():
    """test the 4x4 gaussian blur kernel using simulated bitstreams"""
    N = 256
    num_trials = 5
    ds = [1, 2]
    img_dir = "../matlab_test_imgs/gauss_noise/"
    img_dir_orig = "../matlab_test_imgs/original/"
    img_list = os.listdir(img_dir)
    ssims = np.zeros((len(img_list), len(ds), 4)) #4 is the number of MAIN TESTCASEs
    psnrs = np.zeros_like(ssims)
    sccs = np.zeros_like(ssims)
    #img_results = np.zeros(((len(img_list), len(ds), num_trials, 4)), dtype=np.object_)
    for img_idx, img_name in enumerate(img_list):
        print("img: ", img_name)
        #Get the input bitstreams
        img = np.load(img_dir + img_name)*256
        img_orig = np.load(img_dir_orig + img_name)*256
        Npb = np.int32(N/8)
        h, w = img.shape
        for trial_idx in range(num_trials):
            print("trial: ", trial_idx)
            rng = bs.SC_RNG()
            img_bs = img_to_bs(img, rng.bs_lfsr, bs_len=N, lfsr_sz=8)
            const_bs = rng.bs_lfsr_p5_consts(N, 5, 8, add_zero_state=True)

            #MAIN TESTCASE --> Un-optimized design
            #gb4_out = np.zeros((h-3, w-3, Npb), dtype=np.uint8) #32 = N/8 (256/8)
            #for y in range(h-3):
            #    #print("y: ", y)
            #    for x in range(w-3):
            #        gb4_out[y, x, :] = apply_gauss_4x4_ed(img_bs, x, y, const_bs)
            #np.save("gb4_cameraman_bs.npy", gb4_out)
            #gb4_out = np.load("gb4_cameraman_bs.npy")
            #img_gb4 = bs_to_img(gb4_out, bs_mean)
            #disp_img(255-img_gb4)

            #GET THE CORRECT OUTPUT - ground truth edges
            gb4_correct = np.zeros((h-3, w-3))
            for y in range(h-3):
                #print("y: ", y)
                for x in range(w-3):
                    gb4_correct[y, x] = apply_gauss_4x4_ed(img_orig, x, y, const_bs, exact=True)
            #disp_img(255-gb4_correct)

            #ssim_orig = ssim(img_gb4, gb4_correct) #un-optimized gb4
            #psnr_orig = psnr(img_gb4, gb4_correct)

            for d_idx, d in enumerate(ds):
                print("d: ", d)

                #MAIN TESTCASE --> Optimized design
                #gb4_opt_out = np.zeros((h-3, w-3, Npb), dtype=np.uint8)
                #gb4_both_out = np.zeros((h-3, w-3, Npb), dtype=np.uint8)
                gb4_opt_ptm = np.load("gb4_gb2_opt_ptm.npy") #<-- change this

                scc_opt = 0.0
                scc_both = 0.0
                for y in range(h-3):
                    #print("y: ", y)
                    for x in range(w-3):
                        bs_mat = np.array(extract_gauss_4x4_inputs(img_bs, x, y, const_bs[:4]))
                        gb4_layer_opt = apply_ptm_to_bs(bs_mat, gb4_opt_ptm, packed=True)
                        #gb4_opt_out[y, x, :] = robert_cross_r(gb4_layer_opt[0], gb4_layer_opt[3], gb4_layer_opt[1], gb4_layer_opt[2], const_bs[4])

                        #MAIN TESTCASE --> Optimized design AND sequential recorrelation
                        scc_opt += bs_scc(gb4_layer_opt[0], gb4_layer_opt[3], bs_len=N)
                        reco_00, reco_11 = fsm_reco_d(gb4_layer_opt[0], gb4_layer_opt[3], d, packed=True)
                        #reco_01, reco_10 = fsm_reco_d(gb4_layer_opt[1], gb4_layer_opt[2], d, packed=True)
                        #gb4_both_out[y, x, :] = robert_cross_r(reco_00, reco_11, reco_01, reco_10, const_bs[4])
                        scc_both += bs_scc(reco_00, reco_11, bs_len=N)
                scc_opt /= (h-3)*(w-3)
                scc_both /= (h-3)*(w-3)
                        
                #img_gb4_opt = bs_to_img(gb4_opt_out, bs_mean)
                #img_gb4_both = bs_to_img(gb4_both_out, bs_mean)
                #disp_img(255-img_gb4_opt)
                #disp_img(255-img_gb4_both)

                #MAIN TESTCASE --> Sequential re-correlation on its own
                #gb4_reco_out = np.zeros((h-3, w-3, Npb), dtype=np.uint8)
                scc_orig = 0.0
                scc_reco = 0.0
                for y in range(h-3):
                    for x in range(w-3):
                        gb4_layer = apply_gauss_4x4(img_bs, x, y, const_bs[:4], N) #shape (2, 2, N/8)
                        scc_orig += bs_scc(gb4_layer[0, 0, :], gb4_layer[1, 1, :], bs_len=N)
                        reco_00, reco_11 = fsm_reco_d(gb4_layer[0, 0, :], gb4_layer[1, 1, :], d, packed=True)
                        #reco_01, reco_10 = fsm_reco_d(gb4_layer[0, 1, :], gb4_layer[1, 0, :], d, packed=True)
                        scc_reco += bs_scc(reco_00, reco_11, bs_len=N)
                        #gb4_reco_out[y, x, :] = robert_cross_r(reco_00, reco_11, reco_01, reco_10, const_bs[4])
                scc_orig /= (h-3)*(w-3)
                scc_reco /= (h-3)*(w-3)
                #img_gb4_reco = bs_to_img(gb4_reco_out, bs_mean)
                #disp_img(img_gb4_reco)

                #ssim_opt = ssim(img_gb4_opt, gb4_correct) #optimized gb4
                #ssim_both = ssim(img_gb4_both, gb4_correct) #optimized gb4 + sequential recorrelation
                #ssim_reco = ssim(img_gb4_reco, gb4_correct) #sequential recorrelation only
                #psnr_opt = psnr(img_gb4_opt, gb4_correct)
                #psnr_both = psnr(img_gb4_both, gb4_correct)
                #psnr_reco = psnr(img_gb4_reco, gb4_correct)

                #print("SSIMs:")
                #print(ssim_orig)
                #print(ssim_opt)
                #print(ssim_both)
                #print(ssim_reco)
                
                #print("psnrs:")
                #print(psnr_orig)
                #print(psnr_opt)
                #print(psnr_both)
                #print(psnr_reco)

                print("SCCs")
                print(scc_orig)
                print(scc_opt)
                print(scc_both)
                print(scc_reco)
                
                #ssims[img_idx, d_idx, 0] += ssim_orig
                #ssims[img_idx, d_idx, 1] += ssim_opt
                #ssims[img_idx, d_idx, 2] += ssim_both
                #ssims[img_idx, d_idx, 3] += ssim_reco

                #psnrs[img_idx, d_idx, 0] += psnr_orig
                #psnrs[img_idx, d_idx, 1] += psnr_opt
                #psnrs[img_idx, d_idx, 2] += psnr_both
                #psnrs[img_idx, d_idx, 3] += psnr_reco

                sccs[img_idx, d_idx, 0] += scc_orig
                sccs[img_idx, d_idx, 1] += scc_opt
                sccs[img_idx, d_idx, 2] += scc_both
                sccs[img_idx, d_idx, 3] += scc_reco

                #img_results[img_idx, d_idx, trial_idx, 0] = gb4_correct
                #img_results[img_idx, d_idx, trial_idx, 1] = img_gb4_opt
                #img_results[img_idx, d_idx, trial_idx, 2] = img_gb4_both
                #img_results[img_idx, d_idx, trial_idx, 3] = img_gb4_reco
    ssims /= num_trials
    psnrs /= num_trials
    #np.save("gb4_ssims_reco_N{}.npy".format(N), ssims)
    #np.save("gb4_psnrs_reco_N{}.npy".format(N), psnrs)
    np.save("gb4_sccs_reco_N{}.npy".format(N), sccs)
    #np.save("gb4_img_results_N{}.npy".format(N), img_results)

def compute_conf_mat_stats(img_str):
    imgs = np.load(img_str, allow_pickle=True)
    num_imgs, num_ds, num_trials, _ = imgs.shape
    for i in range(num_imgs):
        for d in range(num_ds):
            for t in range(num_trials):
                img_orig = imgs[i, d, t, 0]
                disp_img(img_orig)
                pass


def analyze_gb4_ssim_results():
    gb4_ssims_original = np.load("gb4_ssims2_original.npy")
    gb4_ssims_snp = np.load("gb4_ssims2_snp.npy")
    gb4_ssims_gauss = np.load("gb4_ssims2_gauss.npy")
    
    #First bar plot - raw SSIMs
    def plot_ssim_curve(gb4_ssims, color1, color2, input_type, offset):
        ssims_mean = np.mean(gb4_ssims, axis=0)
        plt.bar([6*x+offset for x in range(6)], ssims_mean[:, 0], width=0.4, label=input_type+" original", color=color1)
        plt.bar([6*x+0.4+offset for x in range(6)], ssims_mean[:, 1], width=0.4, label=input_type+" opt", color=color2)

    plot_ssim_curve(gb4_ssims_original, 'royalblue', 'skyblue', 'original', 0)
    plot_ssim_curve(gb4_ssims_snp, 'green', 'lightgreen', 'snp noise', 1)
    plot_ssim_curve(gb4_ssims_gauss, 'firebrick', 'lightcoral', 'gauss noise', 2)
    plt.ylabel("SSIM")
    plt.xlabel("Bitstream length")
    plt.xticks([6*x+1.2 for x in range(6)], ["$2^3$", "$2^4$", "$2^5$", "$2^6$", "$2^7$", "$2^8$"])
    plt.title("3x3 Gauss Blur + Edge Detection SSIM Results")
    plt.legend()
    plt.show()

    #Second bar plot - relative SSIMs, including re-correlation
    #def plot_relative_ssim(gb4_ssims, color1, color2, input_type, offset):
    #    ssims_mean = np.mean(gb4_ssims, axis=0)
    #    opt_relative = ssims_mean[:, 1] / ssims_mean[:, 0]
    #    reco_relative = ssims_mean[:, 2] / ssims_mean[:, 0]
    #    #plt.bar([6*x+offset for x in range(6)], reco_relative, width=0.4, label=input_type+" reco", color=color1)
    #    plt.bar([6*x+0.4+offset for x in range(6)], opt_relative, width=0.4, label=input_type+" opt", color=color2)
    #plot_relative_ssim(gb4_ssims_original, 'royalblue', 'skyblue', 'original', 0)
    #plot_relative_ssim(gb4_ssims_snp, 'cyan', 'lightgreen', 'snp noise', 1)
    #plot_relative_ssim(gb4_ssims_gauss, 'purple', 'lightcoral', 'gauss noise', 2)
    #plt.ylabel("Relative SSIM Improvement")
    #plt.xlabel("Bitstream length")
    #plt.xticks([6*x+1.2 for x in range(6)], ["$2^3$", "$2^4$", "$2^5$", "$2^6$", "$2^7$", "$2^8$"])
    #plt.title("3x3 Gauss Blur + Edge Detection Relative SSIM Improvement")
    #plt.legend(loc="lower center")
    #plt.show()