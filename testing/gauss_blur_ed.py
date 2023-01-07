import os
import numpy as np
from sim.bitstreams import *
from sim.PTM import *
from sim.SEC import *
from sim.circuits import mux_1, maj_1, robert_cross
from sim.espresso import *
from cv.img_io import *
from sim.SEC_opt_macros import *
from skimage.metrics import structural_similarity as ssim

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

def gauss_blur_4x4(
    x11, x12, x13, x14,
    x21, x22, x23, x24,
    x31, x32, x33, x34,
    x41, x42, x43, x44,
    c0, c1, c2, c3,
    exact=False
):

    a1 = gauss_blur_3x3( #ul
        x11, x12, x13,
        x21, x22, x23,
        x31, x32, x33,
        c0, c1, c2, c3,
        exact=exact
    )
    a2 = gauss_blur_3x3( #ur
        x12, x13, x14,
        x22, x23, x24,
        x32, x33, x34,
        c0, c1, c2, c3,
        exact=exact
    )
    a3 = gauss_blur_3x3( #ll
        x21, x22, x23,
        x31, x32, x33,
        x41, x42, x43,
        c0, c1, c2, c3,
        exact=exact
    )
    a4 = gauss_blur_3x3( #lr
        x22, x23, x24,
        x32, x33, x34,
        x42, x43, x44,
        c0, c1, c2, c3,
        exact=exact
    )
    return a1, a2, \
           a3, a4

def gauss_blur_ED(
    x11, x12, x13, x14,
    x21, x22, x23, x24,
    x31, x32, x33, x34,
    x41, x42, x43, x44,
    c0, c1, c2, c3, c4,
    exact=False
):
    a1, a2, a3, a4 = gauss_blur_4x4(
        x11, x12, x13, x14,
        x21, x22, x23, x24,
        x31, x32, x33, x34,
        x41, x42, x43, x44,
        c0, c1, c2, c3,
        exact=exact
    )
    return robert_cross_r(a1, a2, a3, a4, c4, exact=exact)

def extract_gauss_4x4_inputs(img, x, y, consts):
    imgs = [img[y, x], img[y, x+1], img[y, x+2], img[y, x+3],
    img[y+1, x], img[y+1, x+1], img[y+1, x+2], img[y+1, x+3],
    img[y+2, x], img[y+2, x+1], img[y+2, x+2], img[y+2, x+3],
    img[y+3, x], img[y+3, x+1], img[y+3, x+2], img[y+3, x+3]]
    return imgs + list(consts) 

def apply_gauss_4x4(img, x, y, consts, exact=False):
    inputs = extract_gauss_4x4_inputs(img, x, y, consts)
    gb4_out = np.array(gauss_blur_4x4(*inputs, exact=exact))
    if exact:
        return gb4_out.reshape(2, 2) 
    else:
        return gb4_out.reshape(2, 2, 32)

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

def test_gauss_blur_4x4():
    #num_tests = 252 ** 2
    num_tests = 1000
    gb4_ptm = np.load("gb4_ptm.npy")
    #gb4_ptm = get_func_mat(gauss_blur_4x4, 20, 4)
    #np.save("gb4_ptm.npy", gb4_ptm)

    print("Get opt ptm")
    rced_ptm = get_func_mat(robert_cross_r, 5, 1)
    rced_ptm = reduce_func_mat(rced_ptm, 4, 0.5)
    A = gb4_ptm @ B_mat(4)
    Ks = []
    Ks_opt = []
    for i in range(4):
        K = A[:, i].reshape(2**4, 2**16).T
        K_opt = opt_K_max(K)
        Ks.append(K)
        Ks_opt.append(K_opt)

    for i in range(4):
        weights = np.sum(Ks[i], axis=1)
        print(np.unique(weights, return_counts=True))

    Ks_opt_area_aware = opt_K_max_area_aware_multi(Ks)
    gb4_ptm_opt = Ks_to_Mf(Ks_opt)
    gb4_ptm_opt_a = Ks_to_Mf(Ks_opt_area_aware)
    np.save("gb4_opt_a.npy", gb4_ptm_opt_a)

    print(compare_Kmat_hamming_dist(Ks, Ks_opt))
    print(compare_Kmat_hamming_dist(Ks, Ks_opt_area_aware))

    #print(espresso_get_SOP_area(Ks_to_A(Ks), "gb4.in", is_A=True))
    #print(espresso_get_SOP_area(Ks_to_A(Ks_opt), "gb4.in", is_A=True))
    print(espresso_get_SOP_area(Ks_to_A(Ks_opt_area_aware), "gb4.in", is_A=True))

    avg_corr = np.zeros((4,4))
    avg_corr_opt = np.zeros((4,4))
    avg_corr_opt_a = np.zeros((4,4))
    B4 = B_mat(4)
    gk = np.array([0.25, 0.5, 0.25])

    unopt_err = []
    opt_err = []
    opt_a_err = []
    img = load_img("./img/cameraman.png", gs=True)
    v0 = get_vin_mc0(np.array([0.5, 0.5, 0.5, 0.5]))
    
    #correct_img = np.zeros((256, 256))
    #out_img = np.zeros((256, 256))
    #out_img_opt = np.zeros((256, 256))
    for i in range(num_tests):
    #for i in range(252):
    #    print(i)
    #    for j in range(252):
        print(i)
        px = np.random.rand(4, 4)
        #px = img[i:i+4, j:j+4] / 256

        #Correct result computation
        c1 = gk.T @ px[0:3, 0:3] @ gk
        c2 = gk.T @ px[0:3, 1:4] @ gk
        c3 = gk.T @ px[1:4, 0:3] @ gk
        c4 = gk.T @ px[1:4, 1:4] @ gk
        rced_correct = 0.5*(np.abs(c1-c4) + np.abs(c2-c3))

        v_in = np.kron(v0, get_vin_mc1(px.reshape(16, )))
        result_ptv = gb4_ptm.T @ v_in
        result_ptv_opt = gb4_ptm_opt.T @ v_in
        result_ptv_a = gb4_ptm_opt_a.T @ v_in
        rced_out_ptv = rced_ptm.T @ result_ptv
        rced_out_ptv_opt = rced_ptm.T @ result_ptv_opt
        rced_out_ptv_opt_a = rced_ptm.T @ result_ptv_a

        unopt_err.append(np.abs(rced_out_ptv[1] - rced_correct))
        opt_err.append(np.abs(rced_out_ptv_opt[1] - rced_correct))
        opt_a_err.append(np.abs(rced_out_ptv_opt_a[1] - rced_correct))
        #correct_img[i, j] = rced_correct
        #out_img[i, j] = rced_out_ptv[1]
        #out_img_opt[i, j] = rced_out_ptv_opt[1]

        #result_pout = B4.T @ result_ptv
        #result_pout_opt = B4.T @ result_ptv_opt
        #assert np.all(np.isclose(result_pout, result_pout_opt))

        cout = get_corr_mat_paper(result_ptv)
        cout_opt = get_corr_mat_paper(result_ptv_opt)
        cout_opt_a = get_corr_mat_paper(result_ptv_a)
        avg_corr += cout
        avg_corr_opt += cout_opt
        avg_corr_opt_a += cout_opt_a

    avg_corr /= num_tests
    avg_corr_opt /= num_tests
    avg_corr_opt_a /= num_tests
    print(avg_corr)
    print(avg_corr_opt)
    print(avg_corr_opt_a)
    print(np.mean(unopt_err))
    print(np.mean(opt_err))
    print(np.mean(opt_a_err))

def test_gauss_blur_img_bs():
    """test the 4x4 gaussian blur kernel using simulated bitstreams"""
    Ns = [8, 16, 32, 64, 128, 256]
    img_dir = "../tim_pcc_run2/img/"
    img_list = os.listdir(img_dir)
    ssims = np.zeros((len(img_list), len(Ns),  2))
    for img_idx, img_name in enumerate(img_list):
        print("img: ", img_name)
        for N_idx, N in enumerate(Ns):
            print("N: ", N)
            #Get the input bitstreams
            #img = load_img("./img/cameraman.png", gs=True)
            img = np.load(img_dir + img_name)*256
            Npb = np.int32(N/8)
            #disp_img(img)
            h, w = img.shape
            rng = bs.SC_RNG()
            img_bs = img_to_bs(img, rng.bs_lfsr, bs_len=N, lfsr_sz=8)
            const_bs = rng.bs_lfsr_p5_consts(N, 5, 8, add_zero_state=True)

            #Get the output bitstreams - un-optimized design
            gb4_out = np.zeros((h-3, w-3, Npb), dtype=np.uint8) #32 = N/8 (256/8)
            for y in range(h-3):
                #print("y: ", y)
                for x in range(w-3):
                    gb4_out[y, x, :] = apply_gauss_4x4_ed(img_bs, x, y, const_bs)
            #np.save("gb4_cameraman_bs.npy", gb4_out)
            #gb4_out = np.load("gb4_cameraman_bs.npy")
            img_gb4 = bs_to_img(gb4_out, bs_mean)
            #disp_img(img_gb4)

            #Get the output bitstreams - optimized design
            gb4_opt_out = np.zeros((h-3, w-3, Npb), dtype=np.uint8)
            gb4_opt_ptm = np.load("gb4_opt_ptm.npy")
            for y in range(h-3):
                #print("y: ", y)
                for x in range(w-3):
                    bs_mat = np.array(extract_gauss_4x4_inputs(img_bs, x, y, const_bs[:4]))
                    gb4_out = apply_ptm_to_bs(bs_mat, gb4_opt_ptm, packed=True)
                    gb4_opt_out[y, x, :] = robert_cross_r(gb4_out[0], gb4_out[1], gb4_out[2], gb4_out[3], const_bs[4])
            img_gb4_opt = bs_to_img(gb4_opt_out, bs_mean)
            #disp_img(img_gb4_opt)

            #Get the correct output
            gb4_correct = np.zeros((h-3, w-3))
            for y in range(h-3):
                #print("y: ", y)
                for x in range(w-3):
                    gb4_correct[y, x] = apply_gauss_4x4_ed(img, x, y, const_bs, exact=True)
            #disp_img(gb4_correct)
            ssim_orig = ssim(
                img_gb4,
                gb4_correct,
                data_range=255,
                gaussian_weights=True,
                win_size=11,
                K1=0.01,
                K2=0.03
            )
            ssim_opt = ssim(
                img_gb4_opt,
                gb4_correct,
                data_range=255,
                gaussian_weights=True,
                win_size=11,
                K1=0.01,
                K2=0.03
            )
            print(ssim_orig)
            print(ssim_opt)
            ssims[img_idx, N_idx, 0] = ssim_orig
            ssims[img_idx, N_idx, 1] = ssim_opt
    np.save("gb4_ssims.npy", ssims)

def test_gauss_blur_img():
    """test the 4x4 gaussian blur kernel using real images, via PTMs"""
    #num_tests = 252 ** 2
    num_tests = 1000
    gb4_ptm = np.load("gb4_ptm.npy")
    #gb4_ptm = get_func_mat(gauss_blur_4x4, 20, 4)
    #np.save("gb4_ptm.npy", gb4_ptm)

    rced_ptm = get_func_mat(robert_cross_r, 5, 1)
    rced_ptm = reduce_func_mat(rced_ptm, 4, 0.5)
    #A = gb4_ptm @ B_mat(4)
    #Ks = []
    #Ks_opt = []
    #for i in range(4):
    #    K = A[:, i].reshape(2**4, 2**16).T
    #    K_opt = opt_K_max(K)
    #    Ks.append(K)
    #    Ks_opt.append(K_opt)
    #gb4_ptm_opt = Ks_to_Mf(Ks_opt)
    gb4_ptm_opt = np.load("gb4_opt_ptm.npy")
    espresso_get_opt_file(gb4_ptm_opt, "gb4_opt.in", "gb4_opt.out")

    #avg_corr = np.zeros((4,4))
    #avg_corr_opt = np.zeros((4,4))
    #B4 = B_mat(4)
    #gk = np.array([0.25, 0.5, 0.25])

    #unopt_err = []
    #opt_err = []
    img = load_img("./img/cameraman.png", gs=True)
    h, w = img.shape
    v0 = get_vin_mc0(np.array([0.5, 0.5, 0.5, 0.5]))

    #Salt and pepper noise
    p = 1/32
    for i in range(h):
        for j in range(w):
            if p > np.random.uniform():
                if 0.5 > np.random.uniform():
                    img[i, j] = 255
                else:
                    img[i, j] = 0

    disp_img(img)
    
    #correct_img = np.zeros((h, w))
    out_img = np.zeros((h, w))
    out_img_opt = np.zeros((h, w))
    for i in range(128, 138):
        print(i)
        for j in range(w-4):
            px = img[i:i+4, j:j+4] / 256

            #Correct result computation
            #c1 = gk.T @ px[0:3, 0:3] @ gk
            #c2 = gk.T @ px[0:3, 1:4] @ gk
            #c3 = gk.T @ px[1:4, 0:3] @ gk
            #c4 = gk.T @ px[1:4, 1:4] @ gk
            #rced_correct = 0.5*(np.abs(c1-c4) + np.abs(c2-c3))

            v_in = np.kron(v0, get_vin_mc1(px.reshape(16, )))
            result_ptv = gb4_ptm.T @ v_in
            result_ptv_opt = gb4_ptm_opt.T @ v_in
            rced_out_ptv = rced_ptm.T @ result_ptv
            rced_out_ptv_opt = rced_ptm.T @ result_ptv_opt

            #unopt_err.append(np.abs(rced_out_ptv[1] - rced_correct))
            #opt_err.append(np.abs(rced_out_ptv_opt[1] - rced_correct))
            #correct_img[i, j] = rced_correct
            out_img[i, j] = rced_out_ptv[1]
            out_img_opt[i, j] = rced_out_ptv_opt[1]

            #result_pout = B4.T @ result_ptv
            #result_pout_opt = B4.T @ result_ptv_opt
            #assert np.all(np.isclose(result_pout, result_pout_opt))

            #cout = get_corr_mat_paper(result_ptv)
            #cout_opt = get_corr_mat_paper(result_ptv_opt)
            #avg_corr += cout
            #avg_corr_opt += cout_opt

    disp_img(out_img * 256 * 2)
    disp_img(out_img_opt * 256 * 2)
    #avg_corr /= num_tests
    #avg_corr_opt /= num_tests
    #print(avg_corr)
    #print(avg_corr_opt)
    #print(np.mean(unopt_err))
    #print(np.std(unopt_err))
    #print(np.mean(opt_err))
    #print(np.std(opt_err))
    pass