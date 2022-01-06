import os
import numpy as np
from numpy.lib.npyio import load
import sim.bitstreams as bs
from sim.PTM import *
from sim.bitstreams import *
from sim.circuits import *
from cv.img_io import cifar_unpickle, load_img

def uniform_ptv_test(v_c, m, n, N):
    Bn = B_mat(n)
    q = 1.0 / N
    c_mat_avg = np.zeros((n, n))
    c_pred_avg = np.zeros((n, n))
    m_actual = 0
    for i in range(m):
        Px = np.random.uniform(size=n)
        #Px /= np.linalg.norm(Px, 1) #Scale by L1-norm to ensure probabilities sum to 1

        ptv = v_c(Px)
        if ptv is None:
            continue
        if not np.isclose(np.sum(ptv), 1.0):
            print("FAILED: ptv sum is wrong: sum: {}, ptv: {}".format(np.sum(ptv), ptv))
            return False

        #Test that the ptv reduction probabilities match
        Px_test = Bn.T @ ptv
        if not np.all(np.isclose(Px, Px_test)):
            print("Px FAILED: \n Px: {}, \n Px_test: {}, \n ptv: {}".format(Px, Px_test, ptv))
            return False

        #Generate bitstreams from the ptv
        bs_mat = sample_from_ptv(ptv, N)
        _, N_new = bs_mat.shape

        #Skip if any of the bitstreams are 0 or 1
        sums = np.sum(bs_mat, 1)
        if 0 in sums or N_new in sums:
            continue
        m_actual += 1

        #Test that the correlation matches

        #Predicted correlation matrix
        c_pred_avg += get_corr_mat_paper(ptv)

        #Actual correlation matrix
        c_mat_avg += get_corr_mat_np(bs_mat)

    c_pred_avg /= m_actual
    c_mat_avg /= m_actual
    print("Avg predicted corr mat: \n", c_pred_avg)
    print("Avg actual corr mat: \n", c_mat_avg)
    #print("PASSED")
    return True

def test_ptv_gen():
    n = 5
    m = 1000
    N = 1000
    print("Testing +1 PTV generation")
    assert uniform_ptv_test(get_vin_mc1_paper, m, n, N)
    #(Works always)

    print("Testing 0 PTV generation")
    assert uniform_ptv_test(get_vin_mc0, m, n, N)
    #(Works on average, probably if satisfiable)

    print("Test -1 PTV generation")
    assert uniform_ptv_test(get_vin_mcn1, m, n, N)
    #(Works always, if satisfiable)

    print("Test any c PTV generation")
    c = -0.3
    func = lambda Px: get_vin_mc_any(Px, c)
    assert uniform_ptv_test(func, m, n, N)
    #(Works on average, probably if satisfiable)

    print("Test hybrid (contiguous) PTV generation")
    def hybrid1(Px):
        S1 = get_vin_mc1(Px[0:3])
        S2 = get_vin_mc1(Px[3:5])
        return np.kron(S2, S1)
    assert uniform_ptv_test(hybrid1, m, 5, N)
    #(Works, but it appears that the kron is backwards from what I would expect)

    print("Testing hybrid (non-contiguous) PTV generation")
    def hybrid2(Px):
        S1 = get_vin_mc1(np.array([Px[0], Px[2]]))
        S2 = get_vin_mc1(np.array([Px[1], Px[3]]))
        pre_swap = np.kron(S2, S1)
        swap_inds = np.array([0, 2, 1, 3])
        return PTV_swap_cols(pre_swap, swap_inds)
    assert uniform_ptv_test(hybrid2, m, 4, N)
    #(Works when the generated ptv is propery reordered)

def test_rc():
    #Full circuit
    Mf_rc = get_func_mat(robert_cross_mp, 10, 1)
    print(Mf_rc)

    #Same circuit, but layer-by-layer
    Mf1 = get_func_mat(robert_cross_mp_l1, 10, 9) #Layer 1: XORs
    Mf2 = get_func_mat(robert_cross_mp_l2, 9, 4) #Layer 2: MUX/MAJ
    Mf3 = get_func_mat(robert_cross_mp_l3, 4, 1) #Layer 3: OR tree
    assert np.all(Mf_rc == Mf1 @ Mf2 @ Mf3)

    m = 10 #Number of 3x3 patches
    MNIST = False
    RECO = True
    if RECO:
        reco_divisions = 25
        sccs = np.linspace(0, 1, reco_divisions)
        errs = np.zeros(reco_divisions)

    #MNIST
    if MNIST:
        wh = 28
        maxnum = 10
        imgs = []
        img_cnts = np.zeros(maxnum)
        for i in range(maxnum):
            MNIST_dir = './img/mnist_png/testing/{}/'.format(i)
            MNIST_files = os.listdir(MNIST_dir)
            img_cnts[i] = len(MNIST_files)
            imgs.append([load_img(os.path.join(MNIST_dir, f), gs=True) for f in MNIST_files])
        nums = np.random.randint(maxnum, size=m)
    else: #CIFAR-10
        wh = 32
        imgs = cifar_unpickle("./img/cifar-10-batches-py/data_batch_1")
        img_cnt, _ = imgs.shape
        imgs = imgs[:, 0:1024].reshape((img_cnt, wh, wh))

    patches = np.random.randint(wh-2, size=(m, 2))
    B1 = B_mat(1)
    B9 = B_mat(9)
    B4 = B_mat(4)
    B1_M3 = B1.T @ Mf3.T
    M2_M1 = Mf2.T @ Mf1.T
    avg_err = 0.0
    avg_cmat_l1 = np.zeros((9, 9))
    avg_cmat_l2 = np.zeros((4, 4))
    for i in range(m):
        print(i)
        pi, pj = patches[i, :]
        if MNIST:
            num = nums[i]
            img_idx = np.random.randint(img_cnts[num])
            patch = imgs[num][img_idx][pi:pi+3, pj:pj+3]
        else:
            img_idx = np.random.randint(img_cnt)
            patch = imgs[img_idx][pi:pi+3, pj:pj+3]

        pin = patch.flatten() / 255.0
        correct = robert_cross_mp_ideal(*pin)

        if RECO:
            #vin = np.kron(get_vin_mc1(pin), get_vin_mc1(np.array([0.5,])))
            for j in range(reco_divisions):
                c = sccs[j]
                #First layer
                vin = np.kron(get_vin_mc_any(pin, c), get_vin_mc1(np.array([0.5,]))) 
                pout = B1.T @ Mf_rc.T @ vin


                #Second layer
                #v_l1 = Mf1.T @ vin
                #p_l1 = B9.T @ v_l1
                #v_l1 = np.kron(get_vin_mc_any(p_l1[1:], c), get_vin_mc1(np.array([0.5,])))
                #pout = B1_M3 @ Mf2.T @ v_l1

                #Third layer
                #v_l2 = M2_M1 @ vin
                #p_l2 = B4.T @ v_l2
                #v_l2 = get_vin_mc_any(p_l2, c)
                #pout = B1_M3 @ v_l2

                errs[j] += np.abs(pout - correct)
        else:
            vin = np.kron(get_vin_mc1(pin), get_vin_mc1(np.array([0.5,]))) #Hybrid input correlation
            v_l1 = Mf1.T @ vin
            v_l2 = Mf2.T @ v_l1
            pout = B1_M3 @ v_l2
            avg_cmat_l1 += get_corr_mat_paper(v_l1)
            avg_cmat_l2 += get_corr_mat_paper(v_l2)
            #print("Pout: {}".format(pout))
            #print("Correct: {}".format(correct))
            avg_err += np.abs(pout - correct)
    if RECO:
        errs /= m
        print("Average errors: {}".format(errs))
    else:
        avg_err /= m
        avg_cmat_l1 /= m
        avg_cmat_l2 /= m
        print("Average error: {}".format(avg_err))
        print("Average cmat l1: \n{}".format(avg_cmat_l1))
        print("Average cmat l2: \n{}".format(avg_cmat_l2))
        print("l1 scalar avg: {}".format(np.round(np.sum(np.tril(avg_cmat_l1[1:,1:], -1)) / 28, 4)))

#CIFAR-10 - MUX
#Average error: [0.05491592]
#Average cout:
#[[1.         0.48987349 0.41584443 0.06852669]
# [0.48987349 1.         0.07392897 0.41629147]
# [0.41584443 0.07392897 1.         0.48995027]
# [0.06852669 0.41629147 0.48995027 1.        ]]

#CIFAR-10 - MAJ
#Average error: [0.04341882]
#Average cout:
#[[1.         0.62878906 0.55077187 0.25803841]
# [0.62878906 1.         0.25684183 0.55346345]
# [0.55077187 0.25684183 1.         0.62757959]
# [0.25803841 0.55346345 0.62757959 1.        ]]
#0.2986 <-- same for both sub-circuits

#MNIST - MUX
#Average error: [0.03076884]
#Average cout:
#[[1.         0.89105273 0.89386328 0.76687641]
# [0.89105273 1.         0.80579005 0.89435799]
# [0.89386328 0.80579005 1.         0.89122709]
# [0.76687641 0.89435799 0.89122709 1.        ]]

#MNIST - MAJ
#Average error: [0.02699908]
#Average cout:
#[[1.         0.91104547 0.91786564 0.78138266]
# [0.91104547 1.         0.82822179 0.91703435]
# [0.91786564 0.82822179 1.         0.91199656]
# [0.78138266 0.91703435 0.91199656 1.        ]]
#0.8607 <-- same for both sub-circuits

#MUX:
    #First layer:
#[0.6913819  0.66484093 0.63829996 0.61175899 0.58521802 0.55867705 0.53213608 0.50559511 0.47905415 0.45251318 0.42597221 0.39943124 0.37289027 0.3463493  0.31980833 0.29326736 0.26672639 0.24018542 0.21364445 0.18710348 0.16056251 0.13402154 0.10748057 0.0809396  0.05439863]
    #Second layer:
#[0.14122433 0.13585195 0.13047956 0.12510717 0.11973479 0.1143624  0.10899002 0.10361763 0.09824524 0.09287286 0.08750047 0.08212808 0.0767557  0.07138331 0.06601092 0.06063854 0.05526615 0.04989376 0.04452138 0.03914899 0.0337766  0.02840422 0.02303183 0.01765945 0.01228706]
    #Third layer:
#[1.42780590e-01 1.36831399e-01 1.30882207e-01 1.24933016e-01 1.18983825e-01 1.13034634e-01 1.07085442e-01 1.01136251e-01 9.51870599e-02 8.92378686e-02 8.32886774e-02 7.73394861e-02 7.13902949e-02 6.54411036e-02 5.94919124e-02 5.35427212e-02 4.75935299e-02 4.16443387e-02 3.56951474e-02 2.97459562e-02 2.37967650e-02 1.78475737e-02 1.18983825e-02 5.94919124e-03 6.93031216e-13]

#MAJ:
    #First layer:
#[0.56148391 0.53992544 0.51836696 0.49680849 0.47525002 0.45369155 0.43213308 0.41057461 0.38901613 0.36745766 0.34589919 0.32434072 0.30278225 0.28122378 0.2596653  0.23810683 0.21654836 0.19498989 0.17343142 0.15187295 0.13031447 0.108756   0.08719753 0.06563906 0.04408059]
    #Second layer:
#[0.11383345 0.1093313  0.10482914 0.10032699 0.09582484 0.09132268 0.08682053 0.08231838 0.07781622 0.07331407 0.06881192 0.06430976 0.05980761 0.05530545 0.0508033  0.04630115 0.04179899 0.03729684 0.03279469 0.02829253 0.02379038 0.01928823 0.01478607 0.01028392 0.00578176]
    #Third layer:
#[1.44627072e-01 1.38600944e-01 1.32574816e-01 1.26548688e-01 1.20522560e-01 1.14496432e-01 1.08470304e-01 1.02444176e-01 9.64180477e-02 9.03919198e-02 8.43657918e-02 7.83396638e-02 7.23135358e-02 6.62874078e-02 6.02612798e-02 5.42351519e-02 4.82090239e-02 4.21828959e-02 3.61567679e-02 3.01306399e-02 2.41045119e-02 1.80783840e-02 1.20522560e-02 6.02612798e-03 6.93498599e-13]

def testing_for_paper():
    #test_ptv_gen()
    test_rc()