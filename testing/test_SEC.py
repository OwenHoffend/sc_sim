import numpy as np
from sim.bitstreams import *
from sim.SEC import *
import matplotlib.pyplot as plt

def test_get_SEC_class():
    mux_SEC = get_SEC_class(cir.mux_1, 1, 2, 1, np.array([0.5,]))
    maj_SEC = get_SEC_class(cir.maj_1, 1, 2, 1, np.array([0.5,]))
    print("MUX SEC: \n", mux_SEC)
    print("MAJ SEC: \n", maj_SEC)
    print(mux_SEC == maj_SEC)

def test_get_SECs():
    get_SECs(cir.mux_2_joint, 1, 4, 2, np.array([0.5, ]))

def test_SEC_corr_score():
    SEs = get_SECs(cir.mux_2_joint, 1, 4, 2, np.array([0.5, ]))
    max_ = 0
    min_ = np.inf
    for SE in SEs:
        score = SEC_corr_score(SE, 0, 1)
        if score == 9:
            print('hi')
        if score > max_:
            max_ = score
            print("new max: ", max_)
        elif score < min_:
            min_ = score
            print("new min: ", min_)

def test_max_corr_2outputs_restricted():
    consts = [0.8125, 0.3125]
    max_corr_2outputs_restricted(PARALLEL_CONST_MUL(consts, 4))

def test_parallel_MAC_SEC_plots():
    #For a given precision, find all of the possible parallel MAC circuits and compute the number of overlaps

    #Helper function - exhaustive input combinations for constant inputs
    def consts_iter(prec): #really shouldn't run this with prec > 5
        lim = 2 ** prec
        inc = 2 ** (-prec)
        v1, v2, v3, v4 = inc, inc, inc, inc
        for _ in range(lim-1):
            for _ in range(lim-1):
                for _ in range(lim-1):
                    for _ in range(lim-1):
                        yield [v1, v2, v3, v4]
                        v4 += inc
                    v4 = inc
                    v3 += inc
                v3 = inc
                v2 += inc
            v2 = inc
            v1 += inc

    #Main loop
    max_precision = 3
    mux_res = []
    maj_res = []
    opt_res = []
    for consts in consts_iter(max_precision):
        print(consts)
        #Get circuits
        mac_mux = PARALLEL_MAC_2(consts, max_precision, bipolar=True)
        K1_mux, K2_mux = get_K_2outputs(mac_mux)
        K1_opt, K2_opt = opt_K_max(K1_mux), opt_K_max(K2_mux) 
        mac_maj = PARALLEL_MAC_2(consts, max_precision, bipolar=True, maj=True)
        K1_maj, K2_maj = get_K_2outputs(mac_maj)

        actual_precision = mac_mux.actual_precision

        #Can disable these tests later if they work
        #Test that the circuit produces the correct result
        """
        x_vals = np.random.uniform(size=4)
        correct = np.array([
            0.5*(x_vals[0]*consts[0] + x_vals[1]*consts[1]),
            0.5*(x_vals[2]*consts[2] + x_vals[3]*consts[3])
        ])
        px = np.concatenate((x_vals, np.array([0.5 for _ in range(actual_precision + 1)])))
        ptv = get_vin_mc0(px)
        test_mux = B_mat(2).T @ mac_mux.ptm().T @ ptv
        test_maj = B_mat(2).T @ mac_maj.ptm().T @ ptv
        assert np.allclose(correct, test_mux)
        assert np.allclose(correct, test_maj)

        #Test that the optimal PTM matches
        K1_opt_maj, K2_opt_maj = opt_K_max(K1_maj), opt_K_max(K2_maj) 
        assert np.allclose(K1_opt, K1_opt_maj)
        assert np.allclose(K2_opt, K2_opt_maj)
        """

        #Get corr scores
        mux_res.append(SEC_corr_score_K(K1_mux, K2_mux))
        maj_res.append(SEC_corr_score_K(K1_maj, K2_maj))
        opt_res.append(SEC_corr_score_K(K1_opt, K2_opt))

        print(K1_mux)
        print(K1_opt)
        print("hi")

    print(np.mean(np.array(mux_res)))
    print(np.mean(np.array(maj_res)))
    print(np.mean(np.array(opt_res)))
    print(np.std(np.array(mux_res)))
    print(np.std(np.array(maj_res)))
    print(np.std(np.array(opt_res)))
    plt.plot(mux_res, label='mux')
    plt.plot(maj_res, label='maj')
    plt.plot(opt_res, label='opt')
    plt.legend()
    plt.show()
    