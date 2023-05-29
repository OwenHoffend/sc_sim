from sim.streaming_accuracy import *
from sim.seq_recorr import *
import matplotlib.pyplot as plt
from sim.circuits import robert_cross

def basic_SA_test():
    best = np.array([0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0], dtype=np.bool_)
    worst = np.array([1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0], dtype=np.bool_)
    bad = np.array([0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1], dtype=np.bool_)
    better = np.array([0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1], dtype=np.bool_)
    print(SA(best))
    print(SA(worst))
    print(SA(bad))
    print(SA(better))

def test_circ_SA():
    num_tests = 100
    num_sccs = 100
    N = 256
    sccs = np.linspace(-1, 1, num_sccs)
    avg_SAs = np.zeros(num_sccs)
    for scc_idx, scc in enumerate(sccs):
        print(scc_idx)
        avg_SA = 0
        for _ in range(num_tests):
            rng = SC_RNG()

            #FSMR
            ps = np.random.rand(2)
            x1, x2 = gen_correlated(scc, N, ps[0], ps[1], rng.bs_lfsr)
            x1_r, x2_r = fsm_reco_d(x1, x2, 1, packed=True)
            z = x1_r

            #AND/OR gate
            #ps = np.random.rand(2)
            #z = np.bitwise_and(x1, x2)
            #z = np.bitwise_or(x1, x2)
            #z = np.bitwise_xor(x1, x2)

            #RCED
            #sel = rng.bs_lfsr_p5_consts(N, 1, 9)
            #ps = np.random.rand(4)
            #x11, x22 = gen_correlated(scc, N, ps[0], ps[1], rng.bs_lfsr)
            #x12, x21 = gen_correlated(scc, N, ps[0], ps[1], rng.bs_lfsr)
            ##print(get_corr_mat(np.vstack((sel, x11, x22, x12, x21))))
            #z = robert_cross(sel, x11, x22, x12, x21)[0, :]
            
            avg_SA += SA(z)
        avg_SA /= num_tests
        avg_SAs[scc_idx] = avg_SA
    plt.scatter(sccs, avg_SAs)
    plt.title("Streaming accuracy vs. SCC")
    plt.xlabel("Input SCC")
    plt.ylabel("Streaming Accuracy")
    plt.show()
    pass