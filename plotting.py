from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import cv.img_io as img_io
import sim.bitstreams as bs
import sim.scc_sat as ss
import sim.circuits as cir

def plot_img_conv_mse(img_path):
    """Plot the error due to converting to SC bitstreams and back
       plots against bitstream length, on log scale"""
    img = img_io.load_img(img_path, gs=True)
    def mse_trace(rng, func, correlated, color, label): #Helper function - produces a single MSE trace
        xvals = np.linspace(16, 512, num=20, dtype=np.uint16)
        yvals = np.zeros(20)
        for idx, bs_len in enumerate(xvals):
            for _ in range(5):
                img_bs = img_io.img_to_bs(img, func, bs_len=bs_len, correlated=correlated)
                img_recon = img_io.bs_to_img(img_bs, bs.bs_mean)
                yvals[idx] += img_io.img_mse(img_recon, img)
                rng.reset()
            yvals[idx] /= 5

        print([xvals, yvals])
        plt.plot(xvals, yvals, color=color, label=label)

    rng = bs.SC_RNG()
    mse_trace(rng, rng.bs_uniform, False, color='blue', label="Uniform rand, SCC= 0")
    mse_trace(rng, rng.bs_uniform, True, color='red', label="Uniform, SCC= +1")
    mse_trace(rng, rng.bs_lfsr, True, color='green', label="LFSR, SCC= +1")

    plt.legend()
    plt.xlabel("SC bitstream length")
    plt.ylabel("MSE")
    plt.title("Image to SC conversion MSE test")
    plt.show()

def plot_mse_vs_input_2d(sc_func, correct, points=10, samples=100, bp=False):
    """Plot the MSE of a univariate stochastic function wrt. input value"""
    if bp:
        xvals = np.linspace(-1.0, 1.0, num=points)
    else:
        xvals = np.linspace(0.0, 1.0, num=points)
    mse_vals = np.zeros(points)
    mse = 0.0
    for idx, x in enumerate(xvals):
        for _ in range(samples):
            mse_vals[idx] += (sc_func(x) - correct(x)) ** 2 
        mse_vals[idx] /= samples

    plt.plot(xvals, mse_vals)
    plt.xlabel("Input SN Value")
    plt.ylabel("MSE")
    plt.title("MSE vs. Input SN Value")
    plt.show()

def plot_abs_err_vs_input_3d(sc_func, correct, points=20, samples=25, x_bp=False, y_bp=False):
    if x_bp:
        xvals = np.linspace(-1.0, 1.0, num=points)
    else:
        xvals = np.linspace(0.0, 1.0, num=points)
    if y_bp:
        yvals = np.linspace(-1.0, 1.0, num=points)
    else:
        yvals = np.linspace(0.0, 1.0, num=points)
    mse_vals = np.zeros((points, points))
    for idx, x in enumerate(xvals):
        for idy, y in enumerate(yvals):
            for _ in range(samples):
                mse_vals[idy][idx] += abs(sc_func(x, y) - correct(x, y))
            mse_vals[idy][idx] /= samples
    X, Y = np.meshgrid(xvals, yvals)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, mse_vals, cmap=cm.coolwarm)

    ax.set_xlabel('X Input Value')
    ax.set_ylabel('Y Input Value')
    ax.set_zlabel('Mean Abs Error')
    ax.set_zlim(0, 1)
    fig.colorbar(surf , shrink=0.5, aspect=5)
    plt.title("Mean Abs Error vs. Input X and Y Values")
    plt.show()

def plot_output_vs_input_2d(sc_func, correct, points=20, samples=100, bp=False):
    """Plot the output of a univariate stochastic function & its correct function wrt. input value"""
    if bp:
        xvals = np.linspace(-1.0, 1.0, num=points)
    else:
        xvals = np.linspace(0.0, 1.0, num=points)
    zvals = np.zeros(points)
    correct_vals = np.zeros(points)
    for idx, x in enumerate(xvals):
        for _ in range(samples):
            zvals[idx] += sc_func(x)
        yvals[idx] /= samples
        correct_vals[idx] = correct(x)

    plt.plot(xvals, zvals, color='r', label="SN Approximaton")
    plt.plot(xvals, correct_vals, color='b', label="Ideal function")
    plt.legend()
    plt.xlabel("Input Value")
    plt.ylabel("Output Value")
    plt.title("Output vs. Input Value")
    plt.show()

def plot_output_vs_input_3d(sc_func, correct, points=20, samples=20, x_bp=False, y_bp=False):
    if x_bp:
        xvals = np.linspace(-1.0, 1.0, num=points)
    else:
        xvals = np.linspace(0.0, 1.0, num=points)
    if y_bp:
        yvals = np.linspace(-1.0, 1.0, num=points)
    else:
        yvals = np.linspace(0.0, 1.0, num=points)
    zvals = np.zeros((points, points))
    correct_vals = np.zeros((points, points))
    for idx, x in enumerate(xvals):
        for idy, y in enumerate(yvals):
            for _ in range(samples):
                zvals[idy][idx] += sc_func(x, y)
            zvals[idy][idx] /= samples
            correct_vals[idy][idx] = correct(x, y)
    X, Y = np.meshgrid(xvals, yvals)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, correct_vals, color='b')
    ax.plot_wireframe(X, Y, zvals, color='r')

    ax.set_xlabel('X Input Value')
    ax.set_ylabel('Y Input Value')
    ax.set_zlabel('Output Value')
    plt.title("Output vs. Input X and Y Values")
    plt.show()

#plotting functions related to scc satisfaction
def plot_scc_sat_n3_3d(Ns, c, use_zscc=False):
    """Plot scc satisfaction for n=3 for a range of probability values"""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #Get ranges without 0.0 or 1.0 probabilities
    colors = ['blue', 'red', 'green']
    c_mat = bs.mc_mat(c, 3)
    uniques = {}
    for i, N in enumerate(Ns):
        xs, ys, zs = [], [], []
        r = [q / N for q in range(1, N)] # Generate bitstream probabilities without quantization error wrt N
        cnt = 0
        for x in r:
            for y in r:
                for z in r:
                    #if "x{}y{}z{}".format(x,y,z) in uniques.keys():
                    #    continue
                    if ss.corr_sat(N, 3, c_mat, np.array([x, y, z]), print_stat=False, use_zscc=use_zscc):
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)
                        cnt += 1
                    #    uniques["x{}y{}z{}".format(x,y,z)] = ""
        print("cnt for N={} is {}".format(N, cnt))
        ax.scatter(xs, ys, zs, marker='o', color=colors[i], alpha=1, label="N={}".format(N))
    plt.legend(loc="upper center")
    ax.set_xlabel('PX')
    ax.set_ylabel('PY')
    ax.set_zlabel('PZ')
    plt.show()

def plot_num_scc_combs(Ns):
    """Plot the number of ways to make that given scc value"""
    #Choose a good spread of scc values with all factors
    fig, ax = plt.subplots()
    sccs = []
    for denom in range(1, 37):
        for num in range(-denom, denom+1):
            sccs.append(num / denom)
    sccs = np.unique(sccs)
    colors=['black', 'dimgray', 'lightgray']
    hist_sccs = []
    for i, N in enumerate(Ns):
        for c in sccs:
            c_mat = bs.mc_mat(c, 3)
            r = [q / N for q in range(1, N)]
            print(c)
            for x in r:
                for y in r:
                    for z in r:
                        if ss.corr_sat(N, 3, c_mat, np.array([x, y, z]), print_stat=False):
                            hist_sccs.append(c)
        plt.hist(hist_sccs, 50, alpha=0.75, color=colors[i], label="N={}".format(N))
    plt.yscale('log', nonposy='clip')
    plt.legend(loc="upper center")
    plt.xlabel("SCC Value")
    plt.ylabel("# of probability assignments (log scale)")
    plt.grid()
    plt.show()

def scc_in_vs_out(N, p_arr1, p_arr2, sc_func, func_name, trials=10000):
    sccs = []
    for denom in range(1, 101):
        for num in range(-denom, denom+1):
            sccs.append(num / denom)
    sccs = np.unique(sccs)
    out_sccs = []
    outcs = [[] for _ in range(3)]
    for c in sccs:
        c_mat = bs.mc_mat(c, 3)
        g1 = ss.gen_multi_correlated(N, 3, c_mat, p_arr1, pack_output=False, print_stat=False)
        if not g1:
            continue
        bs_arr1 = g1[1]

        g2 = ss.gen_multi_correlated(N, 3, c_mat, p_arr2, pack_output=False, print_stat=False)
        if not g2:
            continue
        bs_arr2 = g2[1]
        print(c)

        out_avg = np.zeros((3, 3))
        out_cnt = 0
        for trial in range(trials):
            np.random.shuffle(bs_arr2.T) #shuffle one
            results = sc_func(bs_arr1, bs_arr2)
            out_corr = bs.get_corr_mat(results, N)
            if np.any(np.isnan(out_corr)):
                continue
            out_avg += out_corr
            out_cnt += 1
        if out_cnt == 0:
            continue
        out_avg /= out_cnt
        outcs[0].append(out_avg[1][0])
        outcs[1].append(out_avg[2][0])
        outcs[2].append(out_avg[2][1])
        out_sccs.append(c)
    fig, ax = plt.subplots()
    plt.plot(out_sccs, outcs[0])
    plt.plot(out_sccs, outcs[1])
    plt.plot(out_sccs, outcs[2])
    plt.xlabel("Input Mutual ZSCC value")
    plt.ylabel("Output ZSCC values for 3 function outputs")
    plt.title("{} ZSCC of outputs vs Mutual ZSCC input. \n N={}, T={}, px1={}, px2={}".format(func_name, N, trials, p_arr1, p_arr2))
    plt.grid()
    plt.show()

def plot_alignments(Nx, Ny, N):
    Mx = np.minimum(Nx, Ny)
    My = np.maximum(Nx, Ny)
    basex = np.array([1 if i < Mx else 0 for i in range(N)])
    basey = np.array([1 if i < My else 0 for i in reversed(range(N))])
    results_scc = []
    results_zce = []
    results_zscc = []
    while True:
        results_scc.append(bs.bs_scc(np.packbits(basex), np.packbits(basey), N))
        results_zce.append(bs.bs_zce(np.packbits(basex), np.packbits(basey), N))
        results_zscc.append(bs.bs_zscc(np.packbits(basex), np.packbits(basey), N))
        if basey[0] != 0:
            break
        basey = np.roll(basey, -1)

    x = list(range(len(results_zscc)))
    plt.plot(x, results_scc, marker='o', label="SCC")
    plt.plot(x, results_zce, marker='o', label="ZCE")
    plt.plot(x, results_zscc, marker='o', label="ZSCC")
    plt.legend()
    plt.xlabel("Alignments")
    plt.ylabel("Correlation Value")
    plt.title("SCC, ZCE, and ZSCC for all Px=21/32 and Py=22/32")
    plt.grid()
    plt.show()

def plot_corr_err(iters, p_arr, func, Nvals):
    rand_arr = np.random.randint(0, high=256, size=(28,28)) #For comparing against random
    res = [0 for _ in range(len(Nvals))]
    res2 = [0 for _ in range(len(Nvals))]
    for idx, N in enumerate(Nvals):
        print("Progress: N={}".format(N))
        for i in range(iters):
            print("Iter: {}".format(i))
            c_mat = func(p_arr, N)
            c_mat2 = func(rand_arr, N)
            print(c_mat)
            print(c_mat2)
            err = bs.mut_corr_err(1, c_mat)
            err2 = bs.mut_corr_err(1, c_mat2)
            res[idx] += err
            res2[idx] += err2
        res[idx] /= iters
        res2[idx] /= iters
    print(res)
    print(res2)
    plt.plot(Nvals, res, label="Real image")
    plt.plot(Nvals, res2, label="Random pixels")
    plt.legend()
    plt.title("Avg mutual corr err vs bitstream length.")
    plt.ylabel("Avg mutual corr err")
    plt.xlabel("Bitstream length")
    plt.show()

if __name__ == "__main__":
    #plot_img_conv_mse("./img/lena256.png")
    #rng = bs.SC_RNG()
    #plot_abs_err_vs_input_3d(
    #    lambda x, y: bs.bs_mean_bp(~(rng.up_to_bp_lfsr(256, x, keep_rng=False) ^ rng.bs_bp_lfsr(256, y, keep_rng=False))),
    #    lambda xc, yc: xc * yc,
    #    y_bp=True
    #)

    #plot_scc_sat_n3_3d([16, 32])
    #plot_num_scc_combs([16, 24, 32])
    #scc_in_vs_out(20, np.array([0.2, 0.2, 0.1]), np.array([0.2, 0.2, 0.2]), np.bitwise_and, "AND Gate") 

    #plot_alignments(21, 22, 32)
    #plot_alignments(1, 4, 6)

    """Correlation preservation analysis"""
    #p_arr = [np.random.random() for _ in range(9)]
    #Nvals = list(range(100, 20000, 100))
    #plot_corr_err(20, p_arr, cir.robert_cross_3x3_to_2x2, Nvals)

    p_arr = img_io.load_img("./img/lena_s2.jpg", gs=True)
    Nvals = [8*x for x in range(2, 28, 2)]
    kernel = np.array([
        [0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1]
    ])
    func = lambda x, y: cir.cnn_kernel_3x3_up(x, kernel, y)
    plot_corr_err(2, p_arr, func, Nvals)