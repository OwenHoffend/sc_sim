from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import cv.img_io as img_io
import sim.bitstreams as bs

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

if __name__ == "__main__":
    #plot_img_conv_mse("./img/lena256.png")
    rng = bs.SC_RNG()
    plot_abs_err_vs_input_3d(
        lambda x, y: bs.bs_mean_bp(~(rng.up_to_bp_lfsr(256, x, keep_rng=False) ^ rng.bs_bp_lfsr(256, y, keep_rng=False))),
        lambda xc, yc: xc * yc,
        y_bp=True
    )

    #Commit test on CAEN #2