import matplotlib.pyplot as plt
import numpy as np
import cv.img_io as img_io
import sim.bitstreams as bitstreams

def plot_conversion_mse(img_path):
    """Plot the error due to converting to SC bitstreams and back
       plots against bitstream length, on log scale"""
    img = img_io.load_img(img_path, gs=True)
    def mse_trace(rng, func, correlated, color, label): #Helper function - produces a single MSE trace
        xvals = np.linspace(16, 512, num=20, dtype=np.uint16)
        yvals = np.zeros(20)
        for idx, bs_len in enumerate(xvals):
            for _ in range(5):
                img_bs = img_io.img_to_bs(img, func, bs_len=bs_len, correlated=correlated)
                img_recon = img_io.bs_to_img(img_bs, bitstreams.bs_mean)
                yvals[idx] += img_io.img_mse(img_recon, img)
                rng.reset()
            yvals[idx] /= 5

        print([xvals, yvals])
        plt.plot(xvals, yvals, color=color, label=label)

    rng = bitstreams.SC_RNG()
    mse_trace(rng, rng.bs_uniform, False, color='blue', label="Uniform rand, SCC= 0")
    mse_trace(rng, rng.bs_uniform, True, color='red', label="Uniform, SCC= +1")
    mse_trace(rng, rng.bs_lfsr, True, color='green', label="LFSR, SCC= +1")

    plt.legend()
    plt.xlabel("SC bitstream length")
    plt.ylabel("MSE")
    plt.title("Image to SC conversion MSE test")
    plt.show()

if __name__ == "__main__":
    plot_conversion_mse("./img/lena256.png")