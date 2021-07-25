import numpy as np
from scipy.stats import hypergeom
import matplotlib.pyplot as plt
from matplotlib import cm
import sim.bitstreams as bs
import sim.circuits as cir

def hypersum(K, N, n):
    result = 0
    rv = hypergeom(N, K, n)
    for k in range(K+1):
        print("First: {}".format(((K - k) / (N - n))))
        print("Second: {}".format(rv.pmf(k)))
        result += ((K - k) / (N - n)) * rv.pmf(k)
    return result

def plot_variance(func, N, samps):
    xy_vals = np.array([a / N for a in range(0, N, 4)])
    s = xy_vals.size
    var_vals_hyper = np.zeros((s, s))
    var_vals_lfsr = np.zeros((s, s))
    var_vals_uniform = np.zeros((s, s))
    rng1 = bs.SC_RNG()
    rng2 = bs.SC_RNG()
    for idx, x in enumerate(xy_vals):
        for idy, y in enumerate(xy_vals):
            #Equations to reproduce fig (a) on Tim's paper (ideal):
            var_vals_hyper[idx][idy] = np.sqrt((x * (1-x) * y * (1-y)) / (N - 1))
            var_vals_uniform[idx][idy] = np.sqrt(((x * y) * (1 - (x * y))) / N) #<-- Theoretical AND gate variance, bernoulli

            for _ in range(samps):
                bsx_lfsr = rng1.bs_lfsr(N, x, keep_rng=False, save_init=True)
                bsy_lfsr = rng2.bs_lfsr(N, y, keep_rng=False, save_init=True)

                #bit-wise variance
                var_vals_lfsr[idx][idy] += np.sqrt(bs.bs_var(func(bsx_lfsr, bsy_lfsr), bs_len=N))

                #stream-wise variance
                #TODO

                #bsx_uniform = rng.bs_uniform(N, x, keep_rng=False)
                #bsy_uniform = rng.bs_uniform(N, y, keep_rng=False)
                #var_vals_uniform[idx][idy] += bs.bs_var(func(bsx_uniform, bsy_uniform), bs_len=N)

            var_vals_lfsr[idx][idy] /= samps
            #var_vals_uniform[idx][idy] /= samps
    X, Y = np.meshgrid(xy_vals, xy_vals)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, var_vals_lfsr, color='r')
    surf2 = ax.plot_surface(X, Y, var_vals_uniform, color='y')
    surf3 = ax.plot_surface(X, Y, var_vals_hyper, color='b')

    ax.set_xlabel('X Input Value')
    ax.set_ylabel('Y Input Value')
    ax.set_zlabel('Std Deviation')
    maxval = np.maximum(np.max(var_vals_lfsr), np.max(var_vals_uniform))
    ax.set_zlim(0, maxval)
    #fig.colorbar(surf , shrink=0.5, aspect=5)
    plt.title("Std Deviation vs. Input X and Y Values")
    plt.show()

if __name__ == "__main__":
    plot_variance(np.bitwise_and, 128)