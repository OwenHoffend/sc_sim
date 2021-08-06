import numpy as np
from scipy.stats import hypergeom
import matplotlib.pyplot as plt
from matplotlib import cm
import sim.circuits as cir
import sim.bitstreams as bs
import sim.corr_preservation as cp
from testing.test_main import profile

def hypersum(K, N, n):
    result = 0
    rv = hypergeom(N, K, n)
    for k in range(K+1):
        print("First: {}".format(((K - k) / (N - n))))
        print("Second: {}".format(rv.pmf(k)))
        result += ((K - k) / (N - n)) * rv.pmf(k)
    return result

@profile
def lfsr_cov_mat_compare(func, num_inputs, num_outputs, p_arr, N, samps, eq_predicted_cov=None):
    """Compare the actual output covariance matrix for a set of lfsr-generated input bitstreams
        to the theoretical one predicted based on the computed input covariance matrix"""
    Mf = cp.get_func_mat(func, num_inputs, num_outputs)
    rng = bs.SC_RNG()
    bs_mat = np.zeros((num_inputs, N), dtype=np.uint8)
    vins = np.zeros((samps, 2 ** num_inputs))
    Pzs = np.zeros((samps, num_outputs))

    #Generate a set of vins
    for i in range(samps):
        for j in range(num_inputs):
            bs_mat[j, :] = rng.bs_lfsr(N, p_arr[j], keep_rng=False, pack=False)
        vins[i, :] = cp.get_actual_vin(bs_mat)
        bs_mat_out = np.vstack(func(*np.split(bs_mat, bs_mat.shape[0], axis=0))[::-1])
        Pzs[i, :] = bs.bs_mean(bs_mat_out, bs_len=N)

        #Assert that we get the correct p_arr back out (just a test)
        assert np.all(np.isclose(p_arr, cp.B_mat(num_inputs).T @ vins[i, :]))

    in_cov = np.cov(vins.T)
    Bk = cp.B_mat(num_outputs)
    A_mat = Bk.T @ Mf.T
    ideal_out_cov = A_mat @ in_cov @ A_mat.T #This is the equation we are testing
    out_cov = np.cov(Pzs.T)

    print(Mf)
    print("'A' Matrix: {}".format(A_mat))
    np.set_printoptions(linewidth=np.inf)
    print("In cov: \n {}".format(in_cov))

    print("Ideal out cov: \n {}".format(ideal_out_cov))
    print("Actual out cov: \n {}".format(out_cov))
    print(np.all(np.isclose(ideal_out_cov, out_cov)))
    if eq_predicted_cov is not None:
        print("Eq-predicted cov: \n {}".format(eq_predicted_cov(*p_arr, N) ** 2))

@profile
def plot_variance(func, ideal_sc_func, uniform_func, hyper_func, N, samps):
    xy_vals = np.array([a / N for a in range(N+1)])
    s = xy_vals.size
    var_vals_hyper = np.zeros((s, s))
    var_vals_lfsr = np.zeros((s, s))
    var_vals_uniform = np.zeros((s, s))
    rng = bs.SC_RNG()
    for idx, x in enumerate(xy_vals):
        print("{} out of {}".format(idx, s))
        for idy, y in enumerate(xy_vals):
            var_vals_uniform[idx][idy] = uniform_func(x, y, N) #Ideal bernoulli variance
            var_vals_hyper[idx][idy] = hyper_func(x, y, N) #Ideal hypergeometric variance

            lfsr_array = np.zeros(samps)
            #uniform_array = np.zeros(samps)
            for i in range(samps):
                bsx_lfsr = rng.bs_lfsr(N, x, keep_rng=False)
                bsy_lfsr = rng.bs_lfsr(N, y, keep_rng=False)
                #bsx_uniform = rng.bs_uniform(N, x, keep_rng=False)
                #bsy_uniform = rng.bs_uniform(N, y, keep_rng=False)

                #bit-wise output variance
                #var_vals_lfsr[idx][idy] += np.sqrt(bs.bs_var(func(bsx_lfsr, bsy_lfsr), bs_len=N))
                #var_vals_uniform[idx][idy] += bs.bs_var(func(bsx_uniform, bsy_uniform), bs_len=N)

                #stream-wise output variance
                lfsr_array[i] = bs.bs_mean(func(bsx_lfsr, bsy_lfsr), bs_len=N)
                #uniform_array[i] = bs.bs_mean(func(bsx_uniform, bsy_uniform), bs_len=N)

            #Bit-wise output variance
            #var_vals_lfsr[idx][idy] /= samps
            #var_vals_uniform[idx][idy] /= samps

            #Stream-wise output variance
            var_vals_lfsr[idx][idy] = np.sqrt(np.sum((lfsr_array - ideal_sc_func(x, y)) ** 2) / samps)
            #var_vals_uniform[idx][idy] = np.sqrt(np.sum((uniform_array - ideal_sc_func(x, y)) ** 2) / samps)

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

def test_hyper_vin(N, samps):
    rng = bs.SC_RNG()
    err = 0.0
    for i in range(samps):
        print("{} out of {}".format(i, samps))
        px = np.random.randint(0, N+1) / N
        py = np.random.randint(0, N+1) / N
        pz = np.random.randint(0, N+1) / N

        #Get vin from lfsr-generated input bitstreams
        bsx = rng.bs_lfsr(N, px, keep_rng=False, pack=False) * 1
        bsy = rng.bs_lfsr(N, py, keep_rng=False, pack=False) * 1
        bsz = rng.bs_lfsr(N, pz, keep_rng=False, pack=False) * 1
        actual_vin = cp.get_actual_vin(np.array([bsx, bsy, bsz]))
        
        #Get ideal vin
        ideal_vin = cp.get_vin_mc0(np.array([px, py, pz]))

        #Compare the two - probably different
        err += np.abs(actual_vin - ideal_vin)
    return np.mean(err / samps)

def ideal_sc_or(x, y):
    return x + y - x*y

def ideal_sc_and(x, y):
    return x*y

#Ideal uniform functions
def uniform_and(x, y, N):
    return np.sqrt(((x * y) * (1 - (x * y))) / N)

#Ideal hypergeometric functions - From Ma paper
def hyper_or(x, y, N):
    return np.sqrt(((1-x) - (1-x) ** 2) * ((1-y) - (1-y) ** 2) / (N-1))

def hyper_and(x, y, N):
    return np.sqrt((x * (1-x) * y * (1-y)) / (N - 1))

if __name__ == "__main__":
    #plot_variance(np.bitwise_and, ideal_sc_and, uniform_and, hyper_and, 15, 500)
    #print(test_hyper_vin(2047, 1000))
    mux = lambda x, y, z: np.bitwise_or(np.bitwise_and(x, z), np.bitwise_and(y, np.bitwise_not(z)))
    func = lambda x, y: (np.bitwise_and(x, y), np.bitwise_or(x, y))
    lfsr_cov_mat_compare(func, 2, 2, np.array([5/15, 10/15]), 15, 100)

 #[[ 4.63114538e-03 -2.98995073e-03 -2.38325339e-03  7.42058739e-04 -1.80727077e-03  1.66076119e-04 -4.40621218e-04  2.08181587e-03]
 #[-2.98995073e-03  5.33694685e-03  8.05413697e-04 -3.15240982e-03  2.02769788e-04 -2.54976591e-03  1.98176724e-03  3.65228878e-04]
 #[-2.38325339e-03  8.05413697e-04  4.30910073e-03 -2.73126104e-03 -4.69850807e-04  2.04769050e-03 -1.45599653e-03 -1.21843162e-04]
 #[ 7.42058739e-04 -3.15240982e-03 -2.73126104e-03  5.14161212e-03  2.07435179e-03  3.35999289e-04 -8.51494927e-05 -2.32520159e-03]
 #[-1.80727077e-03  2.02769788e-04 -4.69850807e-04  2.07435179e-03  3.24468340e-03 -1.64018242e-03 -9.67561823e-04 -6.36939161e-04]
 #[ 1.66076119e-04 -2.54976591e-03  2.04769050e-03  3.35999289e-04 -1.64018242e-03  4.02387221e-03 -5.73584203e-04 -1.81010559e-03]
 #[-4.40621218e-04  1.98176724e-03 -1.45599653e-03 -8.51494927e-05 -9.67561823e-04 -5.73584203e-04  2.86417957e-03 -1.32303355e-03]
 #[ 2.08181587e-03  3.65228878e-04 -1.21843162e-04 -2.32520159e-03 -6.36939161e-04 -1.81010559e-03 -1.32303355e-03  3.77007830e-03]]