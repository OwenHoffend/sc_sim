import numpy as np
from scipy.stats import hypergeom
import matplotlib.pyplot as plt
from matplotlib import cm
import sim.circuits as cir
import sim.bitstreams as bs
import sim.corr_preservation as cp
import symbolic_manip as sm
from testing.test_main import profile

def hypersum(K, N, n):
    result = 0
    rv = hypergeom(N, K, n)
    for k in range(K+1):
        print("First: {}".format(((K - k) / (N - n))))
        print("Second: {}".format(rv.pmf(k)))
        result += ((K - k) / (N - n)) * rv.pmf(k)
    return result

def symbolic_cov_mat_bernoulli(Mf, num_inputs, num_ouputs, corr=0):
    Bk = cp.B_mat(num_ouputs)
    vin_mat = sm.vin_covmat_bernoulli(num_inputs, corr=corr)
    A_mat = sm.scalar_mat_poly((Mf @ Bk) * 1)
    return A_mat.T @ vin_mat @ A_mat

def maj_mux_var_out():
    mux_mf = cp.get_func_mat(cir.mux_maj, 5, 2)
    mux_poly = symbolic_cov_mat_bernoulli(mux_mf, 5, 2)
    sm.mat_sub_scalar(mux_poly, "p0", 0.5)
    print(sm.mat_to_latex(mux_poly))

    maj_mf = cp.get_func_mat(cir.maj_2, 5, 2)
    maj_poly = symbolic_cov_mat_bernoulli(maj_mf, 5, 2)
    sm.mat_sub_scalar(maj_poly, "p0", 0.5)
    print(sm.mat_to_latex(maj_poly))

    cov_diff = maj_poly[1,0] - mux_poly[1,0]
    print(cov_diff.get_latex())

def var2(a, b, N):
    return (a * b * (1-a) * (1-b))/(N-1)

@profile
def lfsr_cov_mat_compare(func, num_inputs, num_outputs, p_arr, N, samps, eq_predicted_cov=None):
    """Compare the actual output covariance matrix for a set of lfsr-generated input bitstreams
        to the theoretical one predicted based on the computed input covariance matrix"""
    Mf = cp.get_func_mat(func, num_inputs, num_outputs)
    rng = bs.SC_RNG()
    bs_mat = np.zeros((num_inputs, N), dtype=np.uint8)
    Pxs = np.zeros((samps, num_inputs))
    vins = np.zeros((samps, 2 ** num_inputs))
    Pzs = np.zeros((samps, num_outputs))

    #Generate a set of vins
    for i in range(samps):
        for j in range(num_inputs):
            #bs_mat[j, :] = rng.bs_lfsr(N, p_arr[j], keep_rng=False, pack=False)
            bs_mat[j, :] = rng.bs_uniform(N, p_arr[j], keep_rng=False, pack=False)
        vins[i, :] = cp.get_actual_vin(bs_mat)
        bs_mat_out = np.vstack(func(*np.split(bs_mat, bs_mat.shape[0], axis=0))[::-1])
        Pxs[i, :] = bs.bs_mean(bs_mat, bs_len=N)
        Pzs[i, :] = bs.bs_mean(bs_mat_out, bs_len=N)

        #Assert that we get the correct p_arr back out (just a test)
        #assert np.all(np.isclose(p_arr, cp.B_mat(num_inputs).T @ vins[i, :]))

    px_cov = np.cov(Pxs.T)
    vin_cov = np.cov(vins.T)
    Bk = cp.B_mat(num_outputs)
    A_mat = Bk.T @ Mf.T
    ideal_out_cov = A_mat @ vin_cov @ A_mat.T #This is the equation we are testing
    out_cov = np.cov(Pzs.T)

    print("'A' Matrix: {}".format(A_mat))
    np.set_printoptions(linewidth=np.inf)
    print("Px cov: \n {}".format(px_cov)) #--> About 0 (all entries) for independent bitstreams, as expected
    print("Vin cov: \n {}".format(vin_cov))

    print("Ideal out cov: \n {}".format(ideal_out_cov))
    print("Actual out cov: \n {}".format(out_cov))
    print(np.all(np.isclose(ideal_out_cov, out_cov)))
    if eq_predicted_cov is not None:
        print("Eq-predicted cov: \n {}".format(eq_predicted_cov(*p_arr, N) ** 2))

def mux_cov(p1, p2, p3, p4):
    return 0.25*p1*p3-0.25*p2*p3-0.25*p1*p4+0.25*p2*p4

def maj_cov(p1, p2, p3, p4):
    return p1*p2*p3*p4-0.5*p2*p3*p4-0.5*p1*p3*p4-0.5*p1*p2*p4+0.25*p2*p4+0.25*p1*p4-0.5*p1*p2*p3+0.25*p2*p3+0.25*p1*p3

def plot_mux_maj():
    xy_vals = np.linspace(0, 1, 100)
    mux = np.zeros((100, 100))
    maj = np.zeros((100, 100))
    for idx, x in enumerate(xy_vals):
        for idy, y in enumerate(xy_vals):
            mux[idx][idy] = mux_cov(0.5, x, 0.5, y)
            maj[idx][idy] = maj_cov(0.5, x, 0.5, y)

    X, Y = np.meshgrid(xy_vals, xy_vals)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, mux, label="mux", color='r')
    surf3 = ax.plot_surface(X, Y, maj, label="maj", color='b')

    ax.set_xlabel('p2 Input Value')
    ax.set_ylabel('p4 Input Value')
    ax.set_zlabel('Covariance')
    maxval = np.maximum(np.max(mux), np.max(maj))
    minval = np.minimum(np.min(mux), np.min(maj))
    ax.set_zlim(minval, maxval)
    #fig.colorbar(surf , shrink=0.5, aspect=5)
    plt.title("Covariance vs. Input p2 and p4 Values")
    plt.show()

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
    #var = symbolic_cov_mat_bernoulli(cp.get_func_mat(cir.and_4_to_2, 4, 2), 4, 2, corr=1)
    #print(sm.mat_to_latex(var))

    and_cov = symbolic_cov_mat_bernoulli(cp.get_func_mat(cir.and_3_to_2, 3, 2), 3, 2, corr=1)
    print(sm.mat_to_latex(and_cov))
    #plot_mux_maj()
    #print(sm.mat_sub_scalar(and_cov, 'p0', 0.5))

    #maj_mux_var_out()

    #Mux 4 to 1:
    #maj_4_to_1_var = symbolic_cov_mat_bernoulli(cp.get_func_mat(cir.maj_4_to_1, 6, 1), 6, 1)
    #reduced = sm.mat_sub_scalar(maj_4_to_1_var, 'p0', 0.5)
    #reduced = sm.mat_sub_scalar(reduced, 'p1', 0.5)
    #print(sm.mat_to_latex(reduced))

    #plot_variance(np.bitwise_and, ideal_sc_and, uniform_and, hyper_and, 15, 500)
    #print(test_hyper_vin(2047, 1000))
    #mux = lambda x, y, z: np.bitwise_or(np.bitwise_and(x, z), np.bitwise_and(y, np.bitwise_not(z)))
    #func = lambda x, y: (np.bitwise_and(x, y), np.bitwise_or(x, y))
    #lfsr_cov_mat_compare(mux, 3, 1, np.array([127/255, 63/255, 101/255]), 255, 1000)

    #test
    #N = 255
    #x = 127/255
    #y = 63/255
    #z = 101/255

    #print(-(1/N) * ((1-x) * y * z) * ((1-x) * (1-y) * z))

    #var_xy = var2(x, y, N)
    #var_yz = var2(y, z, N)
    #var_xz = var2(x, z, N)
    #var_xyz = (1/(N-1)) * x * y * z * (1 - x) * (1 - y) * (1 - z) #--> Naiive extension of the principle that worked for 2 variables
    #var_xyz = (1/(N-1)) * x * y * (1 - (x*y)) * z * (1-z) #--> Solution if setting n=NE[xy]
    #var_xyz = (1/(N-1)) * z * y * (1 - (z*y)) * x * (1-x) #--> Solution if setting n=NE[zy] (most optimistic)

    #var_xyz = (1/(N-1)) * (1-z) * (1-y) * (1 - ((1-z)*(1-y))) * x * (1-x)
    #var_xyz = (1/(N-1)) * (1-x) * (1-z) * (1 - ((1-z)*(1-x))) * y * (1-y)

    #var_xyz = (1/(N-1)) * (z ** 2) * x * y * (1-x) * (1-y) #--> Application of the equation for var(X1*X2*...*Xn) (asymmetric result)

    #def vt(x, y, z):
    #    return (1/(N-1)) * (z ** 2) * x * y * (1-x) * (1-y)
    #var_xyz = max(vt(x, y, z), vt(y, x, z), vt(z, y, x))  #--> Averaging of all 3 asymmetrical results

    #print(var_xyz)

#20k
 #[[ 1.76718292e-05 -1.36172425e-05 -1.24318284e-05  8.37724162e-06 -8.89096819e-06  4.83638146e-06  3.65096732e-06  4.03619412e-07]
 #[-1.36172425e-05  1.37067732e-05  8.41409637e-06 -8.50362703e-06  4.86349423e-06 -4.95302489e-06  3.39651892e-07 -2.50121233e-07]
 #[-1.24318284e-05  8.41409637e-06  1.24927417e-05 -8.47500968e-06  3.71572058e-06  3.02011410e-07 -3.77663389e-06 -2.41098099e-07]
 #[ 8.37724162e-06 -8.50362703e-06 -8.47500968e-06  8.60139508e-06  3.11753381e-07 -1.85367977e-07 -2.13985324e-07  8.75999202e-08]
 #[-8.89096819e-06  4.86349423e-06  3.71572058e-06  3.11753381e-07  9.00159964e-06 -4.97412568e-06 -3.82635203e-06 -2.01121936e-07]
 #[ 4.83638146e-06 -4.95302489e-06  3.02011410e-07 -1.85367977e-07 -4.97412568e-06  5.09076911e-06 -1.64267191e-07  4.76237576e-08]
 #[ 3.65096732e-06  3.39651892e-07 -3.77663389e-06 -2.13985324e-07 -3.82635203e-06 -1.64267191e-07  3.95201859e-06  3.86006228e-08]
 #[ 4.03619412e-07 -2.50121233e-07 -2.41098099e-07  8.75999202e-08 -2.01121936e-07  4.76237576e-08  3.86006228e-08  1.14897556e-07]]