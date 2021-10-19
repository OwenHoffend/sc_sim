import numpy as np
import matplotlib.pyplot as plt
import symbolic_manip as sm
import sim.PTM as pm
import sim.circuits as cir

def symbolic_cov_mat_bernoulli(Mf, num_inputs, num_ouputs, corr=0, custom=None):
    Bk = pm.B_mat(num_ouputs)
    vin_mat = sm.vin_covmat_bernoulli(num_inputs, corr=corr, custom=custom)
    A_mat = sm.scalar_mat_poly((Mf @ Bk) * 1)
    return A_mat.T @ vin_mat @ A_mat

def maj_mux_var_out():
    mux_mf = pm.get_func_mat(cir.mux_2, 6, 2)

    vin_sels = sm.vin_poly_bernoulli_mc1(2, names=["s0", "s1"])
    vin_data = sm.vin_poly_bernoulli_mc1(4, names=["p1", "p2", "p3", "p4"])
    vin = np.kron(vin_sels, vin_data)

    mux_poly = symbolic_cov_mat_bernoulli(mux_mf, 6, 2, corr=1, custom=vin)
    sm.mat_sub_scalar(mux_poly, "s0", 0.5)
    sm.mat_sub_scalar(mux_poly, "s1", 0.5)
    print(sm.mat_to_latex(mux_poly))

    maj_mf = pm.get_func_mat(cir.maj_2, 6, 2)
    maj_poly = symbolic_cov_mat_bernoulli(maj_mf, 6, 2, corr=1, custom=vin)
    sm.mat_sub_scalar(maj_poly, "s0", 0.5)
    sm.mat_sub_scalar(maj_poly, "s1", 0.5)
    print(sm.mat_to_latex(maj_poly))

    cov_diff = maj_poly[1,0] - mux_poly[1,0]
    print(cov_diff.get_latex())

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

def symbolic_manip_main():
    """Old stuff that was in the main function of variance_analysis.py - not sorted"""
    #var = symbolic_cov_mat_bernoulli(pm.get_func_mat(cir.and_4_to_2, 4, 2), 4, 2, corr=1)
    #print(sm.mat_to_latex(var))
    
    vin_top = sm.vin_poly_bernoulli_mc1(2, names=['p0', 'p2'])
    print(sm.mat_to_latex(np.expand_dims(vin_top, axis=1)))
    vin_bot = sm.vin_poly_bernoulli_mc1(2, ordering=[1, 0], names=['p1', 'p3'])
    print(sm.mat_to_latex(np.expand_dims(vin_bot, axis=1)))
    vin = np.kron(vin_top, vin_bot)
    print(sm.mat_to_latex(np.expand_dims(vin, axis=1)))

    and_cov = symbolic_cov_mat_bernoulli(pm.get_func_mat(cir.and_4_to_2, 4, 2), 4, 2, custom=vin)
    print(sm.mat_to_latex(and_cov))
    #plot_mux_maj()
    #print(sm.mat_sub_scalar(and_cov, 'p0', 0.5))

    #print(sm.mat_to_latex(sm.scalar_mat_poly(pm.get_func_mat(cir.even_odd_sorter_4, 4, 4) * 1)))

    #sorter = symbolic_cov_mat_bernoulli(pm.get_func_mat(cir.even_odd_sorter_4, 4, 4), 4, 4, corr=1)
    #print(sm.mat_to_latex(sorter))

    #maj_mux_var_out()

    #Mux 4 to 1:
    #maj_4_to_1_var = symbolic_cov_mat_bernoulli(pm.get_func_mat(cir.maj_4_to_1, 6, 1), 6, 1)
    #reduced = sm.mat_sub_scalar(maj_4_to_1_var, 'p0', 0.5)
    #reduced = sm.mat_sub_scalar(reduced, 'p1', 0.5)
    #print(sm.mat_to_latex(reduced))