from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
import sim.PTM as pm
import sim.circuits as cir

from symbolic_manip import *

def maj_mux_var_out():
    mux_mf = pm.get_func_mat(cir.mux_2, 6, 2)

    vin_sels = vin_poly_bernoulli_mc1(2, names=["s0", "s1"])
    vin_data = vin_poly_bernoulli_mc1(4, names=["p1", "p2", "p3", "p4"])
    vin = np.kron(vin_sels, vin_data)

    mux_poly = symbolic_cov_mat_bernoulli(mux_mf, 6, 2, corr=1, custom=vin)
    mat_sub_scalar(mux_poly, "s0", 0.5)
    mat_sub_scalar(mux_poly, "s1", 0.5)
    print(mat_to_latex(mux_poly))

    maj_mf = pm.get_func_mat(cir.maj_2, 6, 2)
    maj_poly = symbolic_cov_mat_bernoulli(maj_mf, 6, 2, corr=1, custom=vin)
    mat_sub_scalar(maj_poly, "s0", 0.5)
    mat_sub_scalar(maj_poly, "s1", 0.5)
    print(mat_to_latex(maj_poly))

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

#Polynomial creation test:
def test_poly_creation():
    test_poly = Polynomial(poly_string="0.5(x1^1*x2^2)-0.5(x2^2*x1^1)+1.0(x2^2)")
    print(test_poly.poly)

#Test zero multiplication
def test_zero_mult():
    test_poly = Polynomial(poly_string="0.0(@^1)")
    test_poly2 = Polynomial(poly_string="0.5(x1^1)")
    res = test_poly * test_poly2
    print(res.poly)

#Sum test:
def test_sum():
    test_poly = Polynomial(poly_string="0.5(x1^2*x2^2)-1.0(x1^1*x2^1)+1(x1^2*x2^2)+1(@)")
    test_poly2 = Polynomial(poly_string="0.3(x1^1*x2^1)-5(@)")
    res = test_poly + test_poly2

#Sub test:
def test_sub():
    test_poly = Polynomial(poly_string="0.5(x1^1)+0.3(x1^2*x2^1)")
    test_poly2 = Polynomial(poly_string="0.1(x1^1)-5.0(@^1)")
    print(test_poly - test_poly2)

#Mul test:
def test_mul():
    test_poly = Polynomial(poly_string="0.5(a^1)+2(@^1)")
    test_poly2 = Polynomial(poly_string="0.25(a^2*b^1)+0.5(a^1)+6(@^1)")
    res = test_poly2 ** 2
    print(test_poly.poly)
    print(test_poly2.poly)
    print(res.poly)

#Second mul test:
def test_mul_2():
    test_poly = Polynomial(poly_string="1.0(a^1)+1.0(b^1)")
    test_poly2 = Polynomial(poly_string="1.0(b^1)+1.0(a^1)")
    res = test_poly * test_poly2
    print(res.poly)

#sub_scalar test:
def test_sub_scalar():
    test_poly = Polynomial(poly_string="1.0(x1^1*x2^2)+1.0(x2^1*x1^1)")
    test_poly.sub_scalar("x2", 0.5)
    print(test_poly.poly)

#sub_scalar test2:
def test_sub_scalar_2():
    test_poly = Polynomial(poly_string="0.5(x1^1*x2^2)-0.5(x2^2*x1^1)+1.0(x2^2)")
    test_poly.sub_scalar("x2", 0.5)
    print(test_poly.poly)

#get_latex test:
def test_get_latex():
    test_poly = Polynomial(poly_string="0.5(x1^1*x2^2)-0.5(x2^2*x1^1)+1.0(x2^2)")
    print(test_poly.get_latex())

#Numpy multiplication test
def test_np_mul():
    test_poly = Polynomial(poly_string="1.0(a^1)+1.0(b^1)")
    test_poly2 = Polynomial(poly_string="1.0(b^1)+1.0(a^1)")
    test_vec = np.array([[test_poly, test_poly2]], dtype=object)
    result = test_vec @ test_vec.T
    print(result[0,0].poly)

def test_multiple_vars():
    """test that a(1-b)+ab == a"""
    a = Polynomial(poly_string="1.0(a^1)")
    onem_b = Polynomial(poly_string="1.0(@^1)-1.0(b^1)")
    ab = Polynomial(poly_string="1.0(a^1*b^1)")
    result = a * onem_b + ab
    print(result.poly)

#vin_poly_bernoulli_mc0 test
def test_vin_poly_bernoulli_mc0():
    mat = vin_poly_bernoulli_mc0(4)
    print(mat)

#vin_poly_bernoulli_mc1 test
def test_vin_poly_bernoulli_mc1():
    mat = vin_poly_bernoulli_mc1(2, names=['p1', 'p2'])
    print(mat_to_latex(np.expand_dims(mat, axis=1)))

def test_long_variable_name():
    a = Polynomial(poly_string="1.0(min[p_1, p_2]^1)")
    b = Polynomial(poly_string="1.0(min[p_1, p_2]^1)")
    c = Polynomial(poly_string="1.0(min[p_1, p_3]^1)")
    result = a + b + c
    print(result.poly)

#vin_covmat_bernoulli test
def test_vin_covmat_bernoulli():
    mat0 = vin_covmat_bernoulli(2)
    print(mat0)

    mat1 = vin_covmat_bernoulli(2, corr=1)
    print(mat_to_latex(mat1))

#Test using kronecker product to get vin for a general input correlation structure
def test_kron_prod():
    vin_2_a = vin_poly_bernoulli_mc1(2, names=['p0', 'p1'])
    vin_2_b = vin_poly_bernoulli_mc1(1, names=['s0'])
    vin = np.kron(vin_2_b, vin_2_a)
    print(mat_to_latex(np.expand_dims(vin, axis=1)))

#Test matrix product
def test_mat_prod():
    mat = np.array([
        [1.0, 0],
        [0.5, -0.3]
    ])
    test = scalar_mat_poly(mat) @ vin_poly_bernoulli_mc0(1)
    print(mat_to_latex(np.expand_dims(test, axis=1)))

def xor_4_2_under_c1():
    """Test the output correlation of a pair of xor gates under +1 correlation"""
    Mf = pm.get_func_mat(cir.xor_4_to_2, 4, 2)
    print(Mf)
    xor_cov = symbolic_cov_mat_bernoulli(Mf, 4, 2, corr=1)
    print(mat_to_latex(xor_cov))

def mux_4_2_for_all_vin():
    """Given an input vin, get the output vout for the 4->2 mux circuit in symbolic form, in terms of vin entries"""
    Mf = pm.get_func_mat(cir.mux_2_joint_const, 5, 2)
    Mf_poly = scalar_mat_poly(Mf * 1)
    vin = vin_poly(5)
    print(mat_to_latex(vin))
    print(mat_to_latex(Mf_poly))
    print(mat_to_latex(Mf_poly.T @ vin))

def mux_4_2_corr_perturbation_test():
    Mf = pm.reduce_func_mat(pm.get_func_mat(cir.mux_2_joint_const, 5, 2), 4, 0.5)
    Mf_poly = scalar_mat_poly(Mf * 1)

    v1_x0_x2 = vin_poly(2, vname='a')
    #v1_x0_x1 = vin_poly(2, vname=['a'])
    v0_x1_x3 = vin_poly_bernoulli_mc0(2, vnames=['p1', 'p3'])
    vc_px = vin_poly_bernoulli_mc0(4, vnames=['p0', 'p1', 'p2', 'p3'])

    ab = np.kron(v0_x1_x3, v1_x0_x2)
    #ab = np.kron(v0_x2_x3, v1_x0_x1)
    ab_permuted = pm.PTV_swap_cols(ab, [0, 2, 1, 3])
    print(mat_to_latex(Mf_poly.T @ ab_permuted))

    #d = Polynomial(poly_string="1.0(d^1)")
    #onem_d = Polynomial(poly_string="1.0(@^1)-1.0(d^1)")

    #new_vin = np.multiply(onem_d, vc_px) + np.multiply(d, ab_permuted)
    #new_vin = np.multiply(onem_d, vc_px) + np.multiply(d, ab)
    #print(mat_to_latex(Mf_poly.T @ vc_px))
    #print(mat_to_latex(Mf_poly.T @ ab))
    #print(mat_to_latex(Mf_poly.T @ new_vin))

def mux_4_2_corr_perturbation_test_2():
    Mf = pm.reduce_func_mat(pm.get_func_mat(cir.mux_2_joint_const, 5, 2), 4, 0.5)
    Mf_poly = scalar_mat_poly(Mf * 1)

    v1_x0_x3 = vin_poly(2, vname='a')
    v0_x1_x2 = vin_poly_bernoulli_mc0(2, vnames=['p1', 'p2'])
    ab = np.kron(v1_x0_x3, v0_x1_x2)
    ab_permuted = pm.PTV_swap_cols(ab, [3, 1, 2, 0])
    print(mat_to_latex(Mf_poly.T @ ab_permuted))


def symbolic_manip_main():
    """Old stuff that was in the main function of variance_analysis.py - not sorted"""
    #var = sm.symbolic_cov_mat_bernoulli(pm.get_func_mat(cir.and_4_to_2, 4, 2), 4, 2, corr=1)
    #print(sm.mat_to_latex(var))
    
    #vin_top = vin_poly_bernoulli_mc1(2, names=['p0', 'p2'])
    #print(mat_to_latex(np.expand_dims(vin_top, axis=1)))
    #vin_bot = vin_poly_bernoulli_mc1(2, ordering=[1, 0], names=['p1', 'p3'])
    #print(mat_to_latex(np.expand_dims(vin_bot, axis=1)))
    #vin = np.kron(vin_top, vin_bot)
    #print(mat_to_latex(np.expand_dims(vin, axis=1)))

    #and_cov = symbolic_cov_mat_bernoulli(pm.get_func_mat(cir.and_4_to_2, 4, 2), 4, 2, custom=vin)
    #print(mat_to_latex(and_cov))
    
    #plot_mux_maj()
    #print(mat_sub_scalar(and_cov, 'p0', 0.5))

    #print(mat_to_latex(scalar_mat_poly(pm.get_func_mat(cir.even_odd_sorter_4, 4, 4) * 1)))

    #sorter = symbolic_cov_mat_bernoulli(pm.get_func_mat(cir.even_odd_sorter_4, 4, 4), 4, 4, corr=1)
    #print(sm.mat_to_latex(sorter))

    #maj_mux_var_out()

    #Mux 4 to 1:
    #maj_4_to_1_var = symbolic_cov_mat_bernoulli(pm.get_func_mat(cir.maj_4_to_1, 6, 1), 6, 1)
    #reduced = mat_sub_scalar(maj_4_to_1_var, 'p0', 0.5)
    #reduced = mat_sub_scalar(reduced, 'p1', 0.5)
    #print(mat_to_latex(reduced))