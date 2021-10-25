import numpy as np
import torch
from sim.PTM import *

#----------------------------------------------------------------------------------#
# PRESERVATION ERROR MEASUREMENTS
#----------------------------------------------------------------------------------#

def corr_err(Px, Mf1, Mf2, c1, c2, N=128):
    """Get the correlation error produced by a two-layer circuit"""
    def get_vin(Pin, c):
        if c == -1:
            return get_vin_mcn1(Pin)
        elif c == 1:
            return get_vin_mc1(Pin)
        elif c == 0:
            return get_vin_mc0(Pin)
        else:
            return get_vin_iterative(Pin, c, N) #If c is not a scalar constant, interpret as a covariance matrix
    n1, k1 = np.log2(Mf1.shape).astype(np.uint16)
    k_check, k2 = np.log2(Mf2.shape).astype(np.uint16)
    assert k_check == k1
    Vc1 = get_vin(Px, c1)
    Bk1 = B_mat(k1)
    Bk2 = B_mat(k2)
    Vz1_actual = Mf1.T @ Vc1
    Vz1_ideal = get_vin(Bk1.T @ Vz1_actual, c2)
    return np.abs(Bk2.T @ Mf2.T @ (Vz1_actual - Vz1_ideal))

def corr_err_cuda(Px, Mf1, Mf2, c1, c2, N=128):
    """Get the correlation error produced by a two-layer circuit"""
    def get_vin_cuda(Pin, c):
        if c == -1:
            return get_vin_mcn1_cuda(Pin)
        elif c == 1:
            return get_vin_mc1_cuda(Pin)
        elif c == 0:
            return get_vin_mc0_cuda(Pin)
        else:
            raise NotImplementedError
    n1, k1 = np.log2(Mf1.shape).astype(np.uint16)
    k_check, k2 = np.log2(Mf2.shape).astype(np.uint16)
    assert k_check == k1
    Vc1 = get_vin_cuda(Px, c1) #Vc1 old: 2 ** n1 x 1, new: 2 ** n1 x b
    Bk1 = B_mat(k1, cuda=True)
    Bk2 = B_mat(k2, cuda=True)
    Vz1_actual = Mf1.T @ Vc1 #2 ** k1 x b
    Vz1_ideal = get_vin_cuda((Bk1.T @ Vz1_actual).T, c2)
    Vz1_ideal_test = torch.zeros_like(Vz1_ideal)
    for i in range(Px.shape[0]):
        Vz1_ideal_test[:, i] = get_vin_mc0_cuda_test((Bk1.T @ Vz1_actual).T[i, :])
    assert torch.all(Vz1_ideal == Vz1_ideal_test)
    return torch.abs(Bk2.T @ Mf2.T @ (Vz1_actual - Vz1_ideal))

def err_sweep(N, Mf, vin_type, err_type='e', Mf2=None):
    n, k = np.log2(Mf.shape).astype(np.uint16)
    if Mf2 is not None:
        _, k = np.log2(Mf2.shape).astype(np.uint16) #<-- wtf why not k2?
    p_arr = np.zeros(n)
    err = np.zeros((k, k))
    max_err = np.zeros((k, k))
    for i in range((N - 1) ** n):
        for j in range(n):
            p_arr[j] = (np.floor(i / ((N-1) ** j)) % (N - 1) + 1) / N
        if err_type == 'e':
            if vin_type == 1:
                vin = get_vin_mc1(p_arr)
            elif vin_type == -1:
                if np.sum(p_arr) > 1:
                    continue
                vin = get_vin_mcn1(p_arr)
            elif vin_type == 0:
                vin = get_vin_mc0(p_arr)
            else:
                raise ValueError("Not valid vin type")
            new_err = e_err(vin, Mf, N, vin_type)
        elif err_type == 'c':
            new_err = corr_err(p_arr, Mf, Mf2, vin_type, 0, N=N)
        else:
            raise ValueError("Not a valid error type")
        #with open('err_sweep_test.txt', 'a') as outfile:
        #    outfile.write(str(new_err) + '\n')
        err += new_err
        max_err = np.maximum(max_err, new_err)
    return max_err, err / (N - 1) ** n

def corr_uniform_rand_1layer(m, Mf1, vin_func_c1, c2, N):
    """Compute the correlation error for given 1-layer circuit by generating a set of 
        input vectors drawn from a uniform random distribution"""
    n, k1 = np.log2(Mf1.shape).astype(np.uint16)
    Px = np.zeros(n)

    errs = np.zeros_like(c2)
    for i in range(m):
        Px = np.array([0.1, 0.1, 0.2, 0.1])
        Vin = vin_func_c1(Px)
        cout = get_output_corr_mat(Vin, Mf1, N, use_zscc=True)[1]
        errs += cout
    return errs / m

def err_uniform_rand_2layer(m, Mf1, Mf2, vin_func_c1, vin_func_c2):
    """Compute the correlation error for given 2-layer circuit by generating a set of 
        input vectors drawn from a uniform random distribution"""
    
    n, k1 = np.log2(Mf1.shape).astype(np.uint16)
    _, k2 = np.log2(Mf2.shape).astype(np.uint16)
    Bk1 = B_mat(k1)
    Bk2 = B_mat(k2)
    Px = np.zeros(n)
    errs = np.zeros(k2)
    for i in range(m):
        Px = np.random.rand(n)
        Px[0] = 0.5
        Px[1] = 0.5
        vn = vin_func_c1(Px)
        vz1 = Mf1.T @ vn
        Pz1 = Bk1.T @ vz1
        vz1_ideal = vin_func_c2(Pz1)
        Pz2 = Bk2.T @ Mf2.T @ vz1
        Pz2_ideal = Bk2.T @ Mf2.T @ vz1_ideal
        errs += np.abs(Pz2_ideal - Pz2)
    return errs / m

def err_sweep_cuda(N, Mf, vin_type, err_type='e', Mf2=None):
    """CUDA-accelerated version of err_sweep"""
    n, k = np.log2(Mf.shape).astype(np.uint16)
    if Mf2 is not None:
        _, k = np.log2(Mf2.shape).astype(np.uint16)
    batch_size = N-1
    nprobs = (N - 1) ** n
    p_arr = torch.cuda.FloatTensor(nprobs, n).fill_(0)
    err = torch.cuda.FloatTensor(k, k).fill_(0)
    max_err = torch.cuda.FloatTensor(k, k).fill_(0)
    for i in range(nprobs):
        for j in range(n):
            p_arr[i, j] = (np.floor(i / ((N-1) ** j)) % (N - 1) + 1) / N

    for b in range((nprobs / batch_size).astype(np.uint32)):
        p_range = p_arr[b:(b+batch_size), :]
        if err_type == 'e':
            if vin_type == 1:
                vin = get_vin_mc1_cuda(p_range)
            elif vin_type == -1:
                if np.sum(p_range) > 1:
                    continue
                vin = get_vin_mcn1_cuda(p_range)
            elif vin_type == 0:
                vin = get_vin_mc0_cuda(p_range)
            else:
                raise ValueError("Not valid vin type")
            new_err = e_err_cuda(vin, Mf, N, vin_type)
        elif err_type == 'c':
            new_err = corr_err_cuda(p_range, Mf, Mf2, vin_type, 0, N=N)
        else:
            raise ValueError("Not a valid error type")
        #with open('err_sweep_test_2.txt', 'a') as outfile:
        #    for i in range(batch_size):
        #        outfile.write(str(new_err[i]) + '\n')
        err += torch.sum(new_err)
        max_err = torch.maximum(max_err, torch.max(new_err))
    return max_err, err / nprobs

def e_err(Vin, Mf, N, vin_type):
    _, k = np.log2(Mf.shape).astype(np.uint16)
    Bk = B_mat(k)

    Vout = Mf.T @ Vin
    Pout = Bk.T @ Vout

    err = np.zeros((k, k))
    for i in range(k):
        for j in range(i):
            s = np.sum(np.multiply(np.bitwise_and(Bk[:, i], Bk[:, j]), Vout))
            if vin_type == 1:
                correct = np.minimum(Pout[i], Pout[j])
            elif vin_type == -1:
                correct = np.maximum(Pout[i] + Pout[j] - 1, 0)
            else:
                correct = np.floor(Pout[i] * Pout[j] * N + 0.5)/N
            err[i, j] = 2 * np.abs(s - correct) / (s + correct + 1e-15)
    return err

def e_err_cuda(Vin, Mf, N, vin_type):
    raise NotImplementedError

def circular_shift_compare(shifts, k, comp, *x):
    """Simulation of the circular shift alg"""
    vals = []
    x_np = np.array(x)
    for i in range(k):
        x_np_int = np.sum([v * (2 ** ind) for ind, v in enumerate(x_np[::-1])])
        vals.append(comp[i] > x_np_int)
        x_np = np.roll(x_np, -shifts)
    return (*vals,)

def circular_shift_corr_sweep(n, k, shifts):
    Vin = np.array([0, ] + [1 / (2 ** n - 1) for _ in range(2 ** n - 1)])
    scc_mat = np.zeros((k, k))
    for i in range(2 ** n):
        for j in range(2 ** n):
            func = lambda *x: circular_shift_compare(shifts, k, [i, j], *x)
            Mf = get_func_mat(func, n, k)
            scc_mat += np.abs(get_output_corr_mat(Vin, Mf, 2 ** n, use_zscc=False)[1])
        #print(i)
    return scc_mat / ((2 ** n) ** 2)

if __name__ == "__main__":
    pass
    #m = 20000
    #Mf1 = get_func_mat(cir.mux_2, 6, 2)
    #Mf1 = get_func_mat(cir.maj_2, 6, 2)
    #Mf2 = get_func_mat(np.bitwise_and, 2, 1)
    
    #vin_func_c1 = lambda Px: get_vin_mc0(Px) #0
    #vin_func_c1 = lambda Px: np.kron(get_vin_mc1(Px[0:2]), get_vin_mc1(Px[2:6])) #1

    #vin_func_c2 = lambda Pz1: get_vin_mc0(Pz1) #0
    #vin_func_c2 = lambda Pz1: get_vin_mc1(Pz1) #1
    
    #err_mux = err_uniform_rand(m, Mf1, Mf2, vin_func_c1, vin_func_c2)
    #print(np.round(err_mux, 4))

    #1 to 1 preservation of MUX - 0.10538512
    #0 to 0 preservation of MUX - 1.95010512e-17 (preserves)
    #1 to 0 preservation of MUX - 0.02833997
    #0 to 1 preservation of MUX - 0.11504263

    #1 to 1 preservation of MAJ - 0.0838604 (better)
    #0 to 0 preservation of MAJ - 1.95995404e-17
    #1 to 0 preservation of MAJ - 0.03326625 (worse)
    #0 to 1 preservation of MAJ - 0.10243964

    #Reruns with 0.5:
    #1 to 1 preservation of MUX - 0.0490
    #0 to 0 preservation of MUX - 0.0000
    #1 to 0 preservation of MUX - 0.0922
    #0 to 1 preservation of MUX - 0.1326

    #1 to 1 preservation of MAJ - 0.0171 (better)
    #0 to 0 preservation of MAJ - 0.0000
    #1 to 0 preservation of MAJ - 0.1170 (worse)
    #0 to 1 preservation of MAJ - 0.1339