import numpy as np
import torch
import bitstreams as bs
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(bool)

B_mat_dict = {}
def B_mat(n, cuda=False):
    """Generate a 2^n by n matrix of all of the possible values of n bits"""
    if n in B_mat_dict.keys():
        return B_mat_dict[n]
    B = np.zeros((2 ** n, n), dtype=bool)
    for i in range(2 ** n):
        #B[i][:] = bin_array(i, n)[::-1] #Might cause issues with endianness... right now it's 1 --> [True, False, False]
        B[i][:] = bin_array(i, n) #The old one is the line above
    if cuda:
        B = torch.tensor(B.astype(np.float32)).to(device)
    B_mat_dict[n] = B
    return B

def get_vin_mc0(Pin):
    """Generates a Vin vector for bitstreams mutually correlated with ZSCC=0"""
    n = Pin.size
    Bn = B_mat(n)
    return np.prod(Bn * Pin + (1 - Bn) * (1 - Pin), 1)

def sparse_thresh(ptv, l_thresh):
    """Induce sparsity into a PTV by applying a threshold"""
    gt = ptv >= l_thresh
    gt_0 = np.sum(gt)
    if gt_0 > 0:
        s_ptv = ptv * gt
        s_ptv_normed = s_ptv / np.linalg.norm(s_ptv, 1)
    else:
        s_ptv_normed = np.eye(1, ptv.shape[0], 0).reshape(ptv.shape[0])
    assert np.isclose(np.sum(s_ptv_normed), 1, 1e-4)

    #Print some stats about the induced sparsity
    percent_sparse = 1 - gt_0 / ptv.shape[0]
    #print("Percent Sparse: {}".format(percent_sparse))
    return percent_sparse, s_ptv_normed

def sample_from_ptv(ptv, N):
    n = int(np.log2(ptv.shape[0]))
    bs_mat = np.zeros((n, N), dtype=np.uint8)
    for i in range(N):
        sel = np.random.choice(ptv.shape[0], p=ptv)
        bs_mat[:, i] = bin_array(sel, n)
    return bs_mat

def main():
    #Induced sparsity percentage plots
    n = 16
    m = 100
    N = 500
    nthreshs = 8
    threshs = [0.1/(2**x) for x in range(0, nthreshs)]
    nvals = list(range(2, n))
    sparses = [np.zeros(len(nvals)) for _ in range(nthreshs)]
    corr_errs = [np.zeros(len(nvals)) for _ in range(nthreshs)]
    for idx, n in enumerate(nvals):
        print(n)
        avgs = [0 for _ in range(nthreshs)]
        errs = [0 for _ in range(nthreshs)]
        for _ in range(m):
            Px = np.random.beta(8, 2, size=n)
            ptv = get_vin_mc0(Px)
            psparse = [sparse_thresh(ptv, thresh) for thresh in threshs]
            for x in range(nthreshs):
                avgs[x] += psparse[x][0]
                bs_mat = sample_from_ptv(psparse[x][1], N)
                corr = bs.get_corr_mat_np(bs_mat)
                err = bs.mut_corr_err(0, corr)
                #print(err)
                errs[x] += err
        for x in range(nthreshs):
            sparses[x][idx] = avgs[x] / m
            corr_errs[x][idx] = errs[x] / m

    plt.title("Amount of induced sparsity, based on threshold")
    plt.ylabel("Percent Sparse")
    plt.xlabel("Number of bitstreams")
    for x, thresh in enumerate(threshs):
        plt.plot(nvals, sparses[x], label="Thresh: {}".format(np.round(thresh, 4)))
    plt.legend()
    plt.show()

    plt.title("Mutual Correlation Error (from independent), based on threshold")
    plt.ylabel("Err")
    plt.xlabel("Number of bitstreams")
    for x, thresh in enumerate(threshs):
        plt.plot(nvals, corr_errs[x], label="Thresh: {}".format(np.round(thresh, 4)))
    plt.legend()
    plt.show()

    #n = 5
    #m = 25
    #N = 10000
    #avg = np.zeros((n, n))
    #for i in range(m):
    #    Px = np.random.rand(n) 
    #    ptv = get_vin_mc0(Px)
    #    p_ns, ptv_s = sparse_thresh(ptv, 0.1, 0.9)
    #    #bs_mat = sample_from_ptv(ptv_s, N)
    #    #corr = bs.get_corr_mat_np(bs_mat)
    #    #print(p_ns)
    #    #print(corr)
    #    #avg += corr
    #avg /= m
    #print(avg)

if __name__ == "__main__":
    main()