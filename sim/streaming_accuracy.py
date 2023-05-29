import numpy as np
from sim.bitstreams import *

def SA(bs):
    unp = bs_unpack(bs)
    N = unp.size
    px = bs_mean(bs)

    #compute phi
    rsum = 0
    et_err = 0
    for i in range(N):
        rsum += unp[i]
        et_err += np.abs(rsum/(i+1)-px)
    phi = 1 - et_err/N
    #print("phi: ", phi)

    #compute phi_best
    rsum = 0
    et_err = 0
    for i in range(N):
        et_err0 = np.abs(rsum/(i+1)-px)
        et_err1 = np.abs((rsum+1)/(i+1)-px)
        if et_err1 < et_err0:
            rsum += 1
            et_err += et_err1
        else:
            et_err += et_err0
    phi_best = 1 - et_err/N
    #print("phi_best: ", phi_best)

    #compute phi_worst
    N1s = np.sum(unp)
    N0s = N-N1s
    et_err = 0
    rsum = 0
    if px < 0.5:
        for i in range(N1s):
            rsum += 1
            et_err += np.abs(rsum/(i+1)-px)
        for i in range(N0s):
            et_err += np.abs(rsum/(i+1+N1s)-px)
    else:
        et_err += N0s * px
        for i in range(N1s):
            rsum += 1
            et_err += np.abs(rsum/(i+1+N0s)-px)
    phi_worst = 1 - et_err/N
    #print("phi_worst: ", phi_worst)
    
    if phi_best == phi_worst:
        return 1
    return (phi - phi_worst) / (phi_best - phi_worst)