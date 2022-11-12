import numpy as np
import matplotlib.pyplot as plt
import copy
from sim.PTM import *
from sim.circuits_obj import *
from scipy.special import comb
from sim.espresso import espresso_get_SOP_area

def get_SEC_class(func, nc, nv, k, consts):
    #Expects the circuit function's inputs to be ordered with all constants first
    assert len(consts) == nc
    Mf = get_func_mat(func, nc + nv, k)
    for idx, const in enumerate(consts):
        Mf = reduce_func_mat(Mf, idx, const)
    return Mf @ B_mat(k) 

def append_to_all(all_poss, n):
    for a in all_poss:
        a.append(n)
    return all_poss

def add_poss(all_poss, new_poss):
    combined_poss = []
    for nwp in new_poss:
        combined_poss += append_to_all(copy.deepcopy(all_poss), nwp)
    return combined_poss

def get_SECs(func, nc, nv, k, consts):
    #Get all of the stochastically equivalent circuits given a SEC description for a circuit with 1 constant input
    #and n variable inputs.
    SEC_class = get_SEC_class(func, nc, nv, k, consts)
    vc = get_vin_mc0(consts)

    poss_width = k * (2 ** nc)
    poss_bvals = B_mat(poss_width)
    #poss = np.zeros((2 ** poss_width, k))
    poss_dict = {}
    for i in range(2 ** poss_width):
        brow = poss_bvals[i, :].reshape(2 ** nc, k)
        row_key = str(tuple(vc.T @ brow))
        if row_key in poss_dict:
            poss_dict[row_key].append(brow)
        else:
            poss_dict[row_key] = [brow,]
    all_poss = [[]]
    for i, a in enumerate(SEC_class):
        new_poss = poss_dict[str(tuple(a))]
        all_poss = add_poss(all_poss, new_poss)
    return all_poss

def SEC_corr_score(a, o1_idx, o2_idx):
    s = 0
    for a_ in a:
        s += np.sum(a_[:, o1_idx] & a_[:, o2_idx])
    return s

def SEC_num_ovs(K1, K2):
    #Compute the correlation score between K1 and K2 by counting the number of overlaps
    return np.sum(np.bitwise_and(K1, K2))

def SEC_uniform_SCC_score(K1, K2, ptv_func, m=1000):
    ptm = Ks_to_Mf([K1, K2])
    n, _ = np.log2(ptm.shape)
    avg_Cmat = np.zeros((2, 2))
    for _ in range(m):
        Px = np.random.uniform(size=int(n))
        vin = ptv_func(Px)
        vout = ptm.T @ vin
        avg_Cmat += get_corr_mat_paper(vout)
    avg_Cmat /= m
    #print(avg_Cmat)
    return avg_Cmat 

def SEC_uniform_err(ptm, ptv_func, correct_func, m=1000):
    n, _ = np.log2(ptm.shape)
    avg_err = 0
    for _ in range(m):
        Px = np.random.uniform(size=int(n))
        vin = ptv_func(Px)
        vout = ptm.T @ vin
        avg_err += np.abs(vout[0] - correct_func(Px))
    avg_err /= m
    return avg_err

def opt_K_max(K):
    _, tlen = K.shape
    K_sum = np.sum(K, axis=1)
    return np.stack([np.pad(np.ones(t, dtype=np.bool_), (0, tlen-t), 'constant') for t in K_sum])

def opt_K_min(K):
    K_max = opt_K_max(K)
    return np.flip(K_max, axis=1)

def scc(px, py, pxy, N):
    delta = pxy - px*py
    delta_max = min(px, py) - px*py
    delta_min = max(px + py - 1, 0) - px*py

    if delta > 0:
        if delta_max == 0:
            return -1
        return delta/delta_max
    else:
        if delta_min == 0:
            return 1
        return -delta/delta_min

def zscc(px, py, pxy, N):
    delta = pxy - px*py
    delta0 = np.floor(px*py*N + 0.5)/N - px*py
    delta_max = min(px, py) - px*py
    delta_min = max(px + py - 1, 0) - px*py

    if delta > delta0:
        if delta_max-delta0 == 0:
            return 0
        return (delta - delta0)/(delta_max-delta0)
    else:
        if delta0-delta_min == 0:
            return 0
        return (delta - delta0)/(delta0-delta_min)

def scc_inv(px, py, C, N):
    delta_max = min(px, py) - px*py
    delta_min = max(px + py - 1, 0) - px*py
    if delta_max == 0 or delta_min == 0:
        return min(px, py)
    if C > 0:
        return C * delta_max + px * py
    return -C * delta_min + px * py

def zscc_inv(px, py, Z, N):
    delta0 = np.floor(px*py*N + 0.5)/N - px*py
    delta_max = min(px, py) - px*py
    delta_min = max(px + py - 1, 0) - px*py

    if Z > 0:
        return Z * (delta_max - delta0) + delta0 - px*py
    return Z * (delta_min - delta0) + delta0 - px*py

def opt_K_zero(K1, K2, roll_together=False, use_zscc=False):
    """The roll_together option may help to mitigate the HUGE increase in area that the 
        actual optimal SCC=0 circuit has (At the cost of some correlation)"""
    K1_opt = opt_K_max(K1)
    
    nv2, nc2 = K1.shape
    w1 = np.sum(K1, axis=1)
    w2 = np.sum(K2, axis=1)
    novs = np.round((w1*w2)/nc2).astype(np.int32)

    if use_zscc:
        corr = lambda px, py, pxy: zscc(px, py, pxy, nc2)
    else:
        corr = lambda px, py, pxy: scc(px, py, pxy)

    K2_0 = np.zeros_like(K2)
    if roll_together:
        K2_opt = opt_K_max(K2)
        nrolls = 0
        for i in range(nv2):
            s1 = np.sum(K1_opt[i, :])
            s2 = np.sum(K2_opt[i, :])
            if s1 >= s2:
                nrolls += s1 - novs[i]
            else:
                nrolls += novs[i] - s2
        K2_0 = np.roll(K2_opt, np.rint(nrolls / nv2).astype(np.int32))
    else:
        K2_opt = opt_K_max(K2)
        K2_opt_final = np.zeros_like(K2_opt)
        for i in range(nv2):
            px = w1[i]/nc2
            py = w2[i]/nc2
            K2_0[i, :] = K2_opt[i, :]
            K2_opt_final[i, :] = K2_opt[i, :]
            min_c = 1
            for _ in range(nc2):
                pxy = np.sum(K1_opt[i, :] & K2_0[i, :])/nc2
                c = corr(px, py, pxy)
                if np.abs(c) < min_c:
                    min_c = np.abs(c)
                    K2_opt_final[i, :] = K2_0[i, :]
                K2_0[i, :] = np.roll(K2_0[i, :], 1)
    return K1_opt, K2_opt_final

def opt_K_C_paper(F1, F2, C, corr_inv):
    nv2, nc2 = F1.shape
    F1_opt = np.zeros_like(F1)
    F2_opt = np.zeros_like(F2)
    for i in range(nv2):
        n1 = np.sum(F1[i, :])
        n2 = np.sum(F2[i, :])
        for j in range(n1):
            F1_opt[i, j] = 1
        pov = corr_inv(n1/nc2, n2/nc2, C, nc2)
        #nov = np.round(pov*nc2).astype(np.int) #Somehow, this is changing it to be like ZSCC, figure it out later
        nov = (pov * nc2).astype(np.int) #Even worse than the cost-function alg
        for j in range(nov):
            F2_opt[i, j] = 1
        for j in range(nc2-n2+nov, nc2):
            F2_opt[i, j] = 1
        assert sum(F2_opt[i, :]) == n2
        assert sum(np.bitwise_and(F1_opt[i, :], F2_opt[i, :])) == nov
    return F1_opt, F2_opt

def opt_K_C_paper_iterative(F1, F2, C, corr=scc):
    nv2, nc2 = F1.shape
    F1_opt = np.zeros_like(F1)
    F2_opt = np.zeros_like(F2)
    for i in range(nv2):
        b = np.zeros_like(F1[i, :])
        n1 = np.sum(F1[i, :])
        n2 = np.sum(F2[i, :])
        for j in range(n1):
            F1_opt[i, j] = 1
        for j in range(n2):
            b[j] = 1
        emin = np.inf
        p1 = n1/nc2
        p2 = n2/nc2
        pov = min(p1, p2)
        j = 0
        while pov >= max(p1 + p2 - 1, 0):
            err = np.abs(corr(p1, p2, pov, nc2) - C)
            if err < emin:
                emin = err
                F2_opt[i, :] = b
            b[j] = 0
            b[nc2 - j - 1] = 1
            pov -= 1.0/nc2
            j += 1
    return F1_opt, F2_opt

def get_all_rolled(K):
    """Rotate through all possible variants of optimal circuit"""
    #opt = opt_K_max(K)
    for i in range(K.shape[1]):
        yield np.roll(K, i, axis=1)

#def get_all_rolled(K):
#    """Rotate through all possible variants of optimal circuit"""
#    opt = opt_K_max(K)
#    for i in range(opt.shape[1]):
#        yield np.roll(opt, i, axis=1)

def get_num_in_SEC(K):
    """Use the equation from the SEC paper to get the number of equivalent circuits for a given K matrix"""
    nv2, nc2 = K.shape
    prod = 1
    for row in range(nv2):
        weight = np.sum(K[row, :])
        prod *= comb(nc2, weight, exact=True)
    return prod

def get_implicants(costs, perms, nv2, nc2):
    def get_col_idx_rec(row_idx, depth_limit):
        ncols = costs[row_idx].size
        if ncols > depth_limit:
            ncols = depth_limit
        for col_idx in range(ncols):
            if row_idx == nv2-1: #Last row
                    yield [col_idx]
            else:
                for sub_list in get_col_idx_rec(row_idx + 1, depth_limit):
                    yield [col_idx] + sub_list
    
    max_depth_limit = max([costs[x].size for x in range(nv2)])
    for depth_limit in range(3, max_depth_limit):
        print("Depth Limit: ", depth_limit)
        for col_inds in get_col_idx_rec(0, depth_limit):
            current_implicants = np.zeros((nv2, nc2, 2), dtype=np.bool_)
            for row, col in enumerate(col_inds):
                current_implicants[row, :, :] = perms[row][col, :, :]
            yield current_implicants

def opt_area_SECO(K1, K2, cache_file='test.json', print_final_espresso=True, simulated_annealing=False, sort=True):
    from sympy.utilities.iterables import multiset_permutations
    import json
    import os
    """Use SECO-style area optimization method to jointly optimize area and correlation"""
    nv2, nc2 = K1.shape
    max_iters = 2000
        
    #For assertions
    sum_k1 = np.sum(K1, axis=1)
    sum_k2 = np.sum(K2, axis=1)

    if cache_file is not None and os.path.exists(cache_file):
        with open(cache_file, 'r') as r:
            cost_cache = json.load(r)
    else:
        cost_cache = {}
    costs = []
    perms = []
    for row in range(nv2):
        s = np.array([K1[row, :], K2[row, :]])
        s_int = np.array([int_array(s_i) for s_i in s.T])
        print(s_int)
        row_costs = []
        row_perms = []
        for p in multiset_permutations(s_int):
            p_bin = np.array([bin_array(p_i, 2) for p_i in p])
            key = str(list(p_bin))
            if key in cost_cache:
                cost = cost_cache[key]
            else:
                cost = espresso_get_SOP_area(p_bin, "opt_area_SECO.in", is_A=True)
                cost_cache[key] = cost
            row_costs.append(cost)
            row_perms.append(np.flip(p_bin, axis=0))
        row_costs = np.array(row_costs)
        row_perms = np.array(row_perms)
        if sort:
            order = row_costs.argsort()
            row_costs = row_costs[order]
            row_perms = row_perms[order, :]
        costs.append(row_costs)
        perms.append(row_perms)
    
    with open(cache_file, "w") as w:
        json.dump(cost_cache, w)

    best_cost = np.inf
    best_inds = None
    best_ptm = None
    if simulated_annealing:
        running_costs = []
        col_inds = [0 for _ in range(nv2)]
        new_col_inds = [0 for _ in range(nv2)]
        current_implicants = np.zeros((nv2, nc2, 2), dtype=np.bool_)
        increase_cnt = 0
        for i in range(max_iters):
            for row, col in enumerate(new_col_inds):
                current_implicants[row, :, :] = perms[row][col, :, :]
            K1_test, K2_test = current_implicants[:,:,0], current_implicants[:,:,1]
            ptm = Ks_to_Mf([K1_test, K2_test])

            #Equivalence assertion
            assert np.all(np.sum(K1_test, axis=1) == sum_k1)
            assert np.all(np.sum(K2_test, axis=1) == sum_k2)
            
            #Correlation assertion
            for row in range(nv2):
                s = np.array([K1[row, :], K2[row, :]])
                s_test = np.array([K1_test[row, :], K2_test[row, :]])
                s_int = np.array([int_array(s_i) for s_i in s.T])
                s_int_test = np.array([int_array(s_i) for s_i in s_test.T])
                unq, cnt = np.unique(s_int, return_counts=True)
                unq_test, cnt_test = np.unique(s_int_test, return_counts=True)
                assert np.all(unq == unq_test)
                assert np.all(cnt == cnt_test)

            T = 1 - i/max_iters
            cost = espresso_get_SOP_area(ptm, "opt_area_SECO.in")
            if cost < best_cost:
                p = 1.0
                best_cost = cost
                print("New best cost: ", best_cost)
                best_ptm = ptm
                np.save("best_ptm", best_ptm)
                best_inds = copy.copy(new_col_inds)
                increase_cnt = 0
            else:
                increase_cnt += 1
                p = np.exp((best_cost-cost) / (T * best_cost))

            running_costs.append(best_cost)
            if p >= np.random.uniform():
                col_inds = new_col_inds

            #Make a random step
            row = np.random.randint(0, nv2)
            row_max = costs[row].size - 1
            new_col_inds = copy.copy(col_inds)
            if row_max != 0:
                if col_inds[row] == 0:
                    new_col_inds[row] += 1
                elif col_inds[row] == row_max: #or col_inds[row] == max_depth-1:
                    new_col_inds[row] -= 1
                else:
                    new_col_inds[row] += np.random.choice([-1, 1])

            if increase_cnt == 25: #np.rint(T*10):
                new_col_inds = copy.copy(best_inds)
                increase_cnt = 0

            if i % 100 == 0:
                print("Iter: ", i)
        plt.plot(running_costs)
        plt.show()
    else:
        i = 0
        for current_implicants in get_implicants(costs, perms, nv2, nc2):
            ##Combine the implicants and get the espresso area of the result
            K1_test, K2_test = current_implicants[:,:,0], current_implicants[:,:,1]
            ptm = Ks_to_Mf([K1_test, K2_test])

            #Equivalence assertion
            assert np.all(np.sum(K1_test, axis=1) == sum_k1)
            assert np.all(np.sum(K2_test, axis=1) == sum_k2)
            
            #Correlation assertion
            for row in range(nv2):
                s = np.array([K1[row, :], K2[row, :]])
                s_test = np.array([K1_test[row, :], K2_test[row, :]])
                s_int = np.array([int_array(s_i) for s_i in s.T])
                s_int_test = np.array([int_array(s_i) for s_i in s_test.T])
                unq, cnt = np.unique(s_int, return_counts=True)
                unq_test, cnt_test = np.unique(s_int_test, return_counts=True)
                assert np.all(unq == unq_test)
                assert np.all(cnt == cnt_test)

            cost = espresso_get_SOP_area(ptm, "opt_area_SECO.in")
            if cost < best_cost:
                best_cost = cost
                print("New best cost: ", best_cost)
                best_ptm = ptm
            
            i+=1
            if i == max_iters:
                break
            if i % 100 == 0:
                print("Iter: ", i)
        else:
            print("Reached the end of optimization") 

    if print_final_espresso:
        espresso_get_SOP_area(best_ptm, "opt_area_SECO.in", do_print=True)
    print("Opt area: ", best_cost)
    return best_ptm

def A_to_Mf(A, n, k):
    Mf = np.zeros((2**n, 2**k), dtype=np.bool_)
    for i in range(2**n):
        x = int_array(A[i, :].reshape(1, k))[0] #<-- why [0]
        Mf[i, x] = True
    return Mf

def Ks_to_Mf(Ks):
    nc, nv = np.log2(Ks[0].shape)
    n = int(nv + nc)
    k = len(Ks) #Ks is a list of K matrices
    A = np.zeros((2**n, k), dtype=np.bool_)
    for i, K in enumerate(Ks):
        A[:, i] = K.T.reshape(2**n)
    return A_to_Mf(A, n, k)

def get_K_2outputs(cir, o1_idx=0, o2_idx=1):
    Mf = cir.ptm()
    A = Mf @ B_mat(cir.k) #2**(nc+nv) x k
    K1 = A[:, o1_idx].reshape(2**cir.nc, 2**cir.nv).T
    K2 = A[:, o2_idx].reshape(2**cir.nc, 2**cir.nv).T
    return K1, K2

def get_K(cir, o1_idx=0):
    Mf = cir.ptm()
    A = Mf @ B_mat(cir.k) #2**(nc+nv) x k
    K1 = A[:, o1_idx].reshape(2**cir.nc, 2**cir.nv).T
    return K1

def max_corr_2outputs_restricted(cir, o1_idx=0, o2_idx=1):
    #Directly compute a circuit design that maximizes the output correlation between two outputs
    #Restrict the constant inputs to be the value "0.5" only
    K1, K2 = get_K_2outputs(cir, o1_idx, o2_idx)
    K1_max = opt_K_max(K1)
    K2_max = opt_K_max(K2)
    K2_min = opt_K_min(K2)
    print("# overlaps before: ", SEC_num_ovs(K1, K2))
    print("min # overlaps after: ", SEC_num_ovs(K1_max, K2_min))
    print("max # overlaps after: ", SEC_num_ovs(K1_max, K2_max))

    #A_max = np.zeros_like(A)
    #A_max[:, o1_idx] = K1_max.T.reshape(2**(cir.nc + cir.nv))
    #A_max[:, o2_idx] = K2_max.T.reshape(2**(cir.nc + cir.nv))
    print(K1 * 1)
    print(K2 * 1)
    print(K1_max * 1)
    print(K2_max * 1)
    print(K1_max * 1)
    print(K2_min * 1)