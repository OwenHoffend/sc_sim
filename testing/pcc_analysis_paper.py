import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mc
from sim.PTM import *
from sim.SEC import Ks_to_Mf, opt_K_C_paper, opt_K_C_paper_iterative, opt_K_zero, scc_inv, scc, zscc
from sim.circuits import mux_1
from functools import reduce

def pcc_n(*x, n=2):
    assert len(x) == 2*n
    v = x[0:n]
    c = x[n:2*n]
    z = False
    for i in range(n):
        z = mux_1(c[i], z, v[i])
    return z

def shared_pcc_nk(*x, n=2, k=2):
    assert len(x) == (k+1)*n
    c = x[k*n:(k+1)*n]
    pccs = []
    for i in range(k):
        #pcc_x = c + x[(i+1)*n:(i+2)*n]
        pcc_x = x[i*n:(i+1)*n] + c
        pccs.append(pcc_n(*pcc_x, n=n))
    return tuple(pccs)

def heatmap_err(ax, xs, ys, zs, zmin=None, zmax=None):
    cmap = "Reds" #ERRs
    y, x = np.meshgrid(xs, ys)

    ax.pcolormesh(x, y, zs, cmap=cmap, vmin=zmin, vmax=zmax)
    ax.axis([x.min(), x.max(), y.max(), y.min()])
    ax.yaxis.set_ticks([0.5, 1])
    ax.xaxis.set_ticks([0.5, 1])

    # Remove ticks by setting their length to 0
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)

def heatmap_corr(ax, xs, ys, zs, zmin=-1, zmax=1):
    cmap = 'RdBu_r' #SCC
    y, x = np.meshgrid(xs, ys)
    #z_min, z_max = zs.min(), zs.max()

    ax.pcolormesh(x, y, zs, cmap=cmap, vmin=zmin, vmax=zmax)
    ax.axis([x.min(), x.max(), y.max(), y.min()])
    ax.yaxis.set_ticks([0.5, 1])
    ax.xaxis.set_ticks([0.5, 1])

    # Remove ticks by setting their length to 0
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)

def xy_pcc_heatmaps():
    precisions = [2, 3, 4, 5, 6]
    B2 = B_mat(2)
    and_ptm = get_func_mat(np.bitwise_and, 2, 1)
    denoms = [2**n for n in precisions]
    all_corrs = []
    all_errs = []
    for idx, n in enumerate(precisions):
        print("Precision: ", n)
        denom = denoms[idx]
        rs = np.array([1.0/denom for _ in range(denom)]) #ptv for lfsr (constant) inputs to pcc
        pcc = lambda *x: shared_pcc_nk(*x, n=n)
        pcc_ptm = get_func_mat(pcc, 3*n, 2)
        A = pcc_ptm @ B2
        K1 = A[:, 0].reshape(2**n, 2**(2*n)).T
        K2 = A[:, 1].reshape(2**n, 2**(2*n)).T
        #K1_opt0, K2_opt0 = opt_K_zero(K1, K2)
        K1_opt0, K2_opt0 = opt_K_C_paper_iterative(K1, K2, 0, scc)
        #K1_opt0_z, K2_opt0_z = opt_K_zero(K1, K2, use_zscc=True)
        K1_opt0_z, K2_opt0_z = opt_K_C_paper_iterative(K1, K2, 0, zscc)
        pcc_ptm_opt0 = Ks_to_Mf([K1_opt0, K2_opt0])
        pcc_ptm_opt0_z = Ks_to_Mf([K1_opt0_z, K2_opt0_z])
        corrs = np.empty((denom, denom))
        corrs0 = np.empty((denom, denom))
        corrs0_z = np.empty((denom, denom))
        errs = np.empty((denom, denom))
        errs0 = np.empty((denom, denom))
        errs0_z = np.empty((denom, denom))
        for x in range(denom):
            print("x: ", x)
            xs = np.array([1.0 if i == x else 0.0 for i in range(denom)])
            for y in range(denom):
                ys = np.array([1.0 if i == y else 0.0 for i in range(denom)])
                vin = reduce(np.kron, [rs, ys, xs])
                vout = pcc_ptm.T @ vin
                vout_opt0 = pcc_ptm_opt0.T @ vin
                vout_opt0_z = pcc_ptm_opt0_z.T @ vin

                #value verification
                pout = B2.T @ vout
                pout_opt0 = B2.T @ vout_opt0
                pout_opt0_z = B2.T @ vout_opt0_z
                assert np.all(pout == pout_opt0)
                assert np.all(pout == pout_opt0_z)
                assert np.isclose(pout[0], x/denom)
                assert np.isclose(pout[1], y/denom)

                #correlation
                corrs[x, y] = get_corr_mat_paper(vout)[0, 1]
                corrs0[x, y] = get_corr_mat_paper(vout_opt0)[0, 1]
                corrs0_z[x, y] = get_corr_mat_paper(vout_opt0_z)[0, 1]

                #correlation error
                correct = (x*y)/(denom**2)
                errs[x, y] = np.abs(correct - (and_ptm.T @ vout)[1])
                errs0[x, y] = np.abs(correct - (and_ptm.T @ vout_opt0)[1])
                errs0_z[x, y] = np.abs(correct - (and_ptm.T @ vout_opt0_z)[1])
        all_corrs.append([corrs, corrs0])
        all_errs.append([errs, errs0, errs0_z])

    ##ERR HEATMAPS
    fig, axes = plt.subplots(3, 5, sharey=True, sharex=True)

    #get min/max
    zmin = np.inf
    zmax = -np.inf
    for i in range(5):
        for j in range(3):
            zmin = np.minimum(zmin, np.min(all_errs[i][j]))
            zmax = np.maximum(zmax, np.max(all_errs[i][j]))

    #gen heatmaps
    for i in range(5):
        xs = np.array(range(denoms[i]))/denoms[i]
        ys = np.array(range(denoms[i]))/denoms[i]
        for j in range(3):
            heatmap_err(axes[j, i], xs, ys, all_errs[i][j], zmin=zmin, zmax=zmax)
    
    fig.subplots_adjust(bottom=0.15)
    fig.subplots_adjust(hspace=0.04, wspace=0.04)
    norm = mc.Normalize(zmin, zmax)
    cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.025])
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap="Reds"), 
        cax=cbar_ax, # Pass the new axis
        orientation = "horizontal"
    )
    plt.show()

    #Line plot of mean errs
    mean_errs = []
    mean_errs_opt0 = []
    mean_errs_opt0_z = []
    for i in range(5):
        mean_errs.append(np.mean(all_errs[i][0]))
        mean_errs_opt0.append(np.mean(all_errs[i][1]))
        mean_errs_opt0_z.append(np.mean(all_errs[i][2]))

    plt.plot(precisions, mean_errs, marker="o", label="MUX")
    plt.plot(precisions, mean_errs_opt0, marker="o", label="OPT SCC")
    plt.plot(precisions, mean_errs_opt0_z, marker="o", label="OPT ZSCC")
    plt.xticks(precisions)
    plt.grid(True)
    plt.legend()
    plt.xlabel("PCC Precision (bits)")
    plt.ylabel("Average Error (AND gate multiplier)")
    plt.show()

    #SCC Heatmaps
    fig, axes = plt.subplots(2, 5, sharey=True, sharex=True)

    for i in range(5):
        xs = np.array(range(denoms[i]))/denoms[i]
        ys = np.array(range(denoms[i]))/denoms[i]
        for j in range(2):
            heatmap_corr(axes[j, i], xs, ys, all_corrs[i][j])
            print("i: {}, j: {}, mean: {}, std: {}".format(i, j, np.mean(np.abs(all_corrs[i][j])), np.std(np.abs(all_corrs[i][j]))))

    fig.subplots_adjust(bottom=0.15)
    fig.subplots_adjust(hspace=0.04, wspace=0.04)
    norm = mc.Normalize(-1, 1)
    cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.025])
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap="RdBu_r"), 
        cax=cbar_ax, # Pass the new axis
        orientation = "horizontal"
    )
    plt.show()

    plt.bar(precisions, [np.mean(np.abs(all_corrs[i][0])) for i in precisions], width=0.4, label="MUX", yerr=[np.std(np.abs(all_corrs[i][0])) for i in precisions])
    plt.bar(precisions, [np.mean(np.abs(all_corrs[i][1])) for i in precisions], width=0.4, label="OPT", yerr=[np.std(np.abs(all_corrs[i][1])) for i in precisions])
    plt.show()

def pcc_analysis_paper():
    pass