import numpy as np
import subprocess
from sim.PTM import *

def PTM_to_espresso_input(ptm, fn, inames=None, onames=None):
    n, k = np.log2(ptm.shape).astype(np.uint16)
    Bk = B_mat(k)
    A = ptm @ Bk
    A_to_espresso_input(A, fn, inames=inames, onames=onames)

def A_to_espresso_input(A, fn, inames=None, onames=None):
    n = np.log2(A.shape[0]).astype(np.uint16)
    k = A.shape[1]
    Bn = B_mat(n)
    with open(fn, 'w') as f:
        f.write(".i {}\n".format(n))
        f.write(".o {}\n".format(k))
        if inames is not None:
            f.write(".ilb " + "".join([name for name in inames]) + "\n")
        if onames is not None:
            f.write(".ilb " + "".join([name for name in onames]) + "\n")
        for i in range(2 ** n):
            instr = "".join([str(1*x) for x in Bn[i,:]])
            outstr = "".join([str(1*x) for x in A[i, :]])
            f.write(instr + " " + outstr + "\n")
        f.write(".e")

def espresso_get_cost(fn):
    cost = 0
    with open(fn, 'r') as infile:
        for line in infile:
            if not line.startswith('.'):
                line_ = line.split(' ')[0]
                cost += line_.count('1')
                cost += line_.count('0')
            elif line.startswith('.p'):
                cost += int(line.split(' ')[1])
    print(cost)
    return cost

def espresso_opt(cir_spec, fn, inames=None, onames=None, is_A=False):
    if is_A:
        A_to_espresso_input(cir_spec, fn, inames=inames, onames=onames)
    else:
        PTM_to_espresso_input(cir_spec, fn, inames=inames, onames=onames)
    p = subprocess.Popen("./Espresso {}".format(fn), stdout=subprocess.PIPE)
    #p = subprocess.Popen("./espresso {}".format(fn), stdout=subprocess.PIPE) #new espresso executable

    (output, err) = p.communicate()
    p_status = p.wait()
    return output

def espresso_get_SOP_area(cir_spec, fn, inames=None, onames=None, do_print=False, is_A=False):
    output = espresso_opt(cir_spec, fn, inames, onames, is_A)
    cost = 0
    for line in output.decode('utf-8').split('\n'):
        if do_print:
            print(line)
        if not line.startswith('.'):
            line_ = line.split(' ')[0]
            cost += line_.count('1')
            cost += line_.count('0')
        elif line.startswith('.p'):
            cost += int(line.split(' ')[1])
    if cost == 0 and is_A: #Check for errors due to IPC
        assert np.all(cir_spec == 0)
    return cost

def espresso_get_SOP_area_from_file(ifn, do_print=True):
    cost = 0
    with open(ifn) as infile:
        for line in infile.readlines():
            if do_print:
                print(line)
            if not line.startswith('.'):
                line_ = line.split(' ')[0]
                cost += line_.count('1')
                cost += line_.count('0')
            elif line.startswith('.p'):
                cost += int(line.split(' ')[1])
        return cost

def espresso_get_opt_file(cir_spec, ifn, ofn, inames=None, onames=None, do_print=False, is_A=False):
    output = espresso_opt(cir_spec, ifn, inames, onames, is_A)

    with open(ofn, 'w') as outfile: 
        for line in output.decode('utf-8').split('\n'):
            if do_print:
                print(line)
            outfile.write(line)