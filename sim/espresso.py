import numpy as np
import subprocess
from sim.PTM import *

def PTM_to_espresso_input(ptm, fn, inames=None, onames=None):
    n, k = np.log2(ptm.shape).astype(np.uint16)
    Bk = B_mat(k)
    Bn = B_mat(n)
    A = ptm @ Bk
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

def espresso_get_SOP_area(ptm, fn, inames=None, onames=None):
    PTM_to_espresso_input(ptm, fn, inames=inames, onames=onames)
    p = subprocess.Popen("./Espresso {}".format(fn), stdout=subprocess.PIPE)
    (output, err) = p.communicate()
    p_status = p.wait()
    cost = 0
    for line in output.decode('utf-8').split('\n'):
        print(line)
        if not line.startswith('.'):
            line_ = line.split(' ')[0]
            cost += line_.count('1')
            cost += line_.count('0')
        elif line.startswith('.p'):
            cost += int(line.split(' ')[1])
    return cost