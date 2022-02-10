import numpy as np
from sim.bitstreams import *
from sim.SEC import *

def test_get_SEC_class():
    mux_SEC = get_SEC_class(cir.mux_1, 1, 2, 1, np.array([0.5,]))
    maj_SEC = get_SEC_class(cir.maj_1, 1, 2, 1, np.array([0.5,]))
    print("MUX SEC: \n", mux_SEC)
    print("MAJ SEC: \n", maj_SEC)
    print(mux_SEC == maj_SEC)

def test_get_SECs():
    get_SECs(cir.xor_4_to_2, 1, 3, 2, np.array([0.5, ]))
    #get_SECs(cir.and_3_to_2, 2, 1, 2, np.array([0.3, 0.7]))

def test_SEC():
    test_get_SECs()