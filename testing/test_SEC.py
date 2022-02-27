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
    get_SECs(cir.mux_2_joint, 1, 4, 2, np.array([0.5, ]))

def test_SEC_corr_score():
    SEs = get_SECs(cir.mux_2_joint, 1, 4, 2, np.array([0.5, ]))
    max_ = 0
    min_ = np.inf
    for SE in SEs:
        score = SEC_corr_score(SE, 0, 1)
        if score == 9:
            print('hi')
        if score > max_:
            max_ = score
            print("new max: ", max_)
        elif score < min_:
            min_ = score
            print("new min: ", min_)

def test_max_corr_2inputs_restricted():
    max_corr_2inputs_restricted(cir.mux_2_joint_const, 1, 4)
    #max_corr_2inputs_restricted(cir.and_3_to_2_const, 1, 2)