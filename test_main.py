import cProfile
from pstats import Stats

#Test script imports
from testing.corr_preservation_tests import *
from testing.self_learn_tests import self_learn_tests
from testing.symbolic_manip_tests import *
from testing.variance_analysis_tests import *
from testing.ptv_generation_tests import *
from testing.img_proc_testing import *
from testing.max_pooling_tests import *
from testing.testing_for_paper import *
from testing.test_SEC import *
from testing.circuits_obj_tests import *
from testing.circuits_graph_tests import *
from testing.test_ptm_perturbations import *
from plotting import plot_random

from sim.circuits import *
from cv.CNN_classifier import CNN_classifier_main

def profile(func):
    """Decorator for enabling profiling"""
    def wrapper(*args, **kwargs):
        with cProfile.Profile() as pr:
            result = func(*args, **kwargs)
        with open('profiling_stats.txt', 'w') as stream:
            stats = Stats(pr, stream=stream)
            stats.strip_dirs()
            stats.sort_stats('time')
            stats.dump_stats('.prof_stats')
            stats.print_stats()
        return result
    return wrapper

def main():
    """THIS SHOULD BE THE MAIN ENTRY POINT FOR EVERYTHING!"""
    #CNN_classifier_main(cifar=False, train=False, sc_quantize=True, nbits=4)
    #test_MAC_RELU_sim([0.375, 0.9375], [0.625, 0.875, 0.3125], 4)
    #test_PARALLEL_CONST_sim()
    test_parallel_PCC()

if __name__ == "__main__":
    main()