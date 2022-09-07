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
from testing.gauss_blur_ed import *
from testing.kernel_2x2 import *
from testing.tim_pcc_analysis import *
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
    tims_analysis()

if __name__ == "__main__":
    main()