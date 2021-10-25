import cProfile
from pstats import Stats

#Test script imports
from testing.corr_preservation_tests import *
from testing.symbolic_manip_tests import *
from testing.variance_analysis_tests import *
from testing.ptv_generation_tests import *
from testing.img_proc_testing import *

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
    test_roberts_cross_mux_maj()

if __name__ == "__main__":
    main()