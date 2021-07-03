import cProfile
from pstats import Stats

#Test script imports
from testing.corr_preservation_tests import *

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        test_xor_and_corr_pres()

    with open('profiling_stats.txt', 'w') as stream:
        stats = Stats(pr, stream=stream)
        stats.strip_dirs()
        stats.sort_stats('time')
        stats.dump_stats('.prof_stats')
        stats.print_stats()