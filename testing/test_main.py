import cProfile
from pstats import Stats

#Test script imports
from testing.corr_preservation_tests import *

def profile(func):
    """Decorator for enabling profiling"""
    def wrapper(*args, **kwargs):
        with cProfile.Profile() as pr:
            func(*args, **kwargs)
        with open('profiling_stats.txt', 'w') as stream:
            stats = Stats(pr, stream=stream)
            stats.strip_dirs()
            stats.sort_stats('time')
            stats.dump_stats('.prof_stats')
            stats.print_stats()
    return wrapper

if __name__ == "__main__":
    pass