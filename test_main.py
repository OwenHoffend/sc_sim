import cProfile
from pstats import Stats

#Test script imports
from testing.corr_preservation_tests import *
from testing.symbolic_manip_tests import *
from testing.variance_analysis_tests import *
from testing.ptv_generation_tests import *
from testing.img_proc_testing import *
from testing.max_pooling_tests import *
from testing.testing_for_paper import *

from sim.circuits import *

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
    testing_for_paper()
    #max_pooling_test()

    #corrs = cme_propagation()
    #plt.plot(list(range(1, len(corrs) + 1)), list(reversed(corrs)))
    #plt.xlabel("Layer number")
    #plt.ylabel("Avg Corr")
    #plt.title("XOR Tree Correlation Propagation")
    #plt.xticks(list(range(1, len(corrs) + 1)))
    #plt.show()

    #Mf = get_func_mat(mux_1, 3, 1)
    #for i in range(100):
    #    Px = np.random.rand(2)
    #    vin = np.kron(get_vin_mc1(np.array([0.5,])), get_vin_mc1(Px)) #<-- better way to add an independent 0.5 input?
    #    Pout = B_mat(1).T @ Mf.T @ vin
    #    assert np.isclose(Pout, 0.5 * (Px[0] + Px[1]))
    #    print("Iter: {} Correct".format(i))

if __name__ == "__main__":
    main()