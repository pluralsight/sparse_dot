'''Test it all'''

import numpy as np
import sparse_dot
from sparse_dot.testing_utils import (generate_test_set,
    sparse_dot_full_validate_pass, dot_equal_basic, is_naive_same,
    run_timing_test_v1, run_timing_test)

SAF_GOOD = {'locs': np.array([0, 1, 4], dtype=np.uint32),
            'array': np.array([4.2, 9.0, 5.1], dtype=np.float32)}
SAF_BAD_1 = {'locs': np.array([4, 0, 1], dtype=np.uint32),
             'array': np.array([4.2, 9.0, 5.1], dtype=np.float32)}
SAF_BAD_2 = {'locs': np.array([0, 1, 4], dtype=np.uint32),
             'array': [4.2, 9.0, 5.1]}

def test_validate_saf_1():
    assert sparse_dot.validate_saf(sparse_dot.to_saf([0,2,0,1,3,4,2,0,0,1,0]))
    
def test_validate_saf_2():
    assert sparse_dot.validate_saf(SAF_GOOD)
    assert not sparse_dot.validate_saf(SAF_BAD_1, verbose=False)
    assert not sparse_dot.validate_saf(SAF_BAD_2, verbose=False)

def test_sparse_dot_full_validation_1():
    assert sparse_dot_full_validate_pass(sparse_dot.to_saf_list(np.arange(6).reshape(2,3)))

def test_sparse_dot_full_validation_2():
    assert sparse_dot_full_validate_pass([SAF_GOOD])
    assert not sparse_dot_full_validate_pass([SAF_BAD_1])
    assert not sparse_dot_full_validate_pass([SAF_GOOD, SAF_BAD_1])
    assert not sparse_dot_full_validate_pass([SAF_BAD_2, SAF_BAD_1])


def test_sparse_dot_simple():
    assert dot_equal_basic(*np.arange(6).reshape(2,3))

def test_sparse_dot_basic_100():
    assert dot_equal_basic(*generate_test_set(2, 100))

def test_sparse_dot_basic_100_1():
    assert dot_equal_basic(*generate_test_set(2, 100, 1))

def test_sparse_dot_10_100_1():
    assert is_naive_same(generate_test_set(10, 100, 1))

def test_sparse_dot_100_100_0p1():
    assert is_naive_same(generate_test_set(100, 100, 0.1))

def test_cos_distance_using_scipy_1():
    '''Test the cos distance calculation against scipy
       (must be installed for this test)'''
    import scipy.spatial.distance
    a, b = generate_test_set(2, 1000, 1)
    assert np.isclose(scipy.spatial.distance.cosine(a, b),
                      sparse_dot.cos_distance_using_sparse([a, b])['sparse_result'][0])

def test_cos_distance_using_scipy_2():
    '''Test the cos distance calculation against scipy
       (must be installed for this test)'''
    import scipy.spatial.distance
    rows = generate_test_set(100, 1000, 1)
    for i, j, sr in sparse_dot.cos_distance_using_sparse(rows):
        assert np.isclose(sr, scipy.spatial.distance.cosine(rows[i], rows[j]))

def run_timing_test_v1_1000_1000_0p1():
    return run_timing_test_v1(1000, 1000, 0.1)

def run_timing_test_v1_10000_10000_0p1():
    return run_timing_test_v1(10000, 10000, 0.1)

def run_timing_test_1000_1000_100000():
    return run_timing_test(1000, 1000, 100000)

def run_timing_test_10000_10000_10000000():
    return run_timing_test(10000, 10000, 10000000)

def run_timing_test_1000_20000_10000000():
    return run_timing_test(1000, 200000, 10000000)

def run_timing_test_5000_20000_10000():
    return run_timing_test(5000, 200000, 10000)

if __name__ == '__main__':
    test_validate_saf_1()
    test_validate_saf_2()
    test_sparse_dot_full_validation_1()
    test_sparse_dot_full_validation_2
    test_sparse_dot_simple()
    test_sparse_dot_basic_100()
    test_sparse_dot_basic_100_1()
    test_sparse_dot_10_100_1()
    test_sparse_dot_100_100_0p1()
    
    is_naive_same(generate_test_set(100, 100, 0.1), print_time=True)
    is_naive_same(generate_test_set(1000, 1000, 0.1), print_time=True)
    
    test_cos_distance_using_scipy_1()
    test_cos_distance_using_scipy_2()

    print run_timing_test_v1_1000_1000_0p1()
    print run_timing_test_1000_1000_100000()
    
    # These are all run in the benchmarks instead:
    #print run_timing_test_v1_10000_10000_0p1() # ~100s
    #print run_timing_test_10000_10000_10000000() # ~100s
    #print run_timing_test_1000_20000_10000000() # 10s
    #print run_timing_test_5000_20000_10000() # LOL, only 0.1s to run but 8s to generate the initial data :-P
    
    # FAILS:
    #print run_timing_test_v1_10000_1000000_0p01() # Memory Error
    #print run_timing_test_10000_1000000_100000000() # Memory Error
    #print run_timing_test_10000_20000_10000000() # Ouch
    
