'''Test it all'''

import numpy as np
import sparse_dot
from sparse_dot.testing_utils import (generate_test_set, dot_equal_basic,
                                      is_naive_same, run_timing_test_v1, run_timing_test)

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
    test_sparse_dot_simple()
    test_sparse_dot_basic_100()
    test_sparse_dot_basic_100_1()
    test_sparse_dot_10_100_1()
    test_sparse_dot_100_100_0p1()
    
    is_naive_same(generate_test_set(100, 100, 0.1), print_time=True)
    is_naive_same(generate_test_set(1000, 1000, 0.1), print_time=True)
    
    print run_timing_test_v1_1000_1000_0p1()
    print run_timing_test_1000_1000_100000()
    #print run_timing_test_v1_10000_10000_0p1() # ~100s
    #print run_timing_test_10000_10000_10000000() # ~37s
    #print run_timing_test_1000_20000_10000000() # 3s
    #print run_timing_test_5000_20000_10000() # LOL, only 0.05s to run but 60s to generate the initial data :-P
    
    # FAILS:
    #print run_timing_test_v1_10000_1000000_0p01() # Memory Error
    #print run_timing_test_10000_1000000_100000000() # Memory Error
    #print run_timing_test_10000_20000_10000000() # Ouch
    
