'''Test it all'''

import numpy as np
import sparse_dot
from sparse_dot.testing_utils import (generate_test_set, dot_equal_basic,
                                      is_naive_same, run_timing_test)

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

is_naive_same(generate_test_set(100, 100, 0.1), print_time=True)

def run_timing_test_1000_1000_0p1():
    run_timing_test(1000, 1000, 0.1)

if __name__ == '__main__':
    test_sparse_dot_simple()
    test_sparse_dot_basic_100()
    test_sparse_dot_basic_100_1()
    test_sparse_dot_10_100_1()
    test_sparse_dot_100_100_0p1()
    run_timing_test_1000_1000_0p1
