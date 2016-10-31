'''Utility functions for use in testing'''

import time
import numpy as np
import sparse_dot

def generate_test_set(num_rows=100,
                      num_cols=100,
                      sparsity=0.1,
                      seed=0,
                     ):
    '''Generate a test set to pass to sparse_dot
       sparsity sets to amount of zeros in the set (zero means all zeros, 1 means no zeros)'''
    np.random.seed(seed)
    arr = np.zeros(num_rows * num_cols, dtype=np.float32)
    num_nonzero = int(num_rows * num_cols * sparsity)
    arr[:num_nonzero] = np.random.random(num_nonzero)
    np.random.shuffle(arr)
    arr = arr.reshape((num_rows, num_cols))
    return arr

def naive_dot(arr2d):
    '''arr2d is an 2d array (or list of arrays)
       Uses a naive numpy-based algorithm to compute the dot products
       Returns a matrix of dot products in sparse form, a list of tuples
       like (i, j, dot)
       Zero entries are skipped'''
    l = len(arr2d)
    dots = []
    for i in range(l):
        for j in range(i+1, l):
            dot = np.dot(arr2d[i], arr2d[j])
            if dot>0:
                dots.append((i, j, dot))
    return dots

def dot_equal_basic(a1, a2):
    d = np.dot(a1, a2)
    sd = sparse_dot.dot_full_using_sparse([a1, a2])['sparse_result'][0]
    print d
    print sd
    return np.isclose(d, sd)


def is_naive_same(test_set, print_time=False, verbose=False):
    '''Compare (sorted) results for naive_dot and dot_full_using_sparse'''
    t = time.time()
    sd = sparse_dot.dot_full_using_sparse(test_set)
    if print_time:
        print 'sparse_dot speed:', time.time()-t
    sd = np.sort(sd)
    
    t = time.time()
    d = naive_dot(test_set) # No need to sort, save the time
    if print_time:
        print 'naive speed:', time.time()-t
    
    if verbose:
        print 'test_set', test_set
        print 'sparse_dot result', sd
        print 'naive result', d
    
    return (([(i,j) for i,j,k in d] == [(i,j) for i,j,k in sd]) and
            np.all(np.isclose([k for i,j,k in d], sd['sparse_result'])))

def run_timing_test(*args, **kwds):
    '''Generate a test set and run sparse_dot
       Time both steps and print the output
       kwds:
           verbose=True -> print the test set and the result from sparse_dot
       
       all other args and kwds are passed to generate_test_set
       '''
    verbose = kwds.pop('verbose', False)
    
    t = time.time()
    test_set = generate_test_set(*args, **kwds)
    generate_time = time.time()-t
    if verbose:
        print test_set
    
    t = time.time()
    sd = sparse_dot.dot_full_using_sparse(test_set)
    process_time = time.time()-t
    if verbose:
        print sd
    
    # Printing/returning section:
    return generate_time, process_time
