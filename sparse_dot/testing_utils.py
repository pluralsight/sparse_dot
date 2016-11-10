'''Utility functions for use in testing'''

import time
from itertools import izip
import numpy as np
import sparse_dot
import np_utils

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

def _generate_saf_from_ind_group(ind_group, big_locs, big_vals, num_cols):
    '''Helper function for generate_test_saf_list'''
    locs = big_locs[ind_group] % num_cols
    arg_sort = np.argsort(locs)
    return {'locs': locs[arg_sort],
            'array': big_vals[ind_group][arg_sort]}
                

def generate_test_saf_list(num_rows=100,
                           num_cols=100,
                           num_entries=1000,
                           seed=0,
                          ):
    '''Generate a test set (saf_list) to pass to sparse_dot
       num_entries determines the number of non-zero entries in the whole set
       Returns a saf list, aka a list of dictionaries with entries like:
       {'locs': array([...]),
        'array': array([...])}
       '''
    np.random.seed(seed)
    big_locs = np.sort(np.random.choice(num_rows * num_cols, num_entries, replace=False).astype(np.uint32))
    big_vals = np.random.random(num_entries).astype(np.float32)
    keys, ind_groups = np_utils.get_index_groups(big_locs // num_cols)
    saf_list = [_generate_saf_from_ind_group(g, big_locs, big_vals, num_cols)
                for g in ind_groups]
    return saf_list

def sparse_dot_full_validate_pass(saf_list):
    '''Returns a boolean denoting whether or not validation passed when
       trying this argument in sparse_dot_full'''
    success = None
    try:
        sparse_dot.sparse_dot_full(saf_list, verbose=False)
        success = True
    except:
        success = False
    return success

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
    '''Generate a test set and run sparse_dot_full
       Time both steps and print the output
       kwds:
           verbose=True -> print the test set and the result from sparse_dot
       
       all other args and kwds are passed to generate_test_set
       '''
    verbose = kwds.pop('verbose', False)
    
    t = time.time()
    test_set = generate_test_saf_list(*args, **kwds)
    generate_time = time.time()-t
    if verbose:
        print test_set
    
    t = time.time()
    sd = sparse_dot.sparse_dot_full(test_set)
    process_time = time.time()-t
    if verbose:
        print sd
    
    # Printing/returning section:
    return generate_time, process_time

def run_timing_test_v1(*args, **kwds):
    '''Generate a test set and run dot_full_using_sparse
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
