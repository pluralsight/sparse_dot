'''The main script'''

# To update with any Cython changes, just run:
# python setup.py build_ext --inplace

import numpy as np
import cy_sparse_dot

def to_saf(arr1d):
    arr1d = np.asanyarray(arr1d)
    locs = np.nonzero(arr1d)
    return {'locs': locs[0].astype(np.uint32),
            'array': arr1d[locs].astype(np.float32)}

def to_saf_list(arr2d):
    return map(to_saf, arr2d)

def sparse_dot_full(saf_list):
    '''Takes a list of arrays in locs/array dict form and '''
    return cy_sparse_dot.cy_sparse_dot_full(saf_list)

def dot_full_using_sparse(arr):
    '''Takes a 2d array and runs dot products against every
       combination of rows'''
    return sparse_dot_full(to_saf_list(arr))

def sparse_cos_distance(saf_list):
    norms = np.array([np.linalg.norm(saf['array']) for saf in saf_list])
    dots = sparse_dot_full(saf_list)
    norm_i, norm_j = norms[(dots['i'],)], norms[(dots['j'],)]
    dots['sparse_result'] = 1 - dots['sparse_result'] / (norm_i * norm_j)
    return dots

def cos_distance_using_sparse(arr):
    return sparse_cos_distance(to_saf_list(arr))

if __name__ == '__main__':
    r = dot_full_using_sparse([[1, 0, 0, 1, 3, 1],
                               [2, 0, 0, 0, 1, 5]])
    print r
