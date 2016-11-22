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

def validate_saf(saf, verbose=True):
    '''True if the locs (indices) in a saf are ordered
       AND the data types of the arrays are uint32 and float32 respectively'''
    def vpr(x):
        if verbose:
            print x
    if not ('locs' in saf and 'array' in saf):
        vpr('missing members')
        return False
    if not (hasattr(saf['locs'], 'dtype') and
            hasattr(saf['array'], 'dtype')):
        vpr('members not arrays')
        return False
    if not (saf['locs'].dtype == np.uint32 and
            saf['array'].dtype == np.float32):
        vpr('bad dtype')
        return False
    if not np.all(saf['locs'][1:] > saf['locs'][:-1]):
        vpr('locs not ordered')
        return False
    
    return True

def sparse_dot_full(saf_list, validate=True, verbose=True):
    '''Takes a list of arrays in locs/array dict form and '''
    if validate:
        assert all(validate_saf(saf, verbose=verbose) for saf in saf_list)
    return cy_sparse_dot.cy_sparse_dot_full(saf_list)

def dot_full_using_sparse(arr):
    '''Takes a 2d array and runs dot products against every
       combination of rows'''
    return sparse_dot_full(to_saf_list(arr), validate=False)

def sparse_cos_similarity(saf_list, validate=True, verbose=True):
    norms = np.array([np.linalg.norm(saf['array']) for saf in saf_list])
    dots = sparse_dot_full(saf_list, validate=validate, verbose=verbose)
    norm_i, norm_j = norms[(dots['i'],)], norms[(dots['j'],)]
    dots['sparse_result'] /= norm_i * norm_j
    return dots

def sparse_cos_distance(saf_list, validate=True, verbose=True):
    dots = sparse_cos_similarity(saf_list, validate=validate, verbose=verbose)
    dots['sparse_result'] *= -1
    dots['sparse_result'] += 1
    return dots

def cos_similarity_using_sparse(arr):
    return sparse_cos_similarity(to_saf_list(arr))

def cos_distance_using_sparse(arr):
    return sparse_cos_distance(to_saf_list(arr))

if __name__ == '__main__':
    r = dot_full_using_sparse([[1, 0, 0, 1, 3, 1],
                               [2, 0, 0, 0, 1, 5]])
    print r
