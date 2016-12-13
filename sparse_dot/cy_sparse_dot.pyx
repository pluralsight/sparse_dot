import numpy as np
cimport numpy as np

import multiprocessing
from cython.parallel cimport parallel
from cython.parallel import prange
cimport openmp

ctypedef unsigned int UINT32
ctypedef unsigned long int UINT64

cdef extern from "stdlib.h":
    ctypedef int size_t
    void free(void* ptr)
    void* malloc(size_t size)
    void* realloc(void* ptr, size_t size)

cdef extern from "sparse_dot_core.h":
    ctypedef struct SparseArrayF:
        const UINT32* locs
        const float* array
        UINT32 len

    ctypedef struct SparseResult:
        UINT32 i
        UINT32 j
        float dot

    float sparse_dot(SparseArrayF * saf_ptr_a,
                     SparseArrayF * saf_ptr_b) nogil

    int sparse_dot_multi(SparseArrayF * saf_rows,
                         int i,
                         int num_rows,
                         SparseResult * sparse_results) nogil
    

SPARSE_RESULTS_DTYPE = [('i', np.uint32),
                        ('j', np.uint32),
                        ('sparse_result', np.float32)]

cdef void set_SAF_from_py_dict(SparseArrayF * saf, input_dict, saf_ind=0):
    '''Takes a python dictionary of arrays and fill a SparseArrayF
       The expected input format for the input_dict is:
       {'locs': <numpy array of uint32>,
        'array': <numpy array of float32>}
       '''
    cdef np.ndarray locs_cn = input_dict['locs']
    cdef np.ndarray array_cn = input_dict['array']
    
    saf[saf_ind].locs = <const UINT32 *> locs_cn.data
    saf[saf_ind].array = <const float *> array_cn.data
    saf[saf_ind].len = len(locs_cn)

def cy_sparse_dot_full(py_saf_list):
    '''Run sparse_dot between every possible pair of sparse arrays in a list
       Each input (py_saf_A_list / py_saf_B_list) is a list of sparse arrays
       A sparse array is defined as a dictionary containing two 1d arrays of the same length:
        * locs: (uint32)
        * array: (float32)
       
       Returns a numpy structured array with the following fields:
        * i, j: pair of indices into py_saf_list
        * dot: the dot product of py_saf_list[i] with py_saf_list[j] (float32)
       Missing entries have a zero value
    '''
    
    cdef int i
    
    num_items = len(py_saf_list)
    
    cdef int num_threads = multiprocessing.cpu_count()
    cdef int num_items_c = num_items
    
    # Map the numpy arrays directly to C pointers
    cdef SparseArrayF * saf_pointer = <SparseArrayF *> malloc(num_items * sizeof(SparseArrayF))
    
    for i in range(num_items):
        set_SAF_from_py_dict(saf_pointer, py_saf_list[i], saf_ind=i)
    
    # Run sparse_dot_multi on the generated pointers:
    
    # Set up results buffer and pointers
    # This section:
    # * Makes a num_threads by num_items numpy array
    # * Makes a C double pointer (array-of-arrays) of SparseResult's
    # * Sets the pointers to each of the numpy sub-arrays (each is num_items)
    # Each thread then gets it's own num_items length buffer to store its result
    sparse_results_tmp_arr = np.zeros((num_threads, num_items),        # Make sure each thread uses separate memory
                                      dtype=SPARSE_RESULTS_DTYPE)
    cdef np.ndarray sparse_results_cn
    cdef SparseResult ** sparse_results_pointer_arr
    sparse_results_pointer_arr = <SparseResult **> malloc(num_threads * sizeof(SparseResult *))
    for i in range(num_threads):
        sparse_results_cn = sparse_results_tmp_arr[i]
        sparse_results_pointer_arr[i] = <SparseResult*> sparse_results_cn.data
    
    cdef int num_sparse_results
    cdef int thread_number
    sparse_results_list = [None] * num_items
    for i in prange(num_items_c, nogil=True, chunksize=1, num_threads=num_threads, schedule='static'):
        thread_number = openmp.omp_get_thread_num()
        num_sparse_results = sparse_dot_multi(saf_pointer,
                                              i,
                                              num_items_c,
                                              sparse_results_pointer_arr[thread_number])
        with gil:
            sparse_results_list[i] = np.array(sparse_results_tmp_arr[thread_number][:num_sparse_results])

    # Collect the results and return
    sparse_results = np.concatenate(sparse_results_list)
    return sparse_results

