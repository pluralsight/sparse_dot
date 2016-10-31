#ifndef SPARSE_DOT_CORE_H
#define SPARSE_DOT_CORE_H

#define CONSTANT const

//#define GLOBAL
//#define KERNEL
//#define GET_GLOBAL_ID 0

//For OpenCL, these needs to be:
// #define KERNEL __kernel
// #define CONSTANT __constant
// #define GLOBAL __global
// #define POPCOUNT popcount
// #define GET_GLOBAL_ID get_global_id(0)
// #define GENERATED=GENERATED_STRING,


typedef unsigned int UINT32;
typedef unsigned long int UINT64;

typedef struct {
    CONSTANT UINT32* locs;
    CONSTANT float* array; // or double??
    UINT32 len;
} SparseArrayF;

typedef struct {
    UINT32 i;
    UINT32 j;
    float dot;
} SparseResult;


float sparse_dot(SparseArrayF * saf_ptr_a,
                 SparseArrayF * saf_ptr_b);

int sparse_dot_multi(SparseArrayF * saf_rows,
                     int i,
                     int num_rows,
                     SparseResult * sparse_results);

#endif
