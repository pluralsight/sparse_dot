#include <stdio.h>
#include <stdbool.h>
#include "sparse_dot_core.h"

float sparse_dot(SparseArrayF * saf_ptr_a,
                 SparseArrayF * saf_ptr_b) {
/*Compute the dot product of two SparseArrayF's
  Unpack all the struct members to save lookups
  and make the code below more readable.
*/
    int i, i_a=0, i_b=0, l_a, l_b;
    float sum=0;
    UINT32 * locs_a = saf_ptr_a->locs;
    UINT32 * locs_b = saf_ptr_b->locs;
    float * dat_a = saf_ptr_a->array;
    float * dat_b = saf_ptr_b->array;
    int len_a = saf_ptr_a->len;
    int len_b = saf_ptr_b->len;
    
    for(i=0; i < len_a + len_b; i++) {
        if((i_a >= len_a) || (i_b >= len_b)) {
            break;
        }
        l_a = locs_a[i_a];
        l_b = locs_b[i_b];
        
        if(l_b < l_a) {
            i_b++;
        } else if(l_a < l_b) {
            i_a++;
        } else { // l_a == l_b
            sum += dat_a[i_a] * dat_b[i_b];
            i_a++;
            i_b++;
        }
    }
    return(sum);
}

//Old version:
//void sparse_dot_multi(SparseArrayF * saf_ptr_a,
//                      SparseArrayF * saf_rows_b,
//                      int num_rows_b,
//                      float * sparse_results) {
// /*Compute the sparse_dot of A with each of element of B
//  (saf_rows_b[i] where 0<=i<num_rows)
//  sparse_results must be pre-allocated with a length of num_rows_b
//*/
//    int i;
//    for(i=0;i<num_rows_b;i++) {
//        sparse_results[i] = sparse_dot(saf_ptr_A, &saf_rows_B[i]);
//    }
//}

int sparse_dot_multi(SparseArrayF * saf_rows,
                     int i,
                     int num_rows,
                     SparseResult * sparse_results) {
/*Compute the sparse_dot of saf_rows[i] with each of element of saf_rows[j]
  (where i<j<num_rows)
  sparse_results must be pre-allocated with a length of num_rows
*/
    int j;
    float result;
    SparseResult * sparse_result_ptr;
    int num_results = 0;
    for(j = i+1; j < num_rows; j++) {
        result = sparse_dot(&saf_rows[i], &saf_rows[j]);
        if(result != 0) {
            sparse_result_ptr = &sparse_results[num_results];
            sparse_result_ptr->i = i;
            sparse_result_ptr->j = j;
            sparse_result_ptr->dot = result;
            num_results++;
        }
    }
    return num_results;
}
