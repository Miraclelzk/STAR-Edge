/**
 * @file pml.c
 * @brief Source code for generating table of associated Legendre functions (Pml).
 */

#include "s2kit/pml.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "s2kit/chebyshev_nodes.h"
#include "s2kit/pmm.h"

#include "util/l2_norms.h"
#include "util/vector_funcs.h"

/**
 * @brief Generates all of the Pmls for a specified value of @p m.
 * 
 * @c P(m,l,j) respresents the associated Legendre function <tt>P_l^m</tt> evaluated at the j-th Chebyshev
 * point (for the bandwidth @p bw): <tt>cos((2 * j + 1) * PI / (2 * bw))</tt>.
 *
 * The array is placed in @p pml_table as follows:
 * @code
 * P(m,m,0)      P(m,m,1)    ...  P(m,m,2*bw-1)
 * P(m,m+1,0)    P(m,m+1,1)  ...  P(m,m+1,2*bw-1)
 * P(m,m+2,0)    P(m,m+2,1)  ...  P(m,m+2,2*bw-1)
 *   ...
 * P(m,bw-1,0)   P(m,bw-1,1) ...  P(m,bw-1,2*bw-1)
 * @endcode
 * This array will eventually be used by the naive transform algorithm (see naive.c).\n
 * This function will precompute the arrays necessary for the algorithm.
 *
 * @param bw bandwidth
 * @param m order
 * @param pml_table result array with generated Pmls of size @c 2*bw*(bw-m)
 * @param workspace space for computations of size @c 16*bw
 */
void GeneratePmlTable(const int bw, const int m, double* pml_table, double* workspace) {
    int size = 2 * bw;

    double* prevprev = workspace;
    double* prev = prevprev + size;
    double* temp1 = prev + size;
    double* temp2 = temp1 + size;
    double* temp3 = temp2 + size;
    double* temp4 = temp3 + size;
    double* x_i = temp4 + size;
    double* eval_args = x_i + size;

    // get the evaluation nodes
    ChebyshevNodes(size, x_i);
    AcosOfChebyshevNodes(size, eval_args);

    // set initial values of first two Pmls
    memset(prevprev, 0, sizeof(double) * size);

    if (!m)
        for (int i = 0; i < size; ++i)
            prev[i] = M_SQRT1_2;
    else
        Pmm_L2(m, eval_args, size, prev);

    memcpy(pml_table, prev, sizeof(double) * size);

    for (int i = 0; i < bw - m - 1; ++i) {
        vec_mul(L2_cn(m, m + i), prevprev, temp1, size);
        vec_dot(prev, x_i, temp2, size);
        vec_mul(L2_an(m, m + i), temp2, temp3, size);
        vec_add(temp3, temp1, temp4, size); // temp4 now contains P(m,m+i+1)

        pml_table += size;
        memcpy(pml_table, temp4, sizeof(double) * size);
        memcpy(prevprev, prev, sizeof(double) * size);
        memcpy(prev, temp4, sizeof(double) * size);
    }
}
