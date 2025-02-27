/**
 * @file chebyshev_nodes.c
 * @brief Utils for generating Chebyshev nodes.
 */

#include "s2kit/chebyshev_nodes.h"

#include <math.h>

/**
 * @brief Generates an array of the angular arguments of @p n Chebyshev nodes.
 *
 * @param n length of @p eval_points
 * @param eval_points array of result arguments
 */
void AcosOfChebyshevNodes(const int n, double* eval_points) {
    double denominator = 2. * n;

    for (int i = 0; i < n; ++i)
        eval_points[i] = (2. * i + 1.) * M_PI / denominator;
}

/**
 * @brief Generates an array of @p n Chebyshev nodes.
 *
 * @param n length of @p eval_points
 * @param eval_points array of result nodes
 */
void ChebyshevNodes(const int n, double* eval_points) {
    double denominator = 2. * n;

    for (int i = 0; i < n; ++i)
        eval_points[i] = cos((2. * i + 1.) * M_PI / denominator);
}
