/**
 * @file pmm.c
 * @brief Source code for generating L2-normed associated Legendre functions Pmm.
 */

#include "s2kit/pmm.h"

#include <math.h>

/**
 * @brief Generates L2-normed Pmm.
 *
 * The norming constant can be found in Sean's PhD thesis.
 * // TODO find out the thesis
 *
 * @param m order
 * @param eval_points array of the angular arguments of evaluation points
 * @param n length of @p eval_points @b and of @p result
 * @param result array of generated L2-normed Pmm
 */
void Pmm_L2(const int m, double* eval_points, const int n, double* result) {
    double normed_coeff = sqrt(m + 0.5);

    for (int i = 0; i < m; ++i)
        normed_coeff *= sqrt((m - (i / 2.)) / ((double)m - i));

    if (m)
        normed_coeff *= pow(2., -m / 2.);
    if (m % 2)
        normed_coeff *= -1.;

    for (int i = 0; i < n; ++i)
        result[i] = normed_coeff * pow(sin(eval_points[i]), m);
}
