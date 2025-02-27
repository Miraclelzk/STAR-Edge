/**
 * @file l2_norms.c
 * @brief Recurrence coefficients for L2-normed associated Legendre recurrence.
 *
 * @note When using these coeffs, make sure that inital Pmm function is also L2-normed.
 */

#include "l2_norms.h"

#include <math.h>

/**
 * @param m order
 * @param l degree
 */
double L2_an(const int m, const int l) {
    return (
        sqrt(
            ((2. * l + 3.) / (2. * l + 1.)) * ((l - m + 1.) / (l + m + 1.))
        ) * ((2. * l + 1.) / (l - m + 1.))
    );
}

/**
 * @param m order
 * @param l degree
 */
double L2_cn(const int m, const int l) {
    if (!l) {
        return 0.;
    }

    return (
        -1.0 * sqrt(
            ((2. * l + 3.) / (2. * l - 1.)) * ((l - m + 1.) / (l + m + 1.)) * (((double)l - m) / ((double)l + m))
        ) * ((l + m) / (l - m + 1.))
    );
}

/**
 * @param m order
 * @param l degree
 * 
 * @note Use this function, instead of <tt>1/L2_cn(m,l)</tt>
 */
double L2_cn_inv(const int m, const int l) {
    return (
        -(1.0 + (1. - 2. * m) / ((double)m + l)) *
            sqrt(
                ((-1. + 2. * l) / (3. + 2. * l)) * (
                    (l + l * l + m + 2. * l * m + m * m) /
                        (l + l * l - m - 2. * l * m + m * m)
                )
            )
    );
}

/**
 * @param m order
 * @param l degree
 * 
 * @note Use this function, instead of <tt>-L2_an(m,l)/L2_cn(m,l)</tt>
 */
double L2_ancn(const int m, const int l) {
    return (
        sqrt(4. + ((4. * m * m - 1.) / ((double)l * l - m * m)))
    );
}
