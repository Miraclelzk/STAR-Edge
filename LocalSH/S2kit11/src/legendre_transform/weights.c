/**
 * @file weights.c
 * @brief Contains the function that generates the weights for a Legendre transform.
 *
 * Basically, it contains the implementation of the formula as defined in the tensor paper,
 * and also given in the so(3) paper and is mentioned in the s^2 paper! // TODO find these pages
 *
 * This formula is slightly different from the one given in the original Driscoll and Healy paper
 * because they were sampling at the poles, and now we're not.
 *
 * In pseudo-TeX, the formula for the bandwidth B weights is:
 * @code
 * w_B(j) = 2/B sin((pi*(2j+1))/(4B)) *
 *     sum_{k=0}^{B-1} 1/(2k+1)*sin((2j+1)(2k+1)pi/(4B))
 * where j = 0, 1, ..., 2 B - 1 @endcode
 */

#include "s2kit/weights.h"

#include <math.h>

/**
 * @brief Generates weights for both even and odd order Legendre transforms for a given bandwidth.
 * 
 * @param bw bandwidth of transform
 * @param weights array of size @c 4*bw which will contain the weights for both even (starting at
 * <tt>weights[0]</tt>) and odd (<tt>weights[2*bw]</tt>) transforms
 * @note If you want to use these weights for an @b odd order transform, given the way the code is
 * set up, you have to multiply the j-th weight by <tt>sin(pi*(2j+1)/(4B))</tt>.\n
 * It is taken into account in this library functions.
 */
void GenerateWeightsForDLT(const int bw, double* weights) {
    double coeff = M_PI / (4. * bw);

    for (int i = 0; i < 2 * bw; ++i) {
        double sum = 0.;
        double k = 2. * i + 1.;

        for (int j = 0; j < bw; ++j)
            sum += 1. / (2. * j + 1.) * sin(k * (2. * j + 1.) * coeff);

        sum *= 2. * sin(k * coeff) / bw;

        weights[i] = sum;
        weights[i + 2 * bw] = sum * sin(k * coeff);
    }
}
