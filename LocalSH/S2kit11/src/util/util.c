/**
 * @file util.c
 * @brief Utility functions needed for transforms.
 */

#include "s2kit/util.h"

#include <math.h>
#include <stdlib.h>

/**
 * @brief Multiplies two complex numbers.
 * 
 * The result is stored separately for real and imaginary part.
 *
 * @param x real part of the first number
 * @param y imaginary part of the first number
 * @param u real part of the second number
 * @param v imaginary part of the second number
 * @param real_result real part of the result
 * @param imag_result imaginary part of the result
 */

void ComplexMult(const double x, const double y, const double u, const double v,
                        double* real_result, double* imag_result) {
    *real_result = x * u - y * v;
    *imag_result = x * v - y * u;
}


/**
 * @brief Gives the position of the coefficient <tt>f-hat(m,l)</tt> in the one-row array.
 * 
 * Returns the position of the coefficient <tt>f-hat(m,l)</tt> in the one-row array with
 * the spherical coefficients. It helps to preserve the symmetry that the coefficients have:
 * @code
 * f-hat(l,-m) = (-1)^m * conjugate( f-hat(l,m) )
 * @endcode
 *
 * @param m order
 * @param l degree
 * @param bw bandwidth
 */
int IndexOfHarmonicCoeff(const int m, const int l, const int bw) {
    int bigL = bw - 1;

    if (m >= 0)
        return (m * (bigL + 1) - ((m * (m - 1)) / 2) + (l - m));

    return (((bigL * (bigL + 3)) / 2) + 1 + ((bigL + m) * (bigL + m + 1) / 2) + (l - abs(m)));
}

/**
 * @brief Multiplies harmonic coefficients of a function and a filter.
 * 
 * See convolution theorem of Driscoll and Healy for details.
 *
 * @param rdatacoeffs real data coefficients
 * @param idatacoeffs imaginary data coefficients
 * @param rfiltercoeffs real filter coefficients
 * @param ifiltercoeffs imaginary filter coefficients
 * @param rres array of real result
 * @param ires array of imaginary result
 * @param bw bandwidth of problem
 * 
 * @note @p datacoeffs should be output of an SHT (e.g. FSTSemiMemo())
 * @note @p filtercoeffs should be output of an SHT (e.g. FZTSemiMemo())
 * @note All arrays must be of length @c bw*bw
 */
void TransMult(double* rdatacoeffs, double* idatacoeffs, double* rfiltercoeffs, double* ifiltercoeffs, double* rres,
               double* ires, const int bw) {
    double* rdptr = rdatacoeffs;
    double* idptr = idatacoeffs;
    double* rrptr = rres;
    double* irptr = ires;

    for (int m = 0; m < bw; ++m) {
        for (int l = m; l < bw; ++l) {
            ComplexMult(rfiltercoeffs[l], ifiltercoeffs[l], rdptr[l - m], idptr[l - m], rrptr + l - m, irptr + l - m);

            rrptr[l - m] *= sqrt(4. * M_PI / (2. * l + 1.));
            irptr[l - m] *= sqrt(4. * M_PI / (2. * l + 1.));
        }
        rdptr += bw - m;
        idptr += bw - m;
        rrptr += bw - m;
        irptr += bw - m;
    }

    int size = 2 * bw;
    for (int m = bw + 1; m < size; ++m) {
        for (int l = size - m; l < bw; ++l) {
            ComplexMult(rfiltercoeffs[l], ifiltercoeffs[l], rdptr[l - size + m], idptr[l - size + m],
                        rrptr + l - size + m, irptr + l - size + m);

            rrptr[l - size + m] *= sqrt(4. * M_PI / (2. * l + 1.));
            irptr[l - size + m] *= sqrt(4. * M_PI / (2. * l + 1.));
        }

        rdptr += m - bw;
        idptr += m - bw;
        rrptr += m - bw;
        irptr += m - bw;
    }
}
