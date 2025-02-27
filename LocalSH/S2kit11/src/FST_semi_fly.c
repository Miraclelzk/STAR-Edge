/**
 * @file FST_semi_fly.c
 * @brief Routines to perform spherical harmonic transforms and convolutions on the 2-sphere using
 * a combination of semi-naive and naive algorithms.
 *
 * Just like FST_semi_memo.c, except that these routines compute associated Legendre functions on
 * the fly.\n
 * For descriptions on calling these functions, see the documentation preceding each function.
 */

#include "s2kit/FST_semi_fly.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <fftw3.h>

#include "s2kit/cospml.h"
#include "s2kit/pml.h"
#include "s2kit/naive.h"
#include "s2kit/seminaive.h"
#include "s2kit/weights.h"
#include "s2kit/chebyshev_nodes.h"
#include "s2kit/util.h"

/**
 * @brief Computes <b>spherical harmonic transform</b> using the seminaive and naive algorithms.
 *
 * Output ordering of coeffs (in @p rcoeffs and @p icoeffs) <tt>f(m,l)</tt> is:
 * @code
 * f(0,0) f(0,1) f(0,2) ... f(0,bw-1)
 *        f(1,1) f(1,2) ... f(1,bw-1)
 *        etc.
 *            f(bw-2,bw-2), f(bw-2,bw-1)
 *                          f(bw-1,bw-1)
 *                          f(-(bw-1),bw-1)
 *         f(-(bw-2),bw-2), f(-(bw-2),bw-1)
 *        etc.
 *              f(-2,2) ... f(-2,bw-1)
 *      f(-1,1) f(-1,2) ... f(-1,bw-1)
 * @endcode
 *
 * This only requires an array of size @c bw*bw. If zero-padding is used to make the indexing nice,
 * then you need a an <tt>(2bw-1)*bw</tt> array, but that is not done here. Because of the amount
 * of space necessary for doing large transforms, it is important not to use any more than
 * necessary.
 * 
 * @note See an example of use in test_s2_semi_fly.c
 *
 * @param rdata array of length <tt>4*bw*bw</tt> of the real part of the function samples
 * @param idata array of length <tt>4*bw*bw</tt> of the imaginary part of the function samples
 * @param rcoeffs array of length <tt>bw*bw</tt> which will contain the real part of harmonic
 * coefficients in a linearized form
 * @param icoeffs array of length <tt>bw*bw</tt> which will contain the imaginary part of harmonic
 * coefficients in a linearized form
 * @param bw bandwidth of problem
 * @param workspace space for computations of size <tt>(10*bw*bw)+(21*bw)</tt>
 * @param data_format determines the format of the computed data
 * @param cutoff determines order to switch from seminaive to naive algorithm
 * @param DCT_plan plan for DLT which is used as argument for call DLTSemi()
 * @param FFT_plan plan for FFT
 * @param weights array which is used as argument for call DLTSemi() and DLTNaive()
 *
 * @note See more info about @p DCT_plan and @p weights in DLTSemi().
 */
void FSTSemiFly(double* rdata, double* idata, double* rcoeffs, double* icoeffs, const int bw, double* workspace,
                DataFormat data_format, const int cutoff, fftw_plan* DCT_plan, fftw_plan* FFT_plan, double* weights) {
    // TODO add check that cutoff<bw?
    int size = 2 * bw;

    double* rres = workspace;                  // needs (size * size) = (4 * bw^2)
    double* ires = rres + (size * size);       // needs (size * size) = (4 * bw^2)
    double* fltres = ires + (size * size);     // needs bw
    double* sin_values = fltres + bw;          // needs (2 * bw)
    double* eval_pts = sin_values + (size);    // needs (2 * bw)
    double* pmls = eval_pts + (size);          // needs (2 * bw * bw)
    double* scratchpad = pmls + (2 * bw * bw); // needs (16 * bw)

    // do the FFTs along phi
    fftw_execute_split_dft(*FFT_plan, rdata, idata, rres, ires);

    // normalize
    double normed_coeff = sqrt(2. * M_PI) / size;
    for (int i = 0; i < size * size; ++i) {
        rres[i] *= normed_coeff;
        ires[i] *= normed_coeff;
    }

    // point to start of output data buffers
    double* rdataptr = rcoeffs;
    double* idataptr = icoeffs;

    for (int m = 0; m < cutoff; ++m) { // semi-naive part
        // generate cosine series of pmls
        GenerateCosPmlTable(bw, m, pmls, scratchpad);

        // real part
        DLTSemi(rres + (m * size), bw, m, fltres, scratchpad, pmls, weights, DCT_plan);
        // load real part of coefficients into output space
        memcpy(rdataptr, fltres, sizeof(double) * (bw - m));
        rdataptr += bw - m;

        // imaginary part
        DLTSemi(ires + (m * size), bw, m, fltres, scratchpad, pmls, weights, DCT_plan);
        // load imaginary part of coefficients into output space
        memcpy(idataptr, fltres, sizeof(double) * (bw - m));
        idataptr += bw - m;
    }

    for (int m = cutoff; m < bw; ++m) { // naive part
        // generate pmls, note we are using CosPmls array
        GeneratePmlTable(bw, m, pmls, scratchpad);

        // real part
        DLTNaive(rres + (m * size), bw, m, weights, fltres, pmls, scratchpad);
        memcpy(rdataptr, fltres, sizeof(double) * (bw - m));
        rdataptr += bw - m;

        // imaginary part
        DLTNaive(ires + (m * size), bw, m, weights, fltres, pmls, scratchpad);
        memcpy(idataptr, fltres, sizeof(double) * (bw - m));
        idataptr += bw - m;
    }

    // upper coefficients
    /*
        If the data is real, we don't have to compute the coeffs whose order is less than 0,
        i.e. since the data is real, we know that

        f-hat(l,-m) = (-1)^m * conjugate(f-hat(l,m)),

        so use that to get the rest of the coefficients
    */
    if (data_format == REAL) {
        double coeff = 1.;
        for (int i = 1; i < bw; ++i) {
            coeff *= -1.0;
            for (int j = i; j < bw; ++j) {
                int index0 = IndexOfHarmonicCoeff(i, j, bw);
                int index1 = IndexOfHarmonicCoeff(-i, j, bw);

                rcoeffs[index1] = coeff * rcoeffs[index0];
                icoeffs[index1] = -coeff * icoeffs[index0];
            }
        }

        return;
    }


    // complex data
    /* 
        Note that `m` is greater than `bw` here, but this is for
        purposes of indexing the input data arrays. The "true"
        value of `m` as a parameter for Pml is `size-m` 
    */
    for (int m = bw + 1; m <= size - cutoff; ++m) { // naive part
        // generate pmls, note we are using CosPmls array
        GeneratePmlTable(bw, size - m, pmls, scratchpad);

        // real part
        DLTNaive(rres + (m * size), bw, size - m, weights, fltres, pmls, scratchpad);
        // load real part of coefficients into output space
        if (m % 2) {
            for (int i = 0; i < m - bw; ++i)
                rdataptr[i] = -fltres[i];
        } else {
            memcpy(rdataptr, fltres, sizeof(double) * (m - bw));
        }
        rdataptr += m - bw;

        // imaginary part
        DLTNaive(ires + (m * size), bw, size - m, weights, fltres, pmls, scratchpad);
        // load imaginary part of coefficients into output space
        if (m % 2) {
            for (int i = 0; i < m - bw; ++i)
                idataptr[i] = -fltres[i];
        } else {
            memcpy(idataptr, fltres, sizeof(double) * (m - bw));
        }
        idataptr += m - bw;
    }

    for (int m = size - cutoff + 1; m < size; ++m) { // semi-naive part
        // generate cosine series of pmls
        GenerateCosPmlTable(bw, size - m, pmls, scratchpad);

        // real part
        DLTSemi(rres + (m * size), bw, size - m, fltres, scratchpad, pmls, weights, DCT_plan);
        // load real part of coefficients into output space
        if (m % 2) {
            for (int i = 0; i < m - bw; ++i)
                rdataptr[i] = -fltres[i];
        } else {
            memcpy(rdataptr, fltres, sizeof(double) * (m - bw));
        }
        rdataptr += m - bw;

        // imaginary part
        DLTSemi(ires + (m * size), bw, size - m, fltres, scratchpad, pmls, weights, DCT_plan);
        // load imaginary part of coefficients into output space
        if (m % 2) {
            for (int i = 0; i < m - bw; ++i)
                idataptr[i] = -fltres[i];
        } else {
            memcpy(idataptr, fltres, sizeof(double) * (m - bw));
        }
        idataptr += m - bw;
    }
}

/**
 * @brief Computes <b>inverse spherical harmonic transform</b>.
 *
 * @note See an example of use in test_s2_semi_fly.c
 *
 * @param rcoeffs array of length <tt>bw*bw</tt> of the real part of harmonic coefficients
 * @param icoeffs array of length <tt>bw*bw</tt> of the imaginary part of harmonic coefficients
 * @param rdata array of length <tt>4*bw*bw</tt> which will contain the real part of transformed
 * result
 * @param idata array of length <tt>4*bw*bw</tt> which will contain the imaginary part of
 * transformed result
 * @param bw bandwidth of problem
 * @param workspace space for computations of size <tt>(10*bw*bw)+(24*bw)</tt>
 * @param data_format determines the format of the computed data
 * @param cutoff determines order to switch from seminaive to naive algorithm
 * @param inv_DCT_plan plan for inverse DLT which is used as argument for call InvDLTSemi()
 * @param inv_FFT_plan plan for inverse FFT
 *
 * @note The real and imaginary part of harmonic coefficients should be stored in @p rcoeffs and
 * @p icoeffs before passing to this function in the same order as it was mentioned in FSTSemiFly().
 * @note See more info about @p inv_DCT_plan in InvDLTSemi().
*/
void InvFSTSemiFly(double* rcoeffs, double* icoeffs, double* rdata, double* idata, const int bw, double* workspace,
                   DataFormat data_format, const int cutoff, fftw_plan* inv_DCT_plan, fftw_plan* inv_FFT_plan) {
    int size = 2 * bw;

    double* rfourdata = workspace;                  // needs (size * size)
    double* ifourdata = rfourdata + (size * size);  // needs (size * size)
    double* rinvfltres = ifourdata + (size * size); // needs (2 * bw)
    double* iminvfltres = rinvfltres + (size);      // needs (2 * bw)
    double* sin_values = iminvfltres + (size);      // needs (2 * bw)
    double* eval_pts = sin_values + (size);         // needs (2 * bw)
    double* pmls = eval_pts + (size);               // needs (2 * bw * bw)
    double* scratchpad = pmls + (2 * bw * bw);      // needs (16 * bw)

    // load up the sin_values array
    AcosOfChebyshevNodes(size, eval_pts);
    for (int i = 0; i < size; ++i)
        sin_values[i] = sin(eval_pts[i]);

    // do all of the inverse Legendre transforms
    double* rdataptr = rcoeffs;
    double* idataptr = icoeffs;

    for (int m = 0; m < cutoff; ++m) { // semi-naive part
        // generate cosine series of pmls
        GenerateCosPmlTable(bw, m, pmls, scratchpad);
        // take transpose
        TransposeCosPmlTable(bw, m, pmls, pmls + TableSize(m, bw));

        // real part
        InvDLTSemi(rdataptr, bw, m, rinvfltres, pmls + TableSize(m, bw), sin_values, scratchpad, inv_DCT_plan);
        // imaginary part
        InvDLTSemi(idataptr, bw, m, iminvfltres, pmls + TableSize(m, bw), sin_values, scratchpad, inv_DCT_plan);

        // store normal, then tranpose before doing inverse fft
        memcpy(rfourdata + (m * size), rinvfltres, sizeof(double) * size);
        memcpy(ifourdata + (m * size), iminvfltres, sizeof(double) * size);

        rdataptr += bw - m;
        idataptr += bw - m;
    }

    for (int m = cutoff; m < bw; ++m) { // naive part
        // generate pmls
        // Note we are using CosPmls array and don't have to transpose
        GeneratePmlTable(bw, m, pmls, scratchpad);

        InvDLTNaive(rdataptr, bw, m, rinvfltres, pmls); // real part
        InvDLTNaive(idataptr, bw, m, iminvfltres, pmls); // imaginary part

        // store normal, then tranpose before doing inverse fft
        memcpy(rfourdata + (m * size), rinvfltres, sizeof(double) * size);
        memcpy(ifourdata + (m * size), iminvfltres, sizeof(double) * size);

        rdataptr += bw - m;
        idataptr += bw - m;
    }

    // fill in zero values where m = bw (from problem definition)
    memset(rfourdata + (bw * size), 0, sizeof(double) * size);
    memset(ifourdata + (bw * size), 0, sizeof(double) * size);

    /*
        If the data is real, we don't have to compute the coeffs whose order is less than 0,
        i.e. since the data is real, we know that

        invf-hat(l,-m) = conjugate(invf-hat(l,m)),

        so use that to get the rest of the coefficients
    */
    // TODO swap if and for? (almost same 2nd part)
    if (data_format == COMPLEX) {
        for (int m = bw + 1; m <= size - cutoff; ++m) { // naive part
            // Note we are using CosPmls array and don't have to transpose
            GeneratePmlTable(bw, size - m, pmls, scratchpad);

            InvDLTNaive(rdataptr, bw, size - m, rinvfltres, pmls);
            InvDLTNaive(idataptr, bw, size - m, iminvfltres, pmls);

            // store normal, then tranpose before doing inverse fft
            if (m % 2)
                for (int i = 0; i < size; ++i) {
                    rinvfltres[i] *= -1.0;
                    iminvfltres[i] *= -1.0;
                }

            memcpy(rfourdata + (m * size), rinvfltres, sizeof(double) * size);
            memcpy(ifourdata + (m * size), iminvfltres, sizeof(double) * size);

            rdataptr += bw - (size - m);
            idataptr += bw - (size - m);
        }

        for (int m = size - cutoff + 1; m < size; ++m) { // semi-naive part
            // generate cosine series of pmls
            GenerateCosPmlTable(bw, size - m, pmls, scratchpad);
            // take transpose
            TransposeCosPmlTable(bw, size - m, pmls, pmls + TableSize(size - m, bw));

            InvDLTSemi(rdataptr, bw, size - m, rinvfltres, pmls + TableSize(size - m, bw), sin_values,
                       scratchpad, inv_DCT_plan);
            InvDLTSemi(idataptr, bw, size - m, iminvfltres, pmls + TableSize(size - m, bw), sin_values,
                       scratchpad, inv_DCT_plan);

            // store normal, then tranpose before doing inverse fft
            if (m % 2)
                for (int i = 0; i < size; ++i) {
                    rinvfltres[i] *= -1.0;
                    iminvfltres[i] *= -1.0;
                }

            memcpy(rfourdata + (m * size), rinvfltres, sizeof(double) * size);
            memcpy(ifourdata + (m * size), iminvfltres, sizeof(double) * size);

            rdataptr += bw - (size - m);
            idataptr += bw - (size - m);
        }
    } else { // real data
        for (int m = bw + 1; m < size; ++m) {
            memcpy(rfourdata + (m * size), rfourdata + ((size - m) * size), sizeof(double) * size);
            memcpy(ifourdata + (m * size), ifourdata + ((size - m) * size), sizeof(double) * size);

            for (int i = 0; i < size; ++i)
                ifourdata[(m * size) + i] *= -1.0;
        }
    }

    // normalize
    double normed_coeff = 1. / (sqrt(2. * M_PI));
    for (int i = 0; i < 4 * bw * bw; ++i) {
        rfourdata[i] *= normed_coeff;
        ifourdata[i] *= normed_coeff;
    }

    fftw_execute_split_dft(*inv_FFT_plan, ifourdata, rfourdata, idata, rdata);
}

/**
 * @brief Computes <b>zonal harmonic transform</b> using seminaive algorithm.
 *
 * This transform is used in convolutions. Only computes spherical harmonics for <tt>m=0</tt>.
 * 
 * @param rdata array of length <tt>4*bw*bw</tt> of the real part of the data samples
 * @param idata array of length <tt>4*bw*bw</tt> of the imaginary part of the data samples
 * @param rres array of length <tt>bw</tt> which will contain the real part of harmonic
 * coefficients
 * @param ires array of length <tt>bw</tt> which will contain the imaginary part of harmonic
 * coefficients
 * @param bw bandwidth of problem
 * @param workspace space for computations of size <tt>12*bw</tt>
 * @param data_format determines the format of the computed data
 * @param DCT_plan plan for DLT which is used as argument for call DLTSemi()
 * @param weights array which is used as argument for call DLTSemi()
 * 
 * @note See more info about @p DCT_plan and @p weights in DLTSemi().
 */
void FZTSemiFly(double* rdata, double* idata, double* rres, double* ires, const int bw, double* workspace,
                DataFormat data_format, fftw_plan* DCT_plan, double* weights) {
    int size = 2 * bw;

    double* r0 = workspace;                    // needs (2 * bw)
    double* i0 = r0 + size;                    // needs (2 * bw)
    double* pmls = i0 + size;                  // needs (2 * bw * bw)
    double* scratchpad = pmls + (2 * bw * bw); // needs (4 * bw)

    double dsize = sqrt(2. * M_PI) / size;
    // compute the m = 0 components
    for (int i = 0; i < size; ++i) {
        double tmpreal = 0.0;
        double tmpimag = 0.0;

        for (int j = 0; j < size; ++j) {
            tmpreal += rdata[(i * size) + j];
            tmpimag += idata[(i * size) + j];
        }

        // normalize
        r0[i] = tmpreal * dsize;
        i0[i] = tmpimag * dsize;
    }

    // generate cosine series of pmls
    GenerateCosPmlTable(bw, 0, pmls, scratchpad);

    // real part
    DLTSemi(r0, bw, 0, rres, scratchpad, pmls, weights, DCT_plan);

    if (data_format == COMPLEX) // if data is complex, do imaginary part
        DLTSemi(i0, bw, 0, ires, scratchpad, pmls, weights, DCT_plan);
    else // otherwise set coefficients to zero
        memset(ires, 0, sizeof(double) * size);
}

/**
 * @brief <b>Convolves</b> two functions defined on the 2-sphere.
 *
 * Uses seminaive algorithms for spherical harmonic transforms.
 *
 * @note See an example of use in test_conv_semi_fly.c
 *
 * Memory requirements for Conv2Sphere:\n
 * @code
 * 4*bw^2 + 2*bw + 10*bw^2 + 24*bw = 14*bw^2 + 26*bw
 * @endcode
 *
 * @param rdata array of length <tt>4*bw*bw</tt> of the real part of sampled function
 * @param idata array of length <tt>4*bw*bw</tt> of the imaginary part of sampled function
 * @param rfilter array of length <tt>4*bw*bw</tt> of the real part of sampled filter function
 * @param ifilter array of length <tt>4*bw*bw</tt> of the imaginary part of sampled filter function
 * @param rres array of length <tt>4*bw*bw</tt> which will contain the real part of result function
 * @param ires array of length <tt>4*bw*bw</tt> which will contain the imaginary part of result
 * function
 * @param bw bandwidth of problem
 * @param workspace space for computations of size <tt>(14*bw*bw)+(26*bw)</tt>
 *
 * @note We assume that data is real.
 * @note This function will do seminaive algorithm for all orders.
 * @note If you want to do multiple convolutions, you can freely reuse once allocated workspace.
 * Same for FSTSemiFly(), FZTSemiFly() and InvFSTSemiFly() functions.
 */
void ConvOn2SphereSemiFly(double* rdata, double* idata, double* rfilter, double* ifilter, double* rres, double* ires,
                          const int bw, double* workspace) {
    int size = 2 * bw;

    double* frres = workspace;            // needs (bw * bw)
    double* fires = frres + (bw * bw);    // needs (bw * bw)
    double* trres = fires + (bw * bw);    // needs (bw * bw)
    double* tires = trres + (bw * bw);    // needs (bw * bw)
    double* filtrres = tires + (bw * bw); // needs bw
    double* filtires = filtrres + bw;     // needs bw
    double* scratchpad = filtires + bw;   // needs (10 * bw^2) + (24 * bw)

    double* weights = (double*)malloc(sizeof(double) * 4 * bw);
    GenerateWeightsForDLT(bw, weights);

    // Make DCT plans. Note that I will be using the GURU interface to execute these plans within the routines
    fftw_plan DCT_plan = fftw_plan_r2r_1d(size, weights, rdata, FFTW_REDFT10, FFTW_ESTIMATE);
    fftw_plan inv_DCT_plan = fftw_plan_r2r_1d(size, weights, rdata, FFTW_REDFT01, FFTW_ESTIMATE);

    // fftw "preamble"
    // Note that FFT plan places the output in a transposed array
    int rank = 1;
    //fftw_iodim dims[rank];
	fftw_iodim dims[1];
    dims[0].n = size;
    dims[0].is = 1;
    dims[0].os = size;

    int howmany_rank = 1;
    //fftw_iodim howmany_dims[howmany_rank];
	fftw_iodim howmany_dims[1];
    howmany_dims[0].n = size;
    howmany_dims[0].is = size;
    howmany_dims[0].os = 1;

    fftw_plan FFT_plan = fftw_plan_guru_split_dft(rank, dims, howmany_rank, howmany_dims, rdata, idata, workspace,
                                                  workspace + (4 * bw * bw), FFTW_ESTIMATE);

    // Note that FFT plans assumes that I'm working with a transposed array, e.g. the inputs for a length 2*bw transform
    // are placed every 2*bw apart, the output will be consecutive entries in the array

    rank = 1;
    dims[0].n = size;
    dims[0].is = size;
    dims[0].os = 1;

    howmany_rank = 1;
    howmany_dims[0].n = size;
    howmany_dims[0].is = 1;
    howmany_dims[0].os = size;

    fftw_plan inv_FFT_plan = fftw_plan_guru_split_dft(rank, dims, howmany_rank, howmany_dims, rdata, idata, workspace,
                                                      workspace + (4 * bw * bw), FFTW_ESTIMATE);

    FSTSemiFly(rdata, idata, frres, fires, bw, scratchpad, 1, bw, &DCT_plan, &FFT_plan, weights);
    FZTSemiFly(rfilter, ifilter, filtrres, filtires, bw, scratchpad, 1, &DCT_plan, weights);

    TransMult(frres, fires, filtrres, filtires, trres, tires, bw);

    InvFSTSemiFly(trres, tires, rres, ires, bw, scratchpad, 1, bw, &inv_DCT_plan, &inv_FFT_plan);

    free(weights);

    fftw_destroy_plan(inv_FFT_plan);
    fftw_destroy_plan(FFT_plan);
    fftw_destroy_plan(inv_DCT_plan);
    fftw_destroy_plan(DCT_plan);
}
