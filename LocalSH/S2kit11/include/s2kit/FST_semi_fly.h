#ifndef _FST_SEMI_FLY_H
#define _FST_SEMI_FLY_H

#include <fftw3.h>

#include "s2kit/util.h"

void FSTSemiFly(double*, double*, double*, double*, const int, double*, DataFormat, const int,
                fftw_plan*, fftw_plan*, double*);

void InvFSTSemiFly(double*, double*, double*, double*, const int, double*, DataFormat, const int,
                   fftw_plan*, fftw_plan*);

void FZTSemiFly(double*, double*, double*, double*, const int, double*, DataFormat, fftw_plan*, double*);

void ConvOn2SphereSemiFly(double*, double*, double*, double*, double*, double*, const int, double*);

#endif // _FST_SEMI_FLY_H
