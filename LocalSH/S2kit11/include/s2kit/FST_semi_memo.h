#ifndef _FST_SEMI_MEMO_H
#define _FST_SEMI_MEMO_H

#include <fftw3.h>

#include "s2kit/util.h"

void FSTSemiMemo(double*, double*, double*, double*, const int, double**, double*, DataFormat, const int,
                   fftw_plan*, fftw_plan*, double*);

void InvFSTSemiMemo(double*, double*, double*, double*, const int, double**, double*, DataFormat, const int,
                      fftw_plan*, fftw_plan*);

void FZTSemiMemo(double*, double*, double*, double*, const int, double*, double*, DataFormat, fftw_plan*, double*);

void ConvOn2SphereSemiMemo(double*, double*, double*, double*, double*, double*, const int, double*);

#endif // _FST_SEMI_MEMO_H
