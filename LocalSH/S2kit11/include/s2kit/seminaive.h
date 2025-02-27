#ifndef _SEMINAIVE_H
#define _SEMINAIVE_H

#include <fftw3.h>

void DLTSemi(double*, const int, const int, double*, double*, double*, double*, fftw_plan*);

void InvDLTSemi(double*, const int, const int, double*, double*, double*, double*, fftw_plan*);

#endif // _SEMINAIVE_H
