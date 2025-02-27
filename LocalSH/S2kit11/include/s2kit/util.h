#ifndef _UTIL_H
#define _UTIL_H

// TODO move to source?
/**
 * @brief Result data format for FST functions.
 *
 * DataFormat is used in FST functions (e.g. FSTSemiMemo()) to determine samples' data format.
 */
typedef enum {
    COMPLEX = 0, /**< Data is complex. */
    REAL         /**< Data is real. */
} DataFormat; 

int IndexOfHarmonicCoeff(const int, const int, const int);

void TransMult(double*, double*, double*, double*, double*, double*, const int);

#endif // _UTIL_H
