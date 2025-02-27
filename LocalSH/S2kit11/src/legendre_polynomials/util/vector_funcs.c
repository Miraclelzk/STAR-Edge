/**
 * @file vector_funcs.c
 * @brief Vector arithmetic operations.
 */

#include "vector_funcs.h"

/**
 * @brief Adds two vectors into a third one.
 *
 * <tt>@b result = @b v1 + @b v2</tt>
 * 
 * @param v1 first vector
 * @param v2 second vector
 * @param result result vector
 * @param len length of all vectors
*/
void vec_add(double* v1, double* v2, double* result, const int len) {
    for (int i = 0; i < len; ++i)
        result[i] = v1[i] + v2[i];
}

/**
 * @brief Multiplies the vector @p <b>v</b> by @p scalar.
 *
 * <tt>@b result = scalar * @b v</tt>
 * 
 * @param scalar scalar
 * @param v vector
 * @param result result vector
 * @param len length of both vectors
*/
void vec_mul(const double scalar, double* v, double* result, const int len) {
    for (int i = 0; i < len; ++i)
        result[i] = scalar * v[i];
}

/* 
    Returns dot product of `v1` and `v2` in `result`

    Note: `result` and `v{1,2}` must be vectors of length `len`
*/
/**
 * @brief Performs dot product of @p <b>v1</b> and @p <b>v2</b>.
 *
 * <tt>@b result = @b v1 * @b v2</tt>
 * 
 * @param v1 first vector
 * @param v2 second vector
 * @param result result vector
 * @param len length of all vectors
*/
void vec_dot(double* v1, double* v2, double* result, const int len) {
    for (int i = 0; i < len; ++i)
        result[i] = v1[i] * v2[i];
}
