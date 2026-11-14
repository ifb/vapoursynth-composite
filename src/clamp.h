/*
 * Rounding and saturating integer clamps for the output quantizers.
 */

#ifndef COMP_CLAMP_H
#define COMP_CLAMP_H

#include <math.h>
#include <stdint.h>

/* Round to nearest even, as lrintf does under the default rounding mode,
 * for a result that fits in int32. gcc vectorizes neither lrintf (its
 * long result) nor, at baseline x86-64, (int32_t)rintf (no SSE2 vector
 * round); __builtin_irintf maps to cvtps2dq at every x86 level. gcc's
 * aarch64 backend vectorizes only the rintf form, as does clang. */
static inline int32_t round_i32(float v)
{
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
    return __builtin_irintf(v);
#else
    return (int32_t)rintf(v);
#endif
}

static inline int16_t clamp_i16(int32_t v)
{
    return (int16_t)(v < -32768 ? -32768 : v > 32767 ? 32767 : v);
}

#endif
