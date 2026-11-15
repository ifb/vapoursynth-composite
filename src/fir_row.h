/*
 * Q15 FIR row kernels: 32-bit samples for the decoder's chroma
 * filters, 16-bit for the encoder's chroma low-pass.
 */

#ifndef COMP_FIR_ROW_H
#define COMP_FIR_ROW_H

#include <stdint.h>

/* Round a row length up to the SIMD write bound: FIR-row tiers may
 * write out and read in up to this many samples, so row buffers are
 * sized with it. */
#define COMP_FIR_ROW_ALIGN(w) (((w) + 15) & ~15)

/* One Q15 FIR pass over a zero-padded row:
 * out[x] = (int32_t)((sum_j (int64_t)coef[j] * in[x + j] + 16384) >> 15)
 * for 0 <= x < w. in holds w + taps - 1 elements (the caller centers
 * its data and zeroes the wings, reproducing the C bounds check).
 * SIMD tiers write out and read in through COMP_FIR_ROW_ALIGN(w)
 * samples, so out has that capacity and in that plus taps - 1; the
 * padding lanes of out are scratch. Accumulator contract:
 * |acc| < 2^45 for any x in the padded span. The taps must be odd in
 * number and symmetric (coef[j] == coef[taps - 1 - j]), and |in| < 2^30
 * so a mirrored pair's sum fits int32: each pair is multiplied once. */
typedef void (*comp_fir_row_q15_fn)(int32_t *out, const int32_t *in,
                                    const int32_t *coef, int taps, int w);

comp_fir_row_q15_fn comp_get_fir_row_q15_fn(unsigned cpu);

/* The same Q15 FIR over int16 samples and taps, accumulating in int32:
 * out[x] = (sum_j coef[j] * in[x + j] + 16384) >> 15 for 0 <= x < w.
 * Taps pair up, so coef holds taps rounded up to even with a zero pad,
 * and the caller guarantees sum|coef| * max|in| + 2^14 < 2^31 (any
 * int16 row when sum|coef| < 2^16, a bounded one otherwise). With wide
 * set the bound is looser: the pairs (coef[2i], coef[2i+1]) alternate
 * between two sums, each of which must fit int32 on its own, and
 * sum|coef| * max|in| + 2^14 < 2^32; the two are joined exactly without
 * being added. SIMD tiers
 * write out through COMP_FIR_S16_ALIGN(w) and read in through
 * COMP_FIR_S16_SPAN(w, taps), the wings and padding being the caller's
 * zeros. The span is a whole block of 64 samples past the last one
 * whatever the tap count, as the zmm tier loads its blocks whole, and
 * for the same reason taps is at most COMP_FIR_S16_MAXTAPS; in is best
 * 64-byte aligned. */
#define COMP_FIR_S16_MAXTAPS 32
#define COMP_FIR_S16_ALIGN(w) (((w) + 31) & ~31)
#define COMP_FIR_S16_SPAN(w, taps) (COMP_FIR_S16_ALIGN(w) + COMP_FIR_S16_MAXTAPS)

typedef void (*comp_fir_row_s16_fn)(int32_t *out, const int16_t *in,
                                    const int16_t *coef, int taps, int w,
                                    int wide);

comp_fir_row_s16_fn comp_get_fir_row_s16_fn(unsigned cpu);

#endif
