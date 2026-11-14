#ifndef COMP_ENCODE_H
#define COMP_ENCODE_H

#include <stddef.h>
#include <stdint.h>

#include "fir_row.h"
#include "subcarrier.h"

/* the wider of the two U/V low-pass lengths (PAL's 13; NTSC's is 9) */
#define COMP_UV_MAXTAPS 13

/* legal composite sample range, 10-bit levels in a 16-bit container
 * (value << 6) [EBU Tech 3280; SMPTE 244M] */
#define COMP_LEVEL_MIN 0x0100
#define COMP_LEVEL_MAX 0xFEFF

/* Modulate one row's chroma onto the carrier and add the luma, in one
 * vectorized fixed-point pass:
 *   cu = (uf[x]*ku + 16384) >> 15;  cv = (vf[x]*kv + 16384) >> 15
 *   dst[x] = clamp(luma[x] + ((cu*s4[x&3] + cv*c4[x&3] + 16384) >> 15),
 *                  COMP_LEVEL_MIN, COMP_LEVEL_MAX)
 * s4/c4 are the 4-sample carrier sign pattern (absolute phase, x = 0 at
 * the first sample). luma is the level-mapped Y (scalar). All arithmetic is int32 -- the intermediates
 * fit for the encoder's chroma range -- so the SIMD tiers are
 * byte-identical. */
typedef void (*comp_encode_mod_row_fn)(uint16_t *dst, const int32_t *luma,
                                       const int32_t *uf, const int32_t *vf,
                                       int32_t ku, int32_t kv,
                                       const int32_t *s4, const int32_t *c4,
                                       int w);
comp_encode_mod_row_fn comp_get_encode_mod_row_fn(unsigned cpu);

typedef struct comp_encode_t comp_encode_t;

struct comp_encode_t {
    int standard;
    int width;
    int den;
    int precomb;
    const int16_t *uv_taps;
    int uv_ntaps;
    int16_t uv_taps16[COMP_UV_MAXTAPS + 1]; /* uv_taps, zero-padded to even */
    comp_fir_row_s16_fn fir_row;
    comp_encode_mod_row_fn mod_row;
    int32_t ku, kv;              /* chroma level per Cb/Cr LSB, Q15 */
    int32_t level_black;
    int32_t luma_num, luma_den;  /* level = black + (y - 4096) * num/den */
    uint32_t luma_mul, luma_add; /* that divide as a multiply-shift */
    int16_t sin_q15[COMP_SC_DEN_PAL];
};

/* setup selects the NTSC 7.5 IRE pedestal (ignored for PAL); precomb
 * low-passes U/V vertically across same-field lines before modulation.
 * Returns 0 on success, nonzero if the standard is not supported. */
int comp_encode_init(comp_encode_t *e, int standard, int setup, int precomb);

/* modulate one line of YUV444P16 active picture into composite GRAY16
 * (without precombing; see comp_encode_frame) */
void comp_encode_line(const comp_encode_t *e, uint16_t *dst,
                      const uint16_t *srcy, const uint16_t *srcu,
                      const uint16_t *srcv, comp_sc_line_t sc);

/* modulate a whole frame, applying the vertical U/V precomb when
 * enabled; strides are in samples, rows sit at raster row_off */
void comp_encode_frame(const comp_encode_t *e, int frame, int rows, int row_off,
                       uint16_t *dst, ptrdiff_t dstride,
                       const uint16_t *srcy, ptrdiff_t ystride,
                       const uint16_t *srcu, ptrdiff_t ustride,
                       const uint16_t *srcv, ptrdiff_t vstride);

#endif
