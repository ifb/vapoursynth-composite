/*
 * PAL and NTSC composite encoding (Clarke, BBC RD 1986/2, sections 3.2
 * and 3.3).
 *
 * Fixed-point port of ld-chroma-encoder's PALEncoder/NTSCEncoder
 * encodeLine() for the active picture only: no syncs, burst, blanking,
 * or edge gating. NTSC uses the wideband-U/V mode. Floating point is
 * used only at init.
 */

#include <math.h>
#include <string.h>

#include "cpu.h"
#include "encode.h"
#include "osdep.h"
#include "fir_row.h"

/* 1.3 MHz low-pass Gaussian for U/V at PAL 4xfsc [Clarke 3.2.4: >= -3 dB
 * at 1.3 MHz, <= -20 dB at 4.0 MHz]: ld-chroma-encoder's 13 taps
 * (scipy.signal.gaussian(13, 1.52), normalized) rounded to Q15. The
 * rounded taps sum to exactly 32768, so flat chroma is preserved. */
static const int16_t uv_taps_pal[13] = {
    4, 38, 270, 1226, 3619, 6927, 8600, 6927, 3619, 1226, 270, 38, 4,
};

/* the same template at NTSC 4xfsc: Clarke's Table 1 nine taps as used by
 * ld-chroma-encoder, rounded to Q15 with the center tap adjusted by +2
 * so the taps sum to exactly 32768 */
static const int16_t uv_taps_ntsc[9] = {
    69, 626, 2959, 7563, 10334, 7563, 2959, 626, 69,
};

/* color-difference weights on full-excursion Cb/Cr
 * [Poynton eq 25.5/28.1 via ld-chroma encoder.h] */
#define KB 0.49211104112248356308804691718185
#define KR 0.87728321993817866838972487283129

int comp_encode_init(comp_encode_t *e, int standard, int setup, int precomb)
{
    e->precomb = !!precomb;

    int32_t level_white;

    if (standard == COMP_STD_PAL) {
        e->width = COMP_ACTIVE_WIDTH_PAL;
        e->den = COMP_SC_DEN_PAL;
        e->uv_taps = uv_taps_pal;
        e->uv_ntaps = 13;
        /* black/blanking and white [EBU Tech 3280] */
        e->level_black = 0x4000;
        level_white = 0xD300;
    } else if (standard == COMP_STD_NTSC) {
        e->width = COMP_ACTIVE_WIDTH_NTSC;
        e->den = COMP_SC_DEN_NTSC;
        e->uv_taps = uv_taps_ntsc;
        e->uv_ntaps = 9;
        /* blanking 240 and white 800, as 10-bit << 6 [SMPTE 244M]; the
         * optional 7.5 IRE setup pedestal (SMPTE 170M, 5.6 codes/IRE)
         * lifts black to 282 */
        e->level_black = setup ? 0x4680 : 0x3C00;
        level_white = 0xC800;
    } else {
        return -1;
    }

    e->standard = standard;

    /* luma level slope (white - black) / (219 << 8), reduced by the
     * common factor 256: 147/219 PAL, 140/219 NTSC, 259/438 with setup */
    e->luma_num = (level_white - e->level_black) / 256;
    e->luma_den = 219;
    if (e->luma_num * 256 != level_white - e->level_black) {
        e->luma_num = (level_white - e->level_black) / 128;
        e->luma_den = 438;
    }

    /* the slope's round-half-away divide as a multiply:
     * |q| = (|y - 4096| * mul + add) >> 31 with mul = ceil(num * 2^31 /
     * den), add = round(floor(den/2) * 2^31 / den). Exact for every
     * 16-bit input under the three slopes above. */
    const uint64_t num31 = (uint64_t)e->luma_num << 31;
    const uint64_t half31 = (uint64_t)(e->luma_den / 2) << 31;
    e->luma_mul = (uint32_t)(num31 / e->luma_den + (num31 % e->luma_den != 0));
    e->luma_add = (uint32_t)(half31 / e->luma_den
                             + (half31 % e->luma_den * 2 >= (uint64_t)e->luma_den));

    /* chroma level per full-excursion Cb/Cr LSB, Q15 */
    const double span = level_white - e->level_black;
    e->ku = (int32_t)lrint((1.0 - 0.114) * KB / (112.0 * 256.0) * span * 32768.0);
    e->kv = (int32_t)lrint((1.0 - 0.299) * KR / (112.0 * 256.0) * span * 32768.0);

    comp_sc_sin_q15(e->den, e->sin_q15);

    /* positive taps summing to 32768 hold the int16 FIR's sums within
     * 2^30 */
    for (int j = 0; j < e->uv_ntaps; j++)
        e->uv_taps16[j] = e->uv_taps[j];
    e->uv_taps16[e->uv_ntaps] = 0;
    e->fir_row = comp_get_fir_row_s16_fn(comp_cpu_detect());
    e->mod_row = comp_get_encode_mod_row_fn(comp_cpu_detect());
    return 0;
}

static void encode_mod_row_c(uint16_t *dst, const int32_t *luma,
                             const int32_t *uf, const int32_t *vf,
                             int32_t ku, int32_t kv,
                             const int32_t *s4, const int32_t *c4, int w)
{
    for (int x = 0; x < w; x++) {
        const int32_t cu = (uf[x] * ku + 16384) >> 15;
        const int32_t cv = (vf[x] * kv + 16384) >> 15;
        const int32_t chroma = (cu * s4[x & 3] + cv * c4[x & 3] + 16384) >> 15;
        const int32_t out = luma[x] + chroma;
        dst[x] = (uint16_t)(out < COMP_LEVEL_MIN ? COMP_LEVEL_MIN :
                            out > COMP_LEVEL_MAX ? COMP_LEVEL_MAX : out);
    }
}

#if defined(__x86_64__)
#define ENCODE_MOD_ROW_ASM(isa)                                             \
    void comp_encode_mod_row_##isa(uint16_t *dst, const int32_t *luma,     \
                                   const int32_t *uf, const int32_t *vf,    \
                                   int32_t ku, int32_t kv,                  \
                                   const int32_t *s4, const int32_t *c4,    \
                                   int w)
ENCODE_MOD_ROW_ASM(sse4);
ENCODE_MOD_ROW_ASM(avx2);
ENCODE_MOD_ROW_ASM(avx512);
#endif

comp_encode_mod_row_fn comp_get_encode_mod_row_fn(unsigned cpu)
{
    cpu |= COMP_CPU_BUILTIN;
#if defined(__x86_64__)
    if (cpu & COMP_CPU_AVX512)
        return comp_encode_mod_row_avx512;
    if (cpu & COMP_CPU_AVX2)
        return comp_encode_mod_row_avx2;
    if (cpu & COMP_CPU_SSE41)
        return comp_encode_mod_row_sse4;
#endif
    (void)cpu;
    return encode_mod_row_c;
}

void comp_encode_line(const comp_encode_t *e, uint16_t *dst,
                      const uint16_t *srcy, const uint16_t *srcu,
                      const uint16_t *srcv, comp_sc_line_t sc)
{
    const int w = e->width;
    const int ntaps = e->uv_ntaps;
    const int span = COMP_FIR_S16_SPAN(w, ntaps);
    int32_t uf[COMP_FIR_S16_ALIGN(COMP_ACTIVE_WIDTH_PAL)];
    int32_t vf[COMP_FIR_S16_ALIGN(COMP_ACTIVE_WIDTH_PAL)];
    /* staged FIR input: in[i] = src[i - ntaps/2] - 32768 for i-ntaps/2
     * in [0, w), 0 elsewhere -- centring the row and zeroing the wings
     * reproduces the C bounds check (blanking carries no chroma). The
     * offset lands exactly on int16. */
    _Static_assert(COMP_UV_MAXTAPS + 1 <= COMP_FIR_S16_MAXTAPS, "fir_row_s16 tap count");
    DECLARE_ALIGNED( int16_t us[COMP_FIR_S16_SPAN(COMP_ACTIVE_WIDTH_PAL, COMP_UV_MAXTAPS)], 64 );
    DECLARE_ALIGNED( int16_t vs[COMP_FIR_S16_SPAN(COMP_ACTIVE_WIDTH_PAL, COMP_UV_MAXTAPS)], 64 );
    const int lo = ntaps / 2;

    memset(us, 0, sizeof(int16_t) * span);
    memset(vs, 0, sizeof(int16_t) * span);
    for (int k = 0; k < w; k++) {
        us[k + lo] = (int16_t)(srcu[k] - 32768);
        vs[k + lo] = (int16_t)(srcv[k] - 32768);
    }
    e->fir_row(uf, us, e->uv_taps16, ntaps, w, 0);
    e->fir_row(vf, vs, e->uv_taps16, ntaps, w, 0);

    /* at 4xfsc the carrier advances exactly 90 degrees per sample, so a
     * line needs only its start-phase sine and cosine, cycled through a
     * 4-sample sign/swap pattern; the V-switch (always +1 for NTSC)
     * folds into the cosine */
    const int32_t sn = e->sin_q15[sc.phase];
    const int32_t cs = e->sin_q15[(sc.phase + e->den / 4) % e->den];
    const int32_t s4[4] = { sn, cs, -sn, -cs };
    const int32_t c4[4] = { sc.vswitch * cs, -sc.vswitch * sn,
                            -sc.vswitch * cs, sc.vswitch * sn };

    /* the luma level-map stays scalar; chroma = U sin(a) + V cos(a) Vsw
     * [Clarke 3.1/3.3] plus the add and clamp are the vectorized pass */
    int32_t luma[COMP_ACTIVE_WIDTH_PAL];
    for (int x = 0; x < w; x++) {
        const int32_t yl = srcy[x] - 4096;
        const uint32_t ay = yl < 0 ? (uint32_t)-yl : (uint32_t)yl;
        const int32_t q = (int32_t)(((uint64_t)ay * e->luma_mul + e->luma_add) >> 31);
        luma[x] = e->level_black + (yl < 0 ? -q : q);
    }
    e->mod_row(dst, luma, uf, vf, e->ku, e->kv, s4, c4, w);
}

void comp_encode_frame(const comp_encode_t *e, int frame, int rows, int row_off,
                       uint16_t *dst, ptrdiff_t dstride,
                       const uint16_t *srcy, ptrdiff_t ystride,
                       const uint16_t *srcu, ptrdiff_t ustride,
                       const uint16_t *srcv, ptrdiff_t vstride)
{
    const int w = e->width;

    for (int r = 0; r < rows; r++) {
        const comp_sc_line_t sc = comp_sc_line(e->standard, frame, r + row_off);

        if (!e->precomb) {
            comp_encode_line(e, dst + r * dstride, srcy + r * ystride,
                             srcu + r * ustride, srcv + r * vstride, sc);
            continue;
        }

        /* vertical chroma pre-filter [1 2 1]/4 across the same field's
         * neighboring lines (rows ±2), the lines the decoder combs;
         * edges reuse the center row */
        const int ra = r - 2 >= 0 ? r - 2 : r;
        const int rb = r + 2 < rows ? r + 2 : r;
        const uint16_t *ua = srcu + ra * ustride, *u0 = srcu + r * ustride, *ub = srcu + rb * ustride;
        const uint16_t *va = srcv + ra * vstride, *v0 = srcv + r * vstride, *vb = srcv + rb * vstride;
        uint16_t uf[COMP_ACTIVE_WIDTH_PAL], vf[COMP_ACTIVE_WIDTH_PAL];

        for (int x = 0; x < w; x++) {
            uf[x] = (uint16_t)((ua[x] + 2 * u0[x] + ub[x] + 2) >> 2);
            vf[x] = (uint16_t)((va[x] + 2 * v0[x] + vb[x] + 2) >> 2);
        }
        comp_encode_line(e, dst + r * dstride, srcy + r * ystride, uf, vf, sc);
    }
}
