/*
 * Composite decoding.
 *
 * PAL: 2D Transform PAL chroma separation followed by PALcolour-style
 * demodulation (ports of ld-chroma-decoder's transformpal2d.cpp and
 * palcolour.cpp transform path).
 *
 * NTSC: 2D line comb separation followed by product demodulation and
 * the color low-pass (port of comb.cpp split1D/split2D/filterIQ, with
 * luma reconstructed from resynthesized filtered chroma as in adjustY).
 *
 * The references recover line phase from the color burst or field
 * metadata; this raster carries neither, and both ends of the round
 * trip share the same deterministic subcarrier sequence, so the decoder
 * substitutes the exact values the encoder used. Separation heuristics
 * run in float; demodulation, filtering, and level mapping are fixed
 * point.
 */

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "clamp.h"
#include "cpu.h"
#include "decode.h"
#include "encode.h"

/* This file is compiled with -ffp-contract=off (see meson.build): the
 * float separation heuristics feed discrete choices -- ntsc_comb2d's
 * output drives split3d's candidate compare -- so a contracted a*b+c*d
 * (single-rounded FMA) here would pick different candidates than the
 * SIMD tiers and than a non-FMA build, changing the 3D output. ISO C
 * modes happen to default the contraction off, but the flag pins it so
 * the reference does not depend on -std surviving (the STDC FP_CONTRACT
 * pragma does not override an explicit -ffp-contract=fast on gcc 13).
 * D044/D047; contrast the terminal float kernels where FMA is a
 * welcome accuracy gain. */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FS COMP_DECODE_FILTER_SIZE
#define WIDTH COMP_ACTIVE_WIDTH_PAL

/* all-zero chroma used for lines beyond the edges */
static const int16_t zero_line[WIDTH];
static const float zero_line_f[WIDTH];

#define COLORLP_TAPS COMP_DEMOD_PQ_TAPS
static const int16_t colorlp_q15[COLORLP_TAPS] = { COMP_COLORLP_COEFS };

/* response of a symmetric FIR (Q15 taps) at normalized angular freq w */
static double fir_response_q15(const int16_t *taps, int n, double w)
{
    double sum = 0.0;
    for (int k = 0; k < n; k++)
        sum += taps[k] / 32768.0 * cos(w * (k - (n - 1) / 2.0));
    return sum;
}

/* Design the chroma cascade equalizer: the inverse of the known
 * encode-filter x decode-filter chroma response, boost capped at 12 dB,
 * Hann-windowed to COMP_EQ_TAPS taps with an exactly unity cascade at
 * DC. dec_taps is the decoder's post-demod profile in 1/32768 units. */
static void design_eq(comp_decode_t *d, const int16_t *enc_taps, int enc_n,
                      const double *dec_taps, int dec_n)
{
    enum { M = 256 };
    double heq[M / 2 + 1];
    double h[COMP_EQ_TAPS];
    const int c = COMP_EQ_TAPS / 2;

    for (int m = 0; m <= M / 2; m++) {
        const double w = 2.0 * M_PI * m / M;
        double dec = 0.0;
        for (int k = 0; k < dec_n; k++)
            dec += dec_taps[k] * cos(w * (k - (dec_n - 1) / 2.0));
        const double g = fir_response_q15(enc_taps, enc_n, w) * dec;
        heq[m] = g > 0.25 ? 1.0 / g : 4.0;
        if (heq[m] > 4.0)
            heq[m] = 4.0;
        if (heq[m] < 0.0)
            heq[m] = 0.0;
    }

    double dc = 0.0;
    for (int j = 0; j < COMP_EQ_TAPS; j++) {
        double sum = heq[0];
        for (int m = 1; m < M / 2; m++)
            sum += 2.0 * heq[m] * cos(2.0 * M_PI * m * (j - c) / M);
        sum += heq[M / 2] * cos(M_PI * (j - c));
        const double hann = 0.5 + 0.5 * cos(M_PI * (j - c) / (c + 1));
        h[j] = sum / M * hann;
        dc += h[j];
    }
    for (int j = 0; j < COMP_EQ_TAPS; j++)
        d->eq_q15[j] = (int32_t)lrint(h[j] * (heq[0] / dc) * 32768.0);
}

/* the eq=2 blend's sub-nominal low end: ~600 kHz Hann-windowed sinc
 * on the demodulated U/V, Q15, DC-exact */
static void design_narrow(comp_decode_t *d, double fs_hz)
{
    const double cut = 600000.0 / fs_hz;
    const int c = COMP_NARROW_TAPS / 2;
    double h[COMP_NARROW_TAPS];
    double sum = 0.0;

    for (int j = 0; j < COMP_NARROW_TAPS; j++) {
        const double k = j - c;
        const double sinc = k == 0.0 ? 2.0 * cut
                                     : sin(2.0 * M_PI * cut * k) / (M_PI * k);
        h[j] = sinc * (0.5 + 0.5 * cos(M_PI * k / (c + 1)));
        sum += h[j];
    }
    int32_t total = 0;
    for (int j = 0; j < COMP_NARROW_TAPS; j++) {
        d->narrow_q15[j] = (int32_t)lrint(h[j] / sum * 32768.0);
        total += d->narrow_q15[j];
    }
    d->narrow_q15[c] += 32768 - total;
}

/* stage a row into a zero-winged buffer and run the dispatched FIR;
 * the wings reproduce the former per-sample bounds check exactly */
static void fir_row_staged(const comp_decode_t *d, const int32_t *coef,
                           int taps, const int32_t *in, int32_t *out, int w)
{
    int32_t in_p[COMP_FIR_ROW_ALIGN(COMP_ACTIVE_WIDTH_PAL) + COMP_EQ_TAPS - 1];
    int32_t out_p[COMP_FIR_ROW_ALIGN(COMP_ACTIVE_WIDTH_PAL)];

    memset(in_p, 0, sizeof(in_p));
    memcpy(in_p + taps / 2, in, sizeof(*in) * w);
    d->fir_row(out_p, in_p, coef, taps, w);
    memcpy(out, out_p, sizeof(*out) * w);
}

static void narrow_row(const comp_decode_t *d, const int32_t *in, int32_t *out, int w)
{
    fir_row_staged(d, d->narrow_q15, COMP_NARROW_TAPS, in, out, w);
}

int comp_decode_init(comp_decode_t *d, int standard, double threshold,
                     int nscratch, int setup, int dimensions, int eq, int refine,
                     int use_transform, int level, double evidence, int cti)
{
    if (nscratch < 1 || dimensions < 1 || dimensions > 3 || refine < 0)
        return -1;
    if (cti && dimensions < 2)
        return -1;
    if (evidence < 0.0 || (evidence > 0.0
                           && (standard != COMP_STD_PAL || dimensions < 2)))
        return -1;
    if (use_transform < 0 || use_transform > 2)
        return -1;
    if (use_transform && (standard != COMP_STD_NTSC || dimensions != 3))
        return -1;
    if (level && (dimensions < 2 || (standard == COMP_STD_NTSC && !use_transform)))
        return -1;
    if (eq < 0 || eq > 2)
        return -1;
    if (eq == 2 && !(standard == COMP_STD_PAL ? dimensions >= 2 : use_transform))
        return -1;

    memset(d, 0, sizeof(*d));

    /* levels and chroma scales must match the encoder exactly; the
     * encoder is also the resynthesis step of the refine loop */
    comp_encode_t enc;
    if (comp_encode_init(&enc, standard, setup, 0))
        return -1;
    d->enc = enc;
    d->refine = refine;
    d->standard = standard;
    d->dimensions = dimensions;
    d->use_transform = use_transform;
    d->eq = eq;
    d->cti = !!cti;
    d->pal_demod = comp_get_pal_demod_fn(comp_cpu_detect());
    d->fir_row = comp_get_fir_row_q15_fn(comp_cpu_detect());
    d->split3d_row = comp_get_split3d_row_fn(comp_cpu_detect());
    d->comb2d_row = comp_get_ntsc_comb2d_row_fn(comp_cpu_detect());
    if (eq == 2)
        design_narrow(d, standard == COMP_STD_PAL ? 4.0 * 4433618.75
                                                  : 4.0 * 3579545.0);
    d->width = enc.width;
    d->height = standard == COMP_STD_PAL ? COMP_ACTIVE_HEIGHT_PAL
                                         : COMP_ACTIVE_HEIGHT_NTSC;
    d->den = enc.den;
    d->ku = enc.ku;
    d->kv = enc.kv;
    d->level_black = enc.level_black;
    d->luma_num = enc.luma_den;  /* the inverse slope */
    d->luma_den = enc.luma_num;
    d->demod_quant_row = comp_get_demod_quant_row_fn(comp_cpu_detect());
    d->demod_rotate_row = comp_get_demod_rotate_row_fn(comp_cpu_detect());
    d->demod_pq_row = comp_get_demod_pq_row_fn(comp_cpu_detect());
    comp_demod_pq_init(d->demod_pq_coef, colorlp_q15);
    comp_magic_init(&d->mag_y, d->luma_num, d->luma_den, 4096);
    comp_magic_init(&d->mag_u, 32768, d->ku, 32768);
    comp_magic_init(&d->mag_v, 32768, d->kv, 32768);
    memcpy(d->sin_q15, enc.sin_q15, sizeof(d->sin_q15));

    if (standard == COMP_STD_PAL && dimensions > 1) {
        /* PALcolour's 2D chroma filter: raised-cosine, ~1.18 MHz,
         * horizontal taps 0..FS mirrored, vertical taps at 0/±2/±1/±3
         * field lines (in that array order). Double at init, Q16 use. */
        const double fs_hz = 4.0 * 4433618.75;
        const double bw_hz = 1100000.0 / 0.93;
        const double ca = 0.5 * fs_hz / bw_hz;
        double cfilt[FS + 1][4];
        double cdiv = 0.0;

        for (int f = 0; f <= FS; f++) {
            const double fc   = fmin(ca, (double)f);
            const double ff   = fmin(ca, sqrt((double)f * f + 2 * 2));
            const double fff  = fmin(ca, sqrt((double)f * f + 4 * 4));
            const double ffff = fmin(ca, sqrt((double)f * f + 6 * 6));
            const int div = (f == 0) ? 2 : 1;

            cfilt[f][0] = (1.0 + cos(M_PI * fc   / ca)) / div;
            cfilt[f][2] = (1.0 + cos(M_PI * ff   / ca)) / div;
            cfilt[f][1] = (1.0 + cos(M_PI * fff  / ca)) / div;
            cfilt[f][3] = (1.0 + cos(M_PI * ffff / ca)) / div;

            cdiv += 2 * (cfilt[f][0] + 2 * cfilt[f][2] + 2 * cfilt[f][1] + 2 * cfilt[f][3]);
        }
        int32_t cfilt_q16[FS + 1][4];
        for (int f = 0; f <= FS; f++)
            for (int k = 0; k < 4; k++)
                cfilt_q16[f][k] = (int32_t)lrint(cfilt[f][k] / cdiv * 65536.0);
        comp_pal_demod_init(d->pal_demod_coef, (const int32_t (*)[4])cfilt_q16);

        if (eq) {
            /* the filter's horizontal profile: for vertically flat
             * chroma the U path reduces to these mirrored taps */
            double prof[2 * FS + 1];
            for (int f = -FS; f <= FS; f++) {
                const int a = f < 0 ? -f : f;
                prof[FS + f] = (cfilt[a][0] + 2 * cfilt[a][1]
                              + 2 * cfilt[a][2] + 2 * cfilt[a][3]) / cdiv
                             * (f == 0 ? 2.0 : 1.0);
            }
            design_eq(d, enc.uv_taps, enc.uv_ntaps, prof, 2 * FS + 1);
        }

        if (dimensions == 2
            ? comp_transform2d_init(&d->transform, threshold, level, evidence)
            : comp_transform3d_init(&d->transform3, threshold, standard, level,
                                    evidence))
            return -1;
    } else if (standard == COMP_STD_PAL) {
        /* dimensions=1 uses the crude path only */
        if (eq) {
            double prof[COLORLP_TAPS];
            for (int k = 0; k < COLORLP_TAPS; k++)
                prof[k] = colorlp_q15[k] / 32768.0;
            design_eq(d, enc.uv_taps, enc.uv_ntaps, prof, COLORLP_TAPS);
        }
    } else {
        /* the comb's adaptivity range: 45 IRE of the encoded span */
        d->comb_krange = (int32_t)lrint(45.0 * (0xC800 - d->level_black) / 100.0);

        if (use_transform && comp_transform3d_init(&d->transform3, threshold,
                                                   standard, level, 0.0))
            return -1;

        if (eq) {
            double prof[COLORLP_TAPS];
            for (int k = 0; k < COLORLP_TAPS; k++)
                prof[k] = colorlp_q15[k] / 32768.0;
            design_eq(d, enc.uv_taps, enc.uv_ntaps, prof, COLORLP_TAPS);
        }
    }

    if (dimensions == 3 && (standard == COMP_STD_PAL || use_transform)) {
        /* slab cache sized so concurrent in-flight frames (each drawing
         * on two slabs, consecutive frames sharing them) do not evict
         * each other */
        if (comp_t3d_cache_init(&d->t3cache, nscratch / 2 + 3,
                                d->width, d->height / 2, eq == 2)) {
            comp_transform3d_free(&d->transform3);
            return -1;
        }
    }

    d->scratch = calloc(nscratch, sizeof(*d->scratch));
    if (!d->scratch) {
        comp_transform2d_free(&d->transform);
        comp_t3d_cache_free(&d->t3cache);
        return -1;
    }
    d->nscratch = nscratch;
    for (int i = 0; i < nscratch; i++) {
        d->scratch[i].chroma_f = malloc(sizeof(float) * d->width * d->height);
        d->scratch[i].chroma = malloc(sizeof(int16_t) * d->width * d->height);
        if (!d->scratch[i].chroma_f || !d->scratch[i].chroma) {
            comp_decode_free(d);
            return -1;
        }
        if (standard == COMP_STD_NTSC && dimensions > 1) {
            d->scratch[i].tmp3 = malloc(sizeof(float) * d->width * d->height
                                        * (dimensions == 3 ? 6 : 1));
            if (!d->scratch[i].tmp3) {
                comp_decode_free(d);
                return -1;
            }
        }
        if (eq == 2) {
            d->scratch[i].conf = malloc(sizeof(float) * d->width * d->height);
            if (!d->scratch[i].conf) {
                comp_decode_free(d);
                return -1;
            }
        }
        if (use_transform == 2) {
            d->scratch[i].chroma2 = malloc(sizeof(int16_t) * d->width * d->height);
            d->scratch[i].mask = malloc(d->width * d->height);
            if (!d->scratch[i].chroma2 || !d->scratch[i].mask) {
                comp_decode_free(d);
                return -1;
            }
        }
        if (refine > 0) {
            /* YUV estimate (3 planes), recomposite, crude Y, spare */
            d->scratch[i].refine = malloc(sizeof(uint16_t) * d->width * d->height * 6);
            if (!d->scratch[i].refine) {
                comp_decode_free(d);
                return -1;
            }
        }
    }
    pthread_mutex_init(&d->lock, NULL);
    pthread_cond_init(&d->cond, NULL);
    return 0;
}

void comp_decode_free(comp_decode_t *d)
{
    if (d->scratch) {
        for (int i = 0; i < d->nscratch; i++) {
            free(d->scratch[i].chroma_f);
            free(d->scratch[i].chroma);
            free(d->scratch[i].conf);
            free(d->scratch[i].chroma2);
            free(d->scratch[i].mask);
            free(d->scratch[i].tmp3);
            free(d->scratch[i].refine);
        }
        free(d->scratch);
        d->scratch = NULL;
        pthread_mutex_destroy(&d->lock);
        pthread_cond_destroy(&d->cond);
    }
    if (d->standard == COMP_STD_PAL || d->use_transform) {
        if (d->dimensions == 2) {
            comp_transform2d_free(&d->transform);
        } else if (d->dimensions == 3) {
            comp_transform3d_free(&d->transform3);
            comp_t3d_cache_free(&d->t3cache);
        }
    }
}

int comp_decode_set_thresholds(comp_decode_t *d, const double *t, int n)
{
    if (d->standard != COMP_STD_PAL && !d->use_transform)
        return -1;
    if (d->dimensions == 2 && n == COMP_T2D_NTHRESH) {
        for (int i = 0; i < n; i++)
            d->transform.threshold_sq[i] = (float)(t[i] * t[i]);
        return 0;
    }
    if (d->dimensions == 3
        && n == (d->standard == COMP_STD_PAL ? COMP_T3D_NTHRESH_PAL
                                             : COMP_T3D_NTHRESH)) {
        /* for NTSC these feed the shaped-threshold exponent */
        for (int i = 0; i < n; i++)
            d->transform3.threshold_sq[i] = (float)(t[i] * t[i]);
        return 0;
    }
    return -1;
}

int comp_decode_set_lut(comp_decode_t *d, const double *v, int n)
{
    if (d->standard != COMP_STD_PAL && !d->use_transform)
        return -1;
    if (d->standard == COMP_STD_PAL && d->dimensions == 2
        && n == COMP_T2D_NTHRESH * COMP_LUT_K) {
        comp_transform2d_set_lut(&d->transform, v);
        return 0;
    }
    if (d->dimensions == 3
        && n == (d->standard == COMP_STD_PAL ? COMP_T3D_NTHRESH_PAL
                                             : COMP_T3D_NTHRESH) * COMP_LUT_K) {
        comp_transform3d_set_lut(&d->transform3, v);
        return 0;
    }
    return -1;
}

int comp_decode_look(const comp_decode_t *d)
{
    if (d->dimensions != 3)
        return 0;
    return (d->standard == COMP_STD_PAL || d->use_transform) ? COMP_T3D_LOOK : 1;
}

static comp_decode_scratch_t *scratch_acquire(comp_decode_t *d)
{
    pthread_mutex_lock(&d->lock);
    for (;;) {
        for (int i = 0; i < d->nscratch; i++) {
            if (!d->scratch[i].used) {
                d->scratch[i].used = 1;
                pthread_mutex_unlock(&d->lock);
                return &d->scratch[i];
            }
        }
        pthread_cond_wait(&d->cond, &d->lock);
    }
}

static void scratch_release(comp_decode_t *d, comp_decode_scratch_t *s)
{
    pthread_mutex_lock(&d->lock);
    s->used = 0;
    pthread_cond_signal(&d->cond);
    pthread_mutex_unlock(&d->lock);
}

/* equalize one row of demodulated chroma levels in place */
static void eq_row(const comp_decode_t *d, const int32_t *in, int32_t *out, int w)
{
    fir_row_staged(d, d->eq_q15, COMP_EQ_TAPS, in, out, w);
}

/* Luma-guided chroma transient improvement: at a coincident
 * transition, resynthesize the chroma edge from the decoded luma's
 * transition profile (normalized between its side levels), so the
 * chroma edge lands where the luma edge is and rises as fast. One
 * profile drives both U and V (hue cannot rotate across the edge) and
 * the output is contained between the chroma side levels (cannot
 * overshoot). Flat chroma and chroma bumps without a step fail the
 * gates and pass untouched. */
#define CTI_EDGE_T   3000   /* luma gradient over 2 samples */
#define CTI_STEP_T   3000   /* minimum chroma step, demod level units */
#define CTI_D        8      /* chroma blur half-width at 4x fsc */

static void cti_row(const uint16_t *outy, int32_t *u, int32_t *v, int w)
{
    int x = CTI_D + 3;
    while (x < w - CTI_D - 3) {
        const int32_t gy = (int32_t)outy[x + 1] - (int32_t)outy[x - 1];
        if (gy < CTI_EDGE_T && gy > -CTI_EDGE_T) {
            x++;
            continue;
        }
        /* extend over the contiguous edge run */
        int x1 = x;
        while (x1 + 1 < w - CTI_D - 3) {
            const int32_t g2 = (int32_t)outy[x1 + 2] - (int32_t)outy[x1];
            if (g2 < CTI_EDGE_T && g2 > -CTI_EDGE_T)
                break;
            x1++;
        }
        const int lo = x - CTI_D, hi = x1 + CTI_D;
        if (hi >= w - 3)
            break;

        /* side levels: short averages just beyond the window */
        const int32_t ya = ((int32_t)outy[lo - 2] + outy[lo - 1] + outy[lo]) / 3;
        const int32_t yb = ((int32_t)outy[hi] + outy[hi + 1] + outy[hi + 2]) / 3;
        const int32_t yd = yb - ya;
        if (yd > 4096 || yd < -4096) {
            for (int p = 0; p < 2; p++) {
                int32_t *c = p ? v : u;
                const int32_t a = (c[lo - 2] + c[lo - 1] + c[lo]) / 3;
                const int32_t b = (c[hi] + c[hi + 1] + c[hi + 2]) / 3;
                if (b - a > CTI_STEP_T || a - b > CTI_STEP_T) {
                    for (int k = lo; k <= hi; k++) {
                        int64_t sn = (int64_t)((int32_t)outy[k] - ya) * 32768 / yd;
                        sn = sn < 0 ? 0 : sn > 32768 ? 32768 : sn;
                        c[k] = a + (int32_t)(((int64_t)(b - a) * sn) >> 15);
                    }
                }
            }
        }
        x = hi + 1;
    }
}

static inline uint16_t clamp_u16(int32_t v)
{
    return (uint16_t)(v < 0 ? 0 : v > 65535 ? 65535 : v);
}

/* coefficient row of tap b, side (0: x + b, 1: x - b) and plane k;
 * b = 0 has one position */
#define PAL_ROW(b, side, k) ((b) ? 4 + ((b) - 1) * 8 + (side) * 4 + (k) : (k))

void comp_pal_demod_init(int16_t coef[COMP_PAL_DEMOD_ROWS][32],
                         const int32_t cfilt[COMP_DECODE_FILTER_SIZE + 1][4])
{
    static const int sign[4] = { 1, 1, -1, -1 };  /* sn + cs */
    /* each plane's lines as the reference combines them: line,
     * -(-2 + +2), -1 - +1, -(-3) + +3 */
    static const int vlo[4] = { 1, -1, 1, -1 };
    static const int vhi[4] = { 0, -1, -1, 1 };
    for (int b = 0; b <= FS; b++) {
        for (int side = 0; side < (b ? 2 : 1); side++) {
            for (int k = 0; k < 4; k++) {
                /* at b = 0 the mirrored pair m[x+b] + m[x-b] is 2 m[x] */
                const int32_t c = cfilt[b][k] * (b ? 1 : 2);
                int16_t *row = coef[PAL_ROW(b, side, k)];
                for (int l = 0; l < 16; l++) {
                    const int s = sign[(l + (side ? 4 * FS - b : b)) & 3];
                    row[2 * l] = (int16_t)(c * s * vlo[k]);
                    row[2 * l + 1] = (int16_t)(c * s * vhi[k]);
                }
            }
        }
    }
}

/* Demodulate one PAL field. comp and chroma are field views; the output
 * pointers are frame planes, written at rows 2*fieldline + field.
 * conf (a field view at chroma_stride, may be NULL) is the transform's
 * chroma-confidence map, which scales the eq=2 boost per sample. */
/* C reference for the dispatched demod kernel; see decode.h */
static void pal_demod_row_c(int32_t *u, int32_t *v, const int16_t *pr,
                            ptrdiff_t stride, const int16_t (*coef)[32],
                            int w, int32_t bp, int32_t bq, int32_t vswitch)
{
    enum { CHUNK = 128 };

    for (int x0 = 0; x0 < w; x0 += CHUNK) {
        const int n = w - x0 < CHUNK ? w - x0 : CHUNK;
        /* [c0/c1 or c2/c3 planes][tap parity][x]; rows outer and x
         * inner in steps of the four carrier phases keep each row's
         * coefficients invariant across the inner loop */
        int32_t acc[2][2][CHUNK];
        memset(acc, 0, sizeof(acc));

        for (int b = 0; b <= FS; b++) {
            for (int side = 0; side < (b ? 2 : 1); side++) {
                for (int k = 0; k < 4; k++) {
                    const int16_t *c = coef[PAL_ROW(b, side, k)];
                    const int16_t *pk = pr + 2 * (k * stride + FS + x0 + (side ? -b : b));
                    int32_t *a = acc[k >= 2][b & 1];
                    for (int x = 0; x < n; x += 4)
                        for (int l = 0; l < 4; l++)
                            a[x + l] += c[2 * l] * pk[2 * (x + l)]
                                      + c[2 * l + 1] * pk[2 * (x + l) + 1];
                }
            }
        }

        for (int x = 0; x < n; x++) {
            /* a tap lands on a sine sample when x + b is odd */
            const int sn = !(x & 1);
            const int32_t a_sn = acc[0][sn][x], a_cs = acc[0][!sn][x];
            const int32_t b_sn = acc[1][sn][x], b_cs = acc[1][!sn][x];

            /* Q16 filter -> Q0 products (at half chroma amplitude) */
            const int32_t pu0 = (a_sn + b_cs + 32768) >> 16;
            const int32_t qu0 = (a_cs - b_sn + 32768) >> 16;
            const int32_t pv0 = (a_sn - b_cs + 32768) >> 16;
            const int32_t qv0 = (a_cs + b_sn + 32768) >> 16;

            /* rotate onto the U/V axes and double (the filter recovers
             * chroma at half amplitude); bp/bq are Q15, so >> 14 */
            u[x0 + x] = (int32_t)(-((int64_t)pu0 * bp + (int64_t)qu0 * bq + 8192) >> 14);
            v[x0 + x] = vswitch *
                (int32_t)(-((int64_t)qv0 * bp - (int64_t)pv0 * bq + 8192) >> 14);
        }
    }
}

#if defined(__x86_64__)
#define PAL_DEMOD_ROW_ASM(isa)                                              \
    void comp_pal_demod_row_##isa(int32_t *u, int32_t *v,                   \
                                  const int16_t *pr, ptrdiff_t stride,      \
                                  const int16_t (*coef)[32],                \
                                  int w, int32_t bp, int32_t bq,            \
                                  int32_t vswitch)
PAL_DEMOD_ROW_ASM(sse4);
PAL_DEMOD_ROW_ASM(avx2);
PAL_DEMOD_ROW_ASM(avx512);
#endif

comp_pal_demod_fn comp_get_pal_demod_fn(unsigned cpu)
{
#if defined(__x86_64__)
    if (cpu & COMP_CPU_AVX512)
        return comp_pal_demod_row_avx512;
    if (cpu & COMP_CPU_AVX2)
        return comp_pal_demod_row_avx2;
    if (cpu & COMP_CPU_SSE41)
        return comp_pal_demod_row_sse4;
#endif
    (void)cpu;
    return pal_demod_row_c;
}

static void pal_decode_field(const comp_decode_t *d, int frame, int field,
                             const uint16_t *comp, ptrdiff_t comp_stride,
                             const int16_t *chroma, ptrdiff_t chroma_stride,
                             const float *conf, comp_conf_acc_t *acc,
                             uint16_t *dsty, ptrdiff_t ystride,
                             uint16_t *dstu, ptrdiff_t ustride,
                             uint16_t *dstv, ptrdiff_t vstride)
{
    const int w = d->width;
    const int rows = d->height / 2;

    /* the vertical tap groups as word pairs of the chroma lines:
     * (line, 0), (-2, +2), (-1, +1), (-3, +3) field lines. FS zero
     * pairs each side give the filter its reach without bounds checks;
     * only [FS, FS + w) is rewritten per row. */
    enum { PR_STRIDE = COMP_PAL_DEMOD_ALIGN(WIDTH) + 2 * FS };
    int16_t pr[4][2 * PR_STRIDE];

    memset(pr, 0, sizeof(pr));
    for (int fr = 0; fr < rows; fr++) {
        const int16_t *in0 = chroma + fr * chroma_stride;
        const int16_t *in1 = fr - 1 >= 0   ? chroma + (fr - 1) * chroma_stride : zero_line;
        const int16_t *in2 = fr + 1 < rows ? chroma + (fr + 1) * chroma_stride : zero_line;
        const int16_t *in3 = fr - 2 >= 0   ? chroma + (fr - 2) * chroma_stride : zero_line;
        const int16_t *in4 = fr + 2 < rows ? chroma + (fr + 2) * chroma_stride : zero_line;
        const int16_t *in5 = fr - 3 >= 0   ? chroma + (fr - 3) * chroma_stride : zero_line;
        const int16_t *in6 = fr + 3 < rows ? chroma + (fr + 3) * chroma_stride : zero_line;

        for (int x = 0; x < w; x++) {
            const int i = 2 * (FS + x);
            pr[0][i] = in0[x];
            pr[1][i] = in3[x];
            pr[1][i + 1] = in4[x];
            pr[2][i] = in1[x];
            pr[2][i + 1] = in2[x];
            pr[3][i] = in5[x];
            pr[3][i + 1] = in6[x];
        }

        /* line phase: (bp,bq) is the unit vector the reference derives
         * from the burst (at -U); here it is exact */
        const int row = fr * 2 + field;
        const comp_sc_line_t sc = comp_sc_line(d->standard, frame, row);
        const int32_t bp = -d->sin_q15[(sc.phase + d->den / 4) % d->den];
        const int32_t bq = -d->sin_q15[sc.phase];

        const uint16_t *comp_row = comp + fr * comp_stride;
        uint16_t *outy = dsty + row * ystride;
        uint16_t *outu = dstu + row * ustride;
        uint16_t *outv = dstv + row * vstride;
        int32_t u_row[COMP_PAL_DEMOD_ALIGN(WIDTH)], v_row[COMP_PAL_DEMOD_ALIGN(WIDTH)];
        int32_t u_eq[WIDTH], v_eq[WIDTH];

        d->pal_demod(u_row, v_row, &pr[0][0], PR_STRIDE, d->pal_demod_coef,
                     w, bp, bq, sc.vswitch);

        /* luma is the composite minus the separated chroma */
        int32_t yl_row[WIDTH];
        for (int x = 0; x < w; x++)
            yl_row[x] = (int32_t)comp_row[x] - in0[x] - d->level_black;
        d->demod_quant_row(outy, yl_row, &d->mag_y, w);

        int32_t *u_out = u_row, *v_out = v_row;
        if (d->eq) {
            eq_row(d, u_row, u_eq, w);
            eq_row(d, v_row, v_eq, w);
            if (d->eq == 2 && conf) {
                /* the confidence steers total chroma bandwidth: the
                 * blend runs from a sub-nominal low-pass (marginal or
                 * absent pairs: residual leak narrowed away) up to the
                 * boosted cascade inverse (symmetric pairs, ~0.94 mean
                 * ratio; leak measures 0.3-0.8, and the fourth power
                 * widens the gap). */
                int32_t u_nar[WIDTH], v_nar[WIDTH];
                narrow_row(d, u_row, u_nar, w);
                narrow_row(d, v_row, v_nar, w);
                const float *cw = conf + fr * chroma_stride;
                for (int x = 0; x < w; x++) {
                    const float c2 = cw[x] * cw[x];
                    int32_t wq = (int32_t)lrintf(c2 * c2 * 32768.0f);
                    wq = wq < 0 ? 0 : wq > 32768 ? 32768 : wq;
                    u_eq[x] = u_nar[x]
                        + (int32_t)(((int64_t)wq * (u_eq[x] - u_nar[x])) >> 15);
                    v_eq[x] = v_nar[x]
                        + (int32_t)(((int64_t)wq * (v_eq[x] - v_nar[x])) >> 15);
                }
                if (acc) {
                    uint16_t *om = acc->out_mask
                        ? acc->out_mask + row * acc->mask_stride : NULL;
                    for (int x = 0; x < w; x++) {
                        acc->sum += cw[x];
                        acc->sumsq += (double)cw[x] * cw[x];
                        if (om) {
                            float c = cw[x] < 0.0f ? 0.0f : cw[x] > 1.0f ? 1.0f : cw[x];
                            om[x] = (uint16_t)lrintf((1.0f - c) * 65535.0f);
                        }
                    }
                    acc->n += w;
                }
            }
            u_out = u_eq;
            v_out = v_eq;
        }

        if (d->cti)
            cti_row(outy, u_out, v_out, w);

        /* invert the encoder's level mappings */
        d->demod_quant_row(outu, u_out, &d->mag_u, w);
        d->demod_quant_row(outv, v_out, &d->mag_v, w);
    }
}

/* NTSC 1D bandpass [-0.25, 0, 0.5, 0, -0.25] centered on fsc
 * (comb.cpp split1D) */
static void ntsc_comb1d(const comp_decode_t *d, int rows,
                        const uint16_t *comp, ptrdiff_t comp_stride, float *c1)
{
    const int w = d->width;

    for (int r = 0; r < rows; r++) {
        const uint16_t *line = comp + r * comp_stride;
        float *out = c1 + r * w;
        out[0] = out[1] = out[w - 2] = out[w - 1] = 0.0f;
        for (int x = 2; x < w - 2; x++)
            out[x] = (2.0f * line[x] - line[x - 2] - line[x + 2]) / 4.0f;
    }
}

/* NTSC 3-line adaptive comb on the 1D chroma (comb.cpp split2D): blend
 * the differences against the lines ±2 frame rows away (the same
 * field's neighboring lines, 180 degrees out of chroma phase),
 * weighted by similarity. */
/* C reference for the dispatched NTSC 2D comb row kernel; see decode.h.
 * Writes out[x] for x in [1, w). Compiled with -ffp-contract=off so the
 * SIMD tiers can be bit-exact (this output feeds split3d's compare). */
static void ntsc_comb2d_row_c(float *out, const float *cur, const float *prev,
                              const float *next, const float *krangep, int w)
{
    const float krange = *krangep;
    for (int x = 1; x < w; x++) {
        float kp, kn;

        kp  = fabsf(fabsf(cur[x]) - fabsf(prev[x]));
        kp += fabsf(fabsf(cur[x - 1]) - fabsf(prev[x - 1]));
        kp -= (fabsf(cur[x]) + fabsf(prev[x - 1])) * 0.10f;
        kn  = fabsf(fabsf(cur[x]) - fabsf(next[x]));
        kn += fabsf(fabsf(cur[x - 1]) - fabsf(next[x - 1]));
        kn -= (fabsf(cur[x]) + fabsf(next[x - 1])) * 0.10f;

        kp = 1.0f - kp / krange;
        kp = kp < 0.0f ? 0.0f : kp > 1.0f ? 1.0f : kp;
        kn = 1.0f - kn / krange;
        kn = kn < 0.0f ? 0.0f : kn > 1.0f ? 1.0f : kn;

        float sc = 1.0f;
        if (kn > 0.0f || kp > 0.0f) {
            if (kn > 3.0f * kp)
                kp = 0.0f;
            else if (kp > 3.0f * kn)
                kn = 0.0f;
            sc = 2.0f / (kn + kp);
            if (sc < 1.0f)
                sc = 1.0f;
        } else if (fabsf(fabsf(prev[x]) - fabsf(next[x]))
                   - fabsf((next[x] + prev[x]) * 0.2f) <= 0.0f) {
            kn = kp = 1.0f;
        }

        out[x] = ((cur[x] - prev[x]) * kp * sc
                + (cur[x] - next[x]) * kn * sc) / 4.0f;
    }
}

#if defined(__x86_64__)
#define NTSC_COMB2D_ROW_ASM(isa)                                            \
    void comp_ntsc_comb2d_row_##isa(float *out, const float *cur,           \
                                    const float *prev, const float *next,   \
                                    const float *krange, int w)
NTSC_COMB2D_ROW_ASM(sse4);
NTSC_COMB2D_ROW_ASM(avx2);
NTSC_COMB2D_ROW_ASM(avx512);
#endif

comp_ntsc_comb2d_row_fn comp_get_ntsc_comb2d_row_fn(unsigned cpu)
{
#if defined(__x86_64__)
    if (cpu & COMP_CPU_AVX512)
        return comp_ntsc_comb2d_row_avx512;
    if (cpu & COMP_CPU_AVX2)
        return comp_ntsc_comb2d_row_avx2;
    if (cpu & COMP_CPU_SSE41)
        return comp_ntsc_comb2d_row_sse4;
#endif
    (void)cpu;
    return ntsc_comb2d_row_c;
}

static void ntsc_comb2d(const comp_decode_t *d, int rows, const float *c1, float *c2)
{
    const int w = d->width;
    const float krange = (float)d->comb_krange;

    for (int r = 0; r < rows; r++) {
        const float *cur = c1 + r * w;
        const float *prev = r - 2 >= 0   ? c1 + (r - 2) * w : zero_line_f;
        const float *next = r + 2 < rows ? c1 + (r + 2) * w : zero_line_f;
        float *out = c2 + r * w;

        out[0] = 0.0f;
        d->comb2d_row(out, cur, prev, next, &krange, w);
    }
}


/* Build the magic reciprocal for clamp_u16(base + rdiv(in * k, den)).
 * rdiv is round-half-away, = sign(n) * floor((|n|*k + floor(den/2)) /
 * den); the ratio k/den is magicked directly onto |n| at a fixed
 * COMP_MAGIC_SHIFT (see D048). den and k are small positive constants,
 * so ceil/round in int64 is exact. */
void comp_magic_init(comp_magic_t *m, int32_t k, int32_t den, int32_t base)
{
    const int s = COMP_MAGIC_SHIFT;
    const int64_t c = den / 2;
    const uint64_t mul = ((uint64_t)k << s) / (uint64_t)den
                       + (((uint64_t)k << s) % (uint64_t)den != 0);  /* ceil */
    const uint64_t add = ((uint64_t)c << s) / (uint64_t)den
                       + ((((uint64_t)c << s) % (uint64_t)den) * 2
                          >= (uint64_t)den);                          /* round */
    /* mul < 2^32 for every standard/setup; the asm loads it as a u32 */
    m->mul = (uint32_t)mul;
    m->add = (uint32_t)add;
    m->base = base;
}

/* C reference for the dispatched output-quantize kernel; see decode.h.
 * Terminal (writes a frame plane), so exactness is required only where
 * clamp_u16 does not saturate -- the magic and rdiv agree there. */
static void demod_quant_row_c(uint16_t *out, const int32_t *in,
                              const comp_magic_t *mag, int w)
{
    for (int x = 0; x < w; x++) {
        const int32_t n = in[x];
        const uint32_t an = n < 0 ? (uint32_t)(-(int64_t)n) : (uint32_t)n;
        const uint32_t q = (uint32_t)(((uint64_t)an * mag->mul + mag->add)
                                      >> COMP_MAGIC_SHIFT);
        const int32_t v = mag->base + (n < 0 ? -(int32_t)q : (int32_t)q);
        out[x] = clamp_u16(v);
    }
}

#if defined(__x86_64__)
#define DEMOD_QUANT_ROW_ASM(isa)                                            \
    void comp_demod_quant_row_##isa(uint16_t *out, const int32_t *in,       \
                                    const comp_magic_t *mag, int w)
DEMOD_QUANT_ROW_ASM(sse4);
DEMOD_QUANT_ROW_ASM(avx2);
DEMOD_QUANT_ROW_ASM(avx512);

/* the asm indexes the magic struct with these offsets */
_Static_assert(sizeof(comp_magic_t) == 12, "magic layout");
_Static_assert(offsetof(comp_magic_t, mul) == 0, "magic layout");
_Static_assert(offsetof(comp_magic_t, add) == 4, "magic layout");
_Static_assert(offsetof(comp_magic_t, base) == 8, "magic layout");
#endif

comp_demod_quant_row_fn comp_get_demod_quant_row_fn(unsigned cpu)
{
#if defined(__x86_64__)
    if (cpu & COMP_CPU_AVX512)
        return comp_demod_quant_row_avx512;
    if (cpu & COMP_CPU_AVX2)
        return comp_demod_quant_row_avx2;
    if (cpu & COMP_CPU_SSE41)
        return comp_demod_quant_row_sse4;
#endif
    (void)cpu;
    return demod_quant_row_c;
}

/* C reference for the dispatched NTSC chroma rotation; see decode.h. */
static void demod_rotate_row_c(int32_t *u, int32_t *v, const int32_t *p,
                               const int32_t *q, const int32_t *bpq, int w)
{
    const int32_t bp = bpq[0], bq = bpq[1];
    for (int x = 0; x < w; x++) {
        u[x] = (int32_t)(-((int64_t)p[x] * bp + (int64_t)q[x] * bq + 8192) >> 14);
        v[x] = (int32_t)(-((int64_t)q[x] * bp - (int64_t)p[x] * bq + 8192) >> 14);
    }
}

#if defined(__x86_64__)
#define DEMOD_ROTATE_ROW_ASM(isa)                                           \
    void comp_demod_rotate_row_##isa(int32_t *u, int32_t *v,                \
                                     const int32_t *p, const int32_t *q,    \
                                     const int32_t *bpq, int w)
DEMOD_ROTATE_ROW_ASM(sse4);
DEMOD_ROTATE_ROW_ASM(avx2);
DEMOD_ROTATE_ROW_ASM(avx512);
#endif

comp_demod_rotate_row_fn comp_get_demod_rotate_row_fn(unsigned cpu)
{
#if defined(__x86_64__)
    if (cpu & COMP_CPU_AVX512)
        return comp_demod_rotate_row_avx512;
    if (cpu & COMP_CPU_AVX2)
        return comp_demod_rotate_row_avx2;
    if (cpu & COMP_CPU_SSE41)
        return comp_demod_rotate_row_sse4;
#endif
    (void)cpu;
    return demod_rotate_row_c;
}

/* the pairs' first taps: even j, then odd j */
static const int demod_pq_j[COMP_DEMOD_PQ_PAIRS] = { 0, 4, 8, 12, 16, 1, 5, 9, 13 };

void comp_demod_pq_init(int16_t coef[COMP_DEMOD_PQ_PAIRS][32],
                        const int16_t *taps)
{
    static const int sign[4] = { 1, 1, -1, -1 };  /* sn + cs */
    for (int i = 0; i < COMP_DEMOD_PQ_PAIRS; i++) {
        const int j = demod_pq_j[i];
        for (int l = 0; l < 16; l++) {
            const int ph = l & 3;
            coef[i][2 * l] = (int16_t)(taps[j] * sign[(ph + j) & 3]);
            coef[i][2 * l + 1] = j + 2 < COMP_DEMOD_PQ_TAPS
                ? (int16_t)(taps[j + 2] * sign[(ph + j + 2) & 3]) : 0;
        }
    }
}

/* C reference for the dispatched demod products; see decode.h */
static void demod_pq_row_c(int32_t *p, int32_t *q, const int16_t *s,
                           const int16_t (*coef)[32], int w)
{
    enum { CHUNK = 128 };

    for (int x0 = 0; x0 < w; x0 += CHUNK) {
        const int n = w - x0 < CHUNK ? w - x0 : CHUNK;
        /* [even-j or odd-j taps][x]; pairs outer and x inner in steps of
         * the four carrier phases keep each pair's coefficients
         * invariant across the inner loop */
        int32_t acc[2][CHUNK];
        memset(acc, 0, sizeof(acc));

        for (int i = 0; i < COMP_DEMOD_PQ_PAIRS; i++) {
            const int j = demod_pq_j[i];
            const int16_t *c = coef[i];
            const int16_t *sx = s + x0 + j;
            int32_t *a = acc[j & 1];
            for (int x = 0; x < n; x += 4)
                for (int l = 0; l < 4; l++)
                    a[x + l] += c[2 * l] * sx[x + l] + c[2 * l + 1] * sx[x + l + 2];
        }

        /* a tap feeds p when x + j is odd; chunks start even, and the
         * pair may run one sample into the 32-aligned padding */
        for (int x = 0; x < n; x += 2) {
            p[x0 + x] = (acc[1][x] + 16384) >> 15;
            q[x0 + x] = (acc[0][x] + 16384) >> 15;
            p[x0 + x + 1] = (acc[0][x + 1] + 16384) >> 15;
            q[x0 + x + 1] = (acc[1][x + 1] + 16384) >> 15;
        }
    }
}

#if defined(__x86_64__)
#define DEMOD_PQ_ROW_ASM(isa)                                               \
    void comp_demod_pq_row_##isa(int32_t *p, int32_t *q, const int16_t *s,  \
                                 const int16_t (*coef)[32], int w)
DEMOD_PQ_ROW_ASM(sse4);
DEMOD_PQ_ROW_ASM(avx2);
DEMOD_PQ_ROW_ASM(avx512);
#endif

comp_demod_pq_row_fn comp_get_demod_pq_row_fn(unsigned cpu)
{
#if defined(__x86_64__)
    if (cpu & COMP_CPU_AVX512)
        return comp_demod_pq_row_avx512;
    if (cpu & COMP_CPU_AVX2)
        return comp_demod_pq_row_avx2;
    if (cpu & COMP_CPU_SSE41)
        return comp_demod_pq_row_sse4;
#endif
    (void)cpu;
    return demod_pq_row_c;
}

/* line phase parity, matching the reference's getLinePhase: the NTSC
 * subcarrier advances half a cycle per broadcast line, so the parity of
 * the line count within the color sequence selects one of the two
 * phase classes */
static int ntsc_line_phase(int frame, int raster_row)
{
    const int frame_line = 39 + raster_row;
    const int field_id = ((frame % 2) * 2 + (frame_line & 1)) % 4;
    const int prev_lines = (field_id / 2) * 525 + (field_id % 2) * 263 + frame_line / 2;
    return prev_lines & 1;
}

/* NTSC adaptive 3D comb (comb.cpp split3D/getBestCandidate): pick the
 * most similar sample that should be 180 degrees out of chroma phase,
 * from this line, neighboring lines, or the neighboring fields and
 * frames, and comb against it; if a same-frame candidate wins, reuse
 * the 2D result. Buffers are indexed 0/1/2 = previous/current/next. */
/* C reference for the dispatched candidate-selection kernel; see
 * decode.h. Identical to the reference's per-pixel loop with the
 * row-constant bounds and phase tests already folded into the
 * have_sample/have_penalty flags. */
static void split3d_row_c(int16_t *out, const float *c1c, const float *c2c,
                          const uint16_t *ref, const comp_split3d_cand_t *cand,
                          const double *irescale, int w)
{
    enum { CHUNK = 64 };

    /* candidates outer, samples inner: the have_sample/have_penalty
     * cases are per candidate, so each runs as its own branch-free
     * sample loop, and the running best per sample keeps the scan order
     * (the first candidate, then strictly smaller penalties) */
    for (int x0 = 3; x0 < w - 3; x0 += CHUNK) {
        const int n = w - 3 - x0 < CHUNK ? w - 3 - x0 : CHUNK;
        double best_penalty[CHUNK];
        float best_sample[CHUNK];
        uint8_t best_same[CHUNK];  /* best < 4 */

        for (int i = 0; i < 8; i++) {
            double penalty[CHUNK];
            float sample[CHUNK];
            const int off = x0 + cand[i].off;

            if (!cand[i].have_sample) {
                for (int x = 0; x < n; x++) {
                    penalty[x] = 1000.0;
                    sample[x] = 0.0f;
                }
            } else if (!cand[i].have_penalty) {
                for (int x = 0; x < n; x++) {
                    penalty[x] = 1000.0;
                    sample[x] = cand[i].c1[off + x];
                }
            } else {
                const float *c1 = cand[i].c1 + off;
                const float *c2 = cand[i].c2 + off;
                const uint16_t *line = cand[i].line + off;
                const double bonus = cand[i].bonus;
                for (int x = 0; x < n; x++) {
                    double ypen = 0.0, iqpen = 0.0;
                    for (int o = -1; o <= 1; o++) {
                        const double ref_c = c2c[x0 + x + o];
                        const double cand_c = c2[x + o];
                        ypen += fabs((ref[x0 + x + o] - ref_c)
                                     - (line[x + o] - cand_c));
                        iqpen += fabs(ref_c + cand_c) * (o ? 0.5 : 1.0);
                    }
                    penalty[x] = ypen / 3.0 / *irescale
                               + (iqpen / 2.0 / *irescale) * 0.28
                               + bonus;
                    sample[x] = c1[x];
                }
            }

            if (i == 0) {
                for (int x = 0; x < n; x++) {
                    best_penalty[x] = penalty[x];
                    best_sample[x] = sample[x];
                    best_same[x] = 1;
                }
                continue;
            }
            /* unconditional stores through selects, so the update
             * compiles without branches on the penalty compare */
            for (int x = 0; x < n; x++) {
                const int take = penalty[x] < best_penalty[x];
                best_penalty[x] = take ? penalty[x] : best_penalty[x];
                best_sample[x] = take ? sample[x] : best_sample[x];
                best_same[x] = take ? i < 4 : best_same[x];
            }
        }

        for (int x = 0; x < n; x++) {
            const float tc = best_same[x] ? c2c[x0 + x]
                                          : (c1c[x0 + x] - best_sample[x]) / 2.0f;
            out[x0 + x] = clamp_i16(lrintf(tc));
        }
    }
}

#if defined(__x86_64__)
#define SPLIT3D_ROW_ASM(isa)                                                \
    void comp_split3d_row_##isa(int16_t *out, const float *c1c,             \
                                const float *c2c, const uint16_t *ref,      \
                                const comp_split3d_cand_t *cand,            \
                                const double *irescale, int w)
SPLIT3D_ROW_ASM(sse4);
SPLIT3D_ROW_ASM(avx2);
SPLIT3D_ROW_ASM(avx512);

/* the asm indexes the candidate table with these offsets */
_Static_assert(sizeof(comp_split3d_cand_t) == 48, "split3d cand layout");
_Static_assert(offsetof(comp_split3d_cand_t, off) == 24, "split3d cand layout");
_Static_assert(offsetof(comp_split3d_cand_t, bonus) == 40, "split3d cand layout");
#endif

comp_split3d_row_fn comp_get_split3d_row_fn(unsigned cpu)
{
#if defined(__x86_64__)
    if (cpu & COMP_CPU_AVX512)
        return comp_split3d_row_avx512;
    if (cpu & COMP_CPU_AVX2)
        return comp_split3d_row_avx2;
    if (cpu & COMP_CPU_SSE41)
        return comp_split3d_row_sse4;
#endif
    (void)cpu;
    return split3d_row_c;
}

static void ntsc_split3d(const comp_decode_t *d, int rows, int row_off,
                         const comp_frame_view_t *views, const int *view_frames,
                         const float *const c1[3], const float *const c2[3],
                         int16_t *chroma)
{
    const int w = d->width;
    const double irescale = d->comb_krange / 45.0;
    /* bias bonuses prefer frame over field over line candidates
     * (comb.cpp adaptThreshold = 1.0, chromaWeight = 1.0) */
    static const double LINE_BONUS = -2.0, FIELD_BONUS = -4.0, FRAME_BONUS = -6.0;

    for (int r = 0; r < rows; r++) {
        const int cur_lp = ntsc_line_phase(view_frames[1], r + row_off);
        const float *c1c = c1[1] + r * w;
        const float *c2c = c2[1] + r * w;
        const uint16_t *ref = views[1].data + r * views[1].stride;
        int16_t *out = chroma + r * w;

        /* candidate table: buffer, line, sample offset, bonus; the
         * first four are the same-frame (1D/2D) candidates */
        struct { int k, cr, off; double bonus; } ct[8] = {
            { 1, r, -2, 0.0 },
            { 1, r, +2, 0.0 },
            { 1, r - 2, 0, LINE_BONUS },
            { 1, r + 2, 0, LINE_BONUS },
            { 1, r - 1, 0, FIELD_BONUS },
            { 1, r + 1, 0, FIELD_BONUS },
            { 0, r, 0, FRAME_BONUS },
            { 2, r, 0, FRAME_BONUS },
        };
        /* adjacent-field candidates come from this frame or the
         * neighboring one, whichever gives the matching phase */
        if (cur_lp == ntsc_line_phase(view_frames[1], r + row_off - 1))
            ct[4].k = 0;
        else
            ct[5].k = 2;

        comp_split3d_cand_t cand[8];
        for (int i = 0; i < 8; i++) {
            const int k = ct[i].k, cr = ct[i].cr, off = ct[i].off;
            const int inside = cr >= 0 && cr < rows;
            const int scr = inside ? cr : r;
            /* the reference's per-pixel phase test reduces to a row
             * constant: want - have = 2 + 2*cur_lp - 2*lp_k - off
             * (mod 4), the x parity cancelling */
            const int lpk = ntsc_line_phase(view_frames[k], scr + row_off);
            const int phase_ok =
                (((2 + 2 * cur_lp - 2 * lpk - off) % 4) + 4) % 4 == 0;
            cand[i].c1 = c1[k] + scr * w;
            cand[i].c2 = c2[k] + scr * w;
            cand[i].line = views[k].data + scr * views[k].stride;
            cand[i].off = off;
            cand[i].have_sample = inside;
            cand[i].have_penalty = inside && phase_ok;
            cand[i].bonus = ct[i].bonus;
        }

        d->split3d_row(out, c1c, c2c, ref, cand, &irescale, w);

        for (int x = 0; x < 3; x++)
            out[x] = clamp_i16(lrintf(c2c[x]));
        for (int x = w - 3; x < w; x++)
            out[x] = clamp_i16(lrintf(c2c[x]));
    }
}

/* Demodulate one NTSC line: product demod against the trivial 4xfsc
 * carriers, the reference's color low-pass, rotation onto U/V, and
 * luma as composite minus the resynthesized filtered chroma. conf
 * (may be NULL) is the transform's confidence row for the eq=2 blend. */
static void ntsc_demod_line(const comp_decode_t *d, int frame, int raster_row,
                            int out_row,
                            const uint16_t *comp_row, const int16_t *chroma_row,
                            const float *conf, comp_conf_acc_t *acc,
                            const uint8_t *mask,
                            uint16_t *outy, uint16_t *outu, uint16_t *outv)
{
    const int w = d->width;
    int16_t s[COMP_DEMOD_PQ_SPAN(COMP_ACTIVE_WIDTH_NTSC)];
    int32_t p_row[COMP_DEMOD_PQ_ALIGN(COMP_ACTIVE_WIDTH_NTSC)];
    int32_t q_row[COMP_DEMOD_PQ_ALIGN(COMP_ACTIVE_WIDTH_NTSC)];

    memset(s, 0, sizeof(s));
    memcpy(s + COMP_DEMOD_PQ_WING, chroma_row, sizeof(*chroma_row) * w);

    const comp_sc_line_t sc = comp_sc_line(d->standard, frame, raster_row);
    const int32_t sn0 = d->sin_q15[sc.phase];
    const int32_t cs0 = d->sin_q15[(sc.phase + d->den / 4) % d->den];
    const int32_t bpq[2] = { -cs0, -sn0 };  /* bp, bq */
    const int32_t s4[4] = { sn0, cs0, -sn0, -cs0 };
    const int32_t c4[4] = { cs0, -sn0, -cs0, sn0 };

    int32_t u_row[COMP_ACTIVE_WIDTH_NTSC], v_row[COMP_ACTIVE_WIDTH_NTSC];
    int32_t u_eq[COMP_ACTIVE_WIDTH_NTSC], v_eq[COMP_ACTIVE_WIDTH_NTSC];

    d->demod_pq_row(p_row, q_row, s, d->demod_pq_coef, w);

    d->demod_rotate_row(u_row, v_row, p_row, q_row, bpq, w);

    /* comb mode follows comb.cpp adjustY (subtract the filtered chroma
     * resynthesized on the carrier, using the values before
     * equalization); transform mode subtracts the separated chroma
     * directly, as Transform PAL does. The hybrid picks per sample from
     * the source that produced the chroma. The mode test is
     * row-invariant, so each variant gets its own branch-free
     * (autovectorizable) loop; the hybrid folds its per-sample pick
     * into an and-mask select. */
    int32_t yl_row[COMP_ACTIVE_WIDTH_NTSC];
    if (d->use_transform == 1) {
        for (int x = 0; x < w; x++)
            yl_row[x] = (int32_t)comp_row[x] - chroma_row[x];
    } else {
        int32_t re[COMP_ACTIVE_WIDTH_NTSC];
        for (int x = 0; x < w; x++)
            re[x] = (u_row[x] * s4[x & 3] + v_row[x] * c4[x & 3] + 16384) >> 15;
        if (d->use_transform == 2) {
            for (int x = 0; x < w; x++) {
                const int32_t dif = re[x] - chroma_row[x];
                yl_row[x] = (int32_t)comp_row[x] - chroma_row[x]
                          - (dif & -(int32_t)(mask[x] != 0));
            }
        } else {
            for (int x = 0; x < w; x++)
                yl_row[x] = (int32_t)comp_row[x] - re[x];
        }
    }
    for (int x = 0; x < w; x++)
        yl_row[x] -= d->level_black;
    d->demod_quant_row(outy, yl_row, &d->mag_y, w);

    int32_t *u_out = u_row, *v_out = v_row;
    if (d->eq) {
        eq_row(d, u_row, u_eq, w);
        eq_row(d, v_row, v_eq, w);
        if (d->eq == 2 && conf) {
            /* the PAL blend: confidence steers bandwidth from the
             * sub-nominal low-pass up to the boosted cascade inverse */
            int32_t u_nar[COMP_ACTIVE_WIDTH_NTSC], v_nar[COMP_ACTIVE_WIDTH_NTSC];
            narrow_row(d, u_row, u_nar, w);
            narrow_row(d, v_row, v_nar, w);
            if (acc) {
                uint16_t *om = acc->out_mask
                    ? acc->out_mask + out_row * acc->mask_stride : NULL;
                for (int x = 0; x < w; x++) {
                    acc->sum += conf[x];
                    acc->sumsq += (double)conf[x] * conf[x];
                    if (om) {
                        float c = conf[x] < 0.0f ? 0.0f : conf[x] > 1.0f ? 1.0f : conf[x];
                        om[x] = (uint16_t)lrintf((1.0f - c) * 65535.0f);
                    }
                }
                acc->n += w;
            }
            for (int x = 0; x < w; x++) {
                const float c2 = conf[x] * conf[x];
                int32_t wq = (int32_t)lrintf(c2 * c2 * 32768.0f);
                wq = wq < 0 ? 0 : wq > 32768 ? 32768 : wq;
                u_eq[x] = u_nar[x]
                    + (int32_t)(((int64_t)wq * (u_eq[x] - u_nar[x])) >> 15);
                v_eq[x] = v_nar[x]
                    + (int32_t)(((int64_t)wq * (v_eq[x] - v_nar[x])) >> 15);
            }
        }
        u_out = u_eq;
        v_out = v_eq;
    }

    if (d->cti)
        cti_row(outy, u_out, v_out, w);

    d->demod_quant_row(outu, u_out, &d->mag_u, w);
    d->demod_quant_row(outv, v_out, &d->mag_v, w);
}


/* Crude 1D notch decode (a classic cheap decoder, and the degradation
 * model of the refine loop): bandpass chroma estimate, product demod
 * with the color low-pass, luma = composite minus the estimate. */
static void crude_decode_frame(const comp_decode_t *d, int frame, int rows,
                               int row_off, int use_eq,
                               const uint16_t *comp, ptrdiff_t comp_stride,
                               uint16_t *dsty, ptrdiff_t ystride,
                               uint16_t *dstu, ptrdiff_t ustride,
                               uint16_t *dstv, ptrdiff_t vstride)
{
    const int w = d->width;
    int16_t s[COMP_DEMOD_PQ_SPAN(COMP_ACTIVE_WIDTH_PAL)];
    int16_t *est = s + COMP_DEMOD_PQ_WING;

    memset(s, 0, sizeof(s));
    for (int r = 0; r < rows; r++) {
        const uint16_t *line = comp + r * comp_stride;
        int32_t p_row[COMP_DEMOD_PQ_ALIGN(COMP_ACTIVE_WIDTH_PAL)];
        int32_t q_row[COMP_DEMOD_PQ_ALIGN(COMP_ACTIVE_WIDTH_PAL)];
        int32_t u_row[COMP_ACTIVE_WIDTH_PAL], v_row[COMP_ACTIVE_WIDTH_PAL];
        int32_t u_eq[COMP_ACTIVE_WIDTH_PAL], v_eq[COMP_ACTIVE_WIDTH_PAL];

        /* the numerator spans +-131070 over 16-bit samples, so the
         * truncating quarter stays within +-32767 */
        est[0] = est[1] = est[w - 2] = est[w - 1] = 0;
        for (int x = 2; x < w - 2; x++)
            est[x] = (int16_t)((2 * (int32_t)line[x] - line[x - 2] - line[x + 2]) / 4);

        const comp_sc_line_t sc = comp_sc_line(d->standard, frame, r + row_off);
        const int32_t sn0 = d->sin_q15[sc.phase];
        const int32_t cs0 = d->sin_q15[(sc.phase + d->den / 4) % d->den];

        d->demod_pq_row(p_row, q_row, s, d->demod_pq_coef, w);

        int32_t yl_row[COMP_ACTIVE_WIDTH_PAL];
        for (int x = 0; x < w; x++) {
            const int32_t p0 = p_row[x];
            const int32_t q0 = q_row[x];
            u_row[x] = (int32_t)(((int64_t)p0 * cs0 + (int64_t)q0 * sn0 + 8192) >> 14);
            v_row[x] = sc.vswitch *
                (int32_t)(((int64_t)q0 * cs0 - (int64_t)p0 * sn0 + 8192) >> 14);
            yl_row[x] = (int32_t)line[x] - est[x] - d->level_black;
        }
        d->demod_quant_row(dsty + r * ystride, yl_row, &d->mag_y, w);

        const int32_t *u_out = u_row, *v_out = v_row;
        if (use_eq && d->eq) {
            eq_row(d, u_row, u_eq, w);
            eq_row(d, v_row, v_eq, w);
            u_out = u_eq;
            v_out = v_eq;
        }
        d->demod_quant_row(dstu + r * ustride, u_out, &d->mag_u, w);
        d->demod_quant_row(dstv + r * vstride, v_out, &d->mag_v, w);
    }
}

/* One row of the hybrid's motion routing: the per-sample metric is the
 * largest same-phase frame difference within ±2 samples (edge-clamped),
 * and where it stays under the margin the comb's chroma replaces the
 * transform's. Split into branch-free passes — pairwise difference,
 * fixed-bound sliding max with the borders peeled, and-mask select —
 * so each loop autovectorizes. */
static void hybrid_motion_row(const uint16_t *cur, const uint16_t *pr,
                              const uint16_t *nx, int32_t margin,
                              uint8_t *restrict mk, int16_t *restrict chroma,
                              const int16_t *restrict chroma2, int w)
{
    int32_t pairmax[COMP_ACTIVE_WIDTH_NTSC];

    for (int x = 0; x < w; x++) {
        const int32_t a = abs((int)cur[x] - (int)pr[x]);
        const int32_t b = abs((int)cur[x] - (int)nx[x]);
        pairmax[x] = a > b ? a : b;
    }
    for (int x = 2; x < w - 2; x++) {
        int32_t m = pairmax[x - 2];
        m = pairmax[x - 1] > m ? pairmax[x - 1] : m;
        m = pairmax[x]     > m ? pairmax[x]     : m;
        m = pairmax[x + 1] > m ? pairmax[x + 1] : m;
        m = pairmax[x + 2] > m ? pairmax[x + 2] : m;
        mk[x] = m < margin;
    }
    for (int e = 0; e < 4; e++) {
        const int x = e < 2 ? e : w - 4 + e;
        const int lo = x < 2 ? 0 : x - 2;
        const int hi = x >= w - 2 ? w - 1 : x + 2;
        int32_t m = 0;
        for (int o = lo; o <= hi; o++)
            m = pairmax[o] > m ? pairmax[o] : m;
        mk[x] = m < margin;
    }
    for (int x = 0; x < w; x++)
        chroma[x] ^= (int16_t)(-(int16_t)(mk[x] != 0)
                               & (chroma[x] ^ chroma2[x]));
}

/* Y-only Landweber refinement, anchored to the pre-encode original:
 * iterate the luma estimate so the crude decode of its recomposite
 * converges to the original degraded luma, i.e. deconvolve the crude
 * decoder model. Chroma stays fixed. */
static void refine_luma(const comp_decode_t *d, comp_decode_scratch_t *s,
                        int frame, int rows, int row_off,
                        const uint16_t *orig_y, ptrdiff_t orig_stride,
                        uint16_t *dsty, ptrdiff_t ystride,
                        uint16_t *dstu, ptrdiff_t ustride,
                        uint16_t *dstv, ptrdiff_t vstride,
                        double *out_residual, double *out_correction)
{
    const int w = d->width;
    const size_t plane = (size_t)w * d->height;
    uint16_t *est = s->refine;                   /* current YUV estimate */
    uint16_t *comp2 = s->refine + 3 * plane;     /* recomposite */
    uint16_t *cru = s->refine + 4 * plane;       /* crude Y of recomposite */
    /* the |difference| sums are integers under rows * w * 65535 < 2^53,
     * so an integer total is the exact value a double running sum takes,
     * and integer accumulation leaves the loops vectorizable */
    int64_t residual_sum = 0;

    for (int r = 0; r < rows; r++) {
        memcpy(est + r * w, dsty + r * ystride, sizeof(uint16_t) * w);
        memcpy(est + plane + r * w, dstu + r * ustride, sizeof(uint16_t) * w);
        memcpy(est + 2 * plane + r * w, dstv + r * vstride, sizeof(uint16_t) * w);
    }

    for (int it = 0; it < d->refine; it++) {
        const int last = it == d->refine - 1;
        residual_sum = 0;
        for (int r = 0; r < rows; r++)
            comp_encode_line(&d->enc, comp2 + r * w,
                             est + r * w, est + plane + r * w, est + 2 * plane + r * w,
                             comp_sc_line(d->standard, frame, r + row_off));
        crude_decode_frame(d, frame, rows, row_off, 0, comp2, w,
                           cru, w, cru + plane, w, cru + plane, w);
        for (int r = 0; r < rows; r++) {
            const uint16_t *oy = orig_y + r * orig_stride;
            uint16_t *ey = est + r * w;
            const uint16_t *cy = cru + r * w;
            for (int x = 0; x < w; x++) {
                const int32_t resid = (int32_t)oy[x] - (int32_t)cy[x];
                if (last)               /* model misfit remaining this frame */
                    residual_sum += resid < 0 ? -resid : resid;
                ey[x] = clamp_u16((int32_t)ey[x] + resid);
            }
        }
    }

    /* correction magnitude: how far the loop moved luma from the decode
     * (dsty still holds the pre-refine luma until we overwrite it below) */
    int64_t correction_sum = 0;
    for (int r = 0; r < rows; r++) {
        const uint16_t *in = dsty + r * ystride;
        const uint16_t *ey = est + r * w;
        for (int x = 0; x < w; x++) {
            const int32_t dlt = (int32_t)ey[x] - (int32_t)in[x];
            correction_sum += dlt < 0 ? -dlt : dlt;
        }
    }

    for (int r = 0; r < rows; r++)
        memcpy(dsty + r * ystride, est + r * w, sizeof(uint16_t) * w);

    if (out_residual)
        *out_residual = (double)residual_sum / ((double)rows * w);
    if (out_correction)
        *out_correction = (double)correction_sum / ((double)rows * w);
}

void comp_decode_frame(comp_decode_t *d, int frame, int nframes,
                       int rows, int row_off,
                       const comp_frame_view_t *views, const int *view_frames,
                       int look,
                       const uint16_t *orig_y, ptrdiff_t orig_stride,
                       uint16_t *dsty, ptrdiff_t ystride,
                       uint16_t *dstu, ptrdiff_t ustride,
                       uint16_t *dstv, ptrdiff_t vstride,
                       comp_decode_metrics_t *metrics,
                       uint16_t *out_mask, ptrdiff_t mask_stride, int mask_kind)
{
    comp_decode_scratch_t *s = scratch_acquire(d);
    const int w = d->width;
    const uint16_t *comp = views[look].data;
    const ptrdiff_t comp_stride = views[look].stride;
    comp_conf_acc_t acc = { 0.0, 0.0, 0, NULL, 0 };
    /* the confidence map exists only at eq=2. Run the accumulator when the
     * caller wants the props (metrics) or the confidence mask; the same
     * consumption loop writes 65535*(1-conf) into out_mask when set. */
    const int want_conf_mask = out_mask && mask_kind == COMP_MASK_CONFIDENCE;
    comp_conf_acc_t *pacc = ((metrics || want_conf_mask) && d->eq == 2 && s->conf)
                            ? &acc : NULL;
    if (want_conf_mask && pacc) {
        acc.out_mask = out_mask;
        acc.mask_stride = mask_stride;
    }

    if (d->dimensions == 1) {
        crude_decode_frame(d, frame, rows, row_off, 1, comp, comp_stride,
                           dsty, ystride, dstu, ustride, dstv, vstride);
    } else if (d->standard == COMP_STD_PAL) {
        const int frows = rows / 2;

        if (d->dimensions == 2) {
            /* chroma separation per field; field views interleave rows */
            for (int field = 0; field < 2; field++) {
                comp_transform2d_field(&d->transform,
                                       comp + field * comp_stride, 2 * comp_stride,
                                       w, frows,
                                       s->chroma_f + field * w, 2 * w,
                                       s->conf ? s->conf + field * w : NULL, 2 * w);
            }
            /* quantise the separated chroma for the fixed-point demod;
             * the 3D path below quantises inside the frame sum instead */
            for (int i = 0; i < w * rows; i++)
                s->chroma[i] = clamp_i16(lrintf(s->chroma_f[i]));
        } else {
            /* build the field stack the covering 3D tiles need; the
             * views are edge-clamped frames, so out-of-clip fields
             * resolve to the nearest frame's field of the same parity */
            comp_field_view_t fields[COMP_T3D_ZTILE + COMP_T3D_ZTILE / 2];
            const int fout = frame * 2;
            const int tz_hi = ((fout + 1) / (COMP_T3D_ZTILE / 2)) * (COMP_T3D_ZTILE / 2);
            const int z0 = tz_hi - COMP_T3D_ZTILE / 2;
            const int nfields = COMP_T3D_ZTILE + COMP_T3D_ZTILE / 2;

            for (int i = 0; i < nfields; i++) {
                const int g = z0 + i;
                if (g < 0 || g >= 2 * nframes) {
                    /* outside the clip: black, as the reference pads --
                     * repeating real fields would break the temporal
                     * phase sequence the symmetry test relies on */
                    fields[i].data = NULL;
                    fields[i].stride = 0;
                    continue;
                }
                int k = g / 2 - (frame - look);
                k = k < 0 ? 0 : k > 2 * look ? 2 * look : k;
                fields[i].data = views[k].data + (g & 1) * views[k].stride;
                fields[i].stride = 2 * views[k].stride;
            }
            comp_transform3d_frame(&d->transform3, &d->t3cache,
                                   fields, z0, nfields,
                                   frame, 0, w, frows,
                                   s->chroma, s->chroma + w, 2 * w,
                                   s->conf, s->conf ? s->conf + w : NULL, 2 * w);
        }

        for (int field = 0; field < 2; field++) {
            pal_decode_field(d, frame, field,
                             comp + field * comp_stride, 2 * comp_stride,
                             s->chroma + field * w, 2 * w,
                             s->conf ? s->conf + field * w : NULL, pacc,
                             dsty, ystride, dstu, ustride, dstv, vstride);
        }
    } else {
        /* the comb works on frame rows directly: ±2 rows are the same
         * field's neighboring lines */
        if (d->use_transform) {
            comp_field_view_t fields[COMP_T3D_ZTILE + COMP_T3D_ZTILE / 2];
            const int fout = frame * 2;
            const int tz_hi = ((fout + 1) / (COMP_T3D_ZTILE / 2)) * (COMP_T3D_ZTILE / 2);
            const int z0 = tz_hi - COMP_T3D_ZTILE / 2;
            const int nfields = COMP_T3D_ZTILE + COMP_T3D_ZTILE / 2;

            const int parity = (row_off + 1) & 1;
            for (int i = 0; i < nfields; i++) {
                const int g = z0 + i;
                if (g < 0 || g >= 2 * nframes) {
                    fields[i].data = NULL;
                    fields[i].stride = 0;
                    continue;
                }
                int k = g / 2 - (frame - look);
                k = k < 0 ? 0 : k > 2 * look ? 2 * look : k;
                fields[i].data = views[k].data + ((g ^ parity) & 1) * views[k].stride;
                fields[i].stride = 2 * views[k].stride;
            }
            comp_transform3d_frame(&d->transform3, &d->t3cache,
                                   fields, z0, nfields,
                                   frame, parity, w, rows / 2,
                                   s->chroma, s->chroma + w, 2 * w,
                                   s->conf, s->conf ? s->conf + w : NULL, 2 * w);

            if (d->use_transform == 2) {
                /* the hybrid: run the adaptive comb on the center three
                 * frames and keep its chroma wherever a temporal
                 * candidate won cleanly (static content) */
                const float *c1[3], *c2[3];
                for (int k = 0; k < 3; k++) {
                    float *b1 = s->tmp3 + k * w * d->height;
                    float *b2 = s->tmp3 + (3 + k) * w * d->height;
                    ntsc_comb1d(d, rows, views[look - 1 + k].data,
                                views[look - 1 + k].stride, b1);
                    ntsc_comb2d(d, rows, b1, b2);
                    c1[k] = b1;
                    c2[k] = b2;
                }
                ntsc_split3d(d, rows, row_off, views + look - 1,
                             view_frames + look - 1, c1, c2, s->chroma2);
                /* route on motion, not on the comb's own confidence:
                 * frames two apart share subcarrier phase, so their
                 * difference sees content change and no chroma. Where
                 * the neighborhood is still, the comb is the better
                 * decoder wholesale; where it moves, the transform is.
                 * Baked-in crawl that itself moves reads as motion and
                 * routes to the transform — a conservative choice. */
                const int32_t margin = d->comb_krange / 16;
                const uint16_t *p2 = views[look - 2].data;
                const uint16_t *n2 = views[look + 2].data;
                const ptrdiff_t p2s = views[look - 2].stride;
                const ptrdiff_t n2s = views[look + 2].stride;
                for (int r = 0; r < rows; r++)
                    hybrid_motion_row(comp + r * comp_stride, p2 + r * p2s,
                                      n2 + r * n2s, margin, s->mask + r * w,
                                      s->chroma + r * w, s->chroma2 + r * w, w);
            }
        } else if (d->dimensions == 2) {
            ntsc_comb1d(d, rows, comp, comp_stride, s->chroma_f);
            ntsc_comb2d(d, rows, s->chroma_f, s->tmp3);
            for (int i = 0; i < w * rows; i++)
                s->chroma[i] = clamp_i16(lrintf(s->tmp3[i]));
        } else {
            const float *c1[3], *c2[3];
            for (int k = 0; k < 3; k++) {
                float *b1 = s->tmp3 + k * w * d->height;
                float *b2 = s->tmp3 + (3 + k) * w * d->height;
                ntsc_comb1d(d, rows, views[k].data, views[k].stride, b1);
                ntsc_comb2d(d, rows, b1, b2);
                c1[k] = b1;
                c2[k] = b2;
            }
            ntsc_split3d(d, rows, row_off, views, view_frames, c1, c2,
                         s->chroma);
        }
        for (int r = 0; r < rows; r++)
            ntsc_demod_line(d, frame, r + row_off, r,
                            comp + r * comp_stride, s->chroma + r * w,
                            s->conf ? s->conf + r * w : NULL, pacc,
                            d->use_transform == 2 ? s->mask + r * w : NULL,
                            dsty + r * ystride, dstu + r * ustride,
                            dstv + r * vstride);
    }

    if (d->refine > 0 && orig_y)
        refine_luma(d, s, frame, rows, row_off, orig_y, orig_stride,
                    dsty, ystride, dstu, ustride, dstv, vstride,
                    metrics ? &metrics->refine_residual : NULL,
                    metrics ? &metrics->refine_correction : NULL);

    if (pacc && acc.n && metrics) {     /* pacc can run for the mask alone */
        const double mean = acc.sum / (double)acc.n;
        double var = acc.sumsq / (double)acc.n - mean * mean;
        if (var < 0.0)                  /* guard fp cancellation near zero */
            var = 0.0;
        metrics->conf_mean = mean;
        metrics->conf_std = sqrt(var);
    }

    /* NTSC hybrid: the motion router's per-sample decision (s->mask, 1 =
     * still -> comb, so motion is mask 0), filled only on transform=2.
     * Report the motion fraction, and emit the per-sample motion mask
     * (GRAY16, 65535 = motion) at the raster when a mask output is asked
     * for — matching CompositeMotionFraction's white=motion polarity. */
    const int want_motion_mask = out_mask && mask_kind == COMP_MASK_MOTION;
    if ((metrics || want_motion_mask) && d->standard == COMP_STD_NTSC
        && d->dimensions == 3 && d->use_transform == 2 && s->mask) {
        uint64_t motion = 0;
        for (int r = 0; r < rows; r++) {
            const uint8_t *mk = s->mask + r * w;
            uint16_t *om = want_motion_mask ? out_mask + r * mask_stride : NULL;
            for (int x = 0; x < w; x++) {
                const int is_motion = mk[x] == 0;
                motion += is_motion;
                if (om)
                    om[x] = is_motion ? 65535 : 0;
            }
        }
        if (metrics)
            metrics->motion_fraction = (double)motion / ((double)rows * w);
    }

    scratch_release(d, s);
}
