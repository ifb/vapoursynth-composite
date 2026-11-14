/*
 * Internal fixed-size 3D real FFT for the transform tiles (see fft.h
 * for the contract). Cooley-Tukey radix-2 throughout: the X axis is a
 * 16-point real FFT per row via the pack-to-half-size-complex trick,
 * the Y and Z axes are decimation-in-frequency (forward) and
 * decimation-in-time (inverse) passes whose butterfly elements are the
 * packed five-complex band rows, leaving the (ky, kz) numbering
 * bit-reversed instead of spending permutation passes.
 *
 * Twiddles are embedded literals, so results are identical across
 * platforms and independent of libm.
 */

#include <stddef.h>
#include <string.h>

#include "cpu.h"
#include "fft.h"
#include "subcarrier.h"

#define ZT COMP_T3D_ZTILE
#define XT COMP_T3D_XTILE
#define NCOL COMP_FFT_NCOL
#define RS16 (COMP_FFT_ROWSTRIDE / 2)  /* band row stride, complex */
#define C0 (COMP_T3D_XTILE / 8)  /* first band column */

typedef struct { float re, im; } fcpx;

/* e^{-2 pi i k / N}; tw16 also serves as the 16-point real-FFT
 * untangle table */
static const fcpx tw32[16] = {
    { 1.0f, 0.0f },
    { 0.98078528f, -0.195090322f },
    { 0.923879533f, -0.382683432f },
    { 0.831469612f, -0.555570233f },
    { 0.707106781f, -0.707106781f },
    { 0.555570233f, -0.831469612f },
    { 0.382683432f, -0.923879533f },
    { 0.195090322f, -0.98078528f },
    { 0.0f, -1.0f },
    { -0.195090322f, -0.98078528f },
    { -0.382683432f, -0.923879533f },
    { -0.555570233f, -0.831469612f },
    { -0.707106781f, -0.707106781f },
    { -0.831469612f, -0.555570233f },
    { -0.923879533f, -0.382683432f },
    { -0.98078528f, -0.195090322f },
};
static const fcpx tw16[8] = {
    { 1.0f, 0.0f },
    { 0.923879533f, -0.382683432f },
    { 0.707106781f, -0.707106781f },
    { 0.382683432f, -0.923879533f },
    { 0.0f, -1.0f },
    { -0.382683432f, -0.923879533f },
    { -0.707106781f, -0.707106781f },
    { -0.923879533f, -0.382683432f },
};
static const fcpx tw8[4] = {
    { 1.0f, 0.0f },
    { 0.707106781f, -0.707106781f },
    { 0.0f, -1.0f },
    { -0.707106781f, -0.707106781f },
};

/* 8-point complex FFT, in-place, natural order in and out */
static void cfft8(fcpx *v)
{
    fcpx t;
    t = v[1]; v[1] = v[4]; v[4] = t;
    t = v[3]; v[3] = v[6]; v[6] = t;
    for (int i = 0; i < 8; i += 2) {
        const fcpx a = v[i], b = v[i + 1];
        v[i].re = a.re + b.re;     v[i].im = a.im + b.im;
        v[i + 1].re = a.re - b.re; v[i + 1].im = a.im - b.im;
    }
    for (int i = 0; i < 8; i += 4) {
        fcpx a = v[i], b = v[i + 2];
        v[i].re = a.re + b.re;     v[i].im = a.im + b.im;
        v[i + 2].re = a.re - b.re; v[i + 2].im = a.im - b.im;
        a = v[i + 1]; b = v[i + 3];
        const fcpx bw = { b.im, -b.re };
        v[i + 1].re = a.re + bw.re; v[i + 1].im = a.im + bw.im;
        v[i + 3].re = a.re - bw.re; v[i + 3].im = a.im - bw.im;
    }
    static const float RS = 0.707106781f;
    for (int j = 0; j < 4; j++) {
        const fcpx a = v[j], b = v[j + 4];
        fcpx bw;
        switch (j) {
        case 0: bw = b; break;
        case 1: bw.re = (b.re + b.im) * RS; bw.im = (b.im - b.re) * RS; break;
        case 2: bw.re = b.im; bw.im = -b.re; break;
        default: bw.re = (b.im - b.re) * RS; bw.im = -(b.re + b.im) * RS; break;
        }
        v[j].re = a.re + bw.re;     v[j].im = a.im + bw.im;
        v[j + 4].re = a.re - bw.re; v[j + 4].im = a.im - bw.im;
    }
}

static void icfft8(fcpx *v)  /* unnormalized inverse */
{
    for (int i = 0; i < 8; i++) v[i].im = -v[i].im;
    cfft8(v);
    for (int i = 0; i < 8; i++) v[i].im = -v[i].im;
}

/* 16-point real FFT of one row, band columns only */
static void rfft16_row(const float *x, fcpx *out)
{
    fcpx c[8];
    for (int j = 0; j < 8; j++) {
        c[j].re = x[2 * j];
        c[j].im = x[2 * j + 1];
    }
    cfft8(c);
    for (int k = NCOL; k < RS16; k++)
        out[k].re = out[k].im = 0.0f;
    for (int k = C0; k <= C0 + NCOL - 1; k++) {
        const fcpx ck = c[k & 7];
        const fcpx cn = c[(8 - k) & 7];
        const float er = 0.5f * (ck.re + cn.re);
        const float ei = 0.5f * (ck.im - cn.im);
        const float odr = 0.5f * (ck.im + cn.im);
        const float odi = 0.5f * (cn.re - ck.re);
        const fcpx w = tw16[k];
        out[k - C0].re = er + w.re * odr - w.im * odi;
        out[k - C0].im = ei + w.re * odi + w.im * odr;
    }
}

/* inverse of rfft16_row: band columns in, 16 reals out (x 16) */
static void irfft16_row(const fcpx *F, float *x)
{
    fcpx F9[9], c[8];
    F9[0].re = F9[0].im = F9[1].re = F9[1].im = 0.0f;
    F9[7].re = F9[7].im = F9[8].re = F9[8].im = 0.0f;
    for (int k = 0; k < NCOL; k++)
        F9[C0 + k] = F[k];
    for (int k = 0; k < 8; k++) {
        const fcpx fk = F9[k];
        const fcpx fn = F9[8 - k];
        const float er = 0.5f * (fk.re + fn.re);
        const float ei = 0.5f * (fk.im - fn.im);
        const float qr = 0.5f * (fk.re - fn.re);
        const float qi = 0.5f * (fk.im + fn.im);
        const fcpx w = tw16[k];  /* conjugated on use */
        const float odr = w.re * qr + w.im * qi;
        const float odi = w.re * qi - w.im * qr;
        c[k].re = er - odi;
        c[k].im = ei + odr;
    }
    icfft8(c);
    for (int j = 0; j < 8; j++) {
        x[2 * j] = 2.0f * c[j].re;
        x[2 * j + 1] = 2.0f * c[j].im;
    }
}

/* decimation in frequency: natural-order input, bit-reversed output,
 * elements are the packed band rows */
static void pass_dif(fcpx *base, int n, ptrdiff_t stride, const fcpx *tw)
{
    for (int len = n; len >= 2; len >>= 1) {
        const int half = len >> 1;
        const int tstep = n / len;
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < half; j++) {
                const fcpx w = tw[j * tstep];
                fcpx *a = base + (i + j) * stride;
                fcpx *b = base + (i + j + half) * stride;
                for (int c = 0; c < NCOL; c++) {
                    const float ar = a[c].re, ai = a[c].im;
                    const float dr = ar - b[c].re, di = ai - b[c].im;
                    a[c].re = ar + b[c].re;
                    a[c].im = ai + b[c].im;
                    b[c].re = dr * w.re - di * w.im;
                    b[c].im = dr * w.im + di * w.re;
                }
            }
        }
    }
}

/* decimation in time: bit-reversed input, natural-order output */
static void pass_dit(fcpx *base, int n, ptrdiff_t stride, const fcpx *tw)
{
    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        const int tstep = n / len;
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < half; j++) {
                fcpx w = tw[j * tstep];
                w.im = -w.im;
                fcpx *a = base + (i + j) * stride;
                fcpx *b = base + (i + j + half) * stride;
                for (int c = 0; c < NCOL; c++) {
                    const float br = b[c].re * w.re - b[c].im * w.im;
                    const float bi = b[c].re * w.im + b[c].im * w.re;
                    const float ar = a[c].re, ai = a[c].im;
                    a[c].re = ar + br; a[c].im = ai + bi;
                    b[c].re = ar - br; b[c].im = ai - bi;
                }
            }
        }
    }
}

/* whether NTSC row (z, y) belongs to the tile's own field (see fft.h) */
static int picture_row(int z, int y, unsigned ph)
{
    return (y & 1) == (int)((ph >> z) & 1);
}

static void fft_fwd(float *band, const float *real, int yt, const fcpx *twy,
                    const float *black, unsigned ph)
{
    fcpx *b = (fcpx *)band;
    for (int z = 0; z < ZT; z++) {
        for (int y = 0; y < yt; y++) {
            const int r = z * yt + y;
            if (!black || picture_row(z, y, ph))
                rfft16_row(real + r * XT, b + r * RS16);
            else
                memcpy(b + r * RS16, black + r * COMP_FFT_ROWSTRIDE,
                       sizeof(float) * COMP_FFT_ROWSTRIDE);
        }
    }
    for (int z = 0; z < ZT; z++)
        pass_dif(b + z * yt * RS16, yt, RS16, twy);
    for (int y = 0; y < yt; y++)
        pass_dif(b + y * RS16, ZT, yt * RS16, tw8);
}

static void fft_inv(float *real, float *band, int yt, const fcpx *twy,
                    int ntsc, unsigned ph)
{
    fcpx *b = (fcpx *)band;
    for (int y = 0; y < yt; y++)
        pass_dit(b + y * RS16, ZT, yt * RS16, tw8);
    for (int z = 0; z < ZT; z++)
        pass_dit(b + z * yt * RS16, yt, RS16, twy);
    for (int z = 0; z < ZT; z++)
        for (int y = 0; y < yt; y++)
            if (!ntsc || picture_row(z, y, ph))
                irfft16_row(b + (z * yt + y) * RS16, real + (z * yt + y) * XT);
}

static void fft_fwd_ntsc_c(float *band, const float *real, const float *black,
                           unsigned ph)
{
    fft_fwd(band, real, COMP_T3D_YTILE, tw32, black, ph);
}

static void fft_fwd_pal_c(float *band, const float *real, const float *black,
                          unsigned ph)
{
    (void)black;
    (void)ph;
    fft_fwd(band, real, COMP_T3D_YTILE_PAL, tw16, NULL, 0);
}

static void fft_inv_ntsc_c(float *real, float *band, unsigned ph)
{
    fft_inv(real, band, COMP_T3D_YTILE, tw32, 1, ph);
}

static void fft_inv_pal_c(float *real, float *band, unsigned ph)
{
    (void)ph;
    fft_inv(real, band, COMP_T3D_YTILE_PAL, tw16, 0, 0);
}

#if defined(__x86_64__)
#define FFT_PASS_ASM(name)                                                  \
    void comp_fft_##name##_avx2(float *base, ptrdiff_t stride, int count,   \
                                ptrdiff_t cstride, const float *tw)
FFT_PASS_ASM(dif8);
FFT_PASS_ASM(dit8);
FFT_PASS_ASM(dif16);
FFT_PASS_ASM(dit16);
FFT_PASS_ASM(dif32);
FFT_PASS_ASM(dit32);
void comp_fft_x_fwd_avx2(float *band, const float *real, int nrows);
void comp_fft_x_inv_avx2(float *real, const float *band, int nrows);
void comp_fft_x_fwd2_avx2(float *band, const float *real, int nrows);
void comp_fft_x_inv2_avx2(float *real, const float *band, int nrows);

#define ROWB (COMP_FFT_ROWSTRIDE * 4)  /* band row, bytes */
#define YT COMP_T3D_YTILE

static void fft_fwd_ntsc_avx2(float *band, const float *real, const float *black,
                              unsigned ph)
{
    for (int z = 0; z < ZT; z++) {
        const int p = (ph >> z) & 1;
        const int r0 = z * YT + p;
        comp_fft_x_fwd2_avx2(band + r0 * XT, real + r0 * XT, YT / 2);
        for (int r = z * YT + !p; r < (z + 1) * YT; r += 2)
            memcpy(band + r * XT, black + r * XT, ROWB);
    }
    comp_fft_dif32_avx2(band, ROWB, ZT, YT * ROWB, (const float *)tw32);
    comp_fft_dif8_avx2(band, YT * ROWB, YT, ROWB, (const float *)tw8);
}

static void fft_fwd_pal_avx2(float *band, const float *real, const float *black,
                             unsigned ph)
{
    (void)black;
    (void)ph;
    comp_fft_x_fwd_avx2(band, real, ZT * COMP_T3D_YTILE_PAL);
    comp_fft_dif16_avx2(band, ROWB, ZT, COMP_T3D_YTILE_PAL * ROWB,
                        (const float *)tw16);
    comp_fft_dif8_avx2(band, COMP_T3D_YTILE_PAL * ROWB, COMP_T3D_YTILE_PAL,
                       ROWB, (const float *)tw8);
}

static void fft_inv_ntsc_avx2(float *real, float *band, unsigned ph)
{
    comp_fft_dit8_avx2(band, YT * ROWB, YT, ROWB, (const float *)tw8);
    comp_fft_dit32_avx2(band, ROWB, ZT, YT * ROWB, (const float *)tw32);
    for (int z = 0; z < ZT; z++) {
        const int r0 = z * YT + ((ph >> z) & 1);
        comp_fft_x_inv2_avx2(real + r0 * XT, band + r0 * XT, YT / 2);
    }
}

static void fft_inv_pal_avx2(float *real, float *band, unsigned ph)
{
    (void)ph;
    comp_fft_dit8_avx2(band, COMP_T3D_YTILE_PAL * ROWB, COMP_T3D_YTILE_PAL,
                       ROWB, (const float *)tw8);
    comp_fft_dit16_avx2(band, ROWB, ZT, COMP_T3D_YTILE_PAL * ROWB,
                        (const float *)tw16);
    comp_fft_x_inv_avx2(real, band, ZT * COMP_T3D_YTILE_PAL);
}
#endif

void comp_fft_x_rows_ntsc(float *band, const float *real, unsigned cpu)
{
#if defined(__x86_64__)
    if (cpu & COMP_CPU_AVX2) {
        comp_fft_x_fwd_avx2(band, real, ZT * COMP_T3D_YTILE);
        return;
    }
#endif
    (void)cpu;
    for (int r = 0; r < ZT * COMP_T3D_YTILE; r++)
        rfft16_row(real + r * XT, (fcpx *)band + r * RS16);
}

comp_fft_fwd_fn comp_get_fft_fwd_fn(int standard, unsigned cpu)
{
#if defined(__x86_64__)
    if (cpu & COMP_CPU_AVX2)
        return standard == COMP_STD_PAL ? fft_fwd_pal_avx2 : fft_fwd_ntsc_avx2;
#endif
    (void)cpu;
    return standard == COMP_STD_PAL ? fft_fwd_pal_c : fft_fwd_ntsc_c;
}

comp_fft_inv_fn comp_get_fft_inv_fn(int standard, unsigned cpu)
{
#if defined(__x86_64__)
    if (cpu & COMP_CPU_AVX2)
        return standard == COMP_STD_PAL ? fft_inv_pal_avx2 : fft_inv_ntsc_avx2;
#endif
    (void)cpu;
    return standard == COMP_STD_PAL ? fft_inv_pal_c : fft_inv_ntsc_c;
}
