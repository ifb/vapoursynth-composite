/*
 * 3D Transform chroma separation.
 *
 * After ld-chroma-decoder's TransformPal3D: the 2D tile filter gains a
 * temporal axis, with tiles half-overlapped in every axis and bins
 * kept only if symmetric with their reflection about the chroma
 * carrier.
 *
 * PAL separates on the displaced field-line lattice of GB 2365247 A
 * (Figure 6): each field is displaced vertically by one picture line
 * per field interval (reset per tile, inverted after), which lands all
 * eight fields on one dense grid, so the tiles are 16 samples by 16
 * field lines by 8 fields — half the FFT volume of the reference's
 * black-filled frame-line tiles. On this raster's line-locked grid the
 * carriers are then single exact bins, U at (z,y,x) = (4,12,4) and V
 * at (4,4,4), sharing one reflection map (probe-verified; the patent's
 * own +-12.5 Hz figures correspond to the opposite displacement
 * direction, which splits the carriers here).
 *
 * NTSC keeps the reference's frame-line tiles (16x32x8) with the other
 * field's rows black, so the FFT sees the interlaced lattice directly.
 * The displaced variant was derived and measured for NTSC too (single
 * carrier at temporal Nyquist, reflection ((-z)%8, (-y)%16, 8-x)) and
 * rejected: real-footage chroma flicker was consistently worse, even
 * with retrained tables. NTSC's single shared carrier makes the pair
 * symmetry approximate, and the black-filled lattice's alias-pair
 * redundancy averages that test — a variance reduction the dense
 * displaced lattice gives up; PAL's two exact carriers had nothing to
 * lose.
 */

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "clamp.h"
#include "cpu.h"
#include "fft.h"
#include "osdep.h"
#include "subcarrier.h"
#include "transform3d.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define XTILE COMP_T3D_XTILE
#define YTILE COMP_T3D_YTILE
#define YTILE_PAL COMP_T3D_YTILE_PAL
#define ZTILE COMP_T3D_ZTILE
#define XC COMP_T3D_XCOMPLEX
#define TILE_REAL (ZTILE * YTILE * XTILE)
#define TILE_CPLX (ZTILE * YTILE * XC)
#define TILE_REAL_PAL (ZTILE * YTILE_PAL * XTILE)
#define TILE_CPLX_PAL (ZTILE * YTILE_PAL * XC)

/* largest per-field displacement, in field rows: ceil((ZTILE-1)/2) */
#define SHIFT_MAX (ZTILE / 2)

#define LEVEL_BLACK 16384.0f

/* per-tile frequency buffer: the FFTW layout and the internal FFT's
 * packed band share the storage */
union t3d_freq {
    fftwf_complex cplx[TILE_CPLX];
    float band[ZTILE * YTILE * COMP_FFT_ROWSTRIDE];
};
union t3d_freq_pal {
    fftwf_complex cplx[TILE_CPLX_PAL];
    float band[ZTILE * YTILE_PAL * COMP_FFT_ROWSTRIDE];
};

/* FFTW's planner is not thread-safe; execute-with-new-arrays is */
static pthread_mutex_t planner_lock = PTHREAD_MUTEX_INITIALIZER;

static void t3d_build_tables(comp_transform3d_t *t);

static double compute_window(int element, int limit)
{
    return 0.5 - 0.5 * cos((2.0 * M_PI * (element + 0.5)) / limit);
}

int comp_transform3d_init(comp_transform3d_t *t, double threshold, int standard,
                          int level, double evidence)
{
    if (!(threshold > 0.0 && threshold <= 1.0) || evidence < 0.0)
        return -1;
    t->standard = standard;
    t->level = !!level;
    t->use_lut = 0;
    t->evidence = (float)evidence;
    t->lut_gain = comp_get_lut_gain_fn(comp_cpu_detect());
    t->t3d_mag = comp_get_t3d_mag_fn(comp_cpu_detect());
    t->t3d_apply = comp_get_t3d_apply_fn(comp_cpu_detect());
    if (comp_cpu_detect() & COMP_CPU_AVX2) {
        t->fft_fwd = comp_get_fft_fwd_fn(standard, comp_cpu_detect());
        t->fft_inv = comp_get_fft_inv_fn(standard, comp_cpu_detect());
    } else {
        /* the C transform loses to FFTW; without the avx2 tier the
         * FFTW path stays in use */
        t->fft_fwd = NULL;
        t->fft_inv = NULL;
    }
    t3d_build_tables(t);
    const int ytile = standard == COMP_STD_PAL ? YTILE_PAL : YTILE;
    for (int i = 0; i < COMP_T3D_NTHRESH; i++)
        t->threshold_sq[i] = (float)(threshold * threshold);

    for (int z = 0; z < ZTILE; z++)
        for (int y = 0; y < ytile; y++)
            for (int x = 0; x < XTILE; x++)
                t->window[z][y][x] = (float)(compute_window(z, ZTILE)
                                             * compute_window(y, ytile)
                                             * compute_window(x, XTILE));

    ALIGNED_32( float real[TILE_REAL] );
    ALIGNED_32( fftwf_complex cplx[TILE_CPLX] );
    if (t->fft_fwd && standard == COMP_STD_NTSC) {
        /* the rows an NTSC tile takes from the other field are black,
         * exactly as build_slab_ntsc would fill them */
        for (int z = 0; z < ZTILE; z++)
            for (int y = 0; y < YTILE; y++)
                for (int x = 0; x < XTILE; x++)
                    real[(z * YTILE + y) * XTILE + x] = LEVEL_BLACK * t->window[z][y][x];
        comp_fft_x_rows_ntsc(t->fft_black, real, comp_cpu_detect());
    }
    pthread_mutex_lock(&planner_lock);
    t->forward = fftwf_plan_dft_r2c_3d(ZTILE, ytile, XTILE, real, cplx,
                                       FFTW_MEASURE);
    t->inverse = fftwf_plan_dft_c2r_3d(ZTILE, ytile, XTILE, cplx, real,
                                       FFTW_MEASURE);
    pthread_mutex_unlock(&planner_lock);
    if (!t->forward || !t->inverse) {
        comp_transform3d_free(t);
        return -1;
    }
    return 0;
}

void comp_transform3d_set_lut(comp_transform3d_t *t, const double *v)
{
    const int nbins = t->standard == COMP_STD_PAL ? COMP_T3D_NTHRESH_PAL
                                                  : COMP_T3D_NTHRESH;
    for (int b = 0; b < nbins; b++)
        for (int k = 0; k < COMP_LUT_K; k++)
            t->lut[b][k] = (float)v[b * COMP_LUT_K + k];
    t->use_lut = 1;
}

void comp_transform3d_free(comp_transform3d_t *t)
{
    pthread_mutex_lock(&planner_lock);
    if (t->forward)
        fftwf_destroy_plan(t->forward);
    if (t->inverse)
        fftwf_destroy_plan(t->inverse);
    pthread_mutex_unlock(&planner_lock);
    t->forward = NULL;
    t->inverse = NULL;
}

/* Bin-row geometry for the staged LUT filter: per bin row the float
 * offsets of its own and reflected tile rows, plus the x = XTILE/4
 * self-column specials (the bins that are their own reflection, kept
 * or discarded by the carrier tables of the pair-test filters). */
void comp_t3d_rowpair(int standard, int32_t (*rows)[2])
{
    const int pal = standard == COMP_STD_PAL;
    const int yt = pal ? YTILE_PAL : YTILE;
    int i = 0;

    for (int z = 0; z < ZTILE; z++) {
        const int z_ref = pal ? (ZTILE - z) % ZTILE
                              : ((ZTILE / 2) + ZTILE - z) % ZTILE;
        for (int y = 0; y < yt; y++, i++) {
            const int y_ref = ((yt / 2) + yt - y) % yt;
            rows[i][0] = (z * yt + y) * XC * 2 + (XTILE / 8) * 2;
            rows[i][1] = (z_ref * yt + y_ref) * XC * 2 + (XTILE / 4) * 2;
        }
    }
}

static int bitrev(int v, int bits)
{
    int r = 0;
    for (int b = 0; b < bits; b++)
        r = (r << 1) | ((v >> b) & 1);
    return r;
}

static void t3d_build_tables(comp_transform3d_t *t)
{
    const int pal = t->standard == COMP_STD_PAL;
    const int yt = pal ? YTILE_PAL : YTILE;
    const int ybits = pal ? 4 : 5;
    int bin = 0, i = 0;

    comp_t3d_rowpair(t->standard, t->rowpair);
    t->nspecial = 0;
    for (int z = 0; z < ZTILE; z++) {
        for (int y = 0; y < yt; y++, bin += 3) {
            const int y_ref = ((yt / 2) + yt - y) % yt;
            const int z_ref = pal ? (ZTILE - z) % ZTILE
                                  : ((ZTILE / 2) + ZTILE - z) % ZTILE;

            /* the internal FFT's band layout: packed 64-byte rows at
             * bit-reversed (ky, kz); same natural bin walk, so the
             * LUT indexing and visit order are unchanged */
            t->rowpair_fft[i][0] =
                (bitrev(z, 3) * yt + bitrev(y, ybits)) * COMP_FFT_ROWSTRIDE;
            t->rowpair_fft[i][1] =
                (bitrev(z_ref, 3) * yt + bitrev(y_ref, ybits))
                * COMP_FFT_ROWSTRIDE + (COMP_T3D_XTILE / 4 - COMP_T3D_XTILE / 8) * 2;
            i++;

            const int keep = pal
                ? (y == y_ref && z == z_ref)
                : ((y == YTILE / 4 && z == ZTILE / 4)
                   || (y == 3 * YTILE / 4 && z == 3 * ZTILE / 4));
            const int discard = !pal
                && (((y == 0 || y == YTILE / 2) && (z == 0 || z == ZTILE / 2))
                    || (y == YTILE / 4 && z == 3 * ZTILE / 4)
                    || (y == 3 * YTILE / 4 && z == ZTILE / 4));
            if (keep || discard) {
                t->special[t->nspecial].bin = bin + 2;
                t->special[t->nspecial].off = ((z * yt + y) * XC + XTILE / 4) * 2;
                t->special[t->nspecial].keep = keep;
                t->special_fft[t->nspecial].bin = bin + 2;
                t->special_fft[t->nspecial].off =
                    (bitrev(z, 3) * yt + bitrev(y, ybits)) * COMP_FFT_ROWSTRIDE
                    + (XTILE / 4 - XTILE / 8) * 2;
                t->special_fft[t->nspecial].keep = keep;
                t->nspecial++;
            }
        }
    }
}

/* Per-bin magnitude rows over the bin-row table: three bins per row
 * from the x band, the reflected side read reversed. May write
 * COMP_T3D_BINPAD floats of slack past the bin count. */
static void t3d_mag_c(float *m_in, float *m_ref, const float *in,
                      const int32_t (*rows)[2], int nrows)
{
    for (int i = 0; i < nrows; i++) {
        const float *a = in + rows[i][0];
        const float *b = in + rows[i][1];
        m_in[3 * i + 0] = a[0] * a[0] + a[1] * a[1];
        m_in[3 * i + 1] = a[2] * a[2] + a[3] * a[3];
        m_in[3 * i + 2] = a[4] * a[4] + a[5] * a[5];
        m_ref[3 * i + 0] = b[4] * b[4] + b[5] * b[5];
        m_ref[3 * i + 1] = b[2] * b[2] + b[3] * b[3];
        m_ref[3 * i + 2] = b[0] * b[0] + b[1] * b[1];
    }
}

/* Gain application over the bin-row table, in table order: each row
 * writes its own x band and its reflection's, so the twice-visited
 * x = XTILE/4 column keeps its later-write-wins semantics. */
static void t3d_apply_c(float *out, const float *in, const float *g,
                        const int32_t (*rows)[2], int nrows)
{
    for (int i = 0; i < nrows; i++) {
        const float *a = in + rows[i][0];
        const float *b = in + rows[i][1];
        float *oa = out + rows[i][0];
        float *ob = out + rows[i][1];
        const float g0 = g[3 * i + 0];
        const float g1 = g[3 * i + 1];
        const float g2 = g[3 * i + 2];

        oa[0] = a[0] * g0;
        oa[1] = a[1] * g0;
        oa[2] = a[2] * g1;
        oa[3] = a[3] * g1;
        oa[4] = a[4] * g2;
        oa[5] = a[5] * g2;
        ob[0] = b[0] * g2;
        ob[1] = b[1] * g2;
        ob[2] = b[2] * g1;
        ob[3] = b[3] * g1;
        ob[4] = b[4] * g0;
        ob[5] = b[5] * g0;
    }
}

/* Per-bin trained-LUT gain row: for each bin, the pair-symmetry ratio
 * r = lo/hi (1 when hi is 0) and the linear interpolation of that
 * bin's LUT row at r * (COMP_LUT_K - 1), knots uniform on [0, 1].
 * Arrays are padded so n may be rounded up to 16. The ratio pass holds
 * the divide and has no lookups, so it vectorizes without gathers; the
 * interpolation position waits in g for the lookup pass. */
static void lut_gain_row_c(float *g, float *r, const float *m_in,
                           const float *m_ref,
                           const float (*lut)[COMP_LUT_K], int n)
{
    for (int i = 0; i < n; i++) {
        const float lo = m_in[i] < m_ref[i] ? m_in[i] : m_ref[i];
        const float hi = m_in[i] < m_ref[i] ? m_ref[i] : m_in[i];
        /* divided unconditionally: lo <= hi, so the hi == 0 lanes the
         * select discards are 0/0, and compilers vectorize no divide
         * they would have to guard */
        const float q = lo / hi;
        const float ri = hi > 0.0f ? q : 1.0f;
        r[i] = ri;
        g[i] = ri * (COMP_LUT_K - 1);
    }
    for (int i = 0; i < n; i++) {
        const float pos = g[i];
        int k = (int)pos;
        if (k > COMP_LUT_K - 2)
            k = COMP_LUT_K - 2;
        g[i] = lut[i][k] + (lut[i][k + 1] - lut[i][k]) * (pos - k);
    }
}

#if defined(__x86_64__)
#define LUT_GAIN_ASM(isa)                                                   \
    void comp_lut_gain_##isa(float *g, float *r, const float *m_in,         \
                             const float *m_ref,                            \
                             const float (*lut)[COMP_LUT_K], int n)
LUT_GAIN_ASM(sse4);
LUT_GAIN_ASM(avx2);

#define T3D_ROW_ASM(isa)                                                    \
    void comp_t3d_mag_##isa(float *m_in, float *m_ref, const float *in,     \
                            const int32_t (*rows)[2], int nrows);           \
    void comp_t3d_apply_##isa(float *out, const float *in, const float *g,  \
                              const int32_t (*rows)[2], int nrows)
T3D_ROW_ASM(sse2);
T3D_ROW_ASM(avx2);
#endif

comp_lut_gain_fn comp_get_lut_gain_fn(unsigned cpu)
{
#if defined(__x86_64__)
    if (cpu & COMP_CPU_AVX2)
        return comp_lut_gain_avx2;
    if (cpu & COMP_CPU_SSE41)
        return comp_lut_gain_sse4;
#endif
    (void)cpu;
    return lut_gain_row_c;
}

comp_t3d_mag_fn comp_get_t3d_mag_fn(unsigned cpu)
{
#if defined(__x86_64__)
    if (cpu & COMP_CPU_AVX2)
        return comp_t3d_mag_avx2;
    if (cpu & COMP_CPU_SSE2)
        return comp_t3d_mag_sse2;
#endif
    (void)cpu;
    return t3d_mag_c;
}

comp_t3d_apply_fn comp_get_t3d_apply_fn(unsigned cpu)
{
#if defined(__x86_64__)
    if (cpu & COMP_CPU_AVX2)
        return comp_t3d_apply_avx2;
    if (cpu & COMP_CPU_SSE2)
        return comp_t3d_apply_sse2;
#endif
    (void)cpu;
    return t3d_apply_c;
}

/* Trained-LUT filter, both standards, staged so the divide-and-
 * interpolate gain math runs over linear rows: per-bin magnitudes
 * first, one lut_gain_row pass, then the gain application and
 * confidence sums. The application keeps the exact scalar bin order:
 * in the x = XTILE/4 column both members of a pair are themselves
 * iterated bins carrying different LUT rows, so each such pair is
 * written twice and the later visit must win; the confidence sums are
 * float accumulation, where order changes the result. */
static float apply_filter_lut(const comp_transform3d_t *t,
                              const float *in, float *out,
                              const int32_t (*rows)[2],
                              const comp_t3d_special_t *sp,
                              size_t out_floats)
{
    const int pal = t->standard == COMP_STD_PAL;
    const int yt = pal ? YTILE_PAL : YTILE;
    const int nrows = ZTILE * yt;
    const int nbins = pal ? COMP_T3D_NTHRESH_PAL : COMP_T3D_NTHRESH;
    float m_in[COMP_T3D_NTHRESH + COMP_T3D_BINPAD];
    float m_ref[COMP_T3D_NTHRESH + COMP_T3D_BINPAD];
    float g[COMP_T3D_NTHRESH + COMP_T3D_BINPAD];
    float r[COMP_T3D_NTHRESH + COMP_T3D_BINPAD];
    float e_num[COMP_T3D_NTHRESH], e_den[COMP_T3D_NTHRESH];
    float conf_num = 0.0f, conf_den = 0.0f;

    memset(out, 0, sizeof(float) * out_floats);

    t->t3d_mag(m_in, m_ref, in, rows, nrows);
    t->lut_gain(g, r, m_in, m_ref, t->lut, nbins);

    /* LF-luma evidence scales the gains per bin (PAL option). The
     * lookups address the FFTW tile layout; the internal-FFT path is
     * gated off when evidence is enabled. */
    if (pal && t->evidence > 0.0f) {
        static const int cy[2] = { 12, 4 };
        int bin = 0;
        for (int z = 0; z < ZTILE; z++) {
            for (int y = 0; y < yt; y++) {
                for (int x = XTILE / 8; x <= XTILE / 4; x++, bin++) {
                    float e = 0.0f;
                    for (int c = 0; c < 2; c++) {
                        const float *lf =
                            in + ((((4 - z + ZTILE) % ZTILE) * yt
                                   + (cy[c] - y + yt) % yt) * XC
                                  + (XTILE / 4) - x) * 2;
                        const float ec = lf[0] * lf[0] + lf[1] * lf[1];
                        e = ec > e ? ec : e;
                    }
                    const float hi = m_in[bin] > m_ref[bin] ? m_in[bin]
                                                            : m_ref[bin];
                    const float den = e + t->evidence * hi;
                    if (den > 0.0f)
                        g[bin] *= e / den;
                }
            }
        }
    }

    /* per-bin confidence terms; the specials override theirs, and the
     * kept ones write at unit gain (the identical bits of a copy) */
    for (int i = 0; i < nbins; i++) {
        const float e = g[i] * g[i] * (m_in[i] + m_ref[i]);
        e_num[i] = e * r[i];
        e_den[i] = e;
    }
    for (int s = 0; s < t->nspecial; s++) {
        const int b = sp[s].bin;
        if (sp[s].keep) {
            g[b] = 1.0f;
            e_num[b] = m_in[b];
            e_den[b] = m_in[b];
        } else {
            e_num[b] = 0.0f;
            e_den[b] = 0.0f;
        }
    }

    t->t3d_apply(out, in, g, rows, nrows);

    /* discarded self-column bins stay zero; both their writers are
     * themselves discards, so a post-pass restores the memset state */
    for (int s = 0; s < t->nspecial; s++) {
        if (!sp[s].keep) {
            out[sp[s].off + 0] = 0.0f;
            out[sp[s].off + 1] = 0.0f;
        }
    }

    /* the sums are float accumulation in bin order; a skipped bin and
     * an added +0.0 term are the same bits, so the discards' zero
     * terms preserve exactness */
    for (int i = 0; i < nbins; i++) {
        conf_num += e_num[i];
        conf_den += e_den[i];
    }
    return conf_den > 0.0f ? conf_num / conf_den : 0.0f;
}


/* PAL frequency-domain filter on the displaced field-line lattice: the
 * carriers are single exact bins, U at (z,y,x) = (4,12,4) and V at
 * (4,4,4), and both share the reflection map
 * ((-z) mod 8, (8-y) mod 16, 8-x), whose four self-paired bins are the
 * two carriers plus the two 25 Hz-offset partners (kept, matching the
 * reference's treatment of its self-paired bins). Returns the tile's
 * chroma confidence: the kept output energy's mean pair-symmetry
 * ratio, 0 when nothing was kept. */
static float apply_filter_pal(const comp_transform3d_t *t,
                              const fftwf_complex *in, fftwf_complex *out)
{
    const float *tsq = t->threshold_sq;
    float conf_num = 0.0f, conf_den = 0.0f;

    memset(out, 0, sizeof(fftwf_complex) * TILE_CPLX_PAL);

    for (int z = 0; z < ZTILE; z++) {
        const int z_ref = (ZTILE - z) % ZTILE;
        for (int y = 0; y < YTILE_PAL; y++) {
            const int y_ref = ((YTILE_PAL / 2) + YTILE_PAL - y) % YTILE_PAL;
            const fftwf_complex *bi = in + (z * YTILE_PAL + y) * XC;
            const fftwf_complex *bi_ref = in + (z_ref * YTILE_PAL + y_ref) * XC;
            fftwf_complex *bo = out + (z * YTILE_PAL + y) * XC;
            fftwf_complex *bo_ref = out + (z_ref * YTILE_PAL + y_ref) * XC;

            for (int x = XTILE / 8; x <= XTILE / 4; x++) {
                const int x_ref = (XTILE / 2) - x;
                const float threshold_sq = *tsq++;

                if (x == x_ref && y == y_ref && z == z_ref) {
                    bo[x][0] = bi[x][0];
                    bo[x][1] = bi[x][1];
                    conf_num += bi[x][0] * bi[x][0] + bi[x][1] * bi[x][1];
                    conf_den += bi[x][0] * bi[x][0] + bi[x][1] * bi[x][1];
                    continue;
                }

                const float m_in_sq = bi[x][0] * bi[x][0] + bi[x][1] * bi[x][1];
                const float m_ref_sq = bi_ref[x_ref][0] * bi_ref[x_ref][0]
                                     + bi_ref[x_ref][1] * bi_ref[x_ref][1];
                const float lo = m_in_sq < m_ref_sq ? m_in_sq : m_ref_sq;
                const float hi = m_in_sq < m_ref_sq ? m_ref_sq : m_in_sq;
                const float r = hi > 0.0f ? lo / hi : 1.0f;

                /* LF-luma evidence at the pair's baseband difference
                 * frequency k - c, for the two single carriers */
                float g_e = 1.0f;
                if (t->evidence > 0.0f) {
                    static const int cy[2] = { 12, 4 };
                    float e = 0.0f;
                    for (int c = 0; c < 2; c++) {
                        const fftwf_complex *lf =
                            in + (((4 - z + ZTILE) % ZTILE) * YTILE_PAL
                                  + (cy[c] - y + YTILE_PAL) % YTILE_PAL) * XC
                               + (XTILE / 4) - x;
                        const float ec = (*lf)[0] * (*lf)[0] + (*lf)[1] * (*lf)[1];
                        e = ec > e ? ec : e;
                    }
                    const float den = e + t->evidence * hi;
                    if (den > 0.0f)
                        g_e = e / den;
                }

                if (t->level) {
                    float f_in = g_e, f_ref = g_e;
                    if (m_in_sq > m_ref_sq)
                        f_in *= sqrtf(m_ref_sq / m_in_sq);
                    else if (m_ref_sq > m_in_sq)
                        f_ref *= sqrtf(m_in_sq / m_ref_sq);
                    bo[x][0] = bi[x][0] * f_in;
                    bo[x][1] = bi[x][1] * f_in;
                    bo_ref[x_ref][0] = bi_ref[x_ref][0] * f_ref;
                    bo_ref[x_ref][1] = bi_ref[x_ref][1] * f_ref;
                    conf_num += 2.0f * lo * g_e * g_e * r;
                    conf_den += 2.0f * lo * g_e * g_e;
                    continue;
                }

                if (m_in_sq < m_ref_sq * threshold_sq ||
                    m_ref_sq < m_in_sq * threshold_sq)
                    continue;

                bo[x][0] = bi[x][0] * g_e;
                bo[x][1] = bi[x][1] * g_e;
                bo_ref[x_ref][0] = bi_ref[x_ref][0] * g_e;
                bo_ref[x_ref][1] = bi_ref[x_ref][1] * g_e;
                conf_num += (m_in_sq + m_ref_sq) * g_e * g_e * r;
                conf_den += (m_in_sq + m_ref_sq) * g_e * g_e;
            }
        }
    }

    return conf_den > 0.0f ? conf_num / conf_den : 0.0f;
}

static inline float dist_sq3(float a, float b, float c)
{
    return a * a + b * b + c * c;
}

/* NTSC frequency-domain filter, ported from the ld-decode transform-ntsc
 * branch (TransformNtsc3D). NTSC's U and V share one carrier, so chroma
 * is only approximately symmetric about it; the test compensates with a
 * frequency-shaped threshold (lenient near the chroma carrier at
 * (fsc, 120 c/aph, 15 Hz), strict near luma) and cross-checks each
 * candidate against the energy at its demodulated-luma positions k±c
 * (chroma with no corresponding luma is suspect). A trained LUT
 * replaces both devices — the per-bin gains learn the frequency
 * dependence directly — so the luma-evidence test is not applied in
 * LUT mode. Returns the tile's chroma confidence as apply_filter does. */
static float apply_filter_ntsc(const comp_transform3d_t *t,
                               const fftwf_complex *in, fftwf_complex *out)
{
    const float *tsq = t->threshold_sq;
    float conf_num = 0.0f, conf_den = 0.0f;

    memset(out, 0, sizeof(fftwf_complex) * TILE_CPLX);

    for (int z = 0; z < ZTILE; z++) {
        const int z_ref = ((ZTILE / 2) + ZTILE - z) % ZTILE;
        const int z_lr = (z - ZTILE / 4 + ZTILE) % ZTILE;
        const int z_lrn = (ZTILE - z_lr) % ZTILE;
        const float kz0 = (float)z / ZTILE;

        for (int y = 0; y < YTILE; y++) {
            const int y_ref = ((YTILE / 2) + YTILE - y) % YTILE;
            const int y_lr = (y - YTILE / 4 + YTILE) % YTILE;
            const int y_lrn = (YTILE - y_lr) % YTILE;
            const float ky0 = (float)y / YTILE;

            /* map (ky, kz) into the interlace-equivalence diamond */
            float ky = ky0, kz = kz0;
            if (kz0 + ky0 < 0.5f) {
                kz = kz0 + 0.5f;
                ky = ky0 + 0.5f;
            } else if (kz0 + ky0 > 1.5f) {
                kz = kz0 - 0.5f;
                ky = ky0 - 0.5f;
            } else if (kz0 - ky0 > 0.5f) {
                kz = kz0 - 0.5f;
                ky = ky0 + 0.5f;
            } else if (ky0 - kz0 > 0.5f) {
                kz = kz0 + 0.5f;
                ky = ky0 - 0.5f;
            }
            if (kz + ky > 1.0f) {
                kz = 1.0f - kz;
                ky = 1.0f - ky;
            }

            const fftwf_complex *bi = in + (z * YTILE + y) * XC;
            const fftwf_complex *bi_ref = in + (z_ref * YTILE + y_ref) * XC;
            const fftwf_complex *bi_lr = in + (z_lr * YTILE + y_lr) * XC;
            const fftwf_complex *bi_lrn = in + (z_lrn * YTILE + y_lrn) * XC;
            fftwf_complex *bo = out + (z * YTILE + y) * XC;
            fftwf_complex *bo_ref = out + (z_ref * YTILE + y_ref) * XC;

            for (int x = XTILE / 8; x <= XTILE / 4; x++) {
                const int x_ref = (XTILE / 2) - x;
                const int x_lr = x - XTILE / 4;
                const float kx = (float)x / XTILE;
                const float t0_sq = *tsq++;

                const fftwf_complex *lr1, *lr2;
                if (x_lr >= 0) {
                    lr1 = &bi_lr[x_lr];
                    lr2 = &bi_lrn[XTILE / 2 - x_lr];
                } else {
                    lr1 = &bi_lrn[-x_lr];
                    lr2 = &bi_lr[XTILE / 2 + x_lr];
                }

                if (x == x_ref) {
                    if ((y == YTILE / 4 && z == ZTILE / 4)
                        || (y == 3 * YTILE / 4 && z == 3 * ZTILE / 4)) {
                        /* its own reflection and a carrier: keep */
                        bo[x][0] = bi[x][0];
                        bo[x][1] = bi[x][1];
                        conf_num += bi[x][0] * bi[x][0] + bi[x][1] * bi[x][1];
                        conf_den += bi[x][0] * bi[x][0] + bi[x][1] * bi[x][1];
                        continue;
                    }
                    if (((y == 0 || y == YTILE / 2) && (z == 0 || z == ZTILE / 2))
                        || (y == YTILE / 4 && z == 3 * ZTILE / 4)
                        || (y == 3 * YTILE / 4 && z == ZTILE / 4)) {
                        /* its own reflection but not a carrier: discard */
                        continue;
                    }
                }

                const float m_in = bi[x][0] * bi[x][0] + bi[x][1] * bi[x][1];
                const float m_ref = bi_ref[x_ref][0] * bi_ref[x_ref][0]
                                  + bi_ref[x_ref][1] * bi_ref[x_ref][1];
                const float lo = m_in < m_ref ? m_in : m_ref;
                const float m_max = m_in > m_ref ? m_in : m_ref;
                const float r = m_max > 0.0f ? lo / m_max : 1.0f;

                const float l1 = (*lr1)[0] * (*lr1)[0] + (*lr1)[1] * (*lr1)[1];
                const float l2 = (*lr2)[0] * (*lr2)[0] + (*lr2)[1] * (*lr2)[1];
                const float m_luma = l1 > l2 ? l1 : l2;

                if (t->level) {
                    /* the reference's levelMode: discard the pair when
                     * it lacks luma evidence, pass it untouched inside
                     * a 10x squared-magnitude dead zone, and beyond
                     * that scale the larger down to the smaller */
                    if (m_max > 10.0f * m_luma)
                        continue;
                    float f_in = 1.0f, f_ref = 1.0f;
                    if (m_in > 10.0f * m_ref)
                        f_in = sqrtf(m_ref / m_in);
                    else if (m_ref > 10.0f * m_in)
                        f_ref = sqrtf(m_in / m_ref);
                    bo[x][0] = bi[x][0] * f_in;
                    bo[x][1] = bi[x][1] * f_in;
                    bo_ref[x_ref][0] = bi_ref[x_ref][0] * f_ref;
                    bo_ref[x_ref][1] = bi_ref[x_ref][1] * f_ref;
                    const float e = f_in * f_in * m_in + f_ref * f_ref * m_ref;
                    conf_num += e * r;
                    conf_den += e;
                    continue;
                }

                /* threshold shaped by proximity to chroma vs luma */
                const float k_luma = dist_sq3(kz - 0.5f, ky - 0.5f, kx);
                const float k_chroma = dist_sq3(kz - 0.25f, ky - 0.25f, kx - 0.25f);
                float th_sq = powf(k_chroma / (k_luma + k_chroma), 10.0f * t0_sq);

                if (m_luma < m_max * th_sq)
                    th_sq = 0.5f * (1.0f + th_sq);

                if (m_in < m_ref * th_sq || m_ref < m_in * th_sq)
                    continue;

                bo[x][0] = bi[x][0];
                bo[x][1] = bi[x][1];
                bo_ref[x_ref][0] = bi_ref[x_ref][0];
                bo_ref[x_ref][1] = bi_ref[x_ref][1];
                conf_num += (m_in + m_ref) * r;
                conf_den += m_in + m_ref;
            }
        }
    }

    return conf_den > 0.0f ? conf_num / conf_den : 0.0f;
}

/* the slab origin never takes this value: the grid is anchored at
 * absolute field 0 and the first frame's low slab sits at -ZTILE/2 */
#define SLAB_EMPTY INT_MIN

int comp_t3d_cache_init(comp_t3d_cache_t *c, int nslabs, int width,
                        int field_rows, int with_conf)
{
    if (nslabs < 2 || width < 1 || field_rows < 1)
        return -1;
    memset(c, 0, sizeof(*c));
    pthread_mutex_init(&c->lock, NULL);
    pthread_cond_init(&c->cond, NULL);
    c->slabs = calloc(nslabs, sizeof(*c->slabs));
    if (!c->slabs) {
        pthread_mutex_destroy(&c->lock);
        pthread_cond_destroy(&c->cond);
        return -1;
    }
    c->nslabs = nslabs;
    c->width = width;
    c->field_rows = field_rows;

    const size_t planes = sizeof(float) * ZTILE * field_rows * width;
    for (int i = 0; i < nslabs; i++) {
        c->slabs[i].tz = SLAB_EMPTY;
        c->slabs[i].chroma = malloc(planes);
        if (with_conf)
            c->slabs[i].conf = malloc(planes);
        if (!c->slabs[i].chroma || (with_conf && !c->slabs[i].conf)) {
            comp_t3d_cache_free(c);
            return -1;
        }
    }
    return 0;
}

void comp_t3d_cache_free(comp_t3d_cache_t *c)
{
    if (!c->slabs)
        return;
    for (int i = 0; i < c->nslabs; i++) {
        free(c->slabs[i].chroma);
        free(c->slabs[i].conf);
    }
    free(c->slabs);
    c->slabs = NULL;
    pthread_mutex_destroy(&c->lock);
    pthread_cond_destroy(&c->cond);
}

static comp_t3d_slab_t *cache_lookup(comp_t3d_cache_t *c, int tz, int parity)
{
    for (int i = 0; i < c->nslabs; i++)
        if (c->slabs[i].tz == tz && c->slabs[i].parity == parity)
            return &c->slabs[i];
    return NULL;
}

static comp_t3d_slab_t *cache_victim(comp_t3d_cache_t *c,
                                     const comp_t3d_slab_t *ex1,
                                     const comp_t3d_slab_t *ex2)
{
    comp_t3d_slab_t *v = NULL;
    for (int i = 0; i < c->nslabs; i++) {
        comp_t3d_slab_t *s = &c->slabs[i];
        if (s == ex1 || s == ex2 || s->refs > 0)
            continue;
        if (s->tz == SLAB_EMPTY)
            return s;
        if (!v || s->stamp < v->stamp)
            v = s;
    }
    return v;
}

/* Take references on the slabs for tz1 and tz2, claiming missing ones
 * for the caller to build (*build set, ready cleared). Both slots are
 * secured under one lock acquisition and a thread that cannot get both
 * holds nothing while it waits, so waiters cannot deadlock: builders
 * never block, and every wait is for a builder to finish or a
 * reference to drop. */
static void cache_acquire(comp_t3d_cache_t *c, int parity, int tz1, int tz2,
                          comp_t3d_slab_t **s1, comp_t3d_slab_t **s2,
                          int *build1, int *build2)
{
    pthread_mutex_lock(&c->lock);
    for (;;) {
        comp_t3d_slab_t *a = cache_lookup(c, tz1, parity);
        comp_t3d_slab_t *b = cache_lookup(c, tz2, parity);
        comp_t3d_slab_t *va = a ? NULL : cache_victim(c, b, NULL);
        comp_t3d_slab_t *vb = b ? NULL : cache_victim(c, a, va);

        if ((a || va) && (b || vb)) {
            if (!a) {
                a = va;
                a->tz = tz1;
                a->parity = parity;
                a->ready = 0;
                *build1 = 1;
            }
            if (!b) {
                b = vb;
                b->tz = tz2;
                b->parity = parity;
                b->ready = 0;
                *build2 = 1;
            }
            a->refs++;
            b->refs++;
            a->stamp = ++c->counter;
            b->stamp = ++c->counter;
            *s1 = a;
            *s2 = b;
            pthread_mutex_unlock(&c->lock);
            return;
        }
        pthread_cond_wait(&c->cond, &c->lock);
    }
}

/* Overlap-add every tile at temporal grid position slab->tz into the
 * slab's ZTILE contribution planes. width/field_rows give the frame
 * geometry (at most the cache's capacity) and define the plane layout,
 * which the assembly in comp_transform3d_frame mirrors. */
static void build_slab_ntsc(const comp_transform3d_t *t, comp_t3d_slab_t *s,
                            const comp_field_view_t *fields, int z0, int nfields,
                            int width, int field_rows)
{
    ALIGNED_32( float real[TILE_REAL] );
    ALIGNED_32( union t3d_freq cplx_in );
    ALIGNED_32( union t3d_freq cplx_out );
    const int use_fft = t->fft_fwd && t->use_lut && t->evidence == 0.0f;
    const int frame_lines = field_rows * 2;
    const int tz = s->tz;
    const int parity = s->parity;
    const size_t plane = (size_t)field_rows * width;
    /* bit z: the y parity of field tz + z's rows in every tile (tile_y
     * is even); the internal FFT takes the other rows' spectra from
     * t->fft_black, so they need no filling */
    unsigned ph = 0;
    for (int z = 0; z < ZTILE; z++)
        ph |= (unsigned)((tz + z + parity) & 1) << z;

    memset(s->chroma, 0, sizeof(float) * ZTILE * plane);
    if (s->conf)
        memset(s->conf, 0, sizeof(float) * ZTILE * plane);

    for (int tile_y = -YTILE / 2; tile_y < frame_lines; tile_y += YTILE / 2) {
        const int start_y = tile_y < 0 ? -tile_y : 0;
        const int end_y = frame_lines - tile_y < YTILE ? frame_lines - tile_y : YTILE;

        for (int tile_x = -XTILE / 2; tile_x < width; tile_x += XTILE / 2) {
            const int start_x = tile_x < 0 ? -tile_x : 0;
            const int end_x = width - tile_x < XTILE ? width - tile_x : XTILE;

            /* windowed forward FFT: a row is real data only when it
             * is inside the picture and belongs to field tz+z */
            for (int z = 0; z < ZTILE; z++) {
                const int g = tz + z;
                const comp_field_view_t *fv =
                    (g >= z0 && g < z0 + nfields) ? &fields[g - z0] : NULL;
                if (fv && !fv->data)
                    fv = NULL;
                for (int y = 0; y < YTILE; y++) {
                    if (use_fft && (y & 1) != (int)((ph >> z) & 1))
                        continue;
                    const int fl = tile_y + y;
                    const int usable = fv && y >= start_y && y < end_y
                                       && ((fl + parity) & 1) == (g & 1);
                    const uint16_t *b = usable
                        ? fv->data + (fl >> 1) * fv->stride : NULL;
                    float *dst = &real[(z * YTILE + y) * XTILE];
                    const float *win = t->window[z][y];
                    /* edge tests hoisted so the sample loops are
                     * branch-free and autovectorizable */
                    const int x0 = usable ? start_x : XTILE;
                    const int x1 = usable ? end_x : XTILE;

                    for (int x = 0; x < x0; x++)
                        dst[x] = LEVEL_BLACK * win[x];
                    for (int x = x0; x < x1; x++)
                        dst[x] = (float)b[tile_x + x] * win[x];
                    for (int x = x1; x < XTILE; x++)
                        dst[x] = LEVEL_BLACK * win[x];
                }
            }
            float w;
            if (use_fft) {
                t->fft_fwd(cplx_in.band, real, t->fft_black, ph);
                w = apply_filter_lut(t, cplx_in.band, cplx_out.band,
                                     t->rowpair_fft, t->special_fft,
                                     ZTILE * YTILE * COMP_FFT_ROWSTRIDE);
                t->fft_inv(real, cplx_out.band, ph);
            } else {
                fftwf_execute_dft_r2c(t->forward, real, cplx_in.cplx);
                w = t->use_lut
                    ? apply_filter_lut(t, (const float *)cplx_in.cplx,
                                       (float *)cplx_out.cplx, t->rowpair,
                                       t->special, 2 * TILE_CPLX)
                    : apply_filter_ntsc(t, cplx_in.cplx, cplx_out.cplx);
                fftwf_execute_dft_c2r(t->inverse, cplx_out.cplx, real);
            }

            /* the confidence uses the same unity-sum window weights */
            for (int z = 0; z < ZTILE; z++) {
                const int g = tz + z;
                float *cb = s->chroma + z * plane;
                float *wb = s->conf ? s->conf + z * plane : NULL;
                for (int y = start_y; y < end_y; y++) {
                    const int fl = tile_y + y;
                    if (((fl + parity) & 1) != (g & 1))
                        continue;
                    float *b = cb + (fl >> 1) * width;
                    for (int x = start_x; x < end_x; x++)
                        b[tile_x + x] += real[(z * YTILE + y) * XTILE + x]
                                         / (ZTILE * YTILE * XTILE);
                    if (wb) {
                        float *bw = wb + (fl >> 1) * width;
                        for (int x = start_x; x < end_x; x++)
                            bw[tile_x + x] += w * t->window[z][y][x];
                    }
                }
            }
        }
    }
}

/* PAL slab builder on the displaced lattice: field z is displaced down
 * by z picture lines, i.e. ceil(z/2) field rows, making the stack a
 * dense 16-row field-line lattice; the inverse displacement is folded
 * into the overlap-add row mapping. Rows have no parity test — every
 * tile row belongs to its field. */
static void build_slab_pal(const comp_transform3d_t *t, comp_t3d_slab_t *s,
                           const comp_field_view_t *fields, int z0, int nfields,
                           int width, int field_rows)
{
    ALIGNED_32( float real[TILE_REAL_PAL] );
    ALIGNED_32( union t3d_freq_pal cplx_in );
    ALIGNED_32( union t3d_freq_pal cplx_out );
    const int use_fft = t->fft_fwd && t->use_lut && t->evidence == 0.0f;
    const int tz = s->tz;
    const size_t plane = (size_t)field_rows * width;

    memset(s->chroma, 0, sizeof(float) * ZTILE * plane);
    if (s->conf)
        memset(s->conf, 0, sizeof(float) * ZTILE * plane);

    for (int tile_r = -YTILE_PAL / 2; tile_r < field_rows + SHIFT_MAX;
         tile_r += YTILE_PAL / 2) {
        for (int tile_x = -XTILE / 2; tile_x < width; tile_x += XTILE / 2) {
            const int start_x = tile_x < 0 ? -tile_x : 0;
            const int end_x = width - tile_x < XTILE ? width - tile_x : XTILE;

            for (int z = 0; z < ZTILE; z++) {
                const int g = tz + z;
                const int sz = (z + 1) / 2;
                const comp_field_view_t *fv =
                    (g >= z0 && g < z0 + nfields) ? &fields[g - z0] : NULL;
                if (fv && !fv->data)
                    fv = NULL;
                for (int y = 0; y < YTILE_PAL; y++) {
                    const int r = tile_r + y - sz;
                    const int usable = fv && r >= 0 && r < field_rows;
                    const uint16_t *b = usable ? fv->data + r * fv->stride : NULL;
                    float *dst = &real[(z * YTILE_PAL + y) * XTILE];
                    const float *win = t->window[z][y];
                    /* edge tests hoisted so the sample loops are
                     * branch-free and autovectorizable */
                    const int x0 = usable ? start_x : XTILE;
                    const int x1 = usable ? end_x : XTILE;

                    for (int x = 0; x < x0; x++)
                        dst[x] = LEVEL_BLACK * win[x];
                    for (int x = x0; x < x1; x++)
                        dst[x] = (float)b[tile_x + x] * win[x];
                    for (int x = x1; x < XTILE; x++)
                        dst[x] = LEVEL_BLACK * win[x];
                }
            }
            float w;
            if (use_fft) {
                t->fft_fwd(cplx_in.band, real, NULL, 0);
                w = apply_filter_lut(t, cplx_in.band, cplx_out.band,
                                     t->rowpair_fft, t->special_fft,
                                     ZTILE * YTILE_PAL * COMP_FFT_ROWSTRIDE);
                t->fft_inv(real, cplx_out.band, 0);
            } else {
                fftwf_execute_dft_r2c(t->forward, real, cplx_in.cplx);
                w = t->use_lut
                    ? apply_filter_lut(t, (const float *)cplx_in.cplx,
                                       (float *)cplx_out.cplx, t->rowpair,
                                       t->special, 2 * TILE_CPLX_PAL)
                    : apply_filter_pal(t, cplx_in.cplx, cplx_out.cplx);
                fftwf_execute_dft_c2r(t->inverse, cplx_out.cplx, real);
            }

            for (int z = 0; z < ZTILE; z++) {
                const int sz = (z + 1) / 2;
                float *cb = s->chroma + z * plane;
                float *wb = s->conf ? s->conf + z * plane : NULL;
                for (int y = 0; y < YTILE_PAL; y++) {
                    const int r = tile_r + y - sz;
                    if (r < 0 || r >= field_rows)
                        continue;
                    float *b = cb + (size_t)r * width;
                    for (int x = start_x; x < end_x; x++)
                        b[tile_x + x] += real[(z * YTILE_PAL + y) * XTILE + x]
                                         / (ZTILE * YTILE_PAL * XTILE);
                    if (wb) {
                        float *bw = wb + (size_t)r * width;
                        for (int x = start_x; x < end_x; x++)
                            bw[tile_x + x] += w * t->window[z][y][x];
                    }
                }
            }
        }
    }
}

void comp_transform3d_frame(const comp_transform3d_t *t, comp_t3d_cache_t *c,
                            const comp_field_view_t *fields, int z0, int nfields,
                            int frame, int parity, int width, int field_rows,
                            int16_t *chroma0, int16_t *chroma1,
                            ptrdiff_t chroma_stride,
                            float *conf0, float *conf1, ptrdiff_t conf_stride)
{
    int16_t *chroma[2] = { chroma0, chroma1 };
    float *conf[2] = { conf0, conf1 };
    const int fout = frame * 2;

    /* the two half-overlapped z positions whose tiles cover this frame's
     * fields; the grid is anchored at absolute field 0 */
    const int tz_hi = ((fout + 1) / (ZTILE / 2)) * (ZTILE / 2);
    comp_t3d_slab_t *s1, *s2;
    int build1 = 0, build2 = 0;

    cache_acquire(c, parity, tz_hi - ZTILE / 2, tz_hi, &s1, &s2,
                  &build1, &build2);
    if (build1)
        (t->standard == COMP_STD_PAL ? build_slab_pal : build_slab_ntsc)
            (t, s1, fields, z0, nfields, width, field_rows);
    if (build2)
        (t->standard == COMP_STD_PAL ? build_slab_pal : build_slab_ntsc)
            (t, s2, fields, z0, nfields, width, field_rows);

    pthread_mutex_lock(&c->lock);
    if (build1)
        s1->ready = 1;
    if (build2)
        s2->ready = 1;
    if (build1 || build2)
        pthread_cond_broadcast(&c->cond);
    while (!s1->ready || !s2->ready)
        pthread_cond_wait(&c->cond, &c->lock);
    pthread_mutex_unlock(&c->lock);

    /* each output field is the sum of the two covering slabs' planes */
    const size_t plane = (size_t)field_rows * width;
    for (int f = 0; f < 2; f++) {
        const int g = fout + f;
        const float *p1 = s1->chroma + (g - s1->tz) * plane;
        const float *p2 = s2->chroma + (g - s2->tz) * plane;
        int16_t *dc = chroma[f ^ parity];
        float *dw = conf[f ^ parity];
        /* sum the two covering slabs and quantise to the demod's
         * fixed-point input in one pass -- the float sum is never
         * spilled to a frame plane the demod would re-read */
        for (int r = 0; r < field_rows; r++)
            for (int x = 0; x < width; x++)
                dc[r * chroma_stride + x] =
                    clamp_i16(round_i32(p1[r * width + x] + p2[r * width + x]));
        if (dw) {
            const float *q1 = s1->conf + (g - s1->tz) * plane;
            const float *q2 = s2->conf + (g - s2->tz) * plane;
            for (int r = 0; r < field_rows; r++)
                for (int x = 0; x < width; x++)
                    dw[r * conf_stride + x] = q1[r * width + x] + q2[r * width + x];
        }
    }

    pthread_mutex_lock(&c->lock);
    s1->refs--;
    s2->refs--;
    pthread_cond_broadcast(&c->cond);
    pthread_mutex_unlock(&c->lock);
}
