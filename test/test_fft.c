/*
 * Cross-validate the internal fixed-size 3D real FFT against FFTW on
 * random tiles, for both standards: forward band columns (at
 * bit-reversed ky/kz) and inverse from band-limited spectra.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fftw3.h>

#include "fft.h"
#include "subcarrier.h"

#define ZT COMP_T3D_ZTILE
#define XT COMP_T3D_XTILE
#define XC COMP_T3D_XCOMPLEX
#define NCOL COMP_FFT_NCOL
#define C0 (XT / 8)

static int fail = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        fail = 1; \
    } \
} while (0)

static int bitrev(int v, int bits)
{
    int r = 0;
    for (int b = 0; b < bits; b++)
        r = (r << 1) | ((v >> b) & 1);
    return r;
}

static void test_standard(int standard)
{
    const int pal = standard == COMP_STD_PAL;
    const int yt = pal ? COMP_T3D_YTILE_PAL : COMP_T3D_YTILE;
    const int ybits = pal ? 4 : 5;
    const int nreal = ZT * yt * XT;
    const int ncplx = ZT * yt * XC;

    float *rin = fftwf_alloc_real(nreal);
    float *rout = fftwf_alloc_real(nreal);
    fftwf_complex *cf = fftwf_alloc_complex(ncplx);
    fftwf_complex *cs = fftwf_alloc_complex(ncplx);
    float *band = malloc(sizeof(float) * ZT * yt * COMP_FFT_ROWSTRIDE);
    float *band2 = malloc(sizeof(float) * ZT * yt * COMP_FFT_ROWSTRIDE);
    float *rmine = malloc(sizeof(float) * nreal);

    fftwf_plan pf = fftwf_plan_dft_r2c_3d(ZT, yt, XT, rin, cf, FFTW_ESTIMATE);
    fftwf_plan pi = fftwf_plan_dft_c2r_3d(ZT, yt, XT, cs, rout, FFTW_ESTIMATE);
    comp_fft_fwd_fn fwd = comp_get_fft_fwd_fn(standard, 0);
    comp_fft_inv_fn inv = comp_get_fft_inv_fn(standard, 0);

    /* windowed-tile-like magnitudes */
    srand(pal ? 77 : 42);
    for (int i = 0; i < nreal; i++)
        rin[i] = (float)(rand() % 53248) * 0.5f;

    /* forward: band columns match FFTW at bit-reversed (kz, ky) */
    /* NTSC: rows of the other field come from the black table; built
     * from rin itself, the forward is then the whole transform */
    const unsigned ph = 0x5A;
    float *black = malloc(sizeof(float) * ZT * COMP_T3D_YTILE * COMP_FFT_ROWSTRIDE);
    if (!pal)
        comp_fft_x_rows_ntsc(black, rin, 0);
    fftwf_execute(pf);
    fwd(band, rin, pal ? NULL : black, ph);
    double emax = 0.0, mmax = 0.0;
    for (int z = 0; z < ZT; z++)
        for (int y = 0; y < yt; y++)
            for (int c = 0; c < NCOL; c++) {
                const float *m = band
                    + ((bitrev(z, 3) * yt + bitrev(y, ybits)) * (COMP_FFT_ROWSTRIDE / 2) + c) * 2;
                const fftwf_complex *f = &cf[(z * yt + y) * XC + C0 + c];
                const double dr = m[0] - (*f)[0], di = m[1] - (*f)[1];
                const double e = sqrt(dr * dr + di * di);
                const double mag = sqrt((*f)[0] * (*f)[0] + (*f)[1] * (*f)[1]);
                if (e > emax) emax = e;
                if (mag > mmax) mmax = mag;
            }
    printf("test_fft: %s fwd rel err %.2e\n", pal ? "pal" : "ntsc", emax / mmax);
    CHECK(emax / mmax < 1e-5, "forward error %g too large", emax / mmax);

    /* inverse: band-limited spectrum, FFTW c2r vs ours */
    memset(cs, 0, sizeof(fftwf_complex) * ncplx);
    for (int z = 0; z < ZT; z++)
        for (int y = 0; y < yt; y++)
            for (int c = 0; c < NCOL; c++) {
                const int i = (z * yt + y) * XC + C0 + c;
                cs[i][0] = cf[i][0];
                cs[i][1] = cf[i][1];
                float *m = band2
                    + ((bitrev(z, 3) * yt + bitrev(y, ybits)) * (COMP_FFT_ROWSTRIDE / 2) + c) * 2;
                m[0] = cf[i][0];
                m[1] = cf[i][1];
            }
    fftwf_execute(pi);
    inv(rmine, band2, ph);
    emax = 0.0; mmax = 0.0;
    for (int i = 0; i < nreal; i++) {
        /* NTSC writes only each z's rows of parity bit z of ph */
        const int z = i / (yt * XT), y = (i / XT) % yt;
        if (!pal && (y & 1) != (int)((ph >> z) & 1))
            continue;
        const double e = fabs(rmine[i] - rout[i]);
        if (e > emax) emax = e;
        if (fabs(rout[i]) > mmax) mmax = fabs(rout[i]);
    }
    printf("test_fft: %s inv rel err %.2e\n", pal ? "pal" : "ntsc", emax / mmax);
    CHECK(emax / mmax < 1e-5, "inverse error %g too large", emax / mmax);

    fftwf_destroy_plan(pf);
    fftwf_destroy_plan(pi);
    fftwf_free(rin); fftwf_free(rout); fftwf_free(cf); fftwf_free(cs);
    free(band); free(band2); free(rmine); free(black);
}

int main(void)
{
    test_standard(COMP_STD_NTSC);
    test_standard(COMP_STD_PAL);
    return fail;
}
