#include <stdint.h>
#include <string.h>

#include <checkasm/checkasm.h>
#include <checkasm/test.h>
#include <checkasm/utils.h>

#include "fft.h"
#include "subcarrier.h"

#define ZT COMP_T3D_ZTILE
#define XT COMP_T3D_XTILE
#define NREAL (ZT * COMP_T3D_YTILE * XT)
#define NBAND (ZT * COMP_T3D_YTILE * COMP_FFT_ROWSTRIDE)

static CHECKASM_ALIGN(float rin[NREAL]);
static CHECKASM_ALIGN(float band_c[NBAND]), band_a[NBAND];
static CHECKASM_ALIGN(float rout_c[NREAL]), rout_a[NREAL];
/* NTSC tiles take the other field's rows from fixed spectra; random
 * ones here, so a mixed-up row shows */
static CHECKASM_ALIGN(float black[NBAND]);

static const int standards[] = { COMP_STD_PAL, COMP_STD_NTSC };

/* Inputs are bounded to +-1024, so intermediates stay under 2^22 and
 * one differing float rounding is under 0.5; the asm reorders roughly
 * a dozen roundings per output (FMA and batching), giving the eps
 * bound. The ULP budget covers the large-magnitude bins. */
#define FFT_EPS 8.0f
#define FFT_ULP 64

static void check_floats(const float *c, const float *a, int n, const char *what)
{
    for (int i = 0; i < n; i++)
        if (!checkasm_float_near_abs_eps_ulp(c[i], a[i], FFT_EPS, FFT_ULP)) {
            checkasm_fail_func("%s[%d]: %.9g vs %.9g", what, i, c[i], a[i]);
            return;
        }
}

void checkasm_test_fft(void)
{
    {
        checkasm_declare(void, float *, const float *, const float *, unsigned);
        for (int s = 0; s < 2; s++) {
            const int pal = standards[s] == COMP_STD_PAL;
            const int yt = pal ? COMP_T3D_YTILE_PAL : COMP_T3D_YTILE;
            if (checkasm_check_func(comp_get_fft_fwd_fn(standards[s],
                                                        checkasm_get_cpu_flags()),
                                    pal ? "fft_fwd_pal" : "fft_fwd_ntsc")) {
                for (int i = 0; i < NREAL; i++)
                    rin[i] = (float)((int)(checkasm_rand() & 2047) - 1024);
                for (int i = 0; i < NBAND; i++)
                    black[i] = (float)((int)(checkasm_rand() & 2047) - 1024);
                const unsigned ph = checkasm_rand() & 0xff;
                memset(band_c, 0xaa, sizeof(band_c));
                memset(band_a, 0xaa, sizeof(band_a));
                checkasm_call_ref(band_c, rin, black, ph);
                checkasm_call_new(band_a, rin, black, ph);
                for (int r = 0; r < ZT * yt; r++)
                    check_floats(band_c + r * COMP_FFT_ROWSTRIDE,
                                 band_a + r * COMP_FFT_ROWSTRIDE,
                                 COMP_FFT_NCOL * 2, "band");
                checkasm_bench_new(band_a, rin, black, 0x55u);
            }
        }
    }
    {
        checkasm_declare(void, float *, float *, unsigned);
        for (int s = 0; s < 2; s++) {
            const int pal = standards[s] == COMP_STD_PAL;
            const int yt = pal ? COMP_T3D_YTILE_PAL : COMP_T3D_YTILE;
            if (checkasm_check_func(comp_get_fft_inv_fn(standards[s],
                                                        checkasm_get_cpu_flags()),
                                    pal ? "fft_inv_pal" : "fft_inv_ntsc")) {
                memset(band_c, 0, sizeof(band_c));
                for (int r = 0; r < ZT * yt; r++)
                    for (int c = 0; c < COMP_FFT_NCOL * 2; c++)
                        band_c[r * COMP_FFT_ROWSTRIDE + c] =
                            (float)((int)(checkasm_rand() & 2047) - 1024);
                memcpy(band_a, band_c, sizeof(band_c));
                memset(rout_c, 0xaa, sizeof(rout_c));
                memset(rout_a, 0xaa, sizeof(rout_a));
                /* the inverse consumes its input; each side gets a copy.
                 * NTSC leaves the other field's rows unwritten, so they
                 * keep the same fill on both sides. */
                const unsigned ph = checkasm_rand() & 0xff;
                checkasm_call_ref(rout_c, band_c, ph);
                checkasm_call_new(rout_a, band_a, ph);
                check_floats(rout_c, rout_a, ZT * yt * XT, "real");
                memset(band_a, 0, sizeof(band_a));
                checkasm_bench_new(rout_a, band_a, 0x55u);
            }
        }
    }
    checkasm_report("fft");
}
