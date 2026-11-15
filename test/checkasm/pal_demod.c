#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <checkasm/checkasm.h>
#include <checkasm/test.h>
#include <checkasm/utils.h>

#include "decode.h"
#include "subcarrier.h"

#define FS COMP_DECODE_FILTER_SIZE
#define MAXW 928
#define STRIDE1 (COMP_PAL_DEMOD_ALIGN(MAXW) + 2 * FS)  /* the production layout */
#define STRIDE2 (STRIDE1 + 50)                         /* catch stride bugs */
#define MAXSTRIDE STRIDE2

static CHECKASM_ALIGN(int16_t pr_buf[4 * 2 * MAXSTRIDE]);
static CHECKASM_ALIGN(int16_t coef[COMP_PAL_DEMOD_ROWS][32]);
static CHECKASM_ALIGN(int32_t u_c[COMP_PAL_DEMOD_ALIGN(MAXW)]);
static CHECKASM_ALIGN(int32_t v_c[COMP_PAL_DEMOD_ALIGN(MAXW)]);
static CHECKASM_ALIGN(int32_t u_a[COMP_PAL_DEMOD_ALIGN(MAXW)]);
static CHECKASM_ALIGN(int32_t v_a[COMP_PAL_DEMOD_ALIGN(MAXW)]);

static const int widths[] = { 8, 16, 24, 100, 512, 720, 921, 928 };
#define NWIDTHS (int)(sizeof(widths) / sizeof(widths[0]))

/* the row of tap b, side s (0: x + b, 1: x - b), plane k */
static int row_of(int b, int s, int k)
{
    return b ? 4 + (b - 1) * 8 + s * 4 + k : k;
}

/* every 17th output sees every word under its taps at full scale with
 * the sign of that lane's coefficient, driving both sums to their bound */
static void worst_case(ptrdiff_t stride)
{
    for (int x0 = 0; x0 + 2 * FS < MAXW; x0 += 17) {
        const int ph = x0 & 3;
        for (int b = 0; b <= FS; b++)
            for (int s = 0; s < (b ? 2 : 1); s++)
                for (int k = 0; k < 4; k++) {
                    const int16_t *c = coef[row_of(b, s, k)] + 2 * ph;
                    int16_t *p = pr_buf + 2 * (k * stride + FS + x0 + (s ? -b : b));
                    p[0] = c[0] < 0 ? -32768 : 32767;
                    p[1] = c[1] < 0 ? -32768 : 32767;
                }
    }
}

void checkasm_test_pal_demod(void)
{
    checkasm_declare(void, int32_t *, int32_t *, const int16_t *, ptrdiff_t,
                     const int16_t (*)[32], int, int32_t, int32_t, int32_t);
    if (checkasm_check_func(comp_get_pal_demod_fn(checkasm_get_cpu_flags()),
                            "pal_demod_row")) {
        comp_decode_t *d = calloc(1, sizeof(*d));
        if (!d || comp_decode_init(d, COMP_STD_PAL, 0.5, 1, 0, 2, 0, 0, 0, 0, 0.0, 0)) {
            free(d);
            checkasm_fail_func("decoder init");
            return;
        }
        /* the decoder's filter, then random ones: an output sums at most
         * 56 coefficient words (b = 0 doubled), so |c| < 500 keeps it
         * under 2^30 */
        for (int set = 0; set < 2; set++) {
            if (set == 0) {
                memcpy(coef, d->pal_demod_coef, sizeof(coef));
            } else {
                int32_t cf[FS + 1][4];
                for (int b = 0; b <= FS; b++)
                    for (int k = 0; k < 4; k++)
                        cf[b][k] = (int32_t)(checkasm_rand() % 999) - 499;
                comp_pal_demod_init(coef, (const int32_t (*)[4])cf);
            }
            for (int mode = 0; mode < 3; mode++) {
                for (int i = 0; i < NWIDTHS; i++) {
                    const ptrdiff_t stride = i & 1 ? STRIDE2 : STRIDE1;
                    /* [-32767, 32768], the negated sine table's range,
                     * with the +32768 end forced on two widths */
                    const int32_t bp = i == 2 ? 32768
                                     : (int32_t)(checkasm_rand() % 65536) - 32767;
                    const int32_t bq = i == 3 ? 32768
                                     : (int32_t)(checkasm_rand() % 65536) - 32767;
                    const int32_t vsw = i & 2 ? -1 : 1;
                    for (int k = 0; k < 4 * 2 * MAXSTRIDE; k++)
                        pr_buf[k] = mode == 2 ? -32768
                                  : (int16_t)(checkasm_rand() % 65536 - 32768);
                    if (mode == 1)
                        worst_case(stride);
                    memset(u_c, 0xaa, sizeof(u_c));
                    memset(v_c, 0xaa, sizeof(v_c));
                    memset(u_a, 0x55, sizeof(u_a));
                    memset(v_a, 0x55, sizeof(v_a));
                    checkasm_call_ref(u_c, v_c, pr_buf, stride,
                                      (const int16_t (*)[32])coef,
                                      widths[i], bp, bq, vsw);
                    checkasm_call_new(u_a, v_a, pr_buf, stride,
                                      (const int16_t (*)[32])coef,
                                      widths[i], bp, bq, vsw);
                    checkasm_check1d(int32_t, u_c, u_a, widths[i], "u");
                    checkasm_check1d(int32_t, v_c, v_a, widths[i], "v");
                }
            }
        }
        memcpy(coef, d->pal_demod_coef, sizeof(coef));
        checkasm_bench_new(u_a, v_a, pr_buf, (ptrdiff_t)STRIDE1,
                           (const int16_t (*)[32])coef, MAXW, 12345, -23456, 1);
        comp_decode_free(d);
        free(d);
    }
    checkasm_report("pal_demod");
}
