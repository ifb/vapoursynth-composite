#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <checkasm/checkasm.h>
#include <checkasm/test.h>
#include <checkasm/utils.h>

#include "decode.h"
#include "encode.h"

#define MAXW 928
#define MAXTAPS COMP_EQ_TAPS
#define INLEN (COMP_FIR_ROW_ALIGN(MAXW) + MAXTAPS - 1)

static CHECKASM_ALIGN(int32_t in_buf[INLEN]);
static CHECKASM_ALIGN(int32_t coef_buf[MAXTAPS]);
static CHECKASM_ALIGN(int32_t out_c[COMP_FIR_S16_ALIGN(MAXW)]);
static CHECKASM_ALIGN(int32_t out_a[COMP_FIR_S16_ALIGN(MAXW)]);
/* the eq's int16 taps are the longest fir_row_s16 filter */
_Static_assert(COMP_UV_MAXTAPS <= COMP_EQ_TAPS, "fir_row_s16 test buffers");
#define INLEN16 COMP_FIR_S16_SPAN(MAXW, COMP_EQ_TAPS)
static CHECKASM_ALIGN(int16_t in16[INLEN16]);
static CHECKASM_ALIGN(int16_t coef16[COMP_EQ_TAPS + 1]);

/* the production widths plus ragged and tiny ones */
static const int widths[] = { 1, 7, 8, 16, 100, 501, 754, 758, 928 };
#define NWIDTHS (int)(sizeof(widths) / sizeof(widths[0]))
/* 13 is the decoder's narrow chroma LP, 31 the eq boost; 9 and 17
 * ragged lengths */
static const int tapss[] = { 9, COMP_NARROW_TAPS, 17, COMP_EQ_TAPS };
#define NTAPS (int)(sizeof(tapss) / sizeof(tapss[0]))

/* magnitudes within the kernel contract |acc| < 2^45: up to 31 taps of
 * |coef| < 2^17 (the 12 dB eq boost ceiling) times |in| < 2^22 */
static void fill_inputs(void)
{
    for (int i = 0; i < INLEN; i++)
        in_buf[i] = (int32_t)(checkasm_rand() % 8388605) - 4194302;
    for (int i = 0; i < MAXTAPS; i++)
        coef_buf[i] = (int32_t)(checkasm_rand() % 262141) - 131070;
}

static void check_fir_row_s16(void);

void checkasm_test_fir_row(void)
{
    checkasm_declare(void, int32_t *, const int32_t *, const int32_t *,
                     int, int);
    if (checkasm_check_func(comp_get_fir_row_q15_fn(checkasm_get_cpu_flags()),
                            "fir_row_q15")) {
        for (int t = 0; t < NTAPS; t++) {
            for (int i = 0; i < NWIDTHS; i++) {
                fill_inputs();
                /* the kernel's contract: symmetric taps */
                for (int j = 0; j < tapss[t] / 2; j++)
                    coef_buf[tapss[t] - 1 - j] = coef_buf[j];
                memset(out_c, 0xaa, sizeof(out_c));
                memset(out_a, 0x55, sizeof(out_a));
                checkasm_call_ref(out_c, in_buf, coef_buf, tapss[t], widths[i]);
                checkasm_call_new(out_a, in_buf, coef_buf, tapss[t], widths[i]);
                checkasm_check1d(int32_t, out_c, out_a, widths[i], "out");
            }
        }
        fill_inputs();
        checkasm_bench_new(out_a, in_buf, coef_buf, COMP_EQ_TAPS, MAXW);
    }
    checkasm_report("fir_row");
    check_fir_row_s16();
}

/* the wide mode's contract value, computed in 64 bits */
static int32_t fir_s16_exact(const int16_t *in, const int16_t *coef, int taps)
{
    int64_t acc = 0;
    for (int j = 0; j < taps; j++)
        acc += (int64_t)coef[j] * in[j];
    return (int32_t)((acc + 16384) >> 15);
}

static void check_fir_row_s16(void)
{
    checkasm_declare(void, int32_t *, const int16_t *, const int16_t *,
                     int, int, int);
    if (checkasm_check_func(comp_get_fir_row_s16_fn(checkasm_get_cpu_flags()),
                            "fir_row_s16")) {
        /* the encoders' own taps, then random ones within the contract
         * sum|coef| * 2^15 + 2^14 < 2^31 */
        comp_encode_t enc[2];
        comp_encode_init(&enc[0], COMP_STD_PAL, 0, 0);
        comp_encode_init(&enc[1], COMP_STD_NTSC, 0, 0);
        for (int set = 0; set < 3; set++) {
            const int taps = set < 2 ? enc[set].uv_ntaps : 13;
            if (set < 2) {
                memcpy(coef16, enc[set].uv_taps16, sizeof(coef16[0]) * (taps + 1));
            } else {
                for (int j = 0; j < taps; j++)
                    coef16[j] = (int16_t)(checkasm_rand() % 10001) - 5000;
                coef16[taps] = 0;
            }
            /* random rows, then full scale of each sign */
            for (int mode = 0; mode < 3; mode++) {
                for (int i = 0; i < NWIDTHS; i++) {
                    for (int k = 0; k < INLEN16; k++)
                        in16[k] = mode == 0 ? (int16_t)(checkasm_rand() % 65536 - 32768)
                                : mode == 1 ? 32767 : -32768;
                    memset(out_c, 0xaa, sizeof(out_c));
                    memset(out_a, 0x55, sizeof(out_a));
                    checkasm_call_ref(out_c, in16, coef16, taps, widths[i], 0);
                    checkasm_call_new(out_a, in16, coef16, taps, widths[i], 0);
                    checkasm_check1d(int32_t, out_c, out_a, widths[i], "out");
                }
            }
        }

        /* wide mode on the decoders' eq taps at the wide limit, where
         * the whole sum passes int32 and each alternating-pair half
         * still fits: random rows, both signs at the limit, and every
         * word carrying the sign of the tap over it at one position */
        for (int std = 0; std < 2; std++) {
            comp_decode_t *d = calloc(1, sizeof(*d));
            const int standard = std ? COMP_STD_NTSC : COMP_STD_PAL;
            if (!d || comp_decode_init(d, standard, 0.5, 1, 0, 3, 1, 0,
                                       std ? 1 : 0, 0, 0.0, 0)) {
                free(d);
                checkasm_fail_func("decoder init");
                return;
            }
            const int taps = COMP_EQ_TAPS + 1;
            const int32_t lim = d->eq_lim_wide;
            memcpy(coef16, d->eq_r16, sizeof(coef16[0]) * taps);
            for (int mode = 0; mode < 4; mode++) {
                for (int i = 0; i < NWIDTHS; i++) {
                    for (int k = 0; k < INLEN16; k++) {
                        const int neg = coef16[k % taps] < 0;
                        in16[k] = (int16_t)(mode == 0 ? (int32_t)(checkasm_rand() % (2 * lim + 1)) - lim
                                          : mode == 1 ? (checkasm_rand() & 1 ? lim : -lim)
                                          : mode == 2 ? (neg ? -lim : lim)
                                          : (neg ? lim : -lim));
                    }
                    memset(out_c, 0xaa, sizeof(out_c));
                    memset(out_a, 0x55, sizeof(out_a));
                    checkasm_call_ref(out_c, in16, coef16, taps, widths[i], 1);
                    checkasm_call_new(out_a, in16, coef16, taps, widths[i], 1);
                    checkasm_check1d(int32_t, out_c, out_a, widths[i], "out");
                    for (int x = 0; x < widths[i]; x++) {
                        if (out_c[x] != fir_s16_exact(in16 + x, coef16, taps)) {
                            checkasm_fail_func("wide reference at x = %d", x);
                            break;
                        }
                    }
                }
            }
            comp_decode_free(d);
            free(d);
        }
        /* the decoders' eq is the kernel's longest and most frequent
         * filter; coef16 still holds the NTSC one */
        for (int k = 0; k < INLEN16; k++)
            in16[k] = (int16_t)(checkasm_rand() % 20001 - 10000);
        checkasm_bench_new(out_a, in16, coef16, COMP_EQ_TAPS + 1, MAXW, 0);
    }
    checkasm_report("fir_row_s16");
}
