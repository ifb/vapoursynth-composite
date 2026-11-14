#include <stdint.h>
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
#define INLEN16 COMP_FIR_S16_SPAN(MAXW, COMP_UV_MAXTAPS)
static CHECKASM_ALIGN(int16_t in16[INLEN16]);
static CHECKASM_ALIGN(int16_t coef16[COMP_UV_MAXTAPS + 1]);

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

static void check_fir_row_s16(void)
{
    checkasm_declare(void, int32_t *, const int16_t *, const int16_t *,
                     int, int);
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
                    checkasm_call_ref(out_c, in16, coef16, taps, widths[i]);
                    checkasm_call_new(out_a, in16, coef16, taps, widths[i]);
                    checkasm_check1d(int32_t, out_c, out_a, widths[i], "out");
                }
            }
        }
        checkasm_bench_new(out_a, in16, enc[0].uv_taps16, enc[0].uv_ntaps, MAXW);
    }
    checkasm_report("fir_row_s16");
}
