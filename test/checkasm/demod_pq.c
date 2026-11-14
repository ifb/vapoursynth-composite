#include <stdint.h>
#include <string.h>

#include <checkasm/checkasm.h>
#include <checkasm/test.h>
#include <checkasm/utils.h>

#include "decode.h"

#define MAXW 928

static const int16_t taps[COMP_DEMOD_PQ_TAPS] = { COMP_COLORLP_COEFS };

static CHECKASM_ALIGN(int16_t coef[COMP_DEMOD_PQ_PAIRS][32]);
static CHECKASM_ALIGN(int16_t s[COMP_DEMOD_PQ_SPAN(MAXW)]);
static CHECKASM_ALIGN(int32_t p_c[COMP_DEMOD_PQ_ALIGN(MAXW)]);
static CHECKASM_ALIGN(int32_t p_a[COMP_DEMOD_PQ_ALIGN(MAXW)]);
static CHECKASM_ALIGN(int32_t q_c[COMP_DEMOD_PQ_ALIGN(MAXW)]);
static CHECKASM_ALIGN(int32_t q_a[COMP_DEMOD_PQ_ALIGN(MAXW)]);

/* every 24th output sees all 17 of its taps at full scale with the sign
 * of that lane's coefficient: both parity sums reach their bound */
static void worst_case(void)
{
    for (int x0 = 0; x0 + COMP_DEMOD_PQ_TAPS <= COMP_DEMOD_PQ_SPAN(MAXW); x0 += 24) {
        const int ph = x0 & 3;
        for (int j = 0; j < COMP_DEMOD_PQ_TAPS; j++) {
            const int k = (ph + j) & 3;
            const int neg = (taps[j] < 0) != (k >= 2);
            s[x0 + j] = neg ? -32768 : 32767;
        }
    }
}

void checkasm_test_demod_pq(void)
{
    comp_demod_pq_init(coef, taps);

    checkasm_declare(void, int32_t *, int32_t *, const int16_t *,
                     const int16_t (*)[32], int);
    if (checkasm_check_func(comp_get_demod_pq_row_fn(checkasm_get_cpu_flags()),
                            "demod_pq_row")) {
        static const int widths[] = { 8, 24, 40, 122, 360, 758, 928 };
        const int nw = (int)(sizeof(widths) / sizeof(widths[0]));
        for (int mode = 0; mode < 3; mode++) {
            for (int i = 0; i < nw; i++) {
                const int w = widths[i];
                for (int k = 0; k < COMP_DEMOD_PQ_SPAN(MAXW); k++)
                    s[k] = mode == 2 ? -32768
                                     : (int16_t)(checkasm_rand() % 65536 - 32768);
                if (mode == 1)
                    worst_case();
                memset(p_c, 0x55, sizeof(p_c));
                memset(p_a, 0x55, sizeof(p_a));
                memset(q_c, 0x55, sizeof(q_c));
                memset(q_a, 0x55, sizeof(q_a));
                checkasm_call_ref(p_c, q_c, s, (const int16_t (*)[32])coef, w);
                checkasm_call_new(p_a, q_a, s, (const int16_t (*)[32])coef, w);
                checkasm_check1d(int32_t, p_c, p_a, w, "p");
                checkasm_check1d(int32_t, q_c, q_a, w, "q");
            }
        }
        checkasm_bench_new(p_a, q_a, s, (const int16_t (*)[32])coef, 758);
    }
    checkasm_report("demod_pq");
}
