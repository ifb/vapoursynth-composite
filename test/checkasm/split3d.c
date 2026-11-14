#include <stdint.h>
#include <string.h>

#include <checkasm/checkasm.h>
#include <checkasm/test.h>
#include <checkasm/utils.h>

#include "decode.h"

#define MAXW 758

static CHECKASM_ALIGN(float c1_rows[8][MAXW]);
static CHECKASM_ALIGN(float c2_rows[8][MAXW]);
static CHECKASM_ALIGN(uint16_t lines[8][MAXW]);
static CHECKASM_ALIGN(float c1c[MAXW]);
static CHECKASM_ALIGN(float c2c[MAXW]);
static CHECKASM_ALIGN(uint16_t ref[MAXW]);
static CHECKASM_ALIGN(int16_t out_c[MAXW]);
static CHECKASM_ALIGN(int16_t out_a[MAXW]);

/* w - 6 must be a multiple of 16 per the kernel contract */
static const int widths[] = { 22, 118, 358, 758 };
#define NWIDTHS (int)(sizeof(widths) / sizeof(widths[0]))

static float chroma_val(void)
{
    return (float)((int32_t)(checkasm_rand() % 65535) - 32767)
         * (1.0f / 16.0f) * (float)(checkasm_rand() % 16);
}

static void fill_inputs(comp_split3d_cand_t *cand, int similar)
{
    static const double bonuses[4] = { 0.0, -2.0, -4.0, -6.0 };

    for (int i = 0; i < 8; i++) {
        for (int x = 0; x < MAXW; x++) {
            c1_rows[i][x] = chroma_val();
            c2_rows[i][x] = chroma_val();
            /* similar mode keeps candidate lines close to the
             * reference so penalties race tightly */
            lines[i][x] = similar
                ? (uint16_t)(30000 + checkasm_rand() % 600)
                : (uint16_t)(checkasm_rand() % 65536);
        }
    }
    for (int x = 0; x < MAXW; x++) {
        c1c[x] = chroma_val();
        c2c[x] = chroma_val();
        ref[x] = similar ? (uint16_t)(30000 + checkasm_rand() % 600)
                         : (uint16_t)(checkasm_rand() % 65536);
    }
    for (int i = 0; i < 8; i++) {
        const unsigned fl = checkasm_rand();
        cand[i].c1 = c1_rows[i];
        cand[i].c2 = c2_rows[i];
        cand[i].line = lines[i];
        cand[i].off = i == 0 ? -2 : i == 1 ? 2 : 0;
        cand[i].have_sample = fl & 1;
        cand[i].have_penalty = (fl & 3) == 1;  /* implies have_sample */
        cand[i].bonus = bonuses[(fl >> 2) & 3];
    }
}

void checkasm_test_split3d(void)
{
    comp_split3d_cand_t cand[8];

    checkasm_declare(void, int16_t *, const float *, const float *,
                     const uint16_t *, const comp_split3d_cand_t *,
                     const double *, int);
    if (checkasm_check_func(comp_get_split3d_row_fn(checkasm_get_cpu_flags()),
                            "split3d_row")) {
        /* the decoder's comb_krange / 45, without and with setup */
        const double prod[2] = { 16128.0 / 45.0, 14918.0 / 45.0 };
        for (int i = 0; i < 6 * NWIDTHS; i++) {
            const int w = widths[i % NWIDTHS];
            const int kind = (i / NWIDTHS) % 3;
            const double irescale = kind == 2
                ? 1.0 + (double)(checkasm_rand() % 400) / 100.0 : prod[kind];
            fill_inputs(cand, i & 1);
            /* every candidate scored, so every lane divides */
            if (i >= 3 * NWIDTHS)
                for (int k = 0; k < 8; k++)
                    cand[k].have_sample = cand[k].have_penalty = 1;
            /* the same pattern in both, so untouched edges compare too */
            memset(out_c, 0x77, sizeof(out_c));
            memset(out_a, 0x77, sizeof(out_a));
            checkasm_call_ref(out_c, c1c, c2c, ref, cand, &irescale, w);
            checkasm_call_new(out_a, c1c, c2c, ref, cand, &irescale, w);
            checkasm_check1d(int16_t, out_c, out_a, w, "out");
        }
        {
            const double irescale = 2.0;
            fill_inputs(cand, 1);
            for (int i = 0; i < 8; i++)
                cand[i].have_sample = cand[i].have_penalty = 1;
            checkasm_bench_new(out_a, c1c, c2c, ref, cand, &irescale, MAXW);
        }
    }
    checkasm_report("split3d");
}
