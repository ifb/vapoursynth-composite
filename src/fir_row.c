/*
 * Q15 FIR row kernels: scalar references and SIMD dispatch. Both are
 * pure integer.
 */

#include "cpu.h"
#include "fir_row.h"

static void fir_row_q15_c(int32_t *out, const int32_t *in,
                          const int32_t *coef, int taps, int w)
{
    const int c = taps / 2;
    for (int x = 0; x < w; x++) {
        int64_t acc = (int64_t)coef[c] * in[x + c];
        for (int j = 0; j < c; j++)
            acc += (int64_t)coef[j] * (in[x + j] + in[x + taps - 1 - j]);
        out[x] = (int32_t)((acc + 16384) >> 15);
    }
}

#if defined(__x86_64__)
#define FIR_ROW_Q15_ASM(isa)                                                \
    void comp_fir_row_q15_##isa(int32_t *out, const int32_t *in,            \
                                const int32_t *coef, int taps, int w)
FIR_ROW_Q15_ASM(sse4);
FIR_ROW_Q15_ASM(avx2);
FIR_ROW_Q15_ASM(avx512);
#endif

static void fir_row_s16_c(int32_t *out, const int16_t *in,
                          const int16_t *coef, int taps, int w, int wide)
{
    if (!wide) {
        for (int x = 0; x < w; x++) {
            int32_t acc = 0;
            for (int j = 0; j < taps; j++)
                acc += coef[j] * in[x + j];
            out[x] = (acc + 16384) >> 15;
        }
        return;
    }
    /* coef is zero-padded to an even count, so the last pair is whole */
    const int pairs = (taps + 1) / 2;
    for (int x = 0; x < w; x++) {
        int32_t a = 0, b = 0;
        for (int i = 0; i + 1 < pairs; i += 2) {
            a += coef[2 * i] * in[x + 2 * i] + coef[2 * i + 1] * in[x + 2 * i + 1];
            b += coef[2 * i + 2] * in[x + 2 * i + 2] + coef[2 * i + 3] * in[x + 2 * i + 3];
        }
        if (pairs & 1)
            a += coef[2 * pairs - 2] * in[x + 2 * pairs - 2]
               + coef[2 * pairs - 1] * in[x + 2 * pairs - 1];
        /* (a + b + 16384) >> 15 without forming a + b */
        out[x] = ((a >> 1) + (b >> 1) + (a & b & 1) + 8192) >> 14;
    }
}

#if defined(__x86_64__)
#define FIR_ROW_S16_ASM(isa)                                                \
    void comp_fir_row_s16_##isa(int32_t *out, const int16_t *in,            \
                                const int16_t *coef, int taps, int w,       \
                                int wide)
FIR_ROW_S16_ASM(sse4);
FIR_ROW_S16_ASM(avx2);
FIR_ROW_S16_ASM(avx512);
FIR_ROW_S16_ASM(avxvnni);
FIR_ROW_S16_ASM(avx512icl);
#endif

comp_fir_row_s16_fn comp_get_fir_row_s16_fn(unsigned cpu)
{
    cpu |= COMP_CPU_BUILTIN;
#if defined(__x86_64__)
    if (cpu & COMP_CPU_AVX512ICL)
        return comp_fir_row_s16_avx512icl;
    if (cpu & COMP_CPU_AVX512)
        return comp_fir_row_s16_avx512;
    if (cpu & COMP_CPU_AVXVNNI)
        return comp_fir_row_s16_avxvnni;
    if (cpu & COMP_CPU_AVX2)
        return comp_fir_row_s16_avx2;
    if (cpu & COMP_CPU_SSE41)
        return comp_fir_row_s16_sse4;
#endif
    (void)cpu;
    return fir_row_s16_c;
}

comp_fir_row_q15_fn comp_get_fir_row_q15_fn(unsigned cpu)
{
    cpu |= COMP_CPU_BUILTIN;
#if defined(__x86_64__)
    if (cpu & COMP_CPU_AVX512)
        return comp_fir_row_q15_avx512;
    if (cpu & COMP_CPU_AVX2)
        return comp_fir_row_q15_avx2;
    if (cpu & COMP_CPU_SSE41)
        return comp_fir_row_q15_sse4;
#endif
    (void)cpu;
    return fir_row_q15_c;
}
