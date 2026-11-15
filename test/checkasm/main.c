#include <checkasm/checkasm.h>

#include "cpu.h"

void checkasm_test_pal_demod(void);
void checkasm_test_fir_row(void);
void checkasm_test_split3d(void);
void checkasm_test_comb2d(void);
void checkasm_test_demod_quant(void);
void checkasm_test_demod_rotate(void);
void checkasm_test_demod_pq(void);
void checkasm_test_encode_mod(void);
void checkasm_test_lut_gain(void);
void checkasm_test_t3d_filter(void);
void checkasm_test_fft(void);

static const CheckasmCpuInfo cpu_flags[] = {
    { "SSE2", "sse2", COMP_CPU_SSE2 },
    { "SSE4.1", "sse4", COMP_CPU_SSE41 },
    { "AVX2", "avx2", COMP_CPU_AVX2 },
    { "AVX-VNNI", "avxvnni", COMP_CPU_AVXVNNI },
    { "AVX512", "avx512", COMP_CPU_AVX512 },
    { "AVX512ICL", "avx512icl", COMP_CPU_AVX512ICL },
    {0}
};

static const CheckasmTest tests[] = {
    { "pal_demod", checkasm_test_pal_demod },
    { "fir_row", checkasm_test_fir_row },
    { "split3d", checkasm_test_split3d },
    { "comb2d", checkasm_test_comb2d },
    { "demod_quant", checkasm_test_demod_quant },
    { "demod_rotate", checkasm_test_demod_rotate },
    { "demod_pq", checkasm_test_demod_pq },
    { "encode_mod", checkasm_test_encode_mod },
    { "lut_gain", checkasm_test_lut_gain },
    { "t3d_filter", checkasm_test_t3d_filter },
    { "fft", checkasm_test_fft },
    {0}
};

int main(int argc, const char *argv[])
{
    CheckasmConfig config = {
        .cpu_flags = cpu_flags,
        .tests = tests,
        .cpu = comp_cpu_detect(),
    };
    return checkasm_main(&config, argc, argv);
}
