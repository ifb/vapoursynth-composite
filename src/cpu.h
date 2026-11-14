#ifndef COMP_CPU_H
#define COMP_CPU_H

#define COMP_CPU_SSE2   (1 << 0)
#define COMP_CPU_SSE41  (1 << 1)
#define COMP_CPU_AVX2   (1 << 2)
#define COMP_CPU_AVX512 (1 << 3)

unsigned comp_cpu_detect(void);

/* The tiers the compiler target already guarantees, as dav1d's
 * trim_dsp does. The dispatch getters OR this into their flags, so in a
 * per-CPU build the branches below the guaranteed tier are dead and the
 * C references they would return drop out of the module. The baseline
 * guarantees nothing here: checkasm, linked against it, gets its C
 * reference by passing no flags. */
#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512BW__) && \
    defined(__AVX512DQ__) && defined(__AVX512VL__) && defined(__FMA__) && \
    defined(__BMI2__)
#define COMP_CPU_BUILTIN (COMP_CPU_AVX512 | COMP_CPU_AVX2 | COMP_CPU_SSE41 | COMP_CPU_SSE2)
#elif defined(__AVX2__) && defined(__FMA__) && defined(__BMI2__)
#define COMP_CPU_BUILTIN (COMP_CPU_AVX2 | COMP_CPU_SSE41 | COMP_CPU_SSE2)
#else
#define COMP_CPU_BUILTIN 0
#endif

#endif
