/* x86 flag parsing after dav1d src/x86/cpu.c */

#include <stdint.h>

#include "cpu.h"

#if defined(__x86_64__)

void comp_cpu_cpuid(uint32_t regs[4], unsigned leaf, unsigned subleaf);
uint64_t comp_cpu_xgetbv(unsigned xcr);

#define X(reg, mask) (((reg) & (mask)) == (mask))

unsigned comp_cpu_detect(void)
{
    unsigned flags = 0;
    uint32_t r[4];

    comp_cpu_cpuid(r, 0, 0);
    const unsigned max_leaf = r[0];
    if (max_leaf < 1)
        return 0;

    comp_cpu_cpuid(r, 1, 0);
    if (X(r[3], 0x06008000)) { /* CMOV/SSE/SSE2 */
        flags |= COMP_CPU_SSE2;
        if (X(r[2], 0x00080201)) /* SSE3/SSSE3/SSE4.1 */
            flags |= COMP_CPU_SSE41;
    }

    if (X(r[2], 0x18001000) && max_leaf >= 7) { /* FMA/OSXSAVE/AVX */
        const uint64_t xcr0 = comp_cpu_xgetbv(0);
        if (X(xcr0, 0x00000006)) { /* XMM/YMM state */
            comp_cpu_cpuid(r, 7, 0);
            const unsigned max_subleaf = r[0];
            if (X(r[1], 0x00000128)) /* BMI1/BMI2/AVX2 */
                flags |= COMP_CPU_AVX2;
            if (X(xcr0, 0x000000e0) && /* opmask/ZMM state */
                X(r[1], 0xd0030000)) { /* F/DQ/CD/BW/VL */
                flags |= COMP_CPU_AVX512;
                if (X(r[1], 0x00200000) && /* IFMA */
                    X(r[2], 0x00005f42)) /* VBMI/VBMI2/GFNI/VAES/VPCLMULQDQ/VNNI/BITALG/VPOPCNTDQ */
                    flags |= COMP_CPU_AVX512ICL;
            }
            if ((flags & COMP_CPU_AVX2) && max_subleaf >= 1) {
                comp_cpu_cpuid(r, 7, 1);
                if (X(r[0], 0x00000010)) /* AVX-VNNI */
                    flags |= COMP_CPU_AVXVNNI;
            }
        }
    }
    return flags;
}

#else

unsigned comp_cpu_detect(void)
{
    return 0;
}

#endif
