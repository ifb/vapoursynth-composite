;*****************************************************************************
;* Q15 FIR row kernels
;*****************************************************************************

%include "x86inc.asm"
%include "util.asm"

SECTION_RODATA 64

; arithmetic-shift emulation bias: acc + bias is non-negative for the
; |acc| < 2^45 kernel contract, so a logical shift of the biased value
; equals the arithmetic shift plus a constant the final pack subtracts
q_bias15: times 8 dq (1 << 45) + 16384
d_sub30:  times 16 dd 1 << 30
d_rnd15:  times 16 dd 16384
; qword gathers re-interleaving the lo/hi unpack halves of four chunks
q_ilv0:   dq 0, 1, 8, 9, 2, 3, 10, 11
q_ilv1:   dq 4, 5, 12, 13, 6, 7, 14, 15

SECTION .text

; (e, o) 64-bit accumulator lanes -> packed int32 of (acc + 16384) >> 15
%macro FIR_PACK 3 ; out, e, o (clobbers e, o)
    paddq        %2, [q_bias15]
    paddq        %3, [q_bias15]
    psrlq        %2, 15
    psrlq        %3, 15
    psllq        %3, 32
    BLEND_ODD    %1, %2, %3
    psubd        %1, [d_sub30]
%endmacro

; void fir_row_q15(int32_t *out, const int32_t *in, const int32_t *coef,
;                  int taps, int w)
; per the padding contract, out and in are sized for w rounded up to 16
; samples, so no lane masking is needed anywhere
%macro FIR_ROW_Q15 0
cglobal fir_row_q15, 5, 9, 16, out, in, coef, taps, w, x, pin, pc, j
%if mmsize == 64
    mov          xd, 0xAAAA       ; k2 selects the odd dword lanes
    kmovw        k2, xd
%endif
    xor          xq, xq
.groups:
    pxor         m0, m0
    pxor         m1, m1
%if mmsize < 64
    pxor         m2, m2
    pxor         m3, m3
%endif
    lea          pinq, [inq + xq*4]
    mov          pcq, coefq
    mov          jd, tapsd
.tap:
    SPLATD       m4, [pcq]
    movu         m5, [pinq]
    pshufd       m6, m5, 0xF5
    pmuldq       m5, m4
    paddq        m0, m5
    pmuldq       m6, m4
    paddq        m1, m6
%if mmsize < 64
    ; a second output group amortizes the coefficient broadcast; the
    ; zmm tier is already at the 16-sample padding bound with one
    movu         m7, [pinq + mmsize]
    pshufd       m8, m7, 0xF5
    pmuldq       m7, m4
    paddq        m2, m7
    pmuldq       m8, m4
    paddq        m3, m8
%endif
    add          pcq, 4
    add          pinq, 4
    dec          jd
    jg .tap

    FIR_PACK     m9, m0, m1
    movu         [outq + xq*4], m9
%if mmsize < 64
    FIR_PACK     m9, m2, m3
    movu         [outq + xq*4 + mmsize], m9
    add          xq, mmsize/2
%else
    add          xq, mmsize/4
%endif
    cmp          xd, wd
    jl .groups
    RET
%endmacro

INIT_XMM sse4
FIR_ROW_Q15
INIT_YMM avx2
FIR_ROW_Q15
INIT_ZMM avx512
FIR_ROW_Q15

; void fir_row_s16(int32_t *out, const int16_t *in, const int16_t *coef,
;                  int taps, int w)
; Each block covers mmsize/2 outputs at x. A tap pair j, j+1 loads
; in[x+j..] and in[x+j+1..] and interleaves them into word pairs:
; within each 128-bit chunk c, punpcklwd holds outputs x+8c+0..3 and
; punpckhwd x+8c+4..7, and pmaddwd against the (coef[j], coef[j+1])
; dword sums both taps. The lo/hi sums re-interleave at the store.
%macro FIR_ROW_S16 0
cglobal fir_row_s16, 5, 9, 8, out, in, coef, taps, w, x, pin, pc, j
    movifnidn    wd, wm
    xor          xq, xq
.groups:
    pxor         m0, m0
    pxor         m1, m1
    lea          pinq, [inq + xq*2]
    mov          pcq, coefq
    mov          jd, tapsd
.pair:
    SPLATD       m6, [pcq]
    movu         m4, [pinq]
    movu         m5, [pinq + 2]
    punpckhwd    m7, m4, m5
    punpcklwd    m4, m5
    pmaddwd      m4, m6
    pmaddwd      m7, m6
    paddd        m0, m4
    paddd        m1, m7
    add          pcq, 4
    add          pinq, 4
    sub          jd, 2
    jg .pair

    paddd        m0, [d_rnd15]
    paddd        m1, [d_rnd15]
    psrad        m0, 15
    psrad        m1, 15
%if mmsize == 16
    movu         [outq + xq*4], m0
    movu         [outq + xq*4 + 16], m1
%elif mmsize == 32
    vperm2i128   m2, m0, m1, 0x20
    vperm2i128   m3, m0, m1, 0x31
    movu         [outq + xq*4], m2
    movu         [outq + xq*4 + 32], m3
%else
    movu         m2, [q_ilv0]
    movu         m3, [q_ilv1]
    vpermi2q     m2, m0, m1
    vpermi2q     m3, m0, m1
    movu         [outq + xq*4], m2
    movu         [outq + xq*4 + 64], m3
%endif
    add          xd, mmsize/2
    cmp          xd, wd
    jl .groups
    RET
%endmacro

INIT_XMM sse4
FIR_ROW_S16
INIT_YMM avx2
FIR_ROW_S16
INIT_ZMM avx512
FIR_ROW_S16
