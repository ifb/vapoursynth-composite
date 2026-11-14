;*****************************************************************************
;* PAL demodulation filter kernel
;*****************************************************************************

%include "x86inc.asm"
%include "util.asm"

SECTION_RODATA 64

d_rnd16:   times 16 dd 32768
; ROTATE's shift-emulation biases (see util.asm)
q_nbias14: times 8 dq (1 << 40) - 8192
d_sub26:   times 16 dd 1 << 26

SECTION .text

%define FS 7

; one pair plane at one tap position: the word pairs at x + disp dot the
; row's per-phase (lo, hi) coefficients into the dword sums of acc
%macro TERM 4 ; acc, plane, byte disp, coefficient row
    movu         m4, [%2 + xq*4 + %3]
%if cpuflag(avx)
    pmaddwd      m4, [cfq + (%4)*64]
%else
    ; legacy-SSE memory operands fault unless 16-byte aligned, and the
    ; table lives in a malloc'd decoder at no particular offset
    movu         m5, [cfq + (%4)*64]
    pmaddwd      m4, m5
%endif
    paddd        %1, m4
%endmacro

; all four planes at tap position FS + %1 (coefficient rows from %2)
; into the c0/c1 sums %3 and the c2/c3 sums %4
%macro PLANES 4 ; offset, first row, acc c0/c1, acc c2/c3
    TERM         %3, prq, (FS + %1)*4, %2
    TERM         %3, p1q, (FS + %1)*4, %2 + 1
    TERM         %4, p2q, (FS + %1)*4, %2 + 2
    TERM         %4, p3q, (FS + %1)*4, %2 + 3
%endmacro

; void pal_demod_row(int32_t *u, int32_t *v, const int16_t *pr,
;                    ptrdiff_t stride, const int16_t (*coef)[32], int w,
;                    int32_t bp, int32_t bq, int32_t vswitch)
;
; Sums by coefficient group and tap parity: m0/m1 the c0/c1 planes at
; even/odd b, m2/m3 the c2/c3 planes. A tap lands on a sine sample when
; x + b is odd, so even lanes take their sine sums from the odd-b
; accumulators and odd lanes from the even-b ones.
%macro PAL_DEMOD_ROW 0
cglobal pal_demod_row, 6, 11, 16, u, v, pr, stride, cf, w, p1, p2, p3, x, t
    lea          tq, [strideq*4]
    lea          p1q, [prq + tq]
    lea          p2q, [prq + tq*2]
    lea          p3q, [p1q + tq*2]
%if mmsize == 64
    mov          td, 0xAAAA               ; k2 selects odd dword lanes (BLEND_ODD)
    kmovw        k2, td
    ; vswitch is a uniform +/-1; k3 = all-ones when it is -1, so a
    ; merge-masked 0 - v negates exactly those lanes (there is no EVEX
    ; psignd, and this beats pmulld: 1 uop vs 2 on Intel, 1 on Zen)
    xor          td, td
    cmp          dword r8m, 0
    setl         tb
    neg          td
    kmovw        k3, td
%endif
    xor          xq, xq
.loop:
    pxor         m0, m0
    pxor         m1, m1
    pxor         m2, m2
    pxor         m3, m3
    ; b = 0 reads one position (the coefficients carry the doubling)
    PLANES       0, 0, m0, m2
    ; %%-local so the tap counter cannot leak into the next cglobal's
    ; DEFINE_ARGS cleanup
%assign %%b 1
%rep FS
%if %%b & 1
    PLANES        %%b, 4 + (%%b - 1)*8,     m1, m3
    PLANES       -%%b, 4 + (%%b - 1)*8 + 4, m1, m3
%else
    PLANES        %%b, 4 + (%%b - 1)*8,     m0, m2
    PLANES       -%%b, 4 + (%%b - 1)*8 + 4, m0, m2
%endif
%assign %%b %%b+1
%endrep

    BLEND_ODD    m4, m1, m0               ; c0/c1 on sine samples
    BLEND_ODD    m5, m0, m1               ; c0/c1 on cosine samples
    BLEND_ODD    m6, m3, m2               ; c2/c3 on sine samples
    BLEND_ODD    m7, m2, m3               ; c2/c3 on cosine samples
    ; pu = A_sn + B_cs, pv = A_sn - B_cs, qu = A_cs - B_sn,
    ; qv = A_cs + B_sn; then the Q16 rounding to Q0
    paddd        m0, m4, m7
    psubd        m1, m4, m7
    psubd        m2, m5, m6
    paddd        m3, m5, m6
    mova         m4, [d_rnd16]
    paddd        m0, m4
    paddd        m1, m4
    paddd        m2, m4
    paddd        m3, m4
    psrad        m0, 16                   ; pu0
    psrad        m1, 16                   ; pv0
    psrad        m2, 16                   ; qu0
    psrad        m3, 16                   ; qv0

    SPLATD       m8, dword r6m            ; bp
    SPLATD       m9, dword r7m            ; bq

    ; u = -(pu0*bp + qu0*bq + 8192) >> 14
    ROTATE       m10, m0, m8, m2, m9, paddq
    movu         [uq + xq*4], m10

    ; v = vswitch * (-(qv0*bp - pv0*bq + 8192) >> 14)
    ROTATE       m10, m3, m8, m1, m9, psubq
%if mmsize == 64
    vpxord       m11, m11, m11
    vpsubd       m10{k3}, m11, m10        ; conditional negate by vswitch (k3)
%else
    SPLATD       m11, dword r8m           ; vswitch
    psignd       m10, m11
%endif
    movu         [vq + xq*4], m10

    add          xq, mmsize/4
    cmp          xd, wd
    jl .loop
    RET
%endmacro

INIT_XMM sse4
PAL_DEMOD_ROW
INIT_YMM avx2
PAL_DEMOD_ROW
INIT_ZMM avx512
PAL_DEMOD_ROW
