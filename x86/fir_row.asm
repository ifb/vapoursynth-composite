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
d_rnd14:  times 16 dd 8192
d_one:    times 16 dd 1
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

; one coefficient over one or two output groups: %1 = 1 adds the
; mirrored sample at phi first (a symmetric tap pair), 0 is the center
%macro FIR_TERM 1 ; pair
    SPLATD       m4, [pcq]
    movu         m5, [pinq]
%if %1
    movu         m6, [phiq]
    paddd        m5, m6
%endif
    pshufd       m6, m5, 0xF5
    pmuldq       m5, m4
    paddq        m0, m5
    pmuldq       m6, m4
    paddq        m1, m6
%if mmsize < 64
    ; a second output group amortizes the coefficient broadcast; the
    ; zmm tier is already at the 16-sample padding bound with one
    movu         m7, [pinq + mmsize]
%if %1
    movu         m8, [phiq + mmsize]
    paddd        m7, m8
%endif
    pshufd       m8, m7, 0xF5
    pmuldq       m7, m4
    paddq        m2, m7
    pmuldq       m8, m4
    paddq        m3, m8
%endif
%endmacro

; void fir_row_q15(int32_t *out, const int32_t *in, const int32_t *coef,
;                  int taps, int w)
; per the padding contract, out and in are sized for w rounded up to 16
; samples, so no lane masking is needed anywhere. The taps are odd in
; number and symmetric, so each pair coef[j] = coef[taps-1-j] multiplies
; the sum of its two samples once (exact: the pair sums fit int32 and
; integer accumulation is order-free), then the center tap.
%macro FIR_ROW_Q15 0
cglobal fir_row_q15, 5, 10, 16, out, in, coef, taps, w, x, pin, pc, j, phi
%if mmsize == 64
    mov          xd, 0xAAAA       ; k2 selects the odd dword lanes
    kmovw        k2, xd
%endif
    movsxd       tapsq, tapsd
    xor          xq, xq
.groups:
    pxor         m0, m0
    pxor         m1, m1
%if mmsize < 64
    pxor         m2, m2
    pxor         m3, m3
%endif
    lea          pinq, [inq + xq*4]
    lea          phiq, [pinq + tapsq*4 - 4]
    mov          pcq, coefq
    mov          jd, tapsd
    shr          jd, 1
.tap:
    FIR_TERM     1
    add          pcq, 4
    add          pinq, 4
    sub          phiq, 4
    dec          jd
    jg .tap
    FIR_TERM     0                ; pin, pc now at the center tap

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

; one tap pair at pin + %3 into the even/odd sums %1/%2
%macro S16_PAIR 3 ; acc even, acc odd, byte disp
    SPLATD       m6, [pinq + dcq + %3]
%if cpuflag(avxvnni)
    ; nasm picks the EVEX encoding unless told
    {vex} vpdpwssd %1, m6, [pinq + %3]
    {vex} vpdpwssd %2, m6, [pinq + %3 + 2]
%elif cpuflag(avx)
    pmaddwd      m4, m6, [pinq + %3]
    pmaddwd      m5, m6, [pinq + %3 + 2]
    paddd        %1, m4
    paddd        %2, m5
%else
    ; legacy-SSE memory operands fault unless 16-byte aligned
    movu         m4, [pinq + %3]
    movu         m5, [pinq + %3 + 2]
    pmaddwd      m4, m6
    pmaddwd      m5, m6
    paddd        %1, m4
    paddd        %2, m5
%endif
%endmacro

; void fir_row_s16(int32_t *out, const int16_t *in, const int16_t *coef,
;                  int taps, int w)
; Each block covers mmsize/2 outputs at x. pmaddwd of the words at
; in[x+j..] against the broadcast (coef[j], coef[j+1]) sums the tap pair
; for the even outputs x, x+2, ..., and the same product one word on,
; at in[x+j+1..], for the odd outputs, so the samples are memory
; operands and nothing is shuffled per tap. The even/odd sums
; interleave at the store. The avxvnni tier folds each pmaddwd + paddd
; into vpdpwssd.
;
; The pairs alternate between two sums a and b. With wide set the two
; are joined without forming a + b, which may not fit int32:
; (a + b + 16384) >> 15 = ((a >> 1) + (b >> 1) + (a & b & 1) + 8192) >> 14,
; the first three terms being floor((a + b) / 2) and the dropped low
; bit of a + b too small to carry.
%macro S16_JOIN 3 ; a (out), b, scratch
    pand         %3, %1, %2
    pand         %3, [d_one]
    psrad        %1, 1
    psrad        %2, 1
    paddd        %1, %2
    paddd        %1, %3
    paddd        %1, [d_rnd14]
    psrad        %1, 14
%endmacro

%macro FIR_ROW_S16 0
cglobal fir_row_s16, 6, 10, 8, out, in, coef, taps, w, wide, x, pin, pend, dc
    movifnidn    wd, wm
    movsxd       tapsq, tapsd
    inc          tapsq
    and          tapsq, -2                ; taps rounded up to even
    mov          dcq, coefq
    sub          dcq, inq                 ; coef pair of the tap at pin
    xor          xq, xq
.groups:
    pxor         m0, m0
    pxor         m1, m1
    pxor         m2, m2
    pxor         m3, m3
    lea          pinq, [inq + xq*2]
    lea          pendq, [pinq + tapsq*2 - 4]
    sub          dcq, xq
    sub          dcq, xq
    cmp          pinq, pendq
    jae .last
.pair2:
    ; two pairs a round on separate sums: half the loop branches, and
    ; the vpdpwssd accumulate chains (5 cycles a link) run two abreast
    S16_PAIR     m0, m1, 0
    S16_PAIR     m2, m3, 4
    add          pinq, 8
    cmp          pinq, pendq
    jb .pair2
    ja .sum
.last:
    S16_PAIR     m0, m1, 0
.sum:
    lea          dcq, [dcq + xq*2]
    test         wided, wided
    jnz .join
    paddd        m0, m2
    paddd        m1, m3
    paddd        m0, [d_rnd15]
    paddd        m1, [d_rnd15]
    psrad        m0, 15
    psrad        m1, 15
.store:
    punpckhdq    m7, m0, m1
    punpckldq    m0, m1
%if mmsize == 16
    movu         [outq + xq*4], m0
    movu         [outq + xq*4 + 16], m7
%elif mmsize == 32
    vperm2i128   m2, m0, m7, 0x20
    vperm2i128   m3, m0, m7, 0x31
    movu         [outq + xq*4], m2
    movu         [outq + xq*4 + 32], m3
%else
    movu         m2, [q_ilv0]
    movu         m3, [q_ilv1]
    vpermi2q     m2, m0, m7
    vpermi2q     m3, m0, m7
    movu         [outq + xq*4], m2
    movu         [outq + xq*4 + 64], m3
%endif
    add          xd, mmsize/2
    cmp          xd, wd
    jl .groups
    RET
.join:
    S16_JOIN     m0, m2, m4
    S16_JOIN     m1, m3, m5
    jmp .store
%endmacro

INIT_XMM sse4
FIR_ROW_S16
INIT_YMM avx2
FIR_ROW_S16
INIT_ZMM avx512
FIR_ROW_S16
INIT_YMM avxvnni
FIR_ROW_S16
