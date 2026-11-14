;*****************************************************************************
;* 4xfsc chroma demod products by parity-split taps
;*****************************************************************************

%include "x86inc.asm"
%include "util.asm"

SECTION_RODATA 64

d_rnd15: times 16 dd 16384
; qword gathers re-interleaving the lo/hi unpack halves of four chunks
q_ilv0:  dq 0, 1, 8, 9, 2, 3, 10, 11
q_ilv1:  dq 4, 5, 12, 13, 6, 7, 14, 15

SECTION .text

; Taps j and j+2 of one parity for the block's mmsize/2 outputs at x:
; word pairs (s[x+j+l], s[x+j+2+l]) dot the pair's per-phase row.
; Within each 128-bit chunk c, punpcklwd holds outputs x+8c+0..3 and
; punpckhwd x+8c+4..7, so lane phase is the dword index mod 4 in both.
%macro PAIR 4 ; acc_lo, acc_hi, j, pair index
    movu         m4, [sq + xq*2 + %3*2]
    movu         m5, [sq + xq*2 + %3*2 + 4]
    punpckhwd    m7, m4, m5
    punpcklwd    m4, m5
%if cpuflag(avx)
    pmaddwd      m4, [cfq + %4*64]
    pmaddwd      m7, [cfq + %4*64]
%else
    ; legacy-SSE memory operands fault unless 16-byte aligned, and the
    ; table lives in a malloc'd decoder at no particular offset
    movu         m6, [cfq + %4*64]
    pmaddwd      m4, m6
    pmaddwd      m7, m6
%endif
    paddd        %1, m4
    paddd        %2, m7
%endmacro

; void demod_pq_row(int32_t *p, int32_t *q, const int16_t *s,
;                   const int16_t (*coef)[32], int w)
;
; Writes p, q through COMP_DEMOD_PQ_ALIGN(w) and reads s through
; COMP_DEMOD_PQ_SPAN(w) (see decode.h); blocks start at multiples of 8.
; Even-j pairs sum in m0/m1, odd-j in m2/m3 (lo/hi unpack halves). An
; output's sine product comes from the taps where x + j is odd, so even
; lanes take p from the odd-j sums and odd lanes from the even-j sums.
%macro DEMOD_PQ_ROW 0
cglobal demod_pq_row, 5, 6, 8, p, q, s, cf, w, x
    movifnidn    wd, wm
%if mmsize == 64
    mov          xd, 0xAAAA               ; k2 selects odd dword lanes (BLEND_ODD)
    kmovw        k2, xd
%endif
    xor          xd, xd
.block:
    pxor         m0, m0
    pxor         m1, m1
    pxor         m2, m2
    pxor         m3, m3
    PAIR         m0, m1,  0, 0
    PAIR         m0, m1,  4, 1
    PAIR         m0, m1,  8, 2
    PAIR         m0, m1, 12, 3
    PAIR         m0, m1, 16, 4
    PAIR         m2, m3,  1, 5
    PAIR         m2, m3,  5, 6
    PAIR         m2, m3,  9, 7
    PAIR         m2, m3, 13, 8
    BLEND_ODD    m4, m2, m0               ; p lo
    BLEND_ODD    m5, m3, m1               ; p hi
    BLEND_ODD    m6, m0, m2               ; q lo
    BLEND_ODD    m7, m1, m3               ; q hi
    paddd        m4, [d_rnd15]
    paddd        m5, [d_rnd15]
    paddd        m6, [d_rnd15]
    paddd        m7, [d_rnd15]
    psrad        m4, 15
    psrad        m5, 15
    psrad        m6, 15
    psrad        m7, 15
%if mmsize == 16
    movu         [pq + xq*4], m4
    movu         [pq + xq*4 + 16], m5
    movu         [qq + xq*4], m6
    movu         [qq + xq*4 + 16], m7
%elif mmsize == 32
    vperm2i128   m0, m4, m5, 0x20
    vperm2i128   m1, m4, m5, 0x31
    movu         [pq + xq*4], m0
    movu         [pq + xq*4 + 32], m1
    vperm2i128   m0, m6, m7, 0x20
    vperm2i128   m1, m6, m7, 0x31
    movu         [qq + xq*4], m0
    movu         [qq + xq*4 + 32], m1
%else
    movu         m0, [q_ilv0]
    movu         m1, [q_ilv1]
    vpermi2q     m0, m4, m5
    vpermi2q     m1, m4, m5
    movu         [pq + xq*4], m0
    movu         [pq + xq*4 + 64], m1
    movu         m0, [q_ilv0]
    movu         m1, [q_ilv1]
    vpermi2q     m0, m6, m7
    vpermi2q     m1, m6, m7
    movu         [qq + xq*4], m0
    movu         [qq + xq*4 + 64], m1
%endif
    add          xd, mmsize/2
    cmp          xd, wd
    jl .block
    RET
%endmacro

INIT_XMM sse4
DEMOD_PQ_ROW
INIT_YMM avx2
DEMOD_PQ_ROW
INIT_ZMM avx512
DEMOD_PQ_ROW
