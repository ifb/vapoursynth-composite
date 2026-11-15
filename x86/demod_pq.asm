;*****************************************************************************
;* 4xfsc chroma demod products by parity-split taps
;*****************************************************************************

%include "x86inc.asm"
%include "util.asm"

SECTION_RODATA 64

d_rnd15: times 16 dd 16384
d_lo16:  times 16 dd 0x0000FFFF
d_hi16:  times 16 dd 0xFFFF0000
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

; one tap pair of the avx512icl tier: 16 word pairs out of a stream's
; two vectors, shifted by %4 pairs, dot the coefficients in %5
%macro PQ_ZTERM 5 ; acc, stream lo, stream hi, pair shift, coef
%if %4
    valignd      m4, %3, %2, %4
    vpdpwssd     %1, m4, %5
%else
    vpdpwssd     %1, %2, %5
%endif
%endmacro

; The avx512icl tier. A 64-byte load at s[x+j] crosses a cache line for
; all but one j in 32, and the pairing of taps j and j+2 takes two such
; loads and two unpacks a pair. Here a block loads its 64 words once and
; builds the two streams of word pairs two apart, (s[2i], s[2i+2]) in
; m8/m9 and (s[2i+1], s[2i+3]) in m10/m11. A tap pair's products for
; the even outputs are then one stream shifted by whole pairs, and for
; the odd outputs the other, so the sums come out split by output
; parity and need no blend: p is the odd-j sums on even outputs and the
; even-j sums on odd ones. The coefficient rows are held in m12..m29,
; each split into its even-output lanes (phases 0, 2) and odd-output
; lanes (phases 1, 3). Without vpdpwssd to fold the adds this form is
; slower than the avx512 tier on Tiger Lake, so it is not that tier.
%macro DEMOD_PQ_ROW_ZMM 0
cglobal demod_pq_row, 5, 6, 30, p, q, s, cf, w, x
    movifnidn    wd, wm
%assign %%i 0
%rep 9
    %assign %%e 12 + %%i*2
    %assign %%o 13 + %%i*2
    pshufd       m %+ %%e, [cfq + %%i*64], 0x88
    pshufd       m %+ %%o, [cfq + %%i*64], 0xDD
%assign %%i %%i+1
%endrep
    xor          xd, xd
.block:
    pxor         m0, m0                   ; even j, even outputs: q
    pxor         m1, m1                   ; even j, odd outputs: p
    pxor         m2, m2                   ; odd j, even outputs: p
    pxor         m3, m3                   ; odd j, odd outputs: q
    movu         m6, [sq + xq*2]
    movu         m7, [sq + xq*2 + 64]
    ; the pair after each pair; no tap reaches the last pair of the
    ; second vector
    valignd      m4, m7, m6, 1
    valignd      m5, m7, m7, 1
    pand         m8, m6, [d_lo16]
    pand         m9, m7, [d_lo16]
    pslld        m10, m4, 16
    pslld        m11, m5, 16
    por          m8, m10                  ; (s[2i], s[2i+2])
    por          m9, m11
    psrld        m10, m6, 16
    psrld        m11, m7, 16
    pand         m4, [d_hi16]
    pand         m5, [d_hi16]
    por          m10, m4                  ; (s[2i+1], s[2i+3])
    por          m11, m5
    ; even j = 0, 4, .., 16: both outputs' pairs start j/2 pairs in
%assign %%i 0
%rep 5
    %assign %%e 12 + %%i*2
    %assign %%o 13 + %%i*2
    PQ_ZTERM     m0, m8, m9, %%i*2, m %+ %%e
    PQ_ZTERM     m1, m10, m11, %%i*2, m %+ %%o
%assign %%i %%i+1
%endrep
    ; odd j = 1, 5, 9, 13: the even outputs start on an odd word,
    ; (j-1)/2 pairs into that stream, the odd ones (j+1)/2 into the other
%assign %%i 0
%rep 4
    %assign %%e 22 + %%i*2
    %assign %%o 23 + %%i*2
    PQ_ZTERM     m2, m10, m11, %%i*2, m %+ %%e
    PQ_ZTERM     m3, m8, m9, %%i*2 + 1, m %+ %%o
%assign %%i %%i+1
%endrep
    mova         m4, [d_rnd15]
    paddd        m0, m4
    paddd        m1, m4
    paddd        m2, m4
    paddd        m3, m4
    psrad        m0, 15
    psrad        m1, 15
    psrad        m2, 15
    psrad        m3, 15
    punpckldq    m4, m2, m1               ; p, outputs 0..3 of each eight
    punpckhdq    m5, m2, m1               ; p, outputs 4..7
    punpckldq    m6, m0, m3               ; q
    punpckhdq    m7, m0, m3
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
INIT_ZMM avx512icl
DEMOD_PQ_ROW_ZMM
