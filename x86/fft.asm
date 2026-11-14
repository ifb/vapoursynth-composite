;*****************************************************************************
;* Fixed-size 3D real FFT: tile passes
;*****************************************************************************

%include "x86inc.asm"

SECTION .text

; The y/z passes: butterfly elements are 64-byte band rows (two ymm),
; complex values interleaved re,im, twiddles splatted scalars.

; t = cmul(d, w): even lanes d.re*wr - d.im*wi, odd d.im*wr + d.re*wi
%macro CMUL 4 ; d, t, wr, wi
    vpermilps    %2, %1, 0xB1
    mulps        %2, %4
    vfmaddsub231ps %2, %1, %3
%endmacro

; t = cmul(b, conj w): even b.re*wr + b.im*wi, odd b.im*wr - b.re*wi
%macro CMULC 4 ; b, t, wr, wi
    vpermilps    %2, %1, 0xB1
    mulps        %2, %4
    vfmsubadd231ps %2, %1, %3
%endmacro

; The y/z passes keep two or three radix-2 stages in registers per
; load/store sweep: the tile stays in L1, so the stage-per-sweep form
; spent its time on stores. Each butterfly is the same operation on the
; same operands as a lone stage would perform, and butterflies within a
; stage are independent, so the result does not depend on the grouping.
; A group is 2^k elements at positions e0 + m*sp (m = 0 .. 2^k - 1)
; within one transform, held one row half (four complex columns) at a
; time in m0 .. m(2^k - 1); m8 is the butterfly temp, m10/m11 the
; splatted twiddle.

; DIF butterfly in place: lower = a + b, upper = (a - b) * w
%macro DIF_BF 2 ; lower, upper
    subps        m8, %1, %2
    addps        %1, %1, %2
    CMUL         m8, %2, m10, m11
%endmacro

; DIT butterfly in place: t = b * conj(w), lower = a + t, upper = a - t
%macro DIT_BF 2 ; lower, upper
    CMULC        %2, m8, m10, m11
    subps        %2, %1, m8
    addps        %1, %1, m8
%endmacro

; splat twiddle tw[%1] unless the registers already hold it
%macro TWLOAD 1
%if %1 != twcur
    vbroadcastss m10, [twq + %1*8]
    vbroadcastss m11, [twq + %1*8 + 4]
%assign twcur %1
%endif
%endmacro

; one stage of length %4 over the group: the element at position e
; pairs with e + %4/2 when e mod %4 < %4/2, twiddle index
; (e mod %4) * (n / %4), exactly as the stage-at-a-time loop indexes it
%macro STAGE 6 ; dif, k, n, L, e0, sp
%assign %%h %4 / 2
%assign %%d %%h / %6
%assign %%m 0
%rep 1 << %2
%assign %%e %5 + %%m * %6
%assign %%r %%e - (%%e / %4) * %4
%if %%r < %%h
%assign %%p %%m + %%d
    TWLOAD       %%r * (%3 / %4)
%if %1
    DIF_BF       m %+ %%m, m %+ %%p
%else
    DIT_BF       m %+ %%m, m %+ %%p
%endif
%endif
%assign %%m %%m + 1
%endrep
%endmacro

; element %1's address at byte offset %2: a + m*s1, or a4 + (m-4)*s1
; above four (a4 = a + 4*s1, s3 = 3*s1)
%macro EADDR 2 ; m, byte offset
%if %1 < 4
%xdefine %%b aq
%else
%xdefine %%b a4q
%endif
%assign %%q %1 & 3
%if %%q == 0
%xdefine EA [%%b + %2]
%elif %%q == 1
%xdefine EA [%%b + s1q + %2]
%elif %%q == 2
%xdefine EA [%%b + s1q*2 + %2]
%else
%xdefine EA [%%b + s3q + %2]
%endif
%endmacro

; load, transform and store one group for both row halves. DIF runs
; its stages from the longest (%4) down, DIT from the shortest up.
%macro GROUP 6 ; dif, k, n, longest stage, e0, sp
%assign %%off 0
%rep 2
%assign twcur -1
%assign %%m 0
%rep 1 << %2
    EADDR        %%m, %%off
    movu         m %+ %%m, EA
%assign %%m %%m + 1
%endrep
%assign %%s 0
%rep %2
%if %1
%assign %%L %4 >> %%s
%else
%assign %%L %4 >> (%2 - 1 - %%s)
%endif
    STAGE        %1, %2, %3, %%L, %5, %6
%assign %%s %%s + 1
%endrep
%assign %%m 0
%rep 1 << %2
    EADDR        %%m, %%off
    movu         EA, m %+ %%m
%assign %%m %%m + 1
%endrep
%assign %%off %%off + 32
%endrep
%endmacro

; void fft_dif8 / fft_dit8(float *base, ptrdiff_t stride, int count,
;                          ptrdiff_t cstride, const float *tw)
;
; count independent 8-point transforms, element e of transform r the
; 64-byte row at base + r*cstride + e*stride; all three stages at once.
; DIF: natural-order input, bit-reversed output; DIT the inverse with
; conjugate twiddles, unnormalized. tw holds the 4 twiddles e^{-2pi i k/8}.
%macro FFT8 1 ; 1 = dif, 0 = dit
%if %1
cglobal fft_dif8, 5, 9, 12, base, stride, count, cstride, tw, a, a4, s1, s3
%else
cglobal fft_dit8, 5, 9, 12, base, stride, count, cstride, tw, a, a4, s1, s3
%endif
    mov          s1q, strideq
    lea          s3q, [strideq*3]
.rep:
    mov          aq, baseq
    lea          a4q, [aq + s1q*4]
    GROUP        %1, 3, 8, 8, 0, 1
    add          baseq, cstrideq
    dec          countd
    jg .rep
    RET
%endmacro

; The stages from %1 down to 8 for one transform, as four groups at
; positions j + 4m (j = 0..3); r holds the transform's base, s1/s3 are
; set for a 4-row element spacing.
%macro LONG_STAGES 2 ; dif, n
%assign %%j 0
%rep 4
%if %%j == 0
    mov          aq, rq
%elif %%j == 3
    lea          aq, [rq + strideq*2]
    add          aq, strideq
%else
    lea          aq, [rq + strideq*%%j]
%endif
    lea          a4q, [aq + s1q*4]
%if %2 == 32
    GROUP        %1, 3, %2, %2, %%j, 4
%else
    GROUP        %1, 2, %2, %2, %%j, 4
%endif
%assign %%j %%j + 1
%endrep
%endmacro

; void fft_dif16 / 32, fft_dit16 / 32(float *base, ptrdiff_t stride,
;                                     int count, ptrdiff_t cstride,
;                                     const float *tw)
;
; As FFT8 for n = 16 or 32 points: the stages from n down to 8 in one
; sweep (groups of n/4 elements spaced 4 rows apart), the stages 4 and 2
; in another (groups of 4 consecutive rows). DIT runs the sweeps in the
; opposite order. tw holds the n/2 twiddles e^{-2pi i k/n}.
%macro FFTN 2 ; 1 = dif / 0 = dit, n
%if %1
cglobal fft_dif%2, 5, 12, 12, base, stride, count, cstride, tw, a, a4, s1, s3, r, cnt, b
%else
cglobal fft_dit%2, 5, 12, 12, base, stride, count, cstride, tw, a, a4, s1, s3, r, cnt, b
%endif
%if %1 == 0
    SHORT_SWEEP  %1, %2
%endif
    lea          s1q, [strideq*4]
    lea          s3q, [s1q*3]
    mov          rq, baseq
    mov          cntd, countd
.long:
    LONG_STAGES  %1, %2
    add          rq, cstrideq
    dec          cntd
    jg .long
%if %1
    SHORT_SWEEP  %1, %2
%endif
    RET
%endmacro

; stages 4 and 2 over every run of 4 consecutive rows of every transform
%macro SHORT_SWEEP 2 ; dif, n
    mov          s1q, strideq
    lea          s3q, [strideq*3]
    mov          rq, baseq
    mov          cntd, countd
%%trans:
    mov          aq, rq
    mov          bd, %2 / 4
%%run:
    GROUP        %1, 2, %2, 4, 0, 1
    lea          aq, [aq + s1q*4]
    dec          bd
    jg %%run
    add          rq, cstrideq
    dec          cntd
    jg %%trans
%endmacro

INIT_YMM avx2
FFT8 1
FFT8 0
FFTN 1, 16
FFTN 0, 16
FFTN 1, 32
FFTN 0, 32

SECTION_RODATA 64

; 16-point real-FFT untangle twiddles e^{-2pi i k/16}, k = 2,3,5,6
f_w2r: dd 0.707106781
f_w2i: dd -0.707106781
f_w3r: dd 0.382683432
f_w3i: dd -0.923879533
f_w5r: dd -0.382683432
f_w5i: dd -0.923879533
f_w6r: dd -0.707106781
f_w6i: dd -0.707106781
f_rs:  dd 0.707106781
f_hlf: dd 0.5

SECTION .text

; 8x8 float transpose: rows in m0..m7, columns out in m8..m15
%macro TRANSPOSE8X8 0
    vunpcklps    m8, m0, m1
    vunpckhps    m9, m0, m1
    vunpcklps    m10, m2, m3
    vunpckhps    m11, m2, m3
    vunpcklps    m12, m4, m5
    vunpckhps    m13, m4, m5
    vunpcklps    m14, m6, m7
    vunpckhps    m15, m6, m7
    vshufps      m0, m8, m10, 0x44
    vshufps      m1, m8, m10, 0xEE
    vshufps      m2, m9, m11, 0x44
    vshufps      m3, m9, m11, 0xEE
    vshufps      m4, m12, m14, 0x44
    vshufps      m5, m12, m14, 0xEE
    vshufps      m6, m13, m15, 0x44
    vshufps      m7, m13, m15, 0xEE
    vperm2f128   m8, m0, m4, 0x20
    vperm2f128   m9, m1, m5, 0x20
    vperm2f128   m10, m2, m6, 0x20
    vperm2f128   m11, m3, m7, 0x20
    vperm2f128   m12, m0, m4, 0x31
    vperm2f128   m13, m1, m5, 0x31
    vperm2f128   m14, m2, m6, 0x31
    vperm2f128   m15, m3, m7, 0x31
%endmacro

; Forward radix-2 DIF stages of length 4 then 2 over four complex
; column vectors p0..p3 held as re,im register pairs %1..%8; %9, %10
; are temps. Standard DIF: position p ends holding X[brev(p)].
%macro DIF4 10
    ; len 4: (p0,p2) w = 1; (p1,p3) w = -i
    addps        %9, %1, %5
    addps        %10, %2, %6
    subps        %5, %1, %5
    subps        %6, %2, %6
    mova         %1, %9
    mova         %2, %10
    addps        %9, %3, %7
    addps        %10, %4, %8
    subps        %3, %3, %7
    subps        %4, %4, %8
    mova         %7, %4          ; (d1r,d1i) * -i = (d1i, -d1r)
    pxor         %8, %8
    subps        %8, %3
    mova         %3, %9
    mova         %4, %10
    ; len 2: (p0,p1), (p2,p3), w = 1
    addps        %9, %1, %3
    subps        %3, %1, %3
    mova         %1, %9
    addps        %9, %2, %4
    subps        %4, %2, %4
    mova         %2, %9
    addps        %9, %5, %7
    subps        %7, %5, %7
    mova         %5, %9
    addps        %9, %6, %8
    subps        %8, %6, %8
    mova         %6, %9
%endmacro

; Inverse: DIT stages of length 2 then 4, conjugate twiddles,
; bit-reversed input positions, natural output
%macro DIT4 10
    ; len 2: (p0,p1), (p2,p3), w = 1
    addps        %9, %1, %3
    subps        %3, %1, %3
    mova         %1, %9
    addps        %9, %2, %4
    subps        %4, %2, %4
    mova         %2, %9
    addps        %9, %5, %7
    subps        %7, %5, %7
    mova         %5, %9
    addps        %9, %6, %8
    subps        %8, %6, %8
    mova         %6, %9
    ; len 4: (p0,p2) w = 1; (p1,p3) w = conj(-i) = +i
    addps        %9, %1, %5
    addps        %10, %2, %6
    subps        %5, %1, %5
    subps        %6, %2, %6
    mova         %1, %9
    mova         %2, %10
    ; bt = p3 * i = (-p3i, p3r)
    mova         %9, %7
    pxor         %7, %7
    subps        %7, %8
    mova         %8, %9
    addps        %9, %3, %7
    addps        %10, %4, %8
    subps        %7, %3, %7
    subps        %8, %4, %8
    mova         %3, %9
    mova         %4, %10
%endmacro

; void fft_x_fwd(float *band, const float *real, int nrows)
;
; Batched 16-point real FFTs: 8 rows at a time through an 8x8
; transpose so every arithmetic op works on eight rows' same sample.
; Per row: pack 16 reals as 8 complex, DIF-8 (bit-reversed positions),
; untangle F[2..6] with e^{-2pi i k/16}, write one 64-byte band row
; (columns 5..7 zero). Stack: 8 spill slots + F staging.
%macro FFT_X_FWD 0
cglobal fft_x_fwd, 3, 4, 16, 512, band, real, nrows
.block:
    ; first halves -> columns c0..c3, spilled
%assign %%i 0
%rep 8
    movu         m %+ %%i, [realq + %%i*64]
%assign %%i %%i+1
%endrep
    TRANSPOSE8X8
%assign %%i 0
%rep 8
%assign %%j %%i+8
    mova         [rsp + %%i*32], m %+ %%j
%assign %%i %%i+1
%endrep
    ; second halves -> c4..c7 in m0..m7
%assign %%i 0
%rep 8
    movu         m %+ %%i, [realq + %%i*64 + 32]
%assign %%i %%i+1
%endrep
    TRANSPOSE8X8
%assign %%i 0
%rep 8
%assign %%j %%i+8
    mova         m %+ %%i, m %+ %%j
%assign %%i %%i+1
%endrep
    ; DIF-8 first stage: (spilled c_j, register c_{j+4}), twiddle W8^j
    vbroadcastss m10, [f_rs]
    ; j = 0, w = 1
    mova         m8, [rsp + 0*32]
    mova         m9, [rsp + 1*32]
    addps        m11, m8, m0
    addps        m12, m9, m1
    subps        m0, m8, m0
    subps        m1, m9, m1
    mova         [rsp + 0*32], m11
    mova         [rsp + 1*32], m12
    ; j = 1, w = (RS, -RS): d -> ((dr+di)RS, (di-dr)RS)
    mova         m8, [rsp + 2*32]
    mova         m9, [rsp + 3*32]
    addps        m11, m8, m2
    addps        m12, m9, m3
    subps        m2, m8, m2
    subps        m3, m9, m3
    mova         [rsp + 2*32], m11
    mova         [rsp + 3*32], m12
    addps        m8, m2, m3
    subps        m3, m3, m2
    mulps        m2, m8, m10
    mulps        m3, m10
    ; j = 2, w = -i: d -> (di, -dr)
    mova         m8, [rsp + 4*32]
    mova         m9, [rsp + 5*32]
    addps        m11, m8, m4
    addps        m12, m9, m5
    subps        m4, m8, m4
    subps        m5, m9, m5
    mova         [rsp + 4*32], m11
    mova         [rsp + 5*32], m12
    mova         m8, m5
    pxor         m5, m5
    subps        m5, m4
    mova         m4, m8
    ; j = 3, w = (-RS, -RS): d -> ((di-dr)RS, -(dr+di)RS)
    mova         m8, [rsp + 6*32]
    mova         m9, [rsp + 7*32]
    addps        m11, m8, m6
    addps        m12, m9, m7
    subps        m6, m8, m6
    subps        m7, m9, m7
    mova         [rsp + 6*32], m11
    mova         [rsp + 7*32], m12
    subps        m8, m7, m6
    addps        m9, m6, m7
    mulps        m6, m8, m10
    pxor         m7, m7
    subps        m7, m9
    mulps        m7, m10
    ; remaining stages of the register half: positions p4..p7
    DIF4         m0, m1, m2, m3, m4, m5, m6, m7, m8, m9
    ; p4 = X1 (unused), p5 = (m2,m3) = X5, p6 = (m4,m5) = X3, p7 = X7
    ; untangle F3 (from C3 = p6, C5 = p5) and F5
    addps        m8, m4, m2      ; er2  = c3r + c5r
    subps        m9, m5, m3      ; ei2  = c3i - c5i
    addps        m10, m5, m3     ; odr2 = c3i + c5i
    subps        m11, m2, m4     ; odi2 = c5r - c3r
    vbroadcastss m12, [f_w3r]
    vbroadcastss m13, [f_w3i]
    vbroadcastss m14, [f_hlf]
    mulps        m0, m10, m12
    vfnmadd231ps m0, m13, m11
    addps        m0, m8
    mulps        m0, m14         ; F3r
    mulps        m1, m11, m12
    vfmadd231ps  m1, m13, m10
    addps        m1, m9
    mulps        m1, m14         ; F3i
    vbroadcastss m12, [f_w5r]
    vbroadcastss m13, [f_w5i]
    mulps        m2, m10, m12
    vfmadd231ps  m2, m13, m11
    addps        m2, m8
    mulps        m2, m14         ; F5r
    mulps        m3, m10, m13
    vfnmadd231ps m3, m12, m11
    subps        m3, m9
    mulps        m3, m14         ; F5i
    mova         [rsp + 256 + 2*32], m0
    mova         [rsp + 256 + 3*32], m1
    mova         [rsp + 256 + 4*32], m2
    mova         [rsp + 256 + 5*32], m3
    ; spilled half: positions p0..p3
%assign %%i 0
%rep 8
    mova         m %+ %%i, [rsp + %%i*32]
%assign %%i %%i+1
%endrep
    DIF4         m0, m1, m2, m3, m4, m5, m6, m7, m8, m9
    ; p0 = X0 (unused), p1 = (m2,m3) = X4, p2 = (m4,m5) = X2, p3 = X6
    ; untangle F2 (from C2 = p2, C6 = p3), F6, and F4 = (c4r, -c4i)
    addps        m8, m4, m6
    subps        m9, m5, m7
    addps        m10, m5, m7
    subps        m11, m6, m4
    vbroadcastss m12, [f_w2r]
    vbroadcastss m13, [f_w2i]
    vbroadcastss m14, [f_hlf]
    mulps        m0, m10, m12
    vfnmadd231ps m0, m13, m11
    addps        m0, m8
    mulps        m0, m14         ; F2r
    mulps        m1, m11, m12
    vfmadd231ps  m1, m13, m10
    addps        m1, m9
    mulps        m1, m14         ; F2i
    vbroadcastss m12, [f_w6r]
    vbroadcastss m13, [f_w6i]
    mulps        m4, m10, m12
    vfmadd231ps  m4, m13, m11
    addps        m4, m8
    mulps        m4, m14         ; F6r
    mulps        m5, m10, m13
    vfnmadd231ps m5, m12, m11
    subps        m5, m9
    mulps        m5, m14         ; F6i
    mova         [rsp + 256 + 0*32], m4
    mova         [rsp + 256 + 1*32], m5
    ; transpose inputs: F2r F2i F3r F3i F4r F4i F5r F5i
    mova         m4, m2          ; F4r = c4r
    pxor         m5, m5
    subps        m5, m3          ; F4i = -c4i
    mova         m2, [rsp + 256 + 2*32]
    mova         m3, [rsp + 256 + 3*32]
    mova         m6, [rsp + 256 + 4*32]
    mova         m7, [rsp + 256 + 5*32]
    TRANSPOSE8X8
%assign %%i 0
%rep 8
%assign %%j %%i+8
    movu         [bandq + %%i*64], m %+ %%j
%assign %%i %%i+1
%endrep
    ; column 4 = F6 as per-row pairs, columns 5..7 zero
    mova         m0, [rsp + 256 + 0*32]
    mova         m1, [rsp + 256 + 1*32]
    vunpcklps    m2, m0, m1
    vunpckhps    m3, m0, m1
    pxor         m4, m4
%assign %%i 0
%rep 8
    movu         [bandq + %%i*64 + 32], m4
%assign %%i %%i+1
%endrep
    movlps       [bandq + 0*64 + 32], xm2
    movhps       [bandq + 1*64 + 32], xm2
    movlps       [bandq + 2*64 + 32], xm3
    movhps       [bandq + 3*64 + 32], xm3
    vextractf128 xm2, m2, 1
    vextractf128 xm3, m3, 1
    movlps       [bandq + 4*64 + 32], xm2
    movhps       [bandq + 5*64 + 32], xm2
    movlps       [bandq + 6*64 + 32], xm3
    movhps       [bandq + 7*64 + 32], xm3
    add          realq, 8*64
    add          bandq, 8*64
    sub          nrowsd, 8
    jg .block
    RET
%endmacro

INIT_YMM avx2
FFT_X_FWD

; void fft_x_inv(float *real, const float *band, int nrows)
;
; Batched inverse of fft_x_fwd: rebuild the packed 8-point spectrum
; from the band columns (the 0.5 untangle scale and the final x2
; cancel exactly, both being powers of two), inverse DIT-8, and write
; 16 reals per row. Unnormalized, matching the forward's convention.
%macro FFT_X_INV 0
cglobal fft_x_inv, 3, 4, 16, 512, real, band, nrows
.block:
    ; column 4 (F6) pairs -> two column vectors, staged
    movsd        xm0, [bandq + 0*64 + 32]
    movhps       xm0, [bandq + 1*64 + 32]
    movsd        xm1, [bandq + 2*64 + 32]
    movhps       xm1, [bandq + 3*64 + 32]
    movsd        xm2, [bandq + 4*64 + 32]
    movhps       xm2, [bandq + 5*64 + 32]
    movsd        xm3, [bandq + 6*64 + 32]
    movhps       xm3, [bandq + 7*64 + 32]
    vinsertf128  m0, m0, xm2, 1
    vinsertf128  m1, m1, xm3, 1
    vshufps      m4, m0, m1, 0x88   ; F6r
    vshufps      m5, m0, m1, 0xDD   ; F6i
    mova         [rsp + 0*32], m4
    mova         [rsp + 1*32], m5
    ; columns 0..3 -> F2..F5 column vectors
%assign %%i 0
%rep 8
    movu         m %+ %%i, [bandq + %%i*64]
%assign %%i %%i+1
%endrep
    TRANSPOSE8X8                    ; m8..m15 = F2r F2i F3r F3i F4r F4i F5r F5i
    ; regs half: positions p4 = 0, p5 = C5, p6 = C3, p7 = 0
    ; from F3 = (m10,m11), F5 = (m14,m15), w3, w5 (conjugated on use)
    addps        m0, m10, m14       ; ER
    subps        m1, m11, m15       ; EI
    subps        m2, m10, m14       ; QR
    addps        m3, m11, m15       ; QI
    vbroadcastss m4, [f_w3r]
    vbroadcastss m5, [f_w3i]
    mulps        m6, m2, m4
    vfmadd231ps  m6, m5, m3         ; ODR3 = w3r QR + w3i QI
    mulps        m7, m3, m4
    vfnmadd231ps m7, m5, m2         ; ODI3 = w3r QI - w3i QR
    vbroadcastss m4, [f_w5r]
    vbroadcastss m5, [f_w5i]
    mulps        m8, m3, m5
    vfnmadd231ps m8, m4, m2         ; ODR5 = -w5r QR + w5i QI
    mulps        m9, m3, m4
    vfmadd231ps  m9, m5, m2         ; ODI5 = w5r QI + w5i QR
    ; C3 = (ER - ODI3, EI + ODR3) -> p6; C5 = (ER - ODI5, -EI + ODR5) -> p5
    subps        m4, m0, m7
    addps        m5, m1, m6
    subps        m2, m0, m9
    subps        m3, m8, m1
    pxor         m0, m0
    pxor         m1, m1
    pxor         m6, m6
    pxor         m7, m7
    DIT4         m0, m1, m2, m3, m4, m5, m6, m7, m8, m9
%assign %%i 0
%rep 8
    mova         [rsp + 256 + %%i*32], m %+ %%i
%assign %%i %%i+1
%endrep
    ; spilled half: positions p0 = 0, p1 = C4, p2 = C2, p3 = C6
    ; from F2 = (m8? -- recompute: F2 pair survives in m8,m9? DIT4 used m8,m9 as temps)
    ; F2 = (m8,m9) was clobbered; reload from the band transpose is
    ; avoided by recomputing: F2/F4 columns were m8,m9 / m12,m13 --
    ; m12..m15 survived, m8..m11 did not. Reload columns 0,1.
    movsd        xm8, [bandq + 0*64]
    movhps       xm8, [bandq + 1*64]
    movsd        xm9, [bandq + 2*64]
    movhps       xm9, [bandq + 3*64]
    movsd        xm10, [bandq + 4*64]
    movhps       xm10, [bandq + 5*64]
    movsd        xm11, [bandq + 6*64]
    movhps       xm11, [bandq + 7*64]
    vinsertf128  m8, m8, xm10, 1
    vinsertf128  m9, m9, xm11, 1
    vshufps      m10, m8, m9, 0x88  ; F2r
    vshufps      m11, m8, m9, 0xDD  ; F2i
    ; C4 = (2 F4r, -2 F4i) from the surviving (m12, m13)
    addps        m2, m12, m12
    pxor         m3, m3
    subps        m3, m13
    subps        m3, m13
    ; C2 = (ER - ODI2, EI + ODR2), C6 = (ER - ODI6, -EI + ODR6)
    ; from F2 = (m10, m11) and F6 staged at [rsp]
    mova         m8, [rsp + 0*32]
    mova         m9, [rsp + 1*32]
    addps        m12, m10, m8       ; ER
    subps        m13, m11, m9       ; EI
    subps        m14, m10, m8       ; QR
    addps        m15, m11, m9       ; QI
    vbroadcastss m8, [f_w2r]
    vbroadcastss m9, [f_w2i]
    mulps        m10, m14, m8
    vfmadd231ps  m10, m9, m15       ; ODR2
    mulps        m11, m15, m8
    vfnmadd231ps m11, m9, m14       ; ODI2
    subps        m4, m12, m11
    addps        m5, m13, m10
    vbroadcastss m8, [f_w6r]
    vbroadcastss m9, [f_w6i]
    mulps        m10, m15, m9
    vfnmadd231ps m10, m8, m14       ; ODR6 = -w6r QR + w6i QI
    mulps        m11, m15, m8
    vfmadd231ps  m11, m9, m14       ; ODI6 = w6r QI + w6i QR
    subps        m6, m12, m11
    subps        m7, m10, m13
    pxor         m0, m0
    pxor         m1, m1
    DIT4         m0, m1, m2, m3, m4, m5, m6, m7, m8, m9
    ; final DIT-8 stage across the halves, conjugate W8 twiddles
    ; j = 0
    mova         m8, [rsp + 256 + 0*32]
    mova         m9, [rsp + 256 + 1*32]
    subps        m10, m0, m8
    subps        m11, m1, m9
    mova         [rsp + 256 + 0*32], m10
    mova         [rsp + 256 + 1*32], m11
    addps        m0, m8
    addps        m1, m9
    ; j = 1: bt = ((br-bi)RS, (br+bi)RS)
    mova         m8, [rsp + 256 + 2*32]
    mova         m9, [rsp + 256 + 3*32]
    vbroadcastss m12, [f_rs]
    subps        m10, m8, m9
    addps        m11, m8, m9
    mulps        m10, m12
    mulps        m11, m12
    subps        m8, m2, m10
    subps        m9, m3, m11
    mova         [rsp + 256 + 2*32], m8
    mova         [rsp + 256 + 3*32], m9
    addps        m2, m10
    addps        m3, m11
    ; j = 2: bt = (-bi, br)
    mova         m8, [rsp + 256 + 4*32]
    mova         m9, [rsp + 256 + 5*32]
    pxor         m10, m10
    subps        m10, m9
    subps        m11, m4, m10
    subps        m12, m5, m8
    mova         [rsp + 256 + 4*32], m11
    mova         [rsp + 256 + 5*32], m12
    addps        m4, m10
    addps        m5, m8
    ; j = 3: bt = (-(br+bi)RS, (br-bi)RS)
    mova         m8, [rsp + 256 + 6*32]
    mova         m9, [rsp + 256 + 7*32]
    vbroadcastss m12, [f_rs]
    addps        m10, m8, m9
    subps        m11, m8, m9
    mulps        m10, m12
    mulps        m11, m12
    pxor         m13, m13
    subps        m13, m10
    subps        m8, m6, m13
    subps        m9, m7, m11
    mova         [rsp + 256 + 6*32], m8
    mova         [rsp + 256 + 7*32], m9
    addps        m6, m13
    addps        m7, m11
    ; rows out: columns c0..c3 from registers, c4..c7 from the spill
    TRANSPOSE8X8
%assign %%i 0
%rep 8
%assign %%j %%i+8
    movu         [realq + %%i*64], m %+ %%j
%assign %%i %%i+1
%endrep
%assign %%i 0
%rep 8
    mova         m %+ %%i, [rsp + 256 + %%i*32]
%assign %%i %%i+1
%endrep
    TRANSPOSE8X8
%assign %%i 0
%rep 8
%assign %%j %%i+8
    movu         [realq + %%i*64 + 32], m %+ %%j
%assign %%i %%i+1
%endrep
    add          realq, 8*64
    add          bandq, 8*64
    sub          nrowsd, 8
    jg .block
    RET
%endmacro

INIT_YMM avx2
FFT_X_INV
