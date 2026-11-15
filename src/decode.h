#ifndef COMP_DECODE_H
#define COMP_DECODE_H

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#include "encode.h"
#include "fir_row.h"
#include "subcarrier.h"
#include "transform2d.h"
#include "transform3d.h"

#define COMP_DECODE_FILTER_SIZE 7
#define COMP_EQ_TAPS 31
#define COMP_NARROW_TAPS 13

/* Filter the PAL chroma lines with the 2D chroma filter, demodulate
 * them against the 4xfsc carriers and rotate onto the U/V axes (the hot
 * loop of the PALcolour-style demodulator). pr holds four planes of
 * `stride` int16 word pairs, the vertical tap groups (line, 0),
 * (-2, +2), (-1, +1) and (-3, +3) field lines, with x = 0 at pair
 * COMP_DECODE_FILTER_SIZE and that many zero pairs each side. The
 * chroma-times-carrier products are zero on alternate samples, so each
 * tap feeds the sine or the cosine sums by the parity of its position,
 * and the carrier and vertical signs fold into per-phase coefficient
 * rows (comp_pal_demod_init). The Q16 filter over int16 lines keeps
 * every sum near 2^30, so int32 is exact and all tiers byte-identical.
 * bp/bq is the line's burst-phase unit vector (Q15), vswitch the PAL
 * switch sign. Writes u, v through COMP_PAL_DEMOD_ALIGN(w) and reads
 * pr through that plus 2 * COMP_DECODE_FILTER_SIZE pairs. */
#define COMP_PAL_DEMOD_ROWS (4 + 8 * COMP_DECODE_FILTER_SIZE)
#define COMP_PAL_DEMOD_ALIGN(w) (((w) + 15) & ~15)

typedef void (*comp_pal_demod_fn)(int32_t *u, int32_t *v, const int16_t *pr,
                                  ptrdiff_t stride, const int16_t (*coef)[32],
                                  int w, int32_t bp, int32_t bq,
                                  int32_t vswitch);

comp_pal_demod_fn comp_get_pal_demod_fn(unsigned cpu);

/* the coefficient rows from the Q16 2D chroma filter cfilt[b][k] */
void comp_pal_demod_init(int16_t coef[COMP_PAL_DEMOD_ROWS][32],
                         const int32_t cfilt[COMP_DECODE_FILTER_SIZE + 1][4]);

/* One comb candidate of the NTSC 3D split, prepared per row: the
 * bounds and phase tests of the reference's per-pixel candidate table
 * depend only on the row, so the driver hoists them and the kernel
 * evaluates all eight candidates unconditionally. The layout is fixed
 * for the asm (static asserts in decode.c). */
typedef struct comp_split3d_cand_t {
    const float *c1;      /* candidate row of 1D chroma, indexed x + off */
    const float *c2;      /* candidate row of 2D chroma */
    const uint16_t *line; /* candidate composite row */
    int32_t off;          /* ch - x: -2, 0 or +2 */
    int32_t have_sample;  /* candidate row is inside the frame */
    int32_t have_penalty; /* and its chroma phase opposes the current row */
    double bonus;         /* line/field/frame preference bias */
} comp_split3d_cand_t;

/* Adaptive 3D comb candidate selection for one NTSC row, x in
 * [3, w - 3) (the caller writes the edge samples): pick the
 * lowest-penalty candidate, earliest index winning ties, and emit
 * clamp_i16(round_i32(tc)) with tc = c2c[x] for a same-frame winner or
 * (c1c[x] - sample) / 2 otherwise. irescale is passed by pointer to
 * keep the asm ABI GPR-only. w - 6 must be a multiple of 16. */
typedef void (*comp_split3d_row_fn)(int16_t *out, const float *c1c,
                                    const float *c2c, const uint16_t *ref,
                                    const comp_split3d_cand_t *cand,
                                    const double *irescale, int w);

comp_split3d_row_fn comp_get_split3d_row_fn(unsigned cpu);

/* NTSC 2D adaptive line comb for one row (comb.cpp split2D): compares
 * the current line against the same-phase lines two above and two below
 * and blends whichever pair matches, writing out[x] for x in [1, w).
 * The caller writes out[0] and provides prev/next rows, substituting a
 * zero line beyond the frame edges. krange (the comb adaptivity range)
 * is passed by pointer to keep the asm ABI GPR-only. Bit-exact with the
 * C reference (no FMA): its output feeds split3d's candidate compare,
 * so a single-ULP drift would change the 3D result. The kernel
 * over-reads neither past out[w] nor its inputs past [w]. */
typedef void (*comp_ntsc_comb2d_row_fn)(float *out, const float *cur,
                                        const float *prev, const float *next,
                                        const float *krange, int w);

comp_ntsc_comb2d_row_fn comp_get_ntsc_comb2d_row_fn(unsigned cpu);

/* Precomputed magic reciprocal for one output-quantize divisor. The
 * demod quantize does clamp_u16(base + rdiv(in * K, den)), where den
 * (ku, kv, luma_den) and K are per-decoder invariants; this replaces
 * the per-pixel 64-bit divide with a single 32x32->64 multiply. For
 * input n:  mag = (|n| * mul + add) >> 31;  out = clamp_u16(base +
 * sign(n) * mag). Bit-exact with rdiv wherever the result is not
 * saturated by clamp_u16 (the shared shift is fixed at 31; mul < 2^32
 * for every standard/setup -- see D048 and the static assert). */
typedef struct comp_magic_t {
    uint32_t mul;   /* ceil(K * 2^31 / den) */
    uint32_t add;   /* round(floor(den/2) * 2^31 / den) */
    int32_t base;   /* the +4096 (luma) or +32768 (chroma) offset */
} comp_magic_t;

#define COMP_MAGIC_SHIFT 31

/* Quantize one demod row: out[x] = clamp_u16(base + sign(in[x]) *
 * ((|in[x]| * mul + add) >> 31)). Terminal (writes a frame plane), so
 * the contract is byte-identical to the rdiv reference within the
 * unsaturated input range; mag is passed by pointer (GPR-only ABI). */
typedef void (*comp_demod_quant_row_fn)(uint16_t *out, const int32_t *in,
                                        const comp_magic_t *mag, int w);

comp_demod_quant_row_fn comp_get_demod_quant_row_fn(unsigned cpu);

/* build the magic triple for clamp_u16(base + rdiv(in * k, den)) */
void comp_magic_init(comp_magic_t *m, int32_t k, int32_t den, int32_t base);

/* NTSC chroma rotation onto the U/V axes (the comb.cpp demod step):
 *   u[x] = -(p[x]*bp + q[x]*bq + 8192) >> 14
 *   v[x] = -(q[x]*bp - p[x]*bq + 8192) >> 14
 * with the burst-phase pair bp/bq (Q15) passed as a 2-int array to keep
 * the asm ABI GPR-only. Byte-identical to the C reference (integer,
 * associative). The kernel over-reads/-writes nothing past [w] (an
 * overlapping final vector covers the ragged tail). */
typedef void (*comp_demod_rotate_row_fn)(int32_t *u, int32_t *v,
                                         const int32_t *p, const int32_t *q,
                                         const int32_t *bpq, int w);

comp_demod_rotate_row_fn comp_get_demod_rotate_row_fn(unsigned cpu);

/* Demod products: the 17-tap color low-pass of the chroma times the
 * 4xfsc carriers sn = {0,1,0,-1} and cs = {1,0,-1,0} [x & 3],
 *   p[x] = (sum_j h[j] * s[x+j] * sn(x+j) + 16384) >> 15
 * and q likewise with cs, s being the chroma row behind
 * COMP_DEMOD_PQ_WING zeros. Each tap feeds exactly one of the two
 * (x + j odd: p), so the taps split by parity, the carrier sign folds
 * into per-phase coefficients (comp_demod_pq_init), and taps j, j+2 of
 * one parity pair up over int16 samples. sum|h| * 2^15 < 2^31 - 2^14
 * keeps every sum in int32, so all tiers are byte-identical. */
#define COMP_DEMOD_PQ_TAPS  17
/* that low-pass: deemp.h c_colorlp_b rounded to Q15, keeping the
 * reference's +0.2% DC gain */
#define COMP_COLORLP_COEFS \
    73, 317, 200, -682, -1597, -503, 3726, 9094, 11579, \
    9094, 3726, -503, -1597, -682, 200, 317, 73
#define COMP_DEMOD_PQ_WING  (COMP_DEMOD_PQ_TAPS / 2)
#define COMP_DEMOD_PQ_PAIRS 9
/* p and q are written through ALIGN(w), s is read through SPAN(w) */
#define COMP_DEMOD_PQ_ALIGN(w) (((w) + 31) & ~31)
#define COMP_DEMOD_PQ_SPAN(w)  (COMP_DEMOD_PQ_ALIGN(w) + COMP_DEMOD_PQ_TAPS + 1)

typedef void (*comp_demod_pq_row_fn)(int32_t *p, int32_t *q, const int16_t *s,
                                     const int16_t (*coef)[32], int w);

comp_demod_pq_row_fn comp_get_demod_pq_row_fn(unsigned cpu);

/* per-pair coefficient rows, one (h[j], h[j+2]) word pair per lane of a
 * 64-byte row, signed by the lane's carrier phase */
void comp_demod_pq_init(int16_t coef[COMP_DEMOD_PQ_PAIRS][32],
                        const int16_t *taps);

typedef struct comp_decode_scratch_t comp_decode_scratch_t;

struct comp_decode_scratch_t {
    int used;
    float *chroma_f;    /* one frame, active raster */
    int16_t *chroma;    /* quantised copy for the fixed-point demod */
    float *conf;        /* eq=2: transform chroma-confidence map */
    int16_t *chroma2;   /* hybrid: the comb's chroma estimate */
    uint8_t *mask;      /* hybrid: 1 where the comb's temporal candidate won */
    float *tmp3;        /* NTSC: 2D chroma; in 3D mode, 1D+2D of three frames */
    uint16_t *refine;   /* refine loop: YUV estimate, recomposite, scratch */
};

typedef struct comp_frame_view_t comp_frame_view_t;

struct comp_frame_view_t {
    const uint16_t *data;
    ptrdiff_t stride;
};

/* Per-frame accumulation of the eq=2 separation-confidence map, summed
 * over exactly the samples the confidence blend consumes (immune to the
 * stale/partial-fill hazard of the reused conf scratch). Reduced to the
 * mean and population std-dev reported as frame props. When out_mask is
 * non-NULL the same consumption loop also writes the confidence mask
 * (65535*(1-conf)) there, at the output row it is already producing — so
 * the mask inherits conf's correct per-pixel layout for free. */
typedef struct comp_conf_acc_t {
    double sum;
    double sumsq;
    uint64_t n;
    uint16_t *out_mask;     /* confidence mask base (frame 0,0), or NULL */
    ptrdiff_t mask_stride;
} comp_conf_acc_t;

/* Per-frame difficulty/effort metrics reported as optional frame props.
 * Each field is left at its sentinel (-1) when its path is inactive, so
 * the caller emits a prop only for the metrics that were computed. None
 * is a quality score (no clean reference exists during restoration):
 *  - conf_mean/conf_std: eq=2 separation-confidence map (mean, pop stddev)
 *  - motion_fraction: NTSC hybrid, fraction of samples the motion router
 *    judged to be in motion (routed to the transform); 1 = all motion,
 *    0 = all still
 *  - refine_residual: refine>0, mean |orig_y - crude(encode(Y))| at the
 *    final iteration (model misfit left after refinement)
 *  - refine_correction: refine>0, mean |Y_out - Y_in| (luma moved) */
typedef struct comp_decode_metrics_t {
    double conf_mean, conf_std;
    double motion_fraction;
    double refine_residual, refine_correction;
} comp_decode_metrics_t;

#define COMP_METRICS_INIT { -1.0, -1.0, -1.0, -1.0, -1.0 }

typedef struct comp_decode_t comp_decode_t;

struct comp_decode_t {
    int standard;
    int dimensions;      /* 1 (crude notch), 2, or 3 */
    int use_transform;   /* NTSC 3D: 1 transform, 2 comb/transform hybrid */
    int refine;          /* Y-only Landweber iterations, 0 = off */
    int width;
    int height;          /* full raster height: 576 PAL, 486 NTSC */
    int den;
    int32_t ku, kv;      /* the encoder's chroma scales, Q15 */
    int32_t level_black;
    int32_t luma_num, luma_den;  /* y16 = 4096 + (level - black) * num/den */
    comp_magic_t mag_y, mag_u, mag_v;  /* precomputed output-quantize divides */
    int32_t comb_krange;         /* NTSC 2D comb adaptivity range */
    int eq;                      /* equalizer: 0 off, 1 fixed, 2 leak-aware */
    int cti;                     /* luma-guided chroma transient improvement */
    int32_t eq_q15[COMP_EQ_TAPS];
    int32_t narrow_q15[COMP_NARROW_TAPS];  /* eq=2: sub-nominal chroma LP */
    /* the same filters for fir_row_s16, used on rows whose samples stay
     * within the limit that keeps its int32 sums exact: eq with
     * eq_k * 32768 taken off the center tap (added back as eq_k times
     * the sample), both zero-padded to an even count. A negative limit
     * means the taps do not fit int16. */
    int16_t eq_r16[COMP_EQ_TAPS + 1];
    int16_t narrow16[COMP_NARROW_TAPS + 1];
    int32_t eq_k, eq_lim, narrow_lim;
    int16_t sin_q15[COMP_SC_DEN_PAL];
    int16_t pal_demod_coef[COMP_PAL_DEMOD_ROWS][32];
    comp_pal_demod_fn pal_demod;
    comp_fir_row_q15_fn fir_row;
    comp_fir_row_s16_fn fir_row_s16;
    comp_split3d_row_fn split3d_row;
    comp_ntsc_comb2d_row_fn comb2d_row;
    comp_demod_quant_row_fn demod_quant_row;
    comp_demod_rotate_row_fn demod_rotate_row;
    comp_demod_pq_row_fn demod_pq_row;
    int16_t demod_pq_coef[COMP_DEMOD_PQ_PAIRS][32];
    comp_transform2d_t transform;
    comp_transform3d_t transform3;
    comp_t3d_cache_t t3cache;    /* 3D transform slab cache */
    comp_encode_t enc;   /* for refine resynthesis */

    /* scratch pool, allocated once at init and reused per frame */
    int nscratch;
    comp_decode_scratch_t *scratch;
    pthread_mutex_t lock;
    pthread_cond_t cond;
};

/* nscratch is the maximum number of concurrent frame requests. setup
 * selects the NTSC 7.5 IRE pedestal; threshold is Transform PAL's
 * bin-symmetry ratio. eq enables the chroma cascade equalizer; refine
 * is the number of Y-only Landweber refinement iterations against the
 * dimensions=1 model; use_transform selects Transform NTSC separation
 * instead of the comb (1) or a per-sample comb/transform hybrid (2)
 * (NTSC, dimensions=3 only); level selects the
 * transform's amplitude-limiting mode (threshold then unused).
 * eq: 0 off, 1 the fixed cascade inverse, 2 leak-aware (the transform's
 * confidence map steers chroma bandwidth from a sub-nominal low-pass
 * through nominal to the boosted cascade inverse; needs a transform
 * path). evidence > 0 enables the LF-luma prior (PAL transforms only;
 * see transform2d.h). cti resynthesizes chroma edges from coincident
 * luma transitions (dimensions >= 2). */
int comp_decode_init(comp_decode_t *d, int standard, double threshold,
                     int nscratch, int setup, int dimensions, int eq, int refine,
                     int use_transform, int level, double evidence, int cti);
void comp_decode_free(comp_decode_t *d);

/* override the Transform PAL per-bin thresholds after init; n must be
 * COMP_T2D_NTHRESH (dimensions=2) or COMP_T3D_NTHRESH (dimensions=3) */
int comp_decode_set_thresholds(comp_decode_t *d, const double *t, int n);

/* install a trained per-bin soft-gain LUT (transform separations only),
 * replacing the pair test; n must be the bin count * COMP_LUT_K */
int comp_decode_set_lut(comp_decode_t *d, const double *v, int n);

/* source frames needed each side of the decoded frame */
int comp_decode_look(const comp_decode_t *d);

/* Demodulate one frame of active-picture composite (GRAY16 at composite
 * levels) into YUV444P16; strides are in samples. rows is the frame's
 * height and row_off its position in the raster: a 480-line NTSC frame
 * occupies raster rows row_off..row_off+479.
 * views[]/view_frames[] hold 2*look+1 composite frames centered on
 * `frame`, clip edges clamped (with the clamped frame numbers); nframes
 * is the clip length, so out-of-clip fields become black as in the
 * reference implementation.
 * orig_y, when non-NULL, is the luma plane of the pre-encode degraded
 * picture at the raster; d->refine Landweber iterations then deconvolve
 * the crude-decoder model against it (Y only).
 * metrics, when non-NULL, receives the per-frame difficulty proxies;
 * each field is written only when its path is active (see
 * comp_decode_metrics_t), so the caller must init it to
 * COMP_METRICS_INIT and emit a prop only for fields that changed.
 * out_mask, when non-NULL, receives a per-sample GRAY16 mask selected by
 * mask_kind, one frame at mask_stride, indexed by the same 0-based active
 * rows as dsty (rows beyond `rows` are untouched, so the caller pre-zeros
 * the plane):
 *  - COMP_MASK_MOTION: the NTSC hybrid motion router (65535 = motion /
 *    transform-routed, 0 = still / comb); NTSC dimensions=3 transform=2.
 *  - COMP_MASK_CONFIDENCE: 65535*(1 - conf), i.e. least-confident white,
 *    the inverse of the eq=2 separation-confidence map; any eq=2 path. */
enum { COMP_MASK_NONE = 0, COMP_MASK_MOTION, COMP_MASK_CONFIDENCE };

void comp_decode_frame(comp_decode_t *d, int frame, int nframes,
                       int rows, int row_off,
                       const comp_frame_view_t *views, const int *view_frames,
                       int look,
                       const uint16_t *orig_y, ptrdiff_t orig_stride,
                       uint16_t *dsty, ptrdiff_t ystride,
                       uint16_t *dstu, ptrdiff_t ustride,
                       uint16_t *dstv, ptrdiff_t vstride,
                       comp_decode_metrics_t *metrics,
                       uint16_t *out_mask, ptrdiff_t mask_stride, int mask_kind);

#endif
