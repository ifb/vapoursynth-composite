#ifndef COMP_TRANSFORM3D_H
#define COMP_TRANSFORM3D_H

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#include <fftw3.h>

#include "transform2d.h"  /* COMP_LUT_K */

/* Overlapping-tile geometry: XTILE samples by YTILE frame lines by
 * ZTILE fields, advancing by half a tile in each direction. NTSC
 * assembles frame-line tiles with the other field's rows black; PAL
 * separates on the displaced field-line lattice instead (GB 2365247 A:
 * one picture line of displacement per field interval makes the fields
 * a dense lattice), so its tiles are YTILE/2 field lines — half the
 * FFT volume for the same filter. */
#define COMP_T3D_XTILE     16
#define COMP_T3D_YTILE     32
#define COMP_T3D_YTILE_PAL (COMP_T3D_YTILE / 2)
#define COMP_T3D_ZTILE     8
#define COMP_T3D_XCOMPLEX  (COMP_T3D_XTILE / 2 + 1)

/* fields the tiles covering one output frame can span, each side of it */
#define COMP_T3D_LOOK 3

/* bins the symmetry filter considers: z, y outer, x fsc/2..fsc inner */
#define COMP_T3D_NTHRESH (COMP_T3D_ZTILE * COMP_T3D_YTILE \
                          * (COMP_T3D_XTILE / 4 - COMP_T3D_XTILE / 8 + 1))
#define COMP_T3D_NTHRESH_PAL (COMP_T3D_NTHRESH / 2)

/* one bin row per (z, y): the three x band bins */
#define COMP_T3D_NROWS (COMP_T3D_ZTILE * COMP_T3D_YTILE)

/* the row kernels store four lanes per three-bin row, so per-bin
 * arrays carry this much slack past the bin count */
#define COMP_T3D_BINPAD 4

typedef struct comp_field_view_t comp_field_view_t;

struct comp_field_view_t {
    const uint16_t *data;
    ptrdiff_t stride;
};

/* Dispatched per-bin trained-LUT gain row: for each of n bins, r[i] is
 * the pair-symmetry ratio lo/hi of (m_in[i], m_ref[i]) (1 when hi is
 * 0) and g[i] the linear interpolation of the 16-knot row lut[i] at
 * r[i] * (COMP_LUT_K - 1). n may be rounded up to a multiple of 16;
 * all arrays (and lut rows) must be sized for the rounded count. */
typedef void (*comp_lut_gain_fn)(float *g, float *r, const float *m_in,
                                 const float *m_ref,
                                 const float (*lut)[COMP_LUT_K], int n);
comp_lut_gain_fn comp_get_lut_gain_fn(unsigned cpu);

/* Fill rows[i] = { own, ref } float offsets of bin row i's x bands:
 * own points at the x = XTILE/8 bin of the row, ref at the x = XTILE/4
 * bin of the reflected row. The kernels take all addressing from the
 * table, so a different tile layout is just a different table. */
void comp_t3d_rowpair(int standard, int32_t (*rows)[2]);

/* Dispatched row kernels over the bin-row table. mag fills the linear
 * per-bin magnitude rows, the reflected side read reversed; apply
 * writes the gain-scaled x bands in table order, preserving the
 * twice-visited x = XTILE/4 column's later-write-wins semantics.
 * Both may store up to COMP_T3D_BINPAD floats past 3 * nrows into the
 * per-bin arrays (and apply reads as far into g); tile positions
 * outside the bands are never touched. */
typedef void (*comp_t3d_mag_fn)(float *m_in, float *m_ref, const float *in,
                                const int32_t (*rows)[2], int nrows);
typedef void (*comp_t3d_apply_fn)(float *out, const float *in, const float *g,
                                  const int32_t (*rows)[2], int nrows);
comp_t3d_mag_fn comp_get_t3d_mag_fn(unsigned cpu);
comp_t3d_apply_fn comp_get_t3d_apply_fn(unsigned cpu);

/* internal-FFT entry points (see fft.h); stored here so the filter
 * can route per tile without an include cycle */
typedef void (*comp_fft_fwd_fn)(float *band, const float *real,
                                const float *black, unsigned ph);
typedef void (*comp_fft_inv_fn)(float *real, float *band, unsigned ph);

typedef struct {
    int bin;      /* linear bin index */
    int32_t off;  /* float offset of the bin in the tile */
    int keep;
} comp_t3d_special_t;

typedef struct comp_transform3d_t comp_transform3d_t;

struct comp_transform3d_t {
    fftwf_plan forward;
    fftwf_plan inverse;
    comp_lut_gain_fn lut_gain;
    comp_t3d_mag_fn t3d_mag;
    comp_t3d_apply_fn t3d_apply;
    int standard;
    int level;
    int use_lut;
    float evidence;
    /* arrays sized for the larger NTSC geometry; PAL uses the first
     * YTILE_PAL window rows and NTHRESH_PAL bins */
    float window[COMP_T3D_ZTILE][COMP_T3D_YTILE][COMP_T3D_XTILE];
    float threshold_sq[COMP_T3D_NTHRESH];
    float lut[COMP_T3D_NTHRESH][COMP_LUT_K];
    /* bin-row geometry for the staged LUT filter: per bin row the
     * float offsets of its own and reflected tile rows, and the
     * self-column special bins (kept carriers / discarded non-carriers
     * at x = XTILE/4, where the bin is its own reflection) */
    int32_t rowpair[COMP_T3D_NROWS][2];
    comp_t3d_special_t special[8];
    int nspecial;
    /* the same tables for the internal FFT's band layout: packed
     * 64-byte rows at bit-reversed (ky, kz); fft_fwd/fft_inv are the
     * avx2 transforms, NULL when the FFTW path must be used */
    int32_t rowpair_fft[COMP_T3D_NROWS][2];
    comp_t3d_special_t special_fft[8];
    comp_fft_fwd_fn fft_fwd;
    comp_fft_inv_fn fft_inv;
    /* NTSC: the x spectra of black rows (LEVEL_BLACK times the window),
     * band-row layout, for the rows an NTSC tile takes from the other
     * field */
    float fft_black[COMP_T3D_ZTILE * COMP_T3D_YTILE * COMP_T3D_XTILE];
};

/* PAL chroma on the displaced lattice is symmetric about the measured
 * carrier bins, U at (z,y,x) = (4,12,4) and V at (4,4,4) of the
 * 8x16x16 tile FFT; both share one reflection map. The NTSC variant
 * (after the ld-decode transform-ntsc branch) reflects about
 * (fsc, 120 c/aph, 15 Hz) with luma-reference evidence and a
 * frequency-shaped threshold, since NTSC's shared U/V carrier makes
 * the symmetry only approximate. level selects amplitude limiting
 * instead of the threshold test (threshold then unused). evidence > 0
 * enables the LF-luma prior of US 7,872,689 (PAL only; see
 * transform2d.h). */
int comp_transform3d_init(comp_transform3d_t *t, double threshold, int standard,
                          int level, double evidence);
void comp_transform3d_free(comp_transform3d_t *t);

/* install a trained per-bin soft-gain LUT (COMP_T3D_NTHRESH_PAL or
 * COMP_T3D_NTHRESH bins of COMP_LUT_K values in [0,1], knots
 * innermost), replacing the pair test (and, for NTSC, the shaped
 * threshold and luma-evidence test) */
void comp_transform3d_set_lut(comp_transform3d_t *t, const double *v);

/* Slab cache. A slab holds the accumulated chroma (and confidence)
 * contribution of every tile at one temporal grid position tz to the
 * ZTILE fields those tiles span; the tile grid is anchored at absolute
 * field 0, so a slab's content depends only on (tz, parity) and each
 * slab serves the four output frames whose tiles it provides. Slabs
 * are built once under the cache and reused, cutting the FFT and bin
 * filter work by up to 4x. */
typedef struct {
    int tz;          /* absolute field index of the slab origin */
    int parity;
    int ready;       /* built; slabs are immutable once ready */
    int refs;        /* readers + builder; protects against eviction */
    uint64_t stamp;  /* LRU */
    float *chroma;   /* [ZTILE][field_rows][width] */
    float *conf;     /* likewise; NULL when confidence is not tracked */
} comp_t3d_slab_t;

typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t cond;
    comp_t3d_slab_t *slabs;
    int nslabs;
    int width;       /* capacity; frame calls may use less */
    int field_rows;
    uint64_t counter;
} comp_t3d_cache_t;

/* nslabs >= 2 (a frame draws on two slabs); size it to the number of
 * concurrent frame requests / 2 + 3 so in-flight frames do not evict
 * each other's slabs. width/field_rows are a capacity: frame calls may
 * pass smaller geometry (constant within a clip). with_conf must be
 * set when the frame calls will request confidence maps. */
int comp_t3d_cache_init(comp_t3d_cache_t *c, int nslabs, int width,
                        int field_rows, int with_conf);
void comp_t3d_cache_free(comp_t3d_cache_t *c);

/* Extract the chroma of the two fields of output frame `frame`.
 * fields[] holds views of the absolute field indices [z0, z0 + nfields);
 * out-of-clip fields must be supplied clamped with parity preserved,
 * and the window must cover fields the caller's slabs span so that a
 * slab's content is the same whichever frame builds it. parity gives
 * the raster row parity of even (temporally first) fields: 0 for PAL;
 * NTSC's first field sits on odd rows at row offsets 0/4 and even rows
 * at offset 5. chroma0/chroma1 are the row-parity field buffers matching
 * the raster layout: the two covering slabs are summed and quantised to
 * int16 in one pass, so the demod's fixed-point input needs no separate
 * float staging. chroma_stride is in int16 elements. conf0/conf1, when
 * non-NULL, receive [0,1] chroma-confidence maps (float, same layout,
 * stride conf_stride floats). width and field_rows must not exceed the
 * cache capacity. */
void comp_transform3d_frame(const comp_transform3d_t *t, comp_t3d_cache_t *c,
                            const comp_field_view_t *fields, int z0, int nfields,
                            int frame, int parity, int width, int field_rows,
                            int16_t *chroma0, int16_t *chroma1,
                            ptrdiff_t chroma_stride,
                            float *conf0, float *conf1, ptrdiff_t conf_stride);

#endif
