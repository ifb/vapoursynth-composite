/*
 * PAL/NTSC composite video encode/decode for VapourSynth.
 *
 * Encode() modulates YCbCr to composite; Decode() demodulates it back
 * via 2D Transform PAL (PAL only so far).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <VapourSynth4.h>
#include <VSConstants4.h>
#include <VSHelper4.h>

#include "decode.h"
#include "encode.h"
#include "geom.h"
#include "lut_tables.h"
#include "subcarrier.h"

typedef struct comp_filter_t comp_filter_t;

struct comp_filter_t {
    VSNode *node;
    VSNode *orig_node;   /* Restore: the pre-encode picture at the raster */
    VSVideoInfo vi;
    int standard;
    int in_frames;
    int mask_kind;       /* COMP_MASK_* : attach that mask as a frame prop
                          * (VS has no 4-plane YUVA; PropToClip pulls it
                          * out). 0 = none. */
    int pad;             /* decode: replicated columns either side of the
                          * raster for the resample back; 0 at width=0 */
    comp_encode_t enc;
    comp_decode_t *dec;
};

/* Where the picture sits in the raster. A 480-line NTSC clip occupies
 * the 486-line BFF raster at rows 4..483 when its content is BFF (the
 * DV convention) or rows 5..484 when TFF (ATSC/RP 202) — the odd offset
 * re-aligns TFF content to the BFF raster. Field order comes from the
 * frame's _FieldBased prop; absent or progressive means BFF. */
static int comp_row_offset(const comp_filter_t *f, const VSFrame *frame, const VSAPI *vsapi)
{
    if (f->standard != COMP_STD_NTSC || f->vi.height != 480)
        return 0;

    int err;
    const int64_t fb = vsapi->mapGetInt(vsapi->getFramePropertiesRO(frame), "_FieldBased", 0, &err);
    return (!err && fb == VSC_FIELD_TOP) ? 5 : 4;
}

static const VSFrame *VS_CC comp_encode_get_frame(int n, int activation_reason, void *instance_data,
                                                  void **frame_data, VSFrameContext *frame_ctx,
                                                  VSCore *core, const VSAPI *vsapi)
{
    comp_filter_t *f = instance_data;
    (void)frame_data;

    if (activation_reason == arInitial) {
        vsapi->requestFrameFilter(n, f->node, frame_ctx);
        return NULL;
    }
    if (activation_reason != arAllFramesReady)
        return NULL;

    const VSFrame *src = vsapi->getFrameFilter(n, f->node, frame_ctx);
    VSFrame *dst = vsapi->newVideoFrame(&f->vi.format, f->vi.width, f->vi.height, src, core);

    const uint16_t *srcy = (const uint16_t *)vsapi->getReadPtr(src, 0);
    const uint16_t *srcu = (const uint16_t *)vsapi->getReadPtr(src, 1);
    const uint16_t *srcv = (const uint16_t *)vsapi->getReadPtr(src, 2);
    const ptrdiff_t ystride = vsapi->getStride(src, 0) / 2;
    const ptrdiff_t ustride = vsapi->getStride(src, 1) / 2;
    const ptrdiff_t vstride = vsapi->getStride(src, 2) / 2;
    uint16_t *dstp = (uint16_t *)vsapi->getWritePtr(dst, 0);
    const ptrdiff_t dstride = vsapi->getStride(dst, 0) / 2;

    comp_encode_frame(&f->enc, n, f->vi.height, comp_row_offset(f, src, vsapi),
                      dstp, dstride, srcy, ystride, srcu, ustride, srcv, vstride);

    vsapi->freeFrame(src);
    return dst;
}

#define COMP_MAX_LOOK COMP_T3D_LOOK

/* Replicate each row's first and last raster samples into the pad
 * columns either side. The final BT.601 resample reads a few samples
 * beyond the active raster (the 601 window is wider in time), and zimg
 * fills out-of-bounds taps by MIRRORING, which reflects interior picture
 * onto the frame edges; replicated columns make those taps read the
 * edge value instead. */
static void comp_fill_pad(uint16_t *p, ptrdiff_t stride, int rw, int pad, int h)
{
    for (int r = 0; r < h; r++, p += stride) {
        for (int k = 0; k < pad; k++) {
            p[k] = p[pad];
            p[pad + rw + k] = p[pad + rw - 1];
        }
    }
}

static const VSFrame *VS_CC comp_decode_get_frame(int n, int activation_reason, void *instance_data,
                                                  void **frame_data, VSFrameContext *frame_ctx,
                                                  VSCore *core, const VSAPI *vsapi)
{
    comp_filter_t *f = instance_data;
    const int look = comp_decode_look(f->dec);
    const int last = f->in_frames - 1;
    (void)frame_data;

    if (activation_reason == arInitial) {
        int prev = -1;
        for (int i = -look; i <= look; i++) {
            const int k = VSMIN(VSMAX(n + i, 0), last);
            if (k != prev)
                vsapi->requestFrameFilter(k, f->node, frame_ctx);
            prev = k;
        }
        if (f->orig_node)
            vsapi->requestFrameFilter(n, f->orig_node, frame_ctx);
        return NULL;
    }
    if (activation_reason != arAllFramesReady)
        return NULL;

    const VSFrame *srcs[2 * COMP_MAX_LOOK + 1];
    comp_frame_view_t views[2 * COMP_MAX_LOOK + 1];
    int view_frames[2 * COMP_MAX_LOOK + 1];
    for (int i = 0; i <= 2 * look; i++) {
        const int k = VSMIN(VSMAX(n - look + i, 0), last);
        srcs[i] = vsapi->getFrameFilter(k, f->node, frame_ctx);
        views[i].data = (const uint16_t *)vsapi->getReadPtr(srcs[i], 0);
        views[i].stride = vsapi->getStride(srcs[i], 0) / 2;
        view_frames[i] = k;
    }
    const VSFrame *src = srcs[look];
    const VSFrame *orig = f->orig_node ? vsapi->getFrameFilter(n, f->orig_node, frame_ctx) : NULL;
    VSFrame *dst = vsapi->newVideoFrame(&f->vi.format, f->vi.width, f->vi.height, src, core);
    const int pad = f->pad;
    const int rw = f->vi.width - 2 * pad;  /* the raster */

    /* mask=: a standalone GRAY16 raster frame holds the requested mask,
     * attached to the picture frame under a private property. VS has no
     * 4-plane YUVA; std.PropToClip pulls it back as a separate clip
     * downstream, so both outputs share this one decode. */
    VSFrame *maskf = NULL;
    uint16_t *maskp = NULL;
    ptrdiff_t mask_stride = 0;
    if (f->mask_kind) {
        VSVideoFormat gray;
        vsapi->queryVideoFormat(&gray, cfGray, stInteger, 16, 0, 0, core);
        maskf = vsapi->newVideoFrame(&gray, f->vi.width, f->vi.height, NULL, core);
        maskp = (uint16_t *)vsapi->getWritePtr(maskf, 0);
        mask_stride = vsapi->getStride(maskf, 0) / 2;
        /* the decode writes the mask only over the active picture rows;
         * pre-zero so blanking/edge rows read still / fully-confident (0) */
        for (int r = 0; r < f->vi.height; r++)
            memset(maskp + r * mask_stride, 0, sizeof(uint16_t) * f->vi.width);
    }

    comp_decode_metrics_t m = COMP_METRICS_INIT;
    comp_decode_frame(f->dec, n, f->in_frames, f->vi.height, comp_row_offset(f, src, vsapi),
                      views, view_frames, look,
                      orig ? (const uint16_t *)vsapi->getReadPtr(orig, 0) : NULL,
                      orig ? vsapi->getStride(orig, 0) / 2 : 0,
                      (uint16_t *)vsapi->getWritePtr(dst, 0) + pad, vsapi->getStride(dst, 0) / 2,
                      (uint16_t *)vsapi->getWritePtr(dst, 1) + pad, vsapi->getStride(dst, 1) / 2,
                      (uint16_t *)vsapi->getWritePtr(dst, 2) + pad, vsapi->getStride(dst, 2) / 2,
                      &m, maskp ? maskp + pad : NULL, mask_stride, f->mask_kind);
    if (pad) {
        for (int p = 0; p < 3; p++)
            comp_fill_pad((uint16_t *)vsapi->getWritePtr(dst, p),
                          vsapi->getStride(dst, p) / 2, rw, pad, f->vi.height);
        if (maskp)
            comp_fill_pad(maskp, mask_stride, rw, pad, f->vi.height);
    }

    /* each difficulty prop exists only where its path is active (the field
     * stays at the sentinel otherwise). Restore's edge splice drops them,
     * so its create function copies them back onto the spliced output. */
    VSMap *props = vsapi->getFramePropertiesRW(dst);
    if (m.conf_mean >= 0.0) {
        vsapi->mapSetFloat(props, "CompositeSeparationConfidenceMean", m.conf_mean, maReplace);
        vsapi->mapSetFloat(props, "CompositeSeparationConfidenceStdDev", m.conf_std, maReplace);
    }
    if (m.motion_fraction >= 0.0)
        vsapi->mapSetFloat(props, "CompositeMotionFraction", m.motion_fraction, maReplace);
    if (m.refine_residual >= 0.0)
        vsapi->mapSetFloat(props, "CompositeRefineResidual", m.refine_residual, maReplace);
    if (m.refine_correction >= 0.0)
        vsapi->mapSetFloat(props, "CompositeRefineCorrection", m.refine_correction, maReplace);
    if (maskf) {
        vsapi->mapConsumeFrame(props, "CompositeMask", maskf, maReplace);
        maskf = NULL;
    }

    if (orig)
        vsapi->freeFrame(orig);
    for (int i = 0; i <= 2 * look; i++)
        vsapi->freeFrame(srcs[i]);
    return dst;
}

static void VS_CC comp_free(void *instance_data, VSCore *core, const VSAPI *vsapi)
{
    comp_filter_t *f = instance_data;
    (void)core;
    if (f->dec) {
        comp_decode_free(f->dec);
        free(f->dec);
    }
    if (f->orig_node)
        vsapi->freeNode(f->orig_node);
    vsapi->freeNode(f->node);
    free(f);
}

#define RETERROR(x) do { \
    char msg[256]; \
    snprintf(msg, sizeof(msg), "%s: %s", name, x); \
    vsapi->mapSetError(out, msg); \
    if (d.node) \
        vsapi->freeNode(d.node); \
    return; \
} while (0)

static int comp_parse_standard(const VSMap *in, const VSAPI *vsapi, int *standard)
{
    int err;
    const char *s = vsapi->mapGetData(in, "standard", 0, &err);
    if (err)
        s = "pal";
    if (!strcmp(s, "pal"))
        *standard = COMP_STD_PAL;
    else if (!strcmp(s, "ntsc"))
        *standard = COMP_STD_NTSC;
    else
        return -1;
    return 0;
}


/* Pull the raster motion mask (a GRAY16 frame the decode node attached
 * under CompositeMask) out as its own clip via std.PropToClip, and
 * strip that heavy frame-typed prop off the picture path: *dec is replaced
 * with a RemoveFrameProps-wrapped node so the picture output does not carry
 * the mask frame (Spline36 would otherwise propagate it). Consumes *dec,
 * setting it to the stripped node; returns the mask node (NULL on failure,
 * with *dec freed). Both nodes ultimately hit the one decode getFrame for
 * frame n, which the cache dedups under linear access. */
static VSNode *comp_mask_from_props(VSNode **dec, VSCore *core, const VSAPI *vsapi)
{
    VSPlugin *std = vsapi->getPluginByID("com.vapoursynth.std", core);
    VSNode *d = *dec;
    *dec = NULL;

    VSMap *args = vsapi->createMap();
    vsapi->mapSetNode(args, "clip", d, maReplace);
    vsapi->mapSetData(args, "prop", "CompositeMask", -1, dtUtf8, maReplace);
    VSMap *ret = vsapi->invoke(std, "PropToClip", args);
    vsapi->freeMap(args);
    if (vsapi->mapGetError(ret)) {
        vsapi->freeMap(ret);
        vsapi->freeNode(d);
        return NULL;
    }
    VSNode *mask = vsapi->mapGetNode(ret, "clip", 0, NULL);
    vsapi->freeMap(ret);

    /* strip the attached mask frame off the picture node */
    args = vsapi->createMap();
    vsapi->mapConsumeNode(args, "clip", d, maReplace);
    vsapi->mapSetData(args, "props", "CompositeMask", -1, dtUtf8, maReplace);
    ret = vsapi->invoke(std, "RemoveFrameProps", args);
    vsapi->freeMap(args);
    if (vsapi->mapGetError(ret)) {
        vsapi->freeMap(ret);
        vsapi->freeNode(mask);
        return NULL;
    }
    *dec = vsapi->mapGetNode(ret, "clip", 0, NULL);
    vsapi->freeMap(ret);
    return mask;
}

/* Resample the padded raster mask to the output geometry on the SAME crop
 * as the picture (identical src_left/src_width/width so it stays sample-
 * aligned), but with bilinear — Spline36's lobes ring a 0/1 mask edge.
 * Consumes mask, also on failure. */
static VSNode *comp_resample_mask(VSNode *mask, VSCore *core, const VSAPI *vsapi,
                                  int width, int height,
                                  double src_left, double src_width)
{
    VSPlugin *resize = vsapi->getPluginByID("com.vapoursynth.resize", core);
    VSMap *args = vsapi->createMap();
    vsapi->mapConsumeNode(args, "clip", mask, maReplace);
    vsapi->mapSetInt(args, "width", width, maReplace);
    vsapi->mapSetInt(args, "height", height, maReplace);
    vsapi->mapSetFloat(args, "src_left", src_left, maReplace);
    vsapi->mapSetFloat(args, "src_width", src_width, maReplace);
    VSMap *ret = vsapi->invoke(resize, "Bilinear", args);
    vsapi->freeMap(args);
    if (vsapi->mapGetError(ret)) {
        vsapi->freeMap(ret);
        return NULL;
    }
    VSNode *out = vsapi->mapGetNode(ret, "clip", 0, NULL);
    vsapi->freeMap(ret);
    return out;
}

static void VS_CC comp_encode_create(const VSMap *in, VSMap *out, void *user_data, VSCore *core, const VSAPI *vsapi)
{
    const char *name = user_data;
    comp_filter_t d = {0};
    int err;

    d.node = vsapi->mapGetNode(in, "clip", 0, NULL);
    d.vi = *vsapi->getVideoInfo(d.node);

    if (comp_parse_standard(in, vsapi, &d.standard))
        RETERROR("standard must be pal or ntsc");

    const int setup = !!vsapi->mapGetIntSaturated(in, "setup", 0, &err);
    const int precomb = !!vsapi->mapGetIntSaturated(in, "precomb", 0, &err);

    if (!vsh_isConstantVideoFormat(&d.vi))
        RETERROR("clip must have constant format and dimensions");
    if (d.vi.format.colorFamily != cfYUV)
        RETERROR("clip must be YUV");
    if (d.standard == COMP_STD_PAL) {
        if (d.vi.height != COMP_ACTIVE_HEIGHT_PAL)
            RETERROR("pal input must have 576 lines");
    } else {
        if (d.vi.height != 480 && d.vi.height != COMP_ACTIVE_HEIGHT_NTSC)
            RETERROR("ntsc input must have 480 or 486 lines");
    }

    /* resample to the 4xfsc active raster and 4:4:4, equating the
     * shared time base (see the anchors above); lines map 1:1 */
    VSPlugin *resize = vsapi->getPluginByID("com.vapoursynth.resize", core);
    if (!resize)
        RETERROR("resize plugin not found");

    int width;
    double src_left, src_width;
    comp_encode_resample_params(d.standard, d.vi.width, &width, &src_left, &src_width);

    VSMap *args = vsapi->createMap();
    vsapi->mapConsumeNode(args, "clip", d.node, maReplace);
    d.node = NULL;
    vsapi->mapSetInt(args, "format", pfYUV444P16, maReplace);
    vsapi->mapSetInt(args, "width", width, maReplace);
    vsapi->mapSetInt(args, "height", d.vi.height, maReplace);
    vsapi->mapSetFloat(args, "src_left", src_left, maReplace);
    vsapi->mapSetFloat(args, "src_width", src_width, maReplace);
    VSMap *ret = vsapi->invoke(resize, "Spline36", args);
    vsapi->freeMap(args);

    const char *invoke_err = vsapi->mapGetError(ret);
    if (invoke_err) {
        char msg[256];
        snprintf(msg, sizeof(msg), "%s: %s", name, invoke_err);
        vsapi->mapSetError(out, msg);
        vsapi->freeMap(ret);
        return;
    }
    d.node = vsapi->mapGetNode(ret, "clip", 0, NULL);
    vsapi->freeMap(ret);

    comp_encode_init(&d.enc, d.standard, setup, precomb);
    vsapi->queryVideoFormat(&d.vi.format, cfGray, stInteger, 16, 0, 0, core);
    d.vi.width = width;

    comp_filter_t *data = malloc(sizeof(*data));
    *data = d;

    VSFilterDependency deps[] = {{ data->node, rpStrictSpatial }};
    vsapi->createVideoFilter(out, name, &data->vi, comp_encode_get_frame, comp_free,
                             fmParallel, deps, 1, data, core);
}

static void VS_CC comp_decode_create(const VSMap *in, VSMap *out, void *user_data, VSCore *core, const VSAPI *vsapi)
{
    const char *name = user_data;
    comp_filter_t d = {0};
    int err;

    d.node = vsapi->mapGetNode(in, "clip", 0, NULL);
    d.vi = *vsapi->getVideoInfo(d.node);

    if (comp_parse_standard(in, vsapi, &d.standard))
        RETERROR("standard must be pal or ntsc");

    int width = vsapi->mapGetIntSaturated(in, "width", 0, &err);
    if (err)
        width = 720;
    /* width=0 requests the raw 4xfsc decode raster (SMPTE 244M / EBU 4fsc)
     * with no horizontal resample: the picture and mask come straight off
     * the decode grid, crisp, for callers who resize themselves. */
    if (width != 0 && (width < 16 || width > 8192))
        RETERROR("width must be 0 (raw raster) or between 16 and 8192");

    double threshold = vsapi->mapGetFloat(in, "threshold", 0, &err);
    const int threshold_unset = err;
    if (err)
        threshold = 0.4;
    if (!(threshold > 0.0 && threshold <= 1.0))
        RETERROR("threshold must be in (0, 1]");

    const int setup = !!vsapi->mapGetIntSaturated(in, "setup", 0, &err);

    int dimensions = vsapi->mapGetIntSaturated(in, "dimensions", 0, &err);
    if (err)
        dimensions = 3;
    if (dimensions < 1 || dimensions > 3)
        RETERROR("dimensions must be 1, 2 or 3");

    int eq = vsapi->mapGetIntSaturated(in, "eq", 0, &err);
    const int eq_unset = err;
    if (err)
        eq = 1;
    if (eq < 0 || eq > 2)
        RETERROR("eq must be 0 (off), 1 (fixed) or 2 (leak-aware)");





    int transform = vsapi->mapGetIntSaturated(in, "transform", 0, &err);
    if (err)
        transform = (d.standard == COMP_STD_NTSC && dimensions == 3) ? 2 : 0;
    if (transform < 0 || transform > 2)
        RETERROR("transform must be 0 (comb), 1 (transform) or 2 (hybrid)");
    if (transform && d.standard != COMP_STD_NTSC)
        RETERROR("transform applies to ntsc (pal always uses the transform)");
    if (transform && dimensions != 3)
        RETERROR("transform needs dimensions=3");

    /* unset level/eq adapt to the chosen path: the measured best
     * settings wherever a transform separation is present */
    const int has_transform = d.standard == COMP_STD_PAL ? dimensions >= 2
                                                         : transform != 0;
    if (eq_unset && has_transform)
        eq = 2;
    if (eq == 2 && !has_transform)
        RETERROR("eq=2 needs a transform separation");

    /* an explicit threshold, thresholds, lut, or level selects its own
     * mode; in their absence a transform path defaults to the built-in
     * trained soft-gain table (set after the decoder is created) */
    int level = vsapi->mapGetIntSaturated(in, "level", 0, &err);
    const int builtin_lut = err && has_transform && threshold_unset
                            && vsapi->mapNumElements(in, "thresholds") < 1
                            && vsapi->mapNumElements(in, "lut") < 1;
    if (err)
        level = 0;
    if (level && dimensions < 2)
        RETERROR("level=1 needs dimensions 2 or 3");
    if (level && d.standard == COMP_STD_NTSC && !transform)
        RETERROR("level=1 needs a transform separation for ntsc");

    double evidence = vsapi->mapGetFloat(in, "evidence", 0, &err);
    if (err)
        evidence = 0.0;
    if (evidence < 0.0)
        RETERROR("evidence must be >= 0");
    if (evidence > 0.0 && (d.standard != COMP_STD_PAL || dimensions < 2))
        RETERROR("evidence needs a pal transform (dimensions 2 or 3)");

    const int cti = !!vsapi->mapGetIntSaturated(in, "cti", 0, &err);
    if (cti && dimensions < 2)
        RETERROR("cti needs dimensions 2 or 3");

    const int nlut = vsapi->mapNumElements(in, "lut");
    if (nlut > 0) {
        if (d.standard == COMP_STD_NTSC && !transform)
            RETERROR("lut needs a transform separation");
        if (level)
            RETERROR("lut and level are mutually exclusive");
        if (dimensions == 2 && nlut != COMP_T2D_NTHRESH * COMP_LUT_K)
            RETERROR("dimensions=2 needs 1280 lut values (80 bins x 16 knots)");
        if (dimensions == 3
            && nlut != (d.standard == COMP_STD_PAL ? COMP_T3D_NTHRESH_PAL
                                                   : COMP_T3D_NTHRESH) * COMP_LUT_K)
            RETERROR(d.standard == COMP_STD_PAL
                     ? "pal dimensions=3 needs 6144 lut values (384 bins x 16 knots)"
                     : "ntsc dimensions=3 needs 12288 lut values (768 bins x 16 knots)");
        if (dimensions == 1)
            RETERROR("lut needs dimensions 2 or 3");
        if (vsapi->mapNumElements(in, "thresholds") > 0)
            RETERROR("lut and thresholds are mutually exclusive");
    }

    if (!vsh_isConstantVideoFormat(&d.vi))
        RETERROR("clip must have constant format and dimensions");
    if (d.vi.format.colorFamily != cfGray || d.vi.format.sampleType != stInteger
        || d.vi.format.bitsPerSample != 16)
        RETERROR("clip must be GRAY16 composite");
    if (d.standard == COMP_STD_PAL) {
        if (d.vi.width != COMP_ACTIVE_WIDTH_PAL || d.vi.height != COMP_ACTIVE_HEIGHT_PAL)
            RETERROR("pal composite input must be 928x576");
    } else {
        if (d.vi.width != COMP_ACTIVE_WIDTH_NTSC
            || (d.vi.height != 480 && d.vi.height != COMP_ACTIVE_HEIGHT_NTSC))
            RETERROR("ntsc composite input must be 758x480 or 758x486");
    }

    VSPlugin *resize = vsapi->getPluginByID("com.vapoursynth.resize", core);
    if (!resize)
        RETERROR("resize plugin not found");

    /* mask=: emit a per-sample mask as a second output clip. "motion" is
     * the NTSC hybrid router; "confidence" is the eq=2 separation
     * confidence (any transform path). */
    const char *maskmode = vsapi->mapGetData(in, "mask", 0, &err);
    if (!err && maskmode) {
        if (!strcmp(maskmode, "motion")) {
            if (!(d.standard == COMP_STD_NTSC && dimensions == 3 && transform == 2))
                RETERROR("mask=\"motion\" needs the ntsc hybrid path "
                         "(dimensions=3, transform=2)");
            d.mask_kind = COMP_MASK_MOTION;
        } else if (!strcmp(maskmode, "confidence")) {
            if (eq != 2)
                RETERROR("mask=\"confidence\" needs eq=2 (a transform separation)");
            d.mask_kind = COMP_MASK_CONFIDENCE;
        } else {
            RETERROR("mask must be \"motion\" or \"confidence\"");
        }
    }

    VSCoreInfo info;
    vsapi->getCoreInfo(core, &info);

    d.dec = malloc(sizeof(*d.dec));
    if (!d.dec)
        RETERROR("out of memory");
    if (comp_decode_init(d.dec, d.standard, threshold,
                         info.numThreads < 1 ? 1 : info.numThreads, setup,
                         dimensions, eq, 0, transform, level, evidence,
                         cti)) {
        free(d.dec);
        d.dec = NULL;
        RETERROR("decoder initialisation failed");
    }

    const int nthresh = vsapi->mapNumElements(in, "thresholds");
    if (nthresh > 0) {
        if (d.standard == COMP_STD_NTSC && !transform)
            RETERROR("thresholds needs a transform separation for ntsc");
        if (level)
            RETERROR("thresholds needs the threshold mode (level=0)");
        if (dimensions == 2 && nthresh != COMP_T2D_NTHRESH)
            RETERROR("dimensions=2 needs 80 thresholds");
        if (dimensions == 3
            && nthresh != (d.standard == COMP_STD_PAL ? COMP_T3D_NTHRESH_PAL
                                                      : COMP_T3D_NTHRESH))
            RETERROR(d.standard == COMP_STD_PAL
                     ? "pal dimensions=3 needs 384 thresholds"
                     : "ntsc dimensions=3 needs 768 thresholds");
        if (dimensions == 1)
            RETERROR("thresholds needs dimensions 2 or 3");
        const double *tv = vsapi->mapGetFloatArray(in, "thresholds", &err);
        for (int i = 0; i < nthresh; i++)
            if (!(tv[i] > 0.0 && tv[i] <= 1.0))
                RETERROR("thresholds must be in (0, 1]");
        comp_decode_set_thresholds(d.dec, tv, nthresh);
    }

    if (nlut > 0) {
        const double *lv = vsapi->mapGetFloatArray(in, "lut", &err);
        for (int i = 0; i < nlut; i++)
            if (!(lv[i] >= 0.0 && lv[i] <= 1.0))
                RETERROR("lut values must be in [0, 1]");
        comp_decode_set_lut(d.dec, lv, nlut);
    } else if (builtin_lut) {
        if (d.standard == COMP_STD_PAL)
            comp_decode_set_lut(d.dec,
                                dimensions == 2 ? comp_lut_builtin_pal_2d
                                                : comp_lut_builtin_pal_3d,
                                (dimensions == 2 ? COMP_T2D_NTHRESH
                                                 : COMP_T3D_NTHRESH_PAL) * COMP_LUT_K);
        else
            comp_decode_set_lut(d.dec, comp_lut_builtin_ntsc,
                                COMP_T3D_NTHRESH * COMP_LUT_K);
    }

    vsapi->queryVideoFormat(&d.vi.format, cfYUV, stInteger, 16, 0, 0, core);
    d.in_frames = d.vi.numFrames;
    /* the resample back reads past the raster; the decode writes the
     * replicated columns it needs (width=0 returns the bare raster) */
    if (width != 0) {
        d.pad = COMP_EDGE_PAD;
        d.vi.width += 2 * COMP_EDGE_PAD;
    }

    comp_filter_t *data = malloc(sizeof(*data));
    *data = d;

    VSFilterDependency deps[] = {{ data->node, dimensions == 3 ? rpGeneral : rpStrictSpatial }};
    VSNode *dec_node = vsapi->createVideoFilter2(name, &data->vi, comp_decode_get_frame, comp_free,
                                                 fmParallel, deps, 1, data, core);
    if (!dec_node) {
        vsapi->mapSetError(out, "Decode: failed to create filter");
        return;
    }

    /* mask="motion": pull the attached motion mask out as its own clip
     * before the picture path consumes dec_node; both share one decode */
    VSNode *mask_node = NULL;
    if (data->mask_kind) {
        mask_node = comp_mask_from_props(&dec_node, core, vsapi);
        if (!mask_node) {
            vsapi->freeNode(dec_node);
            vsapi->mapSetError(out, "Decode: mask extraction failed");
            return;
        }
    }

    /* width=0: raw 4xfsc raster, no horizontal resample. The decode node
     * is already the raster picture; return it (and the raw mask) as-is. */
    if (width == 0) {
        vsapi->mapConsumeNode(out, "clip", dec_node, maReplace);
        if (mask_node)
            vsapi->mapConsumeNode(out, "clip", mask_node, maAppend);
        return;
    }

    /* resample back to the BT.601 raster: the exact inverse of Encode's
     * mapping. The 601 window is wider in time than the active raster,
     * so the outermost samples come from replicated edge padding. */
    double dst_left, dst_width;
    comp_decode_resample_params(d.standard, width, &dst_left, &dst_width);
    VSMap *args = vsapi->createMap();
    vsapi->mapConsumeNode(args, "clip", dec_node, maReplace);
    vsapi->mapSetInt(args, "width", width, maReplace);
    vsapi->mapSetInt(args, "height", d.vi.height, maReplace);
    vsapi->mapSetFloat(args, "src_left", dst_left, maReplace);
    vsapi->mapSetFloat(args, "src_width", dst_width, maReplace);
    VSMap *ret = vsapi->invoke(resize, "Spline36", args);
    vsapi->freeMap(args);

    const char *invoke_err = vsapi->mapGetError(ret);
    if (invoke_err) {
        char msg[256];
        snprintf(msg, sizeof(msg), "%s: %s", name, invoke_err);
        vsapi->mapSetError(out, msg);
        vsapi->freeMap(ret);
        return;
    }
    VSNode *res_node = vsapi->mapGetNode(ret, "clip", 0, NULL);
    vsapi->freeMap(ret);
    vsapi->mapConsumeNode(out, "clip", res_node, maReplace);

    /* second output: the motion mask, resampled to the output geometry on
     * the same crop as the picture (bilinear; see comp_resample_mask) */
    if (mask_node) {
        VSNode *rmask = comp_resample_mask(mask_node, core, vsapi, width,
                                           d.vi.height, dst_left, dst_width);
        if (!rmask) {
            vsapi->mapSetError(out, "Decode: mask resample failed");
            return;
        }
        vsapi->mapConsumeNode(out, "clip", rmask, maAppend);
    }
}


/* Restore(): the whole noise-reduction round trip in one filter, with
 * optional Y-only Landweber refinement anchored to the input picture
 * (which Decode alone cannot see). */
static void VS_CC comp_restore_create(const VSMap *in, VSMap *out, void *user_data, VSCore *core, const VSAPI *vsapi)
{
    const char *name = user_data;
    comp_filter_t d = {0};
    int err;

    d.node = vsapi->mapGetNode(in, "clip", 0, NULL);
    d.vi = *vsapi->getVideoInfo(d.node);
    const int src_width = d.vi.width;

    if (comp_parse_standard(in, vsapi, &d.standard))
        RETERROR("standard must be pal or ntsc");
    const int setup = !!vsapi->mapGetIntSaturated(in, "setup", 0, &err);

    int width = vsapi->mapGetIntSaturated(in, "width", 0, &err);
    if (err)
        width = src_width;
    /* width=0 requests the raw 4xfsc decode raster (SMPTE 244M / EBU 4fsc)
     * with no horizontal resample: the picture and mask come straight off
     * the decode grid, crisp, for callers who resize themselves. */
    if (width != 0 && (width < 16 || width > 8192))
        RETERROR("width must be 0 (raw raster) or between 16 and 8192");

    double threshold = vsapi->mapGetFloat(in, "threshold", 0, &err);
    const int threshold_unset = err;
    if (err)
        threshold = 0.4;
    if (!(threshold > 0.0 && threshold <= 1.0))
        RETERROR("threshold must be in (0, 1]");

    int dimensions = vsapi->mapGetIntSaturated(in, "dimensions", 0, &err);
    if (err)
        dimensions = 3;
    if (dimensions < 1 || dimensions > 3)
        RETERROR("dimensions must be 1, 2 or 3");

    int eq = vsapi->mapGetIntSaturated(in, "eq", 0, &err);
    const int eq_unset = err;
    if (err)
        eq = 1;
    if (eq < 0 || eq > 2)
        RETERROR("eq must be 0 (off), 1 (fixed) or 2 (leak-aware)");

    int refine = vsapi->mapGetIntSaturated(in, "refine", 0, &err);
    if (err)
        refine = 1;
    if (refine < 0 || refine > 16)
        RETERROR("refine must be between 0 and 16");

    const int precomb = !!vsapi->mapGetIntSaturated(in, "precomb", 0, &err);

    int transform = vsapi->mapGetIntSaturated(in, "transform", 0, &err);
    if (err)
        transform = (d.standard == COMP_STD_NTSC && dimensions == 3) ? 2 : 0;
    if (transform < 0 || transform > 2)
        RETERROR("transform must be 0 (comb), 1 (transform) or 2 (hybrid)");
    if (transform && d.standard != COMP_STD_NTSC)
        RETERROR("transform applies to ntsc (pal always uses the transform)");
    if (transform && dimensions != 3)
        RETERROR("transform needs dimensions=3");

    /* unset level/eq adapt to the chosen path: the measured best
     * settings wherever a transform separation is present */
    const int has_transform = d.standard == COMP_STD_PAL ? dimensions >= 2
                                                         : transform != 0;
    if (eq_unset && has_transform)
        eq = 2;
    if (eq == 2 && !has_transform)
        RETERROR("eq=2 needs a transform separation");

    /* an explicit threshold, thresholds, lut, or level selects its own
     * mode; in their absence a transform path defaults to the built-in
     * trained soft-gain table (set after the decoder is created) */
    int level = vsapi->mapGetIntSaturated(in, "level", 0, &err);
    const int builtin_lut = err && has_transform && threshold_unset
                            && vsapi->mapNumElements(in, "thresholds") < 1
                            && vsapi->mapNumElements(in, "lut") < 1;
    if (err)
        level = 0;
    if (level && dimensions < 2)
        RETERROR("level=1 needs dimensions 2 or 3");
    if (level && d.standard == COMP_STD_NTSC && !transform)
        RETERROR("level=1 needs a transform separation for ntsc");

    double evidence = vsapi->mapGetFloat(in, "evidence", 0, &err);
    if (err)
        evidence = 0.0;
    if (evidence < 0.0)
        RETERROR("evidence must be >= 0");
    if (evidence > 0.0 && (d.standard != COMP_STD_PAL || dimensions < 2))
        RETERROR("evidence needs a pal transform (dimensions 2 or 3)");

    const int cti = !!vsapi->mapGetIntSaturated(in, "cti", 0, &err);
    if (cti && dimensions < 2)
        RETERROR("cti needs dimensions 2 or 3");

    const int nlut = vsapi->mapNumElements(in, "lut");
    if (nlut > 0) {
        if (d.standard == COMP_STD_NTSC && !transform)
            RETERROR("lut needs a transform separation");
        if (level)
            RETERROR("lut and level are mutually exclusive");
        if (dimensions == 2 && nlut != COMP_T2D_NTHRESH * COMP_LUT_K)
            RETERROR("dimensions=2 needs 1280 lut values (80 bins x 16 knots)");
        if (dimensions == 3
            && nlut != (d.standard == COMP_STD_PAL ? COMP_T3D_NTHRESH_PAL
                                                   : COMP_T3D_NTHRESH) * COMP_LUT_K)
            RETERROR(d.standard == COMP_STD_PAL
                     ? "pal dimensions=3 needs 6144 lut values (384 bins x 16 knots)"
                     : "ntsc dimensions=3 needs 12288 lut values (768 bins x 16 knots)");
        if (dimensions == 1)
            RETERROR("lut needs dimensions 2 or 3");
        if (vsapi->mapNumElements(in, "thresholds") > 0)
            RETERROR("lut and thresholds are mutually exclusive");
    }

    if (!vsh_isConstantVideoFormat(&d.vi))
        RETERROR("clip must have constant format and dimensions");
    if (d.vi.format.colorFamily != cfYUV)
        RETERROR("clip must be YUV");
    if (d.standard == COMP_STD_PAL) {
        if (d.vi.height != COMP_ACTIVE_HEIGHT_PAL)
            RETERROR("pal input must have 576 lines");
    } else {
        if (d.vi.height != 480 && d.vi.height != COMP_ACTIVE_HEIGHT_NTSC)
            RETERROR("ntsc input must have 480 or 486 lines");
    }

    VSPlugin *resize = vsapi->getPluginByID("com.vapoursynth.resize", core);
    if (!resize)
        RETERROR("resize plugin not found");

    /* mask=: emit a per-sample mask as a second output clip. "motion" is
     * the NTSC hybrid router; "confidence" is the eq=2 separation
     * confidence (any transform path). */
    const char *maskmode = vsapi->mapGetData(in, "mask", 0, &err);
    if (!err && maskmode) {
        if (!strcmp(maskmode, "motion")) {
            if (!(d.standard == COMP_STD_NTSC && dimensions == 3 && transform == 2))
                RETERROR("mask=\"motion\" needs the ntsc hybrid path "
                         "(dimensions=3, transform=2)");
            d.mask_kind = COMP_MASK_MOTION;
        } else if (!strcmp(maskmode, "confidence")) {
            if (eq != 2)
                RETERROR("mask=\"confidence\" needs eq=2 (a transform separation)");
            d.mask_kind = COMP_MASK_CONFIDENCE;
        } else {
            RETERROR("mask must be \"motion\" or \"confidence\"");
        }
    }

    /* stage 1: the picture on the 4xfsc raster (also the refine anchor) */
    int rwidth;
    double fwd_left, fwd_width;
    comp_encode_resample_params(d.standard, src_width, &rwidth, &fwd_left, &fwd_width);

    VSNode *source = vsapi->addNodeRef(d.node);
    VSMap *args = vsapi->createMap();
    vsapi->mapConsumeNode(args, "clip", d.node, maReplace);
    d.node = NULL;
    vsapi->mapSetInt(args, "format", pfYUV444P16, maReplace);
    vsapi->mapSetInt(args, "width", rwidth, maReplace);
    vsapi->mapSetInt(args, "height", d.vi.height, maReplace);
    vsapi->mapSetFloat(args, "src_left", fwd_left, maReplace);
    vsapi->mapSetFloat(args, "src_width", fwd_width, maReplace);
    VSMap *ret = vsapi->invoke(resize, "Spline36", args);
    vsapi->freeMap(args);

    const char *invoke_err = vsapi->mapGetError(ret);
    if (invoke_err) {
        char msg[256];
        snprintf(msg, sizeof(msg), "%s: %s", name, invoke_err);
        vsapi->mapSetError(out, msg);
        vsapi->freeMap(ret);
        return;
    }
    VSNode *raster = vsapi->mapGetNode(ret, "clip", 0, NULL);
    vsapi->freeMap(ret);

    /* stage 2: modulator over the raster picture */
    comp_filter_t *edata = malloc(sizeof(*edata));
    memset(edata, 0, sizeof(*edata));
    edata->node = raster;
    edata->vi = *vsapi->getVideoInfo(raster);
    edata->standard = d.standard;
    comp_encode_init(&edata->enc, d.standard, setup, precomb);
    vsapi->queryVideoFormat(&edata->vi.format, cfGray, stInteger, 16, 0, 0, core);

    VSFilterDependency edeps[] = {{ raster, rpStrictSpatial }};
    VSNode *modulated = vsapi->createVideoFilter2(name, &edata->vi, comp_encode_get_frame, comp_free,
                                                  fmParallel, edeps, 1, edata, core);
    if (!modulated) {
        vsapi->mapSetError(out, "Restore: failed to create modulator");
        return;
    }

    /* stage 3: decoder with the raster picture as the refine anchor */
    d.node = modulated;
    d.orig_node = vsapi->addNodeRef(raster);
    d.vi = *vsapi->getVideoInfo(modulated);

    VSCoreInfo info;
    vsapi->getCoreInfo(core, &info);
    d.dec = malloc(sizeof(*d.dec));
    if (!d.dec)
        RETERROR("out of memory");
    if (comp_decode_init(d.dec, d.standard, threshold,
                         info.numThreads < 1 ? 1 : info.numThreads, setup,
                         dimensions, eq, refine, transform, level,
                         evidence, cti)) {
        free(d.dec);
        d.dec = NULL;
        RETERROR("decoder initialisation failed");
    }

    const int nthresh = vsapi->mapNumElements(in, "thresholds");
    if (nthresh > 0) {
        if (d.standard == COMP_STD_NTSC && !transform)
            RETERROR("thresholds needs a transform separation for ntsc");
        if (level)
            RETERROR("thresholds needs the threshold mode (level=0)");
        if (dimensions == 2 && nthresh != COMP_T2D_NTHRESH)
            RETERROR("dimensions=2 needs 80 thresholds");
        if (dimensions == 3
            && nthresh != (d.standard == COMP_STD_PAL ? COMP_T3D_NTHRESH_PAL
                                                      : COMP_T3D_NTHRESH))
            RETERROR(d.standard == COMP_STD_PAL
                     ? "pal dimensions=3 needs 384 thresholds"
                     : "ntsc dimensions=3 needs 768 thresholds");
        if (dimensions == 1)
            RETERROR("thresholds needs dimensions 2 or 3");
        const double *tv = vsapi->mapGetFloatArray(in, "thresholds", &err);
        for (int i = 0; i < nthresh; i++)
            if (!(tv[i] > 0.0 && tv[i] <= 1.0))
                RETERROR("thresholds must be in (0, 1]");
        comp_decode_set_thresholds(d.dec, tv, nthresh);
    }

    if (nlut > 0) {
        const double *lv = vsapi->mapGetFloatArray(in, "lut", &err);
        for (int i = 0; i < nlut; i++)
            if (!(lv[i] >= 0.0 && lv[i] <= 1.0))
                RETERROR("lut values must be in [0, 1]");
        comp_decode_set_lut(d.dec, lv, nlut);
    } else if (builtin_lut) {
        if (d.standard == COMP_STD_PAL)
            comp_decode_set_lut(d.dec,
                                dimensions == 2 ? comp_lut_builtin_pal_2d
                                                : comp_lut_builtin_pal_3d,
                                (dimensions == 2 ? COMP_T2D_NTHRESH
                                                 : COMP_T3D_NTHRESH_PAL) * COMP_LUT_K);
        else
            comp_decode_set_lut(d.dec, comp_lut_builtin_ntsc,
                                COMP_T3D_NTHRESH * COMP_LUT_K);
    }

    vsapi->queryVideoFormat(&d.vi.format, cfYUV, stInteger, 16, 0, 0, core);
    d.in_frames = d.vi.numFrames;
    /* the resample back reads past the raster; the decode writes the
     * replicated columns it needs (width=0 returns the bare raster) */
    if (width != 0) {
        d.pad = COMP_EDGE_PAD;
        d.vi.width += 2 * COMP_EDGE_PAD;
    }

    comp_filter_t *data = malloc(sizeof(*data));
    *data = d;

    VSFilterDependency ddeps[] = {{ data->node, dimensions == 3 ? rpGeneral : rpStrictSpatial },
                                  { data->orig_node, rpStrictSpatial }};
    VSNode *dec_node = vsapi->createVideoFilter2(name, &data->vi, comp_decode_get_frame, comp_free,
                                                 fmParallel, ddeps, 2, data, core);
    if (!dec_node) {
        vsapi->mapSetError(out, "Restore: failed to create filter");
        return;
    }

    /* mask="motion": pull the attached raster motion mask out as its own
     * clip (the picture path below uses dec_node unchanged; both share the
     * one decode). It gets its own bilinear resample on the SAME crop
     * geometry and is appended as a second output clip. */
    VSNode *mask_node = NULL;
    if (data->mask_kind) {
        mask_node = comp_mask_from_props(&dec_node, core, vsapi);
        if (!mask_node) {
            vsapi->freeNode(dec_node);
            vsapi->mapSetError(out, "Restore: mask extraction failed");
            return;
        }
    }

    /* width=0: raw 4xfsc raster, no resample and no edge splice (at raster
     * width nothing falls outside, so there are no un-modeled columns). The
     * scalar difficulty props ride natively on the decode frame. */
    if (width == 0) {
        vsapi->mapConsumeNode(out, "clip", dec_node, maReplace);
        if (mask_node)
            vsapi->mapConsumeNode(out, "clip", mask_node, maAppend);
        return;
    }

    /* the decode node carries the separation-confidence props; the edge
     * splice below (std.StackHorizontal) would inherit props from the
     * propless source-edge clip instead, so keep a reference to copy them
     * back onto the spliced output */
    VSNode *conf_src = vsapi->addNodeRef(dec_node);

    /* stage 4: back to the caller's raster */
    double dst_left, dst_width;
    comp_decode_resample_params(d.standard, width, &dst_left, &dst_width);
    args = vsapi->createMap();
    vsapi->mapConsumeNode(args, "clip", dec_node, maReplace);
    vsapi->mapSetInt(args, "width", width, maReplace);
    vsapi->mapSetInt(args, "height", d.vi.height, maReplace);
    vsapi->mapSetFloat(args, "src_left", dst_left, maReplace);
    vsapi->mapSetFloat(args, "src_width", dst_width, maReplace);
    ret = vsapi->invoke(resize, "Spline36", args);
    vsapi->freeMap(args);

    invoke_err = vsapi->mapGetError(ret);
    if (invoke_err) {
        char msg[256];
        snprintf(msg, sizeof(msg), "%s: %s", name, invoke_err);
        vsapi->mapSetError(out, msg);
        vsapi->freeMap(ret);
        vsapi->freeNode(source);
        return;
    }
    VSNode *res_node = vsapi->mapGetNode(ret, "clip", 0, NULL);
    vsapi->freeMap(ret);

    /* the outermost output columns sample beyond the active raster and
     * cannot be reconstructed from it — they never rode the modeled
     * channel. Splice them through from the source instead. */
    int nl, nr;
    comp_decode_edge_columns(d.standard, width, rwidth, &nl, &nr);
    if (nl > 0 || nr > 0) {
        VSPlugin *std = vsapi->getPluginByID("com.vapoursynth.std", core);
        args = vsapi->createMap();
        vsapi->mapConsumeNode(args, "clip", source, maReplace);
        source = NULL;
        vsapi->mapSetInt(args, "format", pfYUV444P16, maReplace);
        vsapi->mapSetInt(args, "width", width, maReplace);
        vsapi->mapSetInt(args, "height", d.vi.height, maReplace);
        ret = vsapi->invoke(resize, "Spline36", args);
        vsapi->freeMap(args);
        if (vsapi->mapGetError(ret)) {
            vsapi->mapSetError(out, "Restore: source edge conversion failed");
            vsapi->freeMap(ret);
            vsapi->freeNode(res_node);
            return;
        }
        VSNode *orig = vsapi->mapGetNode(ret, "clip", 0, NULL);
        vsapi->freeMap(ret);

        VSNode *parts[3] = { NULL, NULL, NULL };
        int nparts = 0;
        const struct { VSNode *from; int left, right; } cuts[3] = {
            { orig, 0, width - nl },
            { res_node, nl, nr },
            { orig, width - nr, 0 },
        };
        int fail = 0;
        for (int i = 0; i < 3; i++) {
            if ((i == 0 && nl == 0) || (i == 2 && nr == 0))
                continue;
            args = vsapi->createMap();
            vsapi->mapSetNode(args, "clip", cuts[i].from, maReplace);
            if (cuts[i].left)
                vsapi->mapSetInt(args, "left", cuts[i].left, maReplace);
            if (cuts[i].right)
                vsapi->mapSetInt(args, "right", cuts[i].right, maReplace);
            ret = vsapi->invoke(std, "Crop", args);
            vsapi->freeMap(args);
            if (vsapi->mapGetError(ret)) {
                vsapi->freeMap(ret);
                fail = 1;
                break;
            }
            parts[nparts++] = vsapi->mapGetNode(ret, "clip", 0, NULL);
            vsapi->freeMap(ret);
        }
        vsapi->freeNode(orig);
        vsapi->freeNode(res_node);
        res_node = NULL;
        if (!fail) {
            args = vsapi->createMap();
            for (int i = 0; i < nparts; i++)
                vsapi->mapConsumeNode(args, "clips", parts[i], maAppend);
            ret = vsapi->invoke(std, "StackHorizontal", args);
            vsapi->freeMap(args);
            if (vsapi->mapGetError(ret))
                fail = 1;
            else
                res_node = vsapi->mapGetNode(ret, "clip", 0, NULL);
            vsapi->freeMap(ret);
        } else {
            for (int i = 0; i < nparts; i++)
                vsapi->freeNode(parts[i]);
        }
        if (fail || !res_node) {
            vsapi->freeNode(conf_src);
            vsapi->mapSetError(out, "Restore: edge splice failed");
            return;
        }
        /* restore the confidence props the stack dropped (copies only the
         * two named props, leaving _Matrix/_ChromaLocation/etc. intact;
         * a no-op on non-eq2 output where the source has no such props) */
        args = vsapi->createMap();
        vsapi->mapConsumeNode(args, "clip", res_node, maReplace);
        vsapi->mapConsumeNode(args, "prop_src", conf_src, maReplace);
        conf_src = NULL;
        vsapi->mapSetData(args, "props", "CompositeSeparationConfidenceMean", -1,
                          dtUtf8, maReplace);
        vsapi->mapSetData(args, "props", "CompositeSeparationConfidenceStdDev", -1,
                          dtUtf8, maAppend);
        vsapi->mapSetData(args, "props", "CompositeMotionFraction", -1, dtUtf8, maAppend);
        vsapi->mapSetData(args, "props", "CompositeRefineResidual", -1, dtUtf8, maAppend);
        vsapi->mapSetData(args, "props", "CompositeRefineCorrection", -1, dtUtf8, maAppend);
        ret = vsapi->invoke(std, "CopyFrameProps", args);
        vsapi->freeMap(args);
        if (vsapi->mapGetError(ret)) {
            vsapi->mapSetError(out, "Restore: confidence prop copy failed");
            vsapi->freeMap(ret);
            return;
        }
        res_node = vsapi->mapGetNode(ret, "clip", 0, NULL);
        vsapi->freeMap(ret);
    } else {
        vsapi->freeNode(source);
        vsapi->freeNode(conf_src);
    }
    vsapi->mapConsumeNode(out, "clip", res_node, maReplace);

    /* second output: the motion mask, resampled to the output geometry on
     * the same crop as the picture (bilinear; see comp_resample_mask) */
    if (mask_node) {
        VSNode *rmask = comp_resample_mask(mask_node, core, vsapi, width,
                                           d.vi.height, dst_left, dst_width);
        if (!rmask) {
            vsapi->mapSetError(out, "Restore: mask resample failed");
            return;
        }
        vsapi->mapConsumeNode(out, "clip", rmask, maAppend);
    }
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi)
{
    vspapi->configPlugin("com.ifb.composite", "composite",
                         "PAL/NTSC composite video encoder/decoder",
                         VS_MAKE_VERSION(0, 2), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("Encode",
                             "clip:vnode;"
                             "standard:data:opt;"
                             "setup:int:opt;"
                             "precomb:int:opt;",
                             "clip:vnode;",
                             comp_encode_create, (void *)"Encode", plugin);
    vspapi->registerFunction("Decode",
                             "clip:vnode;"
                             "standard:data:opt;"
                             "width:int:opt;"
                             "threshold:float:opt;"
                             "setup:int:opt;"
                             "dimensions:int:opt;"
                             "eq:int:opt;"
                             "thresholds:float[]:opt;"
                             "transform:int:opt;"
                             "level:int:opt;"
                             "lut:float[]:opt;"
                             "evidence:float:opt;"
                             "cti:int:opt;"
                             "mask:data:opt;",
                             "clip:vnode[];",
                             comp_decode_create, (void *)"Decode", plugin);
    vspapi->registerFunction("Restore",
                             "clip:vnode;"
                             "standard:data:opt;"
                             "width:int:opt;"
                             "threshold:float:opt;"
                             "setup:int:opt;"
                             "dimensions:int:opt;"
                             "eq:int:opt;"
                             "refine:int:opt;"
                             "thresholds:float[]:opt;"
                             "precomb:int:opt;"
                             "transform:int:opt;"
                             "level:int:opt;"
                             "lut:float[]:opt;"
                             "evidence:float:opt;"
                             "cti:int:opt;"
                             "mask:data:opt;",
                             "clip:vnode[];",
                             comp_restore_create, (void *)"Restore", plugin);
}
