/*
 * PAL/NTSC composite video encoder/decoder for AviSynth+.
 *
 * Encode() modulates YCbCr to composite (GRAY16); Decode() demodulates it
 * back to YCbCr. They mirror the VapourSynth frontend (composite.c),
 * resampling to/from the 4xfsc active raster through zimg (avsresize's
 * z_ConvertFormat, the same resampler VapourSynth uses) and sharing the
 * geometry with the VS plugin via geom.c so both produce the same raster.
 *
 * This is the AviSynth+ C-interface half of a dual-host module: the same
 * binary also exports VapourSynthPluginInit2 (composite.c), and each host
 * looks up only the entry point it knows. The avs_* API resolves by
 * ordinary dynamic linking against the host.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avisynth_c.h"

#include "decode.h"
#include "encode.h"
#include "geom.h"
#include "lut_tables.h"
#include "subcarrier.h"

/* the widest temporal look (NTSC 3D); bounds the per-frame view arrays */
#define COMP_AVS_MAX_LOOK COMP_T3D_LOOK

typedef struct comp_avs_t comp_avs_t;

struct comp_avs_t {
    comp_encode_t enc;
    int standard;
    int height;   /* raster height of the resampled 4:4:4 clip */
};

typedef struct comp_avs_dec_t comp_avs_dec_t;

struct comp_avs_dec_t {
    comp_decode_t dec;
    int standard;
    int height;      /* raster height of the composite input */
    int in_frames;   /* source clip length, for edge clamping */
    int mask_kind;   /* COMP_MASK_* : emit that mask as the YUVA alpha (0 none) */
    AVS_Clip *anchor; /* Restore: the raster picture for refine, else NULL */
};

/* Where the picture sits in the raster, matching composite.c: a 480-line
 * NTSC clip occupies the 486-line raster at rows 4..483 (BFF, the DV
 * convention) or 5..484 (TFF). Field order comes from the frame's
 * _FieldBased prop (2 = TFF, as in VapourSynth); absent or progressive
 * means BFF. */
static int comp_avs_row_offset(int standard, int height,
                               AVS_ScriptEnvironment *env,
                               const AVS_VideoFrame *frame)
{
    if (standard != COMP_STD_NTSC || height != 480)
        return 0;

    const AVS_Map *props = avs_get_frame_props_ro(env, frame);
    int err = 0;
    const int64_t fb = avs_prop_get_int(env, props, "_FieldBased", 0, &err);
    return (!err && fb == 2) ? 5 : 4; /* 2 = TFF, matching VapourSynth */
}

static AVS_VideoFrame *AVSC_CC comp_avs_get_frame(AVS_FilterInfo *fi, int n)
{
    comp_avs_t *f = fi->user_data;
    AVS_ScriptEnvironment *env = fi->env;

    AVS_VideoFrame *src = avs_get_frame(fi->child, n);
    if (!src)
        return NULL;

    AVS_VideoFrame *dst = avs_new_video_frame_p(env, &fi->vi, src);

    const uint16_t *srcy = (const uint16_t *)avs_get_read_ptr_p(src, AVS_PLANAR_Y);
    const uint16_t *srcu = (const uint16_t *)avs_get_read_ptr_p(src, AVS_PLANAR_U);
    const uint16_t *srcv = (const uint16_t *)avs_get_read_ptr_p(src, AVS_PLANAR_V);
    const ptrdiff_t ystride = avs_get_pitch_p(src, AVS_PLANAR_Y) / 2;
    const ptrdiff_t ustride = avs_get_pitch_p(src, AVS_PLANAR_U) / 2;
    const ptrdiff_t vstride = avs_get_pitch_p(src, AVS_PLANAR_V) / 2;
    uint16_t *dstp = (uint16_t *)avs_get_write_ptr_p(dst, AVS_PLANAR_Y);
    const ptrdiff_t dstride = avs_get_pitch_p(dst, AVS_PLANAR_Y) / 2;

    comp_encode_frame(&f->enc, n, f->height,
                      comp_avs_row_offset(f->standard, f->height, env, src),
                      dstp, dstride, srcy, ystride, srcu, ustride, srcv, vstride);

    avs_release_video_frame(src);
    return dst;
}

static int AVSC_CC comp_avs_set_cache_hints(AVS_FilterInfo *fi, int cachehints,
                                            int frame_range)
{
    (void)fi; (void)frame_range;
    /* the encoder state is read-only after init and the frame callback
     * writes only its own output frame, so one shared instance is
     * reentrant across the host's thread pool */
    return cachehints == AVS_CACHE_GET_MTMODE ? AVS_MT_NICE_FILTER : 0;
}

static void AVSC_CC comp_avs_free(AVS_FilterInfo *fi)
{
    free(fi->user_data);
}

static AVS_VideoFrame *AVSC_CC comp_avs_dec_get_frame(AVS_FilterInfo *fi, int n)
{
    comp_avs_dec_t *f = fi->user_data;
    AVS_ScriptEnvironment *env = fi->env;
    const int look = comp_decode_look(&f->dec);
    const int last = f->in_frames - 1;

    /* fetch the 2*look+1 composite frames centered on n, edges clamped */
    AVS_VideoFrame *srcs[2 * COMP_AVS_MAX_LOOK + 1];
    comp_frame_view_t views[2 * COMP_AVS_MAX_LOOK + 1];
    int view_frames[2 * COMP_AVS_MAX_LOOK + 1];
    for (int i = 0; i <= 2 * look; i++) {
        int k = n - look + i;
        k = k < 0 ? 0 : (k > last ? last : k);
        srcs[i] = avs_get_frame(fi->child, k);
        if (!srcs[i]) {
            for (int j = 0; j < i; j++)
                avs_release_video_frame(srcs[j]);
            return NULL;
        }
        views[i].data = (const uint16_t *)avs_get_read_ptr_p(srcs[i], AVS_PLANAR_Y);
        views[i].stride = avs_get_pitch_p(srcs[i], AVS_PLANAR_Y) / 2;
        view_frames[i] = k;
    }

    AVS_VideoFrame *src = srcs[look];
    AVS_VideoFrame *dst = avs_new_video_frame_p(env, &fi->vi, src);

    /* Restore: the raster anchor picture for the refine loop. It is
     * spatial (frame n only, always in range), unlike the composite
     * window. */
    AVS_VideoFrame *anchor = f->anchor ? avs_get_frame(f->anchor, n) : NULL;

    /* mask=: dst is YUVA (see the create function); the requested mask
     * goes into the alpha plane at the raster. Pre-zero so blanking/edge
     * rows read still / fully-confident (0). The create function
     * resamples it out. */
    uint16_t *maskp = NULL;
    ptrdiff_t mask_stride = 0;
    if (f->mask_kind) {
        maskp = (uint16_t *)avs_get_write_ptr_p(dst, AVS_PLANAR_A);
        mask_stride = avs_get_pitch_p(dst, AVS_PLANAR_A) / 2;
        const int aw = avs_get_row_size_p(dst, AVS_PLANAR_A) / 2;
        const int ah = avs_get_height_p(dst, AVS_PLANAR_A);
        for (int r = 0; r < ah; r++)
            memset(maskp + r * mask_stride, 0, sizeof(uint16_t) * aw);
    }

    comp_decode_metrics_t mt = COMP_METRICS_INIT;
    comp_decode_frame(&f->dec, n, f->in_frames, f->height,
                      comp_avs_row_offset(f->standard, f->height, env, src),
                      views, view_frames, look,
                      anchor ? (const uint16_t *)avs_get_read_ptr_p(anchor, AVS_PLANAR_Y) : NULL,
                      anchor ? avs_get_pitch_p(anchor, AVS_PLANAR_Y) / 2 : 0,
                      (uint16_t *)avs_get_write_ptr_p(dst, AVS_PLANAR_Y),
                      avs_get_pitch_p(dst, AVS_PLANAR_Y) / 2,
                      (uint16_t *)avs_get_write_ptr_p(dst, AVS_PLANAR_U),
                      avs_get_pitch_p(dst, AVS_PLANAR_U) / 2,
                      (uint16_t *)avs_get_write_ptr_p(dst, AVS_PLANAR_V),
                      avs_get_pitch_p(dst, AVS_PLANAR_V) / 2,
                      &mt, maskp, mask_stride, f->mask_kind);

    /* each difficulty prop exists only where its path is active. The
     * resample and the Restore edge-splice drop these downstream; the
     * create functions copy them back from this pre-resample clip
     * (comp_avs_copy_conf_props). */
    AVS_Map *props = avs_get_frame_props_rw(env, dst);
    if (mt.conf_mean >= 0.0) {
        avs_prop_set_float(env, props, "CompositeSeparationConfidenceMean", mt.conf_mean, 0);
        avs_prop_set_float(env, props, "CompositeSeparationConfidenceStdDev", mt.conf_std, 0);
    }
    if (mt.motion_fraction >= 0.0)
        avs_prop_set_float(env, props, "CompositeMotionFraction", mt.motion_fraction, 0);
    if (mt.refine_residual >= 0.0)
        avs_prop_set_float(env, props, "CompositeRefineResidual", mt.refine_residual, 0);
    if (mt.refine_correction >= 0.0)
        avs_prop_set_float(env, props, "CompositeRefineCorrection", mt.refine_correction, 0);

    if (anchor)
        avs_release_video_frame(anchor);
    for (int i = 0; i <= 2 * look; i++)
        avs_release_video_frame(srcs[i]);
    return dst;
}

static int AVSC_CC comp_avs_dec_set_cache_hints(AVS_FilterInfo *fi, int cachehints,
                                                int frame_range)
{
    (void)fi; (void)frame_range;
    /* the scratch pool (mutex/cond guarded, blocking) is the reentrancy
     * mechanism, so one shared decoder instance is thread-safe */
    return cachehints == AVS_CACHE_GET_MTMODE ? AVS_MT_NICE_FILTER : 0;
}

static void AVSC_CC comp_avs_dec_free(AVS_FilterInfo *fi)
{
    comp_avs_dec_t *f = fi->user_data;
    comp_decode_free(&f->dec);
    if (f->anchor)
        avs_release_clip(f->anchor);
    free(f);
}

/* Whether avsresize will resample `clip` field-wise. It does so only for
 * subsampled YUV with _FieldBased set; anything else (the 4:4:4 raster on
 * the way back, the Y16 mask) is resampled as a whole frame even when the
 * flag rides along on the frame props. avsresize reads _FieldBased per
 * frame, but src_height is fixed at construction, so a clip whose
 * field-ness changes mid-stream follows whatever frame 0 said. Does not
 * consume `clip`. */
static int comp_avs_clip_fieldwise(AVS_ScriptEnvironment *env, AVS_Value clip)
{
    AVS_Clip *c = avs_take_clip(clip, env);
    if (!c)
        return 0;
    const AVS_VideoInfo *vi = avs_get_video_info(c);
    if (!avs_is_yuv(vi) || avs_is_y(vi) ||
        (avs_get_plane_width_subsampling(vi, AVS_PLANAR_U) == 0 &&
         avs_get_plane_height_subsampling(vi, AVS_PLANAR_U) == 0)) {
        avs_release_clip(c);
        return 0;
    }
    AVS_VideoFrame *f = avs_get_frame(c, 0);
    int interlaced = 0;
    if (f) {
        int err = 0;
        const int64_t fb = avs_prop_get_int(env, avs_get_frame_props_ro(env, f),
                                            "_FieldBased", 0, &err);
        interlaced = !err && (fb == 1 || fb == 2);
        avs_release_video_frame(f);
    }
    avs_release_clip(c);
    return interlaced;
}

/* Whether this avsresize lets src_left/src_width overwrite the active
 * height of a field-wise resample with the full frame height. Releases from
 * r27 on halve the active region themselves, so pinning src_height to the
 * field there would halve it twice. avsresize has no version query, so ask
 * it directly: crop a black-over-white interlaced 4:2:0 clip horizontally
 * only, and a folding build mirrors the black back into the bottom row. */
static int comp_avs_fieldwise_crop_folds(AVS_ScriptEnvironment *env)
{
    enum { W = 16, H = 16 };
    AVS_Value half[2];
    for (int i = 0; i < 2; i++) {
        AVS_Value a[5] = { avs_new_value_int(1), avs_new_value_int(W),
                           avs_new_value_int(H / 2),
                           avs_new_value_string("YV12"),
                           avs_new_value_int(i ? 0xEB8080 : 0x108080) };
        const char *an[5] = { "length", "width", "height", "pixel_type",
                              "color_yuv" };
        half[i] = avs_invoke(env, "BlankClip", avs_new_value_array(a, 5), an);
        if (avs_is_error(half[i])) {
            if (i)
                avs_release_value(half[0]);
            return 0;
        }
    }
    AVS_Value stacked = avs_invoke(env, "StackVertical",
                                   avs_new_value_array(half, 2), NULL);
    avs_release_value(half[0]);
    avs_release_value(half[1]);
    if (avs_is_error(stacked))
        return 0;

    AVS_Value pa[3] = { stacked, avs_new_value_string("_FieldBased"),
                        avs_new_value_int(2) };
    AVS_Value tff = avs_invoke(env, "propSet", avs_new_value_array(pa, 3), NULL);
    avs_release_value(stacked);
    if (avs_is_error(tff))
        return 0;

    AVS_Value za[6] = { tff, avs_new_value_int(W), avs_new_value_int(H),
                        avs_new_value_string("YV24"), avs_new_value_float(0.0f),
                        avs_new_value_float((float)W) };
    const char *zn[6] = { NULL, "width", "height", "pixel_type", "src_left",
                          "src_width" };
    AVS_Value out = avs_invoke(env, "z_ConvertFormat", avs_new_value_array(za, 6), zn);
    avs_release_value(tff);
    if (avs_is_error(out))
        return 0;

    int folds = 0;
    AVS_Clip *c = avs_take_clip(out, env);
    AVS_VideoFrame *f = avs_get_frame(c, 0);
    if (f) {
        const BYTE *p = avs_get_read_ptr_p(f, AVS_PLANAR_Y);
        folds = p[(H - 1) * avs_get_pitch_p(f, AVS_PLANAR_Y)] < 128;
        avs_release_video_frame(f);
    }
    avs_release_clip(c);
    avs_release_value(out);
    return folds;
}

/* z_ConvertFormat(clip, width=, height=, pixel_type=, src_left=,
 * src_width=): one zimg graph does the format conversion and the subpixel
 * crop-resize, the same fused operation the VS plugin gets from
 * resize.Spline36 (same resampler, chroma placement, and parameters).
 * A NaN src_left/src_width omits the subpixel crop (a plain resize, as
 * the Restore edge-splice needs for the source). `clip` is a non-owning
 * element copy; avs_invoke does not consume it. Returns an error
 * AVS_Value if avsresize is not installed.
 *
 * Interlaced sources resample field-wise, which keeps 4:2:0 chroma
 * upsampling from crossing field lines. avsresize before r27 halves the
 * format height for that but then lets src_left/src_width overwrite the
 * active height back to the full frame, so zimg reads a 480-row window out
 * of a 240-row field and mirror-extends the edge, folding the picture about
 * its midline. src_height is assigned from the same statement, so on those
 * builds setting it pins the active height to the field. */
static AVS_Value comp_avs_zresize_f(AVS_ScriptEnvironment *env, AVS_Value clip,
                                    const char *who, int width, int height,
                                    const char *pixel_type,
                                    double src_left, double src_width,
                                    const char *resample_filter)
{
    if (!avs_function_exists(env, "z_ConvertFormat")) {
        /* the host does not copy the string, so hand it a saved one:
         * a local buffer would be dead by the time it is read */
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "%s: avsresize (z_ConvertFormat) is required but not loaded", who);
        return avs_new_value_error(avs_save_string(env, msg, -1));
    }

    const int crop = src_left == src_left && src_width == src_width; /* !NaN */
    AVS_Value args[8];
    const char *names[8];
    int n = 4;
    args[0] = clip;                              names[0] = NULL;
    args[1] = avs_new_value_int(width);          names[1] = "width";
    args[2] = avs_new_value_int(height);         names[2] = "height";
    args[3] = avs_new_value_string(pixel_type);  names[3] = "pixel_type";
    if (crop) {
        args[n] = avs_new_value_float((float)src_left);  names[n] = "src_left"; n++;
        args[n] = avs_new_value_float((float)src_width); names[n] = "src_width"; n++;
        /* every resample here is horizontal-only, so `height` is also the
         * source height. Halving it on a clip resampled as a whole frame
         * would resample just its top half. */
        if (comp_avs_clip_fieldwise(env, clip) &&
            comp_avs_fieldwise_crop_folds(env)) {
            args[n] = avs_new_value_float((float)(height / 2.0));
            names[n] = "src_height"; n++;
        }
    }
    if (resample_filter) {
        args[n] = avs_new_value_string(resample_filter); names[n] = "resample_filter"; n++;
    }

    AVS_Value argv = avs_new_value_array(args, n);
    return avs_invoke(env, "z_ConvertFormat", argv, (const char **)names);
}

/* the picture resample: zimg's default kernel (matches the VS Spline36 path) */
static AVS_Value comp_avs_zresize(AVS_ScriptEnvironment *env, AVS_Value clip,
                                  const char *who, int width, int height,
                                  const char *pixel_type,
                                  double src_left, double src_width)
{
    return comp_avs_zresize_f(env, clip, who, width, height, pixel_type,
                              src_left, src_width, NULL);
}

/* Forward resample to the 4xfsc active raster and YUV444P16. */
static AVS_Value comp_avs_resample(AVS_ScriptEnvironment *env, AVS_Value clip,
                                   int in_width, int in_height, int standard)
{
    int width;
    double src_left, src_width;
    comp_encode_resample_params(standard, in_width, &width, &src_left, &src_width);
    return comp_avs_zresize(env, clip, "Encode", width, in_height,
                            "YUV444P16", src_left, src_width);
}

/* invoke a named built-in with a positional-only argument array */
static AVS_Value comp_avs_invoke_pos(AVS_ScriptEnvironment *env, const char *fn,
                                     AVS_Value *a, int n)
{
    return avs_invoke(env, fn, avs_new_value_array(a, n), NULL);
}

/* Horizontally pad `clip` with COMP_EDGE_PAD replicated edge columns each
 * side, matching composite.c's comp_pad_h: the inverse resample overreads
 * the active raster, and replicating the edge (rather than zimg's mirror)
 * keeps interior picture off the frame edges. Does not consume `clip`. */
static AVS_Value comp_avs_pad_h(AVS_ScriptEnvironment *env, AVS_Value clip,
                                const char *who)
{
    AVS_Clip *c = avs_take_clip(clip, env);
    const int w = avs_get_video_info(c)->width;
    const int h = avs_get_video_info(c)->height;
    avs_release_clip(c);

    AVS_Value cols[3];
    /* left pad: leftmost column, then PointResize to COMP_EDGE_PAD wide */
    {
        AVS_Value crop[5] = { clip, avs_new_value_int(0), avs_new_value_int(0),
                              avs_new_value_int(1), avs_new_value_int(0) };
        AVS_Value col = comp_avs_invoke_pos(env, "Crop", crop, 5);
        if (avs_is_error(col)) return col;
        AVS_Value rs[3] = { col, avs_new_value_int(COMP_EDGE_PAD),
                            avs_new_value_int(h) };
        cols[0] = comp_avs_invoke_pos(env, "PointResize", rs, 3);
        avs_release_value(col);
        if (avs_is_error(cols[0])) return cols[0];
    }
    cols[1] = clip;
    /* right pad: rightmost column, likewise */
    {
        AVS_Value crop[5] = { clip, avs_new_value_int(w - 1), avs_new_value_int(0),
                              avs_new_value_int(1), avs_new_value_int(0) };
        AVS_Value col = comp_avs_invoke_pos(env, "Crop", crop, 5);
        if (avs_is_error(col)) { avs_release_value(cols[0]); return col; }
        AVS_Value rs[3] = { col, avs_new_value_int(COMP_EDGE_PAD),
                            avs_new_value_int(h) };
        cols[2] = comp_avs_invoke_pos(env, "PointResize", rs, 3);
        avs_release_value(col);
        if (avs_is_error(cols[2])) { avs_release_value(cols[0]); return cols[2]; }
    }

    AVS_Value ret = comp_avs_invoke_pos(env, "StackHorizontal", cols, 3);
    avs_release_value(cols[0]);
    avs_release_value(cols[2]);
    (void)who;
    return ret;
}

/* Resample a raster YUV444P16 clip back to the caller's `out_width` x
 * `height` BT.601 raster: replicate-pad, then the inverse zimg crop. The
 * pad's COMP_EDGE_PAD offset is already folded into src_left by
 * comp_decode_resample_params. Does not consume `clip`. */
static AVS_Value comp_avs_resample_back(AVS_ScriptEnvironment *env, AVS_Value clip,
                                        const char *who, int standard,
                                        int out_width, int height, int want_mask)
{
    double dst_left, dst_width;
    comp_decode_resample_params(standard, out_width, &dst_left, &dst_width);

    if (!want_mask) {
        AVS_Value padded = comp_avs_pad_h(env, clip, who);
        if (avs_is_error(padded))
            return padded;
        AVS_Value ret = comp_avs_zresize(env, padded, who, out_width, height,
                                         "YUV444P16", dst_left, dst_width);
        avs_release_value(padded);
        return ret;
    }

    /* mask="motion": the input is YUVA (alpha = the raster motion mask).
     * VS has one resampler per graph and z_ConvertFormat would apply the
     * picture kernel to alpha (and may not resample it well), so split the
     * mask off and resample it SEPARATELY with bilinear (clean 0/1 ramp,
     * no Spline36 ringing) on the SAME crop, then reattach. Both use the
     * shared pad since the geometry is identical. */
    AVS_Value pic = comp_avs_invoke_pos(env, "RemoveAlphaPlane", &clip, 1);
    if (avs_is_error(pic))
        return pic;
    AVS_Value mask = comp_avs_invoke_pos(env, "ExtractA", &clip, 1);
    if (avs_is_error(mask)) { avs_release_value(pic); return mask; }

    AVS_Value pic_pad = comp_avs_pad_h(env, pic, who);
    avs_release_value(pic);
    if (avs_is_error(pic_pad)) { avs_release_value(mask); return pic_pad; }
    AVS_Value pic_rs = comp_avs_zresize(env, pic_pad, who, out_width, height,
                                        "YUV444P16", dst_left, dst_width);
    avs_release_value(pic_pad);
    if (avs_is_error(pic_rs)) { avs_release_value(mask); return pic_rs; }

    AVS_Value mask_pad = comp_avs_pad_h(env, mask, who);
    avs_release_value(mask);
    if (avs_is_error(mask_pad)) { avs_release_value(pic_rs); return mask_pad; }
    AVS_Value mask_rs = comp_avs_zresize_f(env, mask_pad, who, out_width, height,
                                           "Y16", dst_left, dst_width, "bilinear");
    avs_release_value(mask_pad);
    if (avs_is_error(mask_rs)) { avs_release_value(pic_rs); return mask_rs; }

    AVS_Value comb[2] = { pic_rs, mask_rs };
    AVS_Value ret = comp_avs_invoke_pos(env, "AddAlphaPlane", comb, 2);
    avs_release_value(pic_rs);
    avs_release_value(mask_rs);
    return ret;
}

static AVS_Value AVSC_CC comp_avs_create(AVS_ScriptEnvironment *env,
                                         AVS_Value args, void *user_data)
{
    (void)user_data;

    /* args: clip, standard, setup, precomb */
    AVS_Value clip_v = avs_array_elt(args, 0);

    const char *std_s = avs_defined(avs_array_elt(args, 1))
                        ? avs_as_string(avs_array_elt(args, 1)) : "pal";
    int standard;
    if (!strcmp(std_s, "pal"))
        standard = COMP_STD_PAL;
    else if (!strcmp(std_s, "ntsc"))
        standard = COMP_STD_NTSC;
    else
        return avs_new_value_error("Encode: standard must be pal or ntsc");

    const int setup = avs_defined(avs_array_elt(args, 2))
                      ? avs_as_bool(avs_array_elt(args, 2)) : 0;
    const int precomb = avs_defined(avs_array_elt(args, 3))
                        ? avs_as_bool(avs_array_elt(args, 3)) : 0;

    /* validate the source clip against the same rules as the VS plugin */
    AVS_Clip *src_clip = avs_take_clip(clip_v, env);
    const AVS_VideoInfo *svi = avs_get_video_info(src_clip);
    const int in_width = svi->width;
    const int in_height = svi->height;
    const int is_yuv = avs_is_yuv(svi);
    const int height_ok = standard == COMP_STD_PAL
                          ? in_height == COMP_ACTIVE_HEIGHT_PAL
                          : (in_height == 480 || in_height == COMP_ACTIVE_HEIGHT_NTSC);
    avs_release_clip(src_clip);

    if (!is_yuv)
        return avs_new_value_error("Encode: clip must be YUV");
    if (!height_ok)
        return avs_new_value_error(standard == COMP_STD_PAL
            ? "Encode: pal input must have 576 lines"
            : "Encode: ntsc input must have 480 or 486 lines");

    /* resample to the 4xfsc active raster and 4:4:4 through zimg */
    AVS_Value resampled = comp_avs_resample(env, clip_v, in_width, in_height,
                                            standard);
    if (avs_is_error(resampled))
        return resampled;

    /* build the modulator filter over the resampled clip */
    AVS_FilterInfo *fi;
    AVS_Clip *out = avs_new_c_filter(env, &fi, resampled, 1);
    avs_release_value(resampled);
    if (!out)
        return avs_new_value_error("Encode: failed to create filter");

    comp_avs_t *f = calloc(1, sizeof(*f));
    if (!f) {
        avs_release_clip(out);
        return avs_new_value_error("Encode: out of memory");
    }
    f->standard = standard;
    f->height = fi->vi.height;
    comp_encode_init(&f->enc, standard, setup, precomb);

    /* output is GRAY16 composite at the active raster width */
    fi->vi.pixel_type = AVS_CS_Y16;
    fi->get_frame = comp_avs_get_frame;
    fi->set_cache_hints = comp_avs_set_cache_hints;
    fi->free_filter = comp_avs_free;
    fi->user_data = f;

    AVS_Value ret;
    avs_set_to_clip(&ret, out);
    avs_release_clip(out);
    return ret;
}

/* number of prefetch threads, for sizing the decode scratch pool. The
 * pool blocks when exhausted, so this is a performance hint only; a sane
 * floor covers hosts that report nothing useful this early. */
static int comp_avs_num_threads(AVS_ScriptEnvironment *env)
{
    size_t t = avs_get_env_property(env, AVS_AEP_THREADPOOL_THREADS);
    if (t < 1)
        t = avs_get_env_property(env, AVS_AEP_LOGICAL_CPUS);
    return t < 1 ? 1 : (t > 256 ? 256 : (int)t);
}

/* install the trained built-in soft-gain LUT for the chosen path (the
 * default when no explicit threshold/thresholds/lut/level is given) */
static void comp_avs_set_builtin_lut(comp_decode_t *dec, int standard,
                                     int dimensions)
{
    if (standard == COMP_STD_PAL)
        comp_decode_set_lut(dec,
                            dimensions == 2 ? comp_lut_builtin_pal_2d
                                            : comp_lut_builtin_pal_3d,
                            (dimensions == 2 ? COMP_T2D_NTHRESH
                                             : COMP_T3D_NTHRESH_PAL) * COMP_LUT_K);
    else
        comp_decode_set_lut(dec, comp_lut_builtin_ntsc,
                            COMP_T3D_NTHRESH * COMP_LUT_K);
}

static int comp_avs_opt_int(AVS_Value args, int idx, int def)
{
    AVS_Value v = avs_array_elt(args, idx);
    return avs_defined(v) ? avs_as_int(v) : def;
}

static double comp_avs_opt_float(AVS_Value args, int idx, double def)
{
    AVS_Value v = avs_array_elt(args, idx);
    return avs_defined(v) ? avs_as_float(v) : def;
}

static AVS_Value comp_avs_copy_conf_props(AVS_ScriptEnvironment *env,
                                          AVS_Value dst, AVS_Value src);

/* Decode(clip, standard, width, threshold, setup, dimensions, eq,
 *        thresholds, transform, level, lut, evidence, cti)
 * arg indices:  0      1       2      3       4       5      6
 *               7          8         9      10   11        12  */
static AVS_Value AVSC_CC comp_avs_dec_create(AVS_ScriptEnvironment *env,
                                             AVS_Value args, void *user_data)
{
    (void)user_data;
    AVS_Value clip_v = avs_array_elt(args, 0);

    const char *std_s = avs_defined(avs_array_elt(args, 1))
                        ? avs_as_string(avs_array_elt(args, 1)) : "pal";
    int standard;
    if (!strcmp(std_s, "pal"))
        standard = COMP_STD_PAL;
    else if (!strcmp(std_s, "ntsc"))
        standard = COMP_STD_NTSC;
    else
        return avs_new_value_error("Decode: standard must be pal or ntsc");

    const int width = comp_avs_opt_int(args, 2, 720);
    if (width != 0 && (width < 16 || width > 8192))
        return avs_new_value_error("Decode: width must be 0 (raw raster) "
                                   "or between 16 and 8192");

    const int threshold_unset = !avs_defined(avs_array_elt(args, 3));
    const double threshold = comp_avs_opt_float(args, 3, 0.4);
    if (!(threshold > 0.0 && threshold <= 1.0))
        return avs_new_value_error("Decode: threshold must be in (0, 1]");

    const int setup = comp_avs_opt_int(args, 4, 0);

    const int dimensions = comp_avs_opt_int(args, 5, 3);
    if (dimensions < 1 || dimensions > 3)
        return avs_new_value_error("Decode: dimensions must be 1, 2 or 3");

    const int eq_unset = !avs_defined(avs_array_elt(args, 6));
    int eq = comp_avs_opt_int(args, 6, 1);
    if (eq < 0 || eq > 2)
        return avs_new_value_error("Decode: eq must be 0 (off), 1 (fixed) or 2 (leak-aware)");

    AVS_Value thr_arr = avs_array_elt(args, 7);
    const int nthresh = avs_defined(thr_arr) ? avs_array_size(thr_arr) : 0;

    int transform = comp_avs_opt_int(args, 8, -1);
    if (transform < 0)
        transform = (standard == COMP_STD_NTSC && dimensions == 3) ? 2 : 0;
    if (transform < 0 || transform > 2)
        return avs_new_value_error("Decode: transform must be 0 (comb), 1 (transform) or 2 (hybrid)");
    if (transform && standard != COMP_STD_NTSC)
        return avs_new_value_error("Decode: transform applies to ntsc (pal always uses the transform)");
    if (transform && dimensions != 3)
        return avs_new_value_error("Decode: transform needs dimensions=3");

    const int has_transform = standard == COMP_STD_PAL ? dimensions >= 2
                                                       : transform != 0;
    if (eq_unset && has_transform)
        eq = 2;
    if (eq == 2 && !has_transform)
        return avs_new_value_error("Decode: eq=2 needs a transform separation");

    /* mask=: emit the motion or confidence mask as the alpha (index 13) */
    int mask_kind = COMP_MASK_NONE;
    if (avs_defined(avs_array_elt(args, 13))) {
        const char *mm = avs_as_string(avs_array_elt(args, 13));
        if (mm && !strcmp(mm, "motion")) {
            if (!(standard == COMP_STD_NTSC && dimensions == 3 && transform == 2))
                return avs_new_value_error("Decode: mask=\"motion\" needs the ntsc "
                                           "hybrid path (dimensions=3, transform=2)");
            mask_kind = COMP_MASK_MOTION;
        } else if (mm && !strcmp(mm, "confidence")) {
            if (eq != 2)
                return avs_new_value_error("Decode: mask=\"confidence\" needs eq=2 "
                                           "(a transform separation)");
            mask_kind = COMP_MASK_CONFIDENCE;
        } else {
            return avs_new_value_error("Decode: mask must be \"motion\" or \"confidence\"");
        }
    }
    const int want_mask = mask_kind != COMP_MASK_NONE;

    const int level_unset = !avs_defined(avs_array_elt(args, 9));
    const int level = comp_avs_opt_int(args, 9, 0);

    AVS_Value lut_arr = avs_array_elt(args, 10);
    const int nlut = avs_defined(lut_arr) ? avs_array_size(lut_arr) : 0;

    const int builtin_lut = level_unset && has_transform && threshold_unset
                            && nthresh < 1 && nlut < 1;
    if (level && dimensions < 2)
        return avs_new_value_error("Decode: level=1 needs dimensions 2 or 3");
    if (level && standard == COMP_STD_NTSC && !transform)
        return avs_new_value_error("Decode: level=1 needs a transform separation for ntsc");

    const double evidence = comp_avs_opt_float(args, 11, 0.0);
    if (evidence < 0.0)
        return avs_new_value_error("Decode: evidence must be >= 0");
    if (evidence > 0.0 && (standard != COMP_STD_PAL || dimensions < 2))
        return avs_new_value_error("Decode: evidence needs a pal transform (dimensions 2 or 3)");

    const int cti = comp_avs_opt_int(args, 12, 0);
    if (cti && dimensions < 2)
        return avs_new_value_error("Decode: cti needs dimensions 2 or 3");

    if (nlut > 0) {
        if (standard == COMP_STD_NTSC && !transform)
            return avs_new_value_error("Decode: lut needs a transform separation");
        if (level)
            return avs_new_value_error("Decode: lut and level are mutually exclusive");
        if (dimensions == 1)
            return avs_new_value_error("Decode: lut needs dimensions 2 or 3");
        if (nthresh > 0)
            return avs_new_value_error("Decode: lut and thresholds are mutually exclusive");
        const int want = dimensions == 2 ? COMP_T2D_NTHRESH * COMP_LUT_K
                       : (standard == COMP_STD_PAL ? COMP_T3D_NTHRESH_PAL
                                                   : COMP_T3D_NTHRESH) * COMP_LUT_K;
        if (nlut != want)
            return avs_new_value_error("Decode: lut has the wrong length for this mode");
    }
    if (nthresh > 0) {
        if (standard == COMP_STD_NTSC && !transform)
            return avs_new_value_error("Decode: thresholds needs a transform separation for ntsc");
        if (level)
            return avs_new_value_error("Decode: thresholds needs the threshold mode (level=0)");
        if (dimensions == 1)
            return avs_new_value_error("Decode: thresholds needs dimensions 2 or 3");
        const int want = dimensions == 2 ? COMP_T2D_NTHRESH
                       : (standard == COMP_STD_PAL ? COMP_T3D_NTHRESH_PAL
                                                   : COMP_T3D_NTHRESH);
        if (nthresh != want)
            return avs_new_value_error("Decode: thresholds has the wrong length for this mode");
    }

    /* validate the composite input format: GRAY16 at the active raster */
    AVS_Clip *src_clip = avs_take_clip(clip_v, env);
    const AVS_VideoInfo *svi = avs_get_video_info(src_clip);
    const int in_w = svi->width, in_h = svi->height, in_frames = svi->num_frames;
    const int is_y16 = avs_is_y(svi) && avs_bits_per_component(svi) == 16;
    avs_release_clip(src_clip);
    if (!is_y16)
        return avs_new_value_error("Decode: clip must be GRAY16 composite");
    if (standard == COMP_STD_PAL) {
        if (in_w != COMP_ACTIVE_WIDTH_PAL || in_h != COMP_ACTIVE_HEIGHT_PAL)
            return avs_new_value_error("Decode: pal composite input must be 928x576");
    } else {
        if (in_w != COMP_ACTIVE_WIDTH_NTSC
            || (in_h != 480 && in_h != COMP_ACTIVE_HEIGHT_NTSC))
            return avs_new_value_error("Decode: ntsc composite input must be 758x480 or 758x486");
    }

    /* build the decode filter (composite GRAY16 -> YUV444P16 at raster) */
    AVS_FilterInfo *fi;
    AVS_Clip *dec_clip = avs_new_c_filter(env, &fi, clip_v, 1);
    if (!dec_clip)
        return avs_new_value_error("Decode: failed to create filter");

    comp_avs_dec_t *f = calloc(1, sizeof(*f));
    if (!f) {
        avs_release_clip(dec_clip);
        return avs_new_value_error("Decode: out of memory");
    }
    f->standard = standard;
    f->height = in_h;
    f->in_frames = in_frames;
    f->mask_kind = mask_kind;
    /* free_filter/user_data are not set on the filter yet, so on these
     * error paths releasing dec_clip frees only the empty filter shell;
     * f is freed here by hand */
    if (comp_decode_init(&f->dec, standard, threshold,
                         comp_avs_num_threads(env), setup, dimensions, eq, 0,
                         transform, level, evidence, cti)) {
        free(f);
        avs_release_clip(dec_clip);
        return avs_new_value_error("Decode: decoder initialisation failed");
    }

    if (nthresh > 0) {
        double *tv = malloc(sizeof(double) * nthresh);
        for (int i = 0; i < nthresh; i++) {
            tv[i] = avs_as_float(avs_array_elt(thr_arr, i));
            if (!(tv[i] > 0.0 && tv[i] <= 1.0)) {
                free(tv); comp_decode_free(&f->dec); free(f);
                avs_release_clip(dec_clip);
                return avs_new_value_error("Decode: thresholds must be in (0, 1]");
            }
        }
        comp_decode_set_thresholds(&f->dec, tv, nthresh);
        free(tv);
    }
    if (nlut > 0) {
        double *lv = malloc(sizeof(double) * nlut);
        for (int i = 0; i < nlut; i++) {
            lv[i] = avs_as_float(avs_array_elt(lut_arr, i));
            if (!(lv[i] >= 0.0 && lv[i] <= 1.0)) {
                free(lv); comp_decode_free(&f->dec); free(f);
                avs_release_clip(dec_clip);
                return avs_new_value_error("Decode: lut values must be in [0, 1]");
            }
        }
        comp_decode_set_lut(&f->dec, lv, nlut);
        free(lv);
    } else if (builtin_lut) {
        comp_avs_set_builtin_lut(&f->dec, standard, dimensions);
    }

    /* YUVA when a mask was requested (alpha carries the raster motion
     * mask; the resample splits it off, resamples bilinear, reattaches) */
    fi->vi.pixel_type = want_mask ? AVS_CS_YUVA444P16 : AVS_CS_YUV444P16;
    fi->get_frame = comp_avs_dec_get_frame;
    fi->set_cache_hints = comp_avs_dec_set_cache_hints;
    fi->free_filter = comp_avs_dec_free;
    fi->user_data = f;

    AVS_Value dec_v;
    avs_set_to_clip(&dec_v, dec_clip);
    avs_release_clip(dec_clip);

    /* width=0: raw 4xfsc raster (YUV or YUVA), no resample; the decode
     * clip already carries the props natively */
    if (width == 0)
        return dec_v;

    /* resample back to the caller's BT.601 raster */
    AVS_Value resampled = comp_avs_resample_back(env, dec_v, "Decode", standard,
                                                 width, in_h, want_mask);
    if (avs_is_error(resampled)) {
        avs_release_value(dec_v);
        return resampled;
    }
    /* copy the confidence props the resample may drop, from the pre-resample
     * decode clip (matches Restore and the VS side) */
    AVS_Value ret = comp_avs_copy_conf_props(env, resampled, dec_v);
    avs_release_value(resampled);
    avs_release_value(dec_v);
    return ret;
}

/* validate and install the threshold/lut arrays onto a decoder, or the
 * built-in default. Returns an error string (static) or NULL on success. */
static const char *comp_avs_install_tables(comp_decode_t *dec, int standard,
                                           int dimensions, AVS_Value thr_arr,
                                           int nthresh, AVS_Value lut_arr,
                                           int nlut, int builtin_lut)
{
    if (nthresh > 0) {
        double *tv = malloc(sizeof(double) * nthresh);
        for (int i = 0; i < nthresh; i++) {
            tv[i] = avs_as_float(avs_array_elt(thr_arr, i));
            if (!(tv[i] > 0.0 && tv[i] <= 1.0)) {
                free(tv);
                return "thresholds must be in (0, 1]";
            }
        }
        comp_decode_set_thresholds(dec, tv, nthresh);
        free(tv);
    }
    if (nlut > 0) {
        double *lv = malloc(sizeof(double) * nlut);
        for (int i = 0; i < nlut; i++) {
            lv[i] = avs_as_float(avs_array_elt(lut_arr, i));
            if (!(lv[i] >= 0.0 && lv[i] <= 1.0)) {
                free(lv);
                return "lut values must be in [0, 1]";
            }
        }
        comp_decode_set_lut(dec, lv, nlut);
        free(lv);
    } else if (builtin_lut) {
        comp_avs_set_builtin_lut(dec, standard, dimensions);
    }
    return NULL;
}

/* Splice the outermost output columns through from the source: they
 * sample beyond the active raster and were never reconstructed. `restored`
 * is the width x height decoded output; `src444` is the source resampled
 * plainly to the same geometry. Consumes neither; returns a new clip. */
static AVS_Value comp_avs_edge_splice(AVS_ScriptEnvironment *env, AVS_Value restored,
                                      AVS_Value src444, int width, int nl, int nr)
{
    /* left source part | middle restored part | right source part */
    AVS_Value parts[3];
    int np = 0;
    if (nl > 0) {
        AVS_Value c[5] = { src444, avs_new_value_int(0), avs_new_value_int(0),
                           avs_new_value_int(nl), avs_new_value_int(0) };
        parts[np] = comp_avs_invoke_pos(env, "Crop", c, 5);
        if (avs_is_error(parts[np])) return parts[np];
        np++;
    }
    {
        AVS_Value c[5] = { restored, avs_new_value_int(nl), avs_new_value_int(0),
                           avs_new_value_int(width - nl - nr), avs_new_value_int(0) };
        parts[np] = comp_avs_invoke_pos(env, "Crop", c, 5);
        if (avs_is_error(parts[np])) {
            for (int i = 0; i < np; i++) avs_release_value(parts[i]);
            return parts[np];
        }
        np++;
    }
    if (nr > 0) {
        AVS_Value c[5] = { src444, avs_new_value_int(width - nr), avs_new_value_int(0),
                           avs_new_value_int(nr), avs_new_value_int(0) };
        parts[np] = comp_avs_invoke_pos(env, "Crop", c, 5);
        if (avs_is_error(parts[np])) {
            for (int i = 0; i < np; i++) avs_release_value(parts[i]);
            return parts[np];
        }
        np++;
    }

    AVS_Value ret = comp_avs_invoke_pos(env, "StackHorizontal", parts, np);
    for (int i = 0; i < np; i++)
        avs_release_value(parts[i]);
    return ret;
}

/* Copy the difficulty-metric props (separation confidence, comb fraction,
 * refine residual/correction) from the pre-resample decode clip `src`
 * onto `dst` (merge, so _Matrix/_ChromaLocation stay intact); a no-op for
 * any that are absent. Both the resample (z_ConvertFormat) and
 * the Restore edge splice (StackHorizontal inherits props from its first,
 * propless child) drop these downstream; copying from the pre-resample
 * decode clip — which carries them straight from the getframe — is
 * robust to both, and matches the VS side copying from dec_node.
 * propCopy is per-frame-n and dimension-independent, so the geometry
 * mismatch between src and dst is fine. Consumes neither. */
static AVS_Value comp_avs_copy_conf_props(AVS_ScriptEnvironment *env,
                                          AVS_Value dst, AVS_Value src)
{
    AVS_Value pnames[5] = {
        avs_new_value_string("CompositeSeparationConfidenceMean"),
        avs_new_value_string("CompositeSeparationConfidenceStdDev"),
        avs_new_value_string("CompositeMotionFraction"),
        avs_new_value_string("CompositeRefineResidual"),
        avs_new_value_string("CompositeRefineCorrection"),
    };
    AVS_Value args[4];
    const char *names[4];
    args[0] = dst;                             names[0] = NULL;
    args[1] = src;                             names[1] = NULL;
    args[2] = avs_new_value_bool(1);           names[2] = "merge";
    args[3] = avs_new_value_array(pnames, 5);  names[3] = "props";
    return avs_invoke(env, "propCopy", avs_new_value_array(args, 4), names);
}

/* Restore(clip, standard, width, threshold, setup, dimensions, eq, refine,
 *         thresholds, precomb, transform, level, lut, evidence, cti)
 *  index:  0      1       2      3        4      5       6    7
 *          8          9        10        11     12   13        14 */
static AVS_Value AVSC_CC comp_avs_res_create(AVS_ScriptEnvironment *env,
                                             AVS_Value args, void *user_data)
{
    (void)user_data;
    AVS_Value clip_v = avs_array_elt(args, 0);

    const char *std_s = avs_defined(avs_array_elt(args, 1))
                        ? avs_as_string(avs_array_elt(args, 1)) : "pal";
    int standard;
    if (!strcmp(std_s, "pal"))
        standard = COMP_STD_PAL;
    else if (!strcmp(std_s, "ntsc"))
        standard = COMP_STD_NTSC;
    else
        return avs_new_value_error("Restore: standard must be pal or ntsc");

    /* validate the YUV source; width defaults to the source width */
    AVS_Clip *src_clip = avs_take_clip(clip_v, env);
    const AVS_VideoInfo *svi = avs_get_video_info(src_clip);
    const int src_width = svi->width, in_h = svi->height;
    const int is_yuv = avs_is_yuv(svi);
    avs_release_clip(src_clip);

    const int width = comp_avs_opt_int(args, 2, src_width);
    if (width != 0 && (width < 16 || width > 8192))
        return avs_new_value_error("Restore: width must be 0 (raw raster) "
                                   "or between 16 and 8192");

    const int threshold_unset = !avs_defined(avs_array_elt(args, 3));
    const double threshold = comp_avs_opt_float(args, 3, 0.4);
    if (!(threshold > 0.0 && threshold <= 1.0))
        return avs_new_value_error("Restore: threshold must be in (0, 1]");

    const int setup = comp_avs_opt_int(args, 4, 0);

    const int dimensions = comp_avs_opt_int(args, 5, 3);
    if (dimensions < 1 || dimensions > 3)
        return avs_new_value_error("Restore: dimensions must be 1, 2 or 3");

    const int eq_unset = !avs_defined(avs_array_elt(args, 6));
    int eq = comp_avs_opt_int(args, 6, 1);
    if (eq < 0 || eq > 2)
        return avs_new_value_error("Restore: eq must be 0 (off), 1 (fixed) or 2 (leak-aware)");

    const int refine = comp_avs_opt_int(args, 7, 1);
    if (refine < 0 || refine > 16)
        return avs_new_value_error("Restore: refine must be between 0 and 16");

    AVS_Value thr_arr = avs_array_elt(args, 8);
    const int nthresh = avs_defined(thr_arr) ? avs_array_size(thr_arr) : 0;

    const int precomb = comp_avs_opt_int(args, 9, 0);

    int transform = comp_avs_opt_int(args, 10, -1);
    if (transform < 0)
        transform = (standard == COMP_STD_NTSC && dimensions == 3) ? 2 : 0;
    if (transform < 0 || transform > 2)
        return avs_new_value_error("Restore: transform must be 0 (comb), 1 (transform) or 2 (hybrid)");
    if (transform && standard != COMP_STD_NTSC)
        return avs_new_value_error("Restore: transform applies to ntsc (pal always uses the transform)");
    if (transform && dimensions != 3)
        return avs_new_value_error("Restore: transform needs dimensions=3");

    const int has_transform = standard == COMP_STD_PAL ? dimensions >= 2
                                                       : transform != 0;
    if (eq_unset && has_transform)
        eq = 2;
    if (eq == 2 && !has_transform)
        return avs_new_value_error("Restore: eq=2 needs a transform separation");

    /* mask=: emit the motion or confidence mask as the alpha (index 15) */
    int mask_kind = COMP_MASK_NONE;
    if (avs_defined(avs_array_elt(args, 15))) {
        const char *mm = avs_as_string(avs_array_elt(args, 15));
        if (mm && !strcmp(mm, "motion")) {
            if (!(standard == COMP_STD_NTSC && dimensions == 3 && transform == 2))
                return avs_new_value_error("Restore: mask=\"motion\" needs the ntsc "
                                           "hybrid path (dimensions=3, transform=2)");
            mask_kind = COMP_MASK_MOTION;
        } else if (mm && !strcmp(mm, "confidence")) {
            if (eq != 2)
                return avs_new_value_error("Restore: mask=\"confidence\" needs eq=2 "
                                           "(a transform separation)");
            mask_kind = COMP_MASK_CONFIDENCE;
        } else {
            return avs_new_value_error("Restore: mask must be \"motion\" or \"confidence\"");
        }
    }
    const int want_mask = mask_kind != COMP_MASK_NONE;

    const int level_unset = !avs_defined(avs_array_elt(args, 11));
    const int level = comp_avs_opt_int(args, 11, 0);

    AVS_Value lut_arr = avs_array_elt(args, 12);
    const int nlut = avs_defined(lut_arr) ? avs_array_size(lut_arr) : 0;

    const int builtin_lut = level_unset && has_transform && threshold_unset
                            && nthresh < 1 && nlut < 1;
    if (level && dimensions < 2)
        return avs_new_value_error("Restore: level=1 needs dimensions 2 or 3");
    if (level && standard == COMP_STD_NTSC && !transform)
        return avs_new_value_error("Restore: level=1 needs a transform separation for ntsc");

    const double evidence = comp_avs_opt_float(args, 13, 0.0);
    if (evidence < 0.0)
        return avs_new_value_error("Restore: evidence must be >= 0");
    if (evidence > 0.0 && (standard != COMP_STD_PAL || dimensions < 2))
        return avs_new_value_error("Restore: evidence needs a pal transform (dimensions 2 or 3)");

    const int cti = comp_avs_opt_int(args, 14, 0);
    if (cti && dimensions < 2)
        return avs_new_value_error("Restore: cti needs dimensions 2 or 3");

    if (!is_yuv)
        return avs_new_value_error("Restore: clip must be YUV");
    if (standard == COMP_STD_PAL) {
        if (in_h != COMP_ACTIVE_HEIGHT_PAL)
            return avs_new_value_error("Restore: pal input must have 576 lines");
    } else {
        if (in_h != 480 && in_h != COMP_ACTIVE_HEIGHT_NTSC)
            return avs_new_value_error("Restore: ntsc input must have 480 or 486 lines");
    }
    if (nlut > 0 || nthresh > 0) {
        if (standard == COMP_STD_NTSC && !transform)
            return avs_new_value_error("Restore: thresholds/lut need a transform separation for ntsc");
        if (level)
            return avs_new_value_error("Restore: thresholds/lut and level are mutually exclusive");
        if (dimensions == 1)
            return avs_new_value_error("Restore: thresholds/lut need dimensions 2 or 3");
        if (nlut > 0 && nthresh > 0)
            return avs_new_value_error("Restore: lut and thresholds are mutually exclusive");
        const int wl = dimensions == 2 ? COMP_T2D_NTHRESH
                     : (standard == COMP_STD_PAL ? COMP_T3D_NTHRESH_PAL : COMP_T3D_NTHRESH);
        if (nthresh > 0 && nthresh != wl)
            return avs_new_value_error("Restore: thresholds has the wrong length for this mode");
        if (nlut > 0 && nlut != wl * COMP_LUT_K)
            return avs_new_value_error("Restore: lut has the wrong length for this mode");
    }

    /* stage 1: source -> 4xfsc raster (YUV444P16). This raster clip is
     * both the encode input and the refine anchor, so it needs two
     * independent references. */
    int rwidth;
    double fwd_left, fwd_width;
    comp_encode_resample_params(standard, src_width, &rwidth, &fwd_left, &fwd_width);
    AVS_Value raster = comp_avs_zresize(env, clip_v, "Restore", rwidth, in_h,
                                        "YUV444P16", fwd_left, fwd_width);
    if (avs_is_error(raster))
        return raster;
    AVS_Clip *anchor = avs_take_clip(raster, env); /* second, independent ref */

    /* stage 2: encode filter over the raster picture -> GRAY16 composite */
    AVS_FilterInfo *efi;
    AVS_Clip *enc_clip = avs_new_c_filter(env, &efi, raster, 1);
    avs_release_value(raster);
    if (!enc_clip) {
        avs_release_clip(anchor);
        return avs_new_value_error("Restore: failed to create modulator");
    }
    comp_avs_t *ef = calloc(1, sizeof(*ef));
    if (!ef) {
        avs_release_clip(enc_clip);
        avs_release_clip(anchor);
        return avs_new_value_error("Restore: out of memory");
    }
    ef->standard = standard;
    ef->height = efi->vi.height;
    comp_encode_init(&ef->enc, standard, setup, precomb);
    efi->vi.pixel_type = AVS_CS_Y16;
    efi->get_frame = comp_avs_get_frame;
    efi->set_cache_hints = comp_avs_set_cache_hints;
    efi->free_filter = comp_avs_free;
    efi->user_data = ef;

    AVS_Value enc_v;
    avs_set_to_clip(&enc_v, enc_clip);
    avs_release_clip(enc_clip);

    /* stage 3: decode filter over the composite, with the raster anchor */
    AVS_FilterInfo *dfi;
    AVS_Clip *dec_clip = avs_new_c_filter(env, &dfi, enc_v, 1);
    avs_release_value(enc_v);
    if (!dec_clip) {
        avs_release_clip(anchor);
        return avs_new_value_error("Restore: failed to create decoder");
    }
    comp_avs_dec_t *df = calloc(1, sizeof(*df));
    if (!df) {
        avs_release_clip(dec_clip);
        avs_release_clip(anchor);
        return avs_new_value_error("Restore: out of memory");
    }
    df->standard = standard;
    df->height = in_h;
    df->in_frames = dfi->vi.num_frames;
    df->mask_kind = mask_kind;
    df->anchor = anchor; /* takes ownership of the second ref */
    if (comp_decode_init(&df->dec, standard, threshold,
                         comp_avs_num_threads(env), setup, dimensions, eq,
                         refine, transform, level, evidence, cti)) {
        free(df);
        avs_release_clip(anchor);
        avs_release_clip(dec_clip);
        return avs_new_value_error("Restore: decoder initialisation failed");
    }
    const char *terr = comp_avs_install_tables(&df->dec, standard, dimensions,
                                               thr_arr, nthresh, lut_arr, nlut,
                                               builtin_lut);
    if (terr) {
        comp_decode_free(&df->dec);
        avs_release_clip(anchor);
        free(df);
        avs_release_clip(dec_clip);
        char msg[128];
        snprintf(msg, sizeof(msg), "Restore: %s", terr);
        return avs_new_value_error(avs_save_string(env, msg, -1));
    }
    dfi->vi.pixel_type = want_mask ? AVS_CS_YUVA444P16 : AVS_CS_YUV444P16;
    dfi->get_frame = comp_avs_dec_get_frame;
    dfi->set_cache_hints = comp_avs_dec_set_cache_hints;
    dfi->free_filter = comp_avs_dec_free;
    dfi->user_data = df;

    AVS_Value dec_v;
    avs_set_to_clip(&dec_v, dec_clip);
    avs_release_clip(dec_clip);

    /* width=0: raw 4xfsc raster, no resample and no edge splice (at raster
     * width nothing falls outside). The decode clip carries the props
     * natively; return it directly. */
    if (width == 0)
        return dec_v;

    /* stage 4: back to the caller's raster, then splice the un-modeled
     * outer columns through from the source. */
    AVS_Value restored = comp_avs_resample_back(env, dec_v, "Restore", standard,
                                                width, in_h, want_mask);
    if (avs_is_error(restored)) {
        avs_release_value(dec_v);
        return restored;
    }

    int nl, nr;
    comp_decode_edge_columns(standard, width, rwidth, &nl, &nr);
    AVS_Value spliced;
    if (nl == 0 && nr == 0) {
        spliced = restored;
    } else {
        /* plain resize of the source to the output geometry for the spliced
         * edge columns (no subpixel crop; matches the VS plugin) */
        AVS_Value src444 = comp_avs_zresize(env, clip_v, "Restore", width, in_h,
                                            "YUV444P16", 0.0 / 0.0, 0.0 / 0.0);
        if (avs_is_error(src444)) {
            avs_release_value(restored);
            avs_release_value(dec_v);
            return src444;
        }
        /* when masking, `restored` is YUVA; give the source edge a matching
         * alpha so StackHorizontal formats agree. These columns were never
         * motion-analyzed, so their mask is 0 (still) — opacity=0. */
        if (want_mask) {
            AVS_Value a[2] = { src444, avs_new_value_float(0.0f) };
            const char *an[2] = { NULL, "opacity" };
            AVS_Value ya = avs_invoke(env, "AddAlphaPlane",
                                      avs_new_value_array(a, 2), an);
            avs_release_value(src444);
            if (avs_is_error(ya)) {
                avs_release_value(restored);
                avs_release_value(dec_v);
                return ya;
            }
            src444 = ya;
        }
        spliced = comp_avs_edge_splice(env, restored, src444, width, nl, nr);
        avs_release_value(restored);
        avs_release_value(src444);
        if (avs_is_error(spliced)) {
            avs_release_value(dec_v);
            return spliced;
        }
    }

    /* restore the confidence props both the resample and the splice drop */
    AVS_Value ret = comp_avs_copy_conf_props(env, spliced, dec_v);
    avs_release_value(spliced);
    avs_release_value(dec_v);
    return ret;
}

/* The module is built with hidden default visibility (for the
 * VapourSynth side), and on non-Windows AVSC_EXPORT carries no
 * visibility attribute of its own, so the host would not find this
 * entry point without an explicit one. */
#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((visibility("default")))
#endif
AVSC_EXPORT const char *AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment *env)
{
    avs_add_function(env, "composite_Encode",
                     "c[standard]s[setup]b[precomb]b",
                     comp_avs_create, NULL);
    avs_add_function(env, "composite_Decode",
                     "c[standard]s[width]i[threshold]f[setup]b[dimensions]i"
                     "[eq]i[thresholds]f+[transform]i[level]i[lut]f+"
                     "[evidence]f[cti]b[mask]s",
                     comp_avs_dec_create, NULL);
    avs_add_function(env, "composite_Restore",
                     "c[standard]s[width]i[threshold]f[setup]b[dimensions]i"
                     "[eq]i[refine]i[thresholds]f+[precomb]b[transform]i"
                     "[level]i[lut]f+[evidence]f[cti]b[mask]s",
                     comp_avs_res_create, NULL);
    return "composite: PAL/NTSC composite video encoder/decoder";
}
