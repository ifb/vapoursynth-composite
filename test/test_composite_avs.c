/*
 * AviSynth+ end-to-end test for the composite plugin: the AVS mirror of
 * test_composite.py. Loads the plugin in a real AviSynth+ host and asserts
 * on composite_Encode/_Decode/_Restore output — the round trip, the mask
 * outputs, width=0, and the argument-validation errors.
 *
 *   test_composite_avs <composite.so> [avsresize.so]
 *
 * Needs an AviSynth+ host: this links libavisynth to build (the repo does
 * not vendor one — point the compiler and -rpath at your local
 * libavisynth.so.N, as test/README / the avsdump recipe describes) and
 * needs avsresize (z_ConvertFormat) at runtime for the resampling paths.
 * Pass avsresize's path as the 2nd arg, or preload it, or drop it in the
 * autoload dir.
 *
 * Note: the avsresize build must not crash on load — some builds export
 * AVS_linkage as a GLOBAL symbol and segfault as it is interposed onto
 * libavisynth's read-only copy; relink avsresize with -Wl,-Bsymbolic. The
 * width=0 Decode path is the only one that touches no avsresize, so it is
 * the minimal smoke test when avsresize is unavailable (a full round trip
 * still needs it for Encode's forward resample).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avisynth_c.h"

static int failures = 0;
static AVS_ScriptEnvironment *env;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        failures = 1; \
    } \
} while (0)

/* invoke a named function with a positional-only argument array */
static AVS_Value invoke(const char *fn, AVS_Value *a, int n)
{
    return avs_invoke(env, fn, avs_new_value_array(a, n), NULL);
}

/* BlankClip(width,height,pixel_type,color_yuv,length) then Expr(y,u,v) so
 * the plane values are exact (color_yuv packs 8-bit). Returns a clip value. */
static AVS_Value make_clip(int w, int h, const char *pixel_type, int length,
                           const char *ey, const char *eu, const char *ev,
                           int field_based)
{
    AVS_Value bc[5] = { avs_new_value_int(w), avs_new_value_int(h),
                        avs_new_value_string(pixel_type),
                        avs_new_value_int(0), avs_new_value_int(length) };
    const char *bcn[5] = { "width", "height", "pixel_type", "color_yuv", "length" };
    AVS_Value blank = avs_invoke(env, "BlankClip", avs_new_value_array(bc, 5),
                                 (const char **)bcn);
    if (avs_is_error(blank))
        return blank;
    AVS_Value ex[4] = { blank, avs_new_value_string(ey),
                        avs_new_value_string(eu), avs_new_value_string(ev) };
    AVS_Value out = invoke("Expr", ex, 4);
    avs_release_value(blank);
    if (field_based && !avs_is_error(out)) {
        /* AssumeBFF so a 480-line NTSC clip lands at the BFF raster rows */
        AVS_Value a[1] = { out };
        AVS_Value bff = invoke("AssumeBFF", a, 1);
        avs_release_value(out);
        return bff;
    }
    return out;
}

/* composite_<fn>(clip, standard[, extra named args]); the caller passes an
 * already-built positional/named arg pair. */
static AVS_Value comp(const char *fn, AVS_Value *args, const char **names, int n)
{
    return avs_invoke(env, fn, avs_new_value_array(args, n), (const char **)names);
}

static AVS_Clip *first_clip(AVS_Value v)
{
    if (!avs_is_clip(v))
        return NULL;
    return avs_take_clip(v, env);
}

/* mean of plane 0 over the interior, and min/max, from frame n */
static void plane0_stats(AVS_Clip *clip, int n, int plane,
                         double *mean, int *lo, int *hi)
{
    AVS_VideoFrame *vf = avs_get_frame(clip, n);
    const int rs = avs_get_row_size_p(vf, plane) / 2;   /* uint16 samples */
    const int h = avs_get_height_p(vf, plane);
    const int pitch = avs_get_pitch_p(vf, plane) / 2;
    const uint16_t *p = (const uint16_t *)avs_get_read_ptr_p(vf, plane);
    double sum = 0.0;
    int mn = 65535, mx = 0;
    long count = 0;
    for (int y = 8; y < h - 8; y++) {
        for (int x = 32; x < rs - 32; x++) {
            const int v = p[y * pitch + x];
            sum += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            count++;
        }
    }
    *mean = count ? sum / count : 0.0;
    *lo = mn;
    *hi = mx;
    avs_release_video_frame(vf);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s composite.so [avsresize.so]\n", argv[0]);
        return 2;
    }

    env = avs_create_script_environment(AVISYNTH_INTERFACE_VERSION);
    if (!env) { fprintf(stderr, "create env failed\n"); return 1; }

    /* load avsresize (if given) then the composite plugin */
    if (argc >= 3) {
        AVS_Value a[1] = { avs_new_value_string(argv[2]) };
        AVS_Value r = invoke("LoadPlugin", a, 1);
        if (avs_is_error(r))
            fprintf(stderr, "note: LoadPlugin(avsresize) failed: %s\n", avs_as_string(r));
        avs_release_value(r);
    }
    {
        AVS_Value a[1] = { avs_new_value_string(argv[1]) };
        AVS_Value r = invoke("LoadPlugin", a, 1);
        if (avs_is_error(r)) {
            fprintf(stderr, "LoadPlugin(composite) failed: %s\n", avs_as_string(r));
            return 1;
        }
        avs_release_value(r);
    }
    const int have_resize = avs_function_exists(env, "z_ConvertFormat");

    /* ---- width=0 Decode: no avsresize needed, raw 4fsc raster ---------- */
    /* Encode needs a resample; but Decode(width=0) of a raw composite does
     * not. Build a composite by Encode when we can, else skip to masks. */
    if (have_resize) {
        AVS_Value pal = make_clip(720, 576, "YUV444P16", 1,
                                  "32128", "40960", "28672", 0);
        CHECK(!avs_is_error(pal), "pal clip: %s", avs_as_string(pal));

        AVS_Value ea[2] = { pal, avs_new_value_string("pal") };
        const char *en[2] = { NULL, "standard" };
        AVS_Value enc = comp("composite_Encode", ea, en, 2);
        CHECK(!avs_is_error(enc), "encode: %s", avs_as_string(enc));

        /* Decode -> 720 wide, YUV444P16, flat color recovered in tolerance */
        AVS_Value da[2] = { enc, avs_new_value_string("pal") };
        const char *dn[2] = { NULL, "standard" };
        AVS_Value dec = comp("composite_Decode", da, dn, 2);
        CHECK(!avs_is_error(dec), "decode: %s", avs_as_string(dec));
        AVS_Clip *dc = first_clip(dec);
        CHECK(dc != NULL, "decode result is not a clip");
        if (dc) {
            const AVS_VideoInfo *vi = avs_get_video_info(dc);
            CHECK(vi->width == 720 && vi->height == 576, "decode geom %dx%d",
                  vi->width, vi->height);
            /* flat color is recovered near-exactly in the interior (the
             * chroma leak that pulls Y down only shows at the frame edges,
             * which plane0_stats already excludes) */
            double m; int lo, hi;
            plane0_stats(dc, 0, AVS_PLANAR_Y, &m, &lo, &hi);
            CHECK(m > 32128 - 96 && m < 32128 + 96, "decode Y mean %.0f (want ~32128)", m);
            avs_release_clip(dc);
        }

        /* width=0 -> raw raster, 928 wide for PAL */
        AVS_Value wa[3] = { enc, avs_new_value_string("pal"), avs_new_value_int(0) };
        const char *wn[3] = { NULL, "standard", "width" };
        AVS_Value raw = comp("composite_Decode", wa, wn, 3);
        AVS_Clip *rc = first_clip(raw);
        CHECK(rc != NULL, "width=0 result is not a clip: %s", avs_as_string(raw));
        if (rc) {
            const AVS_VideoInfo *vi = avs_get_video_info(rc);
            CHECK(vi->width == 928 && vi->height == 576, "width=0 geom %dx%d",
                  vi->width, vi->height);
            avs_release_clip(rc);
        }
        avs_release_value(raw);
        avs_release_value(dec);
        avs_release_value(enc);
        avs_release_value(pal);

        /* ---- mask="confidence" on PAL: YUVA out, ExtractA is a soft mask */
        AVS_Value texa = make_clip(360, 576, "YUV444P16", 1,
                                   "30000", "40960", "28672", 0);
        AVS_Value texb = make_clip(360, 576, "YUV444P16", 1,
                                   "30000", "24576", "36864", 0);
        AVS_Value sh[2] = { texa, texb };
        AVS_Value tex = invoke("StackHorizontal", sh, 2);
        CHECK(!avs_is_error(tex), "stack: %s", avs_as_string(tex));

        AVS_Value ca[3] = { tex, avs_new_value_string("pal"),
                            avs_new_value_string("confidence") };
        const char *cn[3] = { NULL, "standard", "mask" };
        AVS_Value cmask = comp("composite_Restore", ca, cn, 3);
        CHECK(!avs_is_error(cmask), "confidence restore: %s", avs_as_string(cmask));
        AVS_Value ea2[1] = { cmask };
        AVS_Value amask = invoke("ExtractA", ea2, 1);
        AVS_Clip *ac = first_clip(amask);
        CHECK(ac != NULL, "ExtractA(confidence) not a clip: %s", avs_as_string(amask));
        if (ac) {
            const AVS_VideoInfo *vi = avs_get_video_info(ac);
            CHECK(vi->width == 720 && vi->height == 576, "conf mask geom %dx%d",
                  vi->width, vi->height);
            double m; int lo, hi;
            plane0_stats(ac, 0, AVS_PLANAR_Y, &m, &lo, &hi);
            /* soft mask: a spread of values, not just {0,65535} */
            CHECK(lo >= 0 && hi <= 65535, "conf mask range [%d,%d]", lo, hi);
            CHECK(hi - lo > 100, "conf mask should be soft (spread %d)", hi - lo);
            avs_release_clip(ac);
        }
        avs_release_value(amask);
        avs_release_value(cmask);
        avs_release_value(tex);
        avs_release_value(texa);
        avs_release_value(texb);
    } else {
        fprintf(stderr, "note: avsresize not loaded — skipping resample paths\n");
    }

    /* ---- interlaced 4:2:0 must not fold or stretch the picture. Only
     * subsampled and interlace-flagged input resamples field-wise, and only
     * vertical structure exposes the damage: a flat clip folds onto itself
     * and every plane statistic stays put. Restore rather than Encode because
     * Encode emits the modulated signal rather than the picture. */
    if (have_resize) {
        /* stacked bands rather than an expression: AviSynth's Expr has no
         * row-coordinate term */
        AVS_Value bands[60];
        int nb = 0;
        for (; nb < 60; nb++) {
            char ey[32];
            snprintf(ey, sizeof(ey), "%d", 16 + nb * 3);
            bands[nb] = make_clip(720, 8, "YV12", 1, ey, "128", "128", 0);
            if (avs_is_error(bands[nb])) {
                CHECK(0, "ramp band %d: %s", nb, avs_as_string(bands[nb]));
                break;
            }
        }
        AVS_Value ramp = nb == 60 ? invoke("StackVertical", bands, 60)
                                  : avs_new_value_error("bands");
        CHECK(!avs_is_error(ramp), "ramp clip: %s", avs_as_string(ramp));
        for (int i = 0; i < nb; i++)
            avs_release_value(bands[i]);

        /* avsresize keys off the frame prop, not the vi field flag. Luma
         * is never resampled vertically, so the interlaced result must
         * match the progressive one row for row; a ramp that merely rises
         * would also pass with the top half stretched over the frame. */
        AVS_Value rest[2];
        AVS_Clip *rc[2];
        AVS_VideoFrame *vf[2];
        for (int i = 0; i < 2; i++) {
            AVS_Value pa[3] = { ramp, avs_new_value_string("_FieldBased"),
                                avs_new_value_int(i ? 2 : 0) };
            const char *pn[3] = { NULL, NULL, NULL };
            AVS_Value tagged = comp("propSet", pa, pn, 3);
            CHECK(!avs_is_error(tagged), "propSet: %s", avs_as_string(tagged));

            AVS_Value ra[3] = { tagged, avs_new_value_string("ntsc"),
                                avs_new_value_int(1) };
            const char *rn[3] = { NULL, "standard", "dimensions" };
            rest[i] = comp("composite_Restore", ra, rn, 3);
            avs_release_value(tagged);
            CHECK(!avs_is_error(rest[i]), "_FieldBased=%d 420 restore: %s",
                  i ? 2 : 0, avs_as_string(rest[i]));
            rc[i] = first_clip(rest[i]);
            CHECK(rc[i] != NULL, "_FieldBased=%d 420 restore is not a clip", i ? 2 : 0);
            vf[i] = rc[i] ? avs_get_frame(rc[i], 0) : NULL;
        }
        if (vf[0] && vf[1]) {
            const int pp = avs_get_pitch_p(vf[0], AVS_PLANAR_Y) / 2;
            const int pi = avs_get_pitch_p(vf[1], AVS_PLANAR_Y) / 2;
            const uint16_t *p = (const uint16_t *)avs_get_read_ptr_p(vf[0], AVS_PLANAR_Y);
            const uint16_t *q = (const uint16_t *)avs_get_read_ptr_p(vf[1], AVS_PLANAR_Y);
            int worst = 0, worst_y = 0;
            for (int y = 0; y < 480; y++) {
                const int d = abs(p[y * pp + 360] - q[y * pi + 360]);
                if (d > worst) {
                    worst = d;
                    worst_y = y;
                }
            }
            CHECK(worst <= 1,
                  "interlaced 4:2:0 luma departs from progressive by %d at row %d "
                  "(fold or stretch)", worst, worst_y);
            /* without this the comparison passes on two flat frames */
            CHECK(p[440 * pp + 360] > p[40 * pp + 360] + 4096,
                  "ramp did not rise: top %d bottom %d",
                  p[40 * pp + 360], p[440 * pp + 360]);
        }
        for (int i = 0; i < 2; i++) {
            if (vf[i])
                avs_release_video_frame(vf[i]);
            if (rc[i])
                avs_release_clip(rc[i]);
            avs_release_value(rest[i]);
        }
        avs_release_value(ramp);
    }

    /* ---- argument validation (no avsresize needed for the error paths) -- */
    AVS_Value bad = make_clip(720, 576, "YUV444P16", 1, "32128", "40960", "28672", 0);
    /* mask="confidence" needs eq=2 */
    AVS_Value va[4] = { bad, avs_new_value_string("pal"),
                        avs_new_value_int(1), avs_new_value_string("confidence") };
    const char *vn[4] = { NULL, "standard", "eq", "mask" };
    AVS_Value verr = comp("composite_Restore", va, vn, 4);
    CHECK(avs_is_error(verr), "expected eq=2 error for mask=confidence");
    if (avs_is_error(verr))
        CHECK(strstr(avs_as_string(verr), "eq=2") != NULL,
              "wrong error: %s", avs_as_string(verr));
    avs_release_value(verr);
    /* mask="motion" off the hybrid path (PAL) */
    AVS_Value ma[3] = { bad, avs_new_value_string("pal"),
                        avs_new_value_string("motion") };
    const char *mn[3] = { NULL, "standard", "mask" };
    AVS_Value merr = comp("composite_Restore", ma, mn, 3);
    CHECK(avs_is_error(merr), "expected hybrid error for mask=motion on pal");
    avs_release_value(merr);
    /* bad mask value */
    AVS_Value ba[3] = { bad, avs_new_value_string("pal"),
                        avs_new_value_string("rainbow") };
    AVS_Value berr = comp("composite_Restore", ba, mn, 3);
    CHECK(avs_is_error(berr), "expected error for mask=rainbow");
    avs_release_value(berr);
    avs_release_value(bad);

    avs_delete_script_environment(env);
    if (failures) {
        fprintf(stderr, "test_composite_avs: FAILURES\n");
        return 1;
    }
    printf("test_composite_avs: all tests passed%s\n",
           have_resize ? "" : " (avsresize absent — error paths only)");
    return 0;
}
