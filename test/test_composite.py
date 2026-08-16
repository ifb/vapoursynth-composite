#!/usr/bin/env python3
# Plugin: registration, Encode()/Decode() PAL behavior, the round trip,
# argument validation.
# usage: test_composite.py path/to/composite.so

import sys

import numpy as np
import vapoursynth as vs

core = vs.core
core.std.LoadPlugin(sys.argv[1])


def flat(color, fmt=vs.YUV444P16, w=720, h=576, length=1):
    return core.std.BlankClip(format=fmt, width=w, height=h, length=length,
                              fpsnum=25, fpsden=1, color=color)


# ---- Encode: geometry, format, rate
out = core.composite.Encode(flat([126, 128, 128], fmt=vs.YUV420P8))
assert (out.width, out.height) == (928, 576)
assert out.format.id == vs.GRAY16
assert (out.fps.numerator, out.fps.denominator) == (25, 1)

# ---- Encode: flat mid-gray is exact and chroma-free
# Y16 = 4096 + 73*384 = 32128 -> level 0x4000 + 49*384 = 35200
out = core.composite.Encode(flat([32128, 32768, 32768]))
fr = out.get_frame(0)
for r in (0, 1, 100, 575):
    row = [fr[0][r, x] for x in range(928)]
    assert set(row) == {35200}, (r, set(row))

# ---- Encode: color modulates a subcarrier with the expected structure
col = core.composite.Encode(flat([32128, 40960, 28672], length=8))
frames = [col.get_frame(n) for n in range(8)]
row0 = [frames[0][0][0, x] for x in range(928)]
assert max(row0) - min(row0) > 8000, "expected chroma amplitude"
# samples two apart are half a subcarrier cycle apart: they sum to 2*luma
# (away from the edges, where the chroma filter tapers into the padding)
for x in range(6, 920):
    assert abs(row0[x] + row0[x + 2] - 2 * 35200) <= 2, (x, row0[x], row0[x + 2])
# the 8-field sequence repeats after 4 frames and not before
assert bytes(frames[4][0]) == bytes(frames[0][0])
assert bytes(frames[1][0]) != bytes(frames[0][0])
assert bytes(frames[2][0]) != bytes(frames[0][0])

# ---- Encode: argument validation
for bad_args, needle in (
    (dict(clip=flat([126, 128, 128], fmt=vs.YUV420P8), standard='secam'), 'standard'),
    (dict(clip=flat([126, 128, 128], fmt=vs.YUV420P8, h=480)), '576'),
    (dict(clip=core.std.BlankClip(format=vs.GRAY8, width=720, height=576)), 'YUV'),
):
    try:
        core.composite.Encode(**bad_args)
    except vs.Error as e:
        assert 'Encode' in str(e) and needle in str(e), (needle, str(e))
    else:
        assert False, f'expected error: {needle}'

# ---- Decode: geometry and format invert Encode
enc = core.composite.Encode(flat([32128, 40960, 28672], length=9))
dec = core.composite.Decode(enc)
assert (dec.width, dec.height) == (720, 576)
assert dec.format.id == vs.YUV444P16
assert (dec.fps.numerator, dec.fps.denominator) == (25, 1)
assert core.composite.Decode(enc, width=928).width == 928

# ---- Round trip: flat color recovered within tolerance in the interior
src_val = (32128, 40960, 28672)
fr = dec.get_frame(4)
for p, want in enumerate(src_val):
    vals = [fr[p][r, x] for r in (40, 288, 535) for x in range(64, 656, 8)]
    worst = max(abs(v - want) for v in vals)
    assert worst <= 64, (p, want, worst)

# ---- Decode: argument validation
gray = core.std.BlankClip(format=vs.GRAY16, width=928, height=576, length=1)
for bad_args, needle in (
    (dict(clip=gray, standard='secam'), 'standard'),
    (dict(clip=gray, threshold=0.0), 'threshold'),
    (dict(clip=gray, width=4), 'width'),
    (dict(clip=core.std.BlankClip(format=vs.GRAY16, width=720, height=576)), '928x576'),
    (dict(clip=flat([126, 128, 128], fmt=vs.YUV420P8)), 'GRAY16'),
):
    try:
        core.composite.Decode(**bad_args)
    except vs.Error as e:
        assert 'Decode' in str(e) and needle in str(e), (needle, str(e))
    else:
        assert False, f'expected error: {needle}'


# ---- NTSC: geometry, both heights, field-order handling
ntsc420 = core.std.BlankClip(format=vs.YUV420P8, width=720, height=480, length=3,
                             fpsnum=30000, fpsden=1001, color=[126, 128, 128])
out = core.composite.Encode(ntsc420, standard='ntsc')
assert (out.width, out.height, out.format.id) == (758, 480, vs.GRAY16)
out486 = core.composite.Encode(
    core.std.BlankClip(format=vs.YUV420P8, width=720, height=486, length=1), standard='ntsc')
assert (out486.width, out486.height) == (758, 486)

# Interlaced 4:2:0 must not fold about the frame midline. The forward
# resample is horizontal-only, but 4:2:0 -> 4:4:4 still upsamples chroma
# vertically, so it runs field-wise off _FieldBased and a wrong active region
# mirror-extends a 240-row field into a 480-row window. Only vertical
# structure exposes that: a flat clip folds onto itself.
ramp = np.repeat(np.linspace(16, 235, 480).astype(np.uint8)[:, None], 720, 1)
ramp420 = core.std.BlankClip(format=vs.YUV420P8, width=720, height=480,
                             length=1, fpsnum=30000, fpsden=1001)


def _fill_ramp(n, f):
    fo = f.copy()
    np.asarray(fo[0])[:] = ramp
    np.asarray(fo[1])[:] = 128
    np.asarray(fo[2])[:] = 128
    return fo


ramp420 = core.std.ModifyFrame(ramp420, ramp420, _fill_ramp)
for fb in (1, 2):
    rest = core.composite.Restore(core.std.SetFrameProps(ramp420, _FieldBased=fb),
                                  standard='ntsc', dimensions=1)
    col = np.asarray(rest.get_frame(0)[0])[:, 360].astype(int)
    assert col[440] > col[40] + 4096, (fb, col[40], col[440])
    assert not (np.diff(col[8:473]) < -400).any(), ('mirror fold', fb)

# per-field chroma must not blend across field lines: alternate the chroma
# rows and check both values survive the 4:2:0 -> 4:4:4 upsample intact
chroma420 = core.std.BlankClip(format=vs.YUV420P8, width=720, height=480,
                               length=1, fpsnum=30000, fpsden=1001)


def _fill_fields(n, f):
    fo = f.copy()
    np.asarray(fo[0])[:] = 126
    for p in (1, 2):
        a = np.asarray(fo[p])
        a[0::2] = 64
        a[1::2] = 192
    return fo


chroma420 = core.std.ModifyFrame(chroma420, chroma420, _fill_fields)
up = core.resize.Spline36(core.std.SetFrameProps(chroma420, _FieldBased=2),
                          format=vs.YUV444P16, width=758, height=480)
assert set(np.asarray(up.get_frame(0)[1])[0:8, 300].tolist()) == {16384, 49152}

# TFF and BFF content occupy different raster rows, so the composites differ
flat480 = core.std.BlankClip(format=vs.YUV444P16, width=720, height=480, length=3,
                             fpsnum=30000, fpsden=1001, color=[32128, 40960, 28672])
bff = core.std.SetFrameProps(flat480, _FieldBased=1)
tff = core.std.SetFrameProps(flat480, _FieldBased=2)
enc_bff = core.composite.Encode(bff, standard='ntsc')
enc_tff = core.composite.Encode(tff, standard='ntsc')
assert bytes(enc_bff.get_frame(0)[0]) != bytes(enc_tff.get_frame(0)[0])

# ---- NTSC round trip, both field orders and with setup
for enc_clip in (enc_bff, enc_tff):
    dec = core.composite.Decode(enc_clip, standard='ntsc', dimensions=2)
    assert (dec.width, dec.height, dec.format.id) == (720, 480, vs.YUV444P16)
    fr = dec.get_frame(1)
    for p, want in enumerate((32128, 40960, 28672)):
        vals = [fr[p][r, x] for r in (30, 240, 445) for x in range(64, 656, 8)]
        worst = max(abs(v - want) for v in vals)
        assert worst <= 96, (p, want, worst)

dec = core.composite.Decode(core.composite.Encode(bff, standard='ntsc', setup=1),
                            standard='ntsc', setup=1, dimensions=2)
fr = dec.get_frame(1)
for p, want in enumerate((32128, 40960, 28672)):
    vals = [fr[p][r, x] for r in (30, 240, 445) for x in range(64, 656, 8)]
    worst = max(abs(v - want) for v in vals)
    assert worst <= 96, ('setup', p, want, worst)

# ---- NTSC argument validation
for bad_args, needle in (
    (dict(clip=core.std.BlankClip(format=vs.YUV420P8, width=720, height=482),
          standard='ntsc'), '480 or 486'),
):
    try:
        core.composite.Encode(**bad_args)
    except vs.Error as e:
        assert needle in str(e), (needle, str(e))
    else:
        assert False, f'expected error: {needle}'
try:
    core.composite.Decode(core.std.BlankClip(format=vs.GRAY16, width=928, height=576),
                          standard='ntsc')
except vs.Error as e:
    assert '758x480 or 758x486' in str(e)
else:
    assert False, 'wrong ntsc composite size accepted'


# ---- 3D decoders: round trip both standards, static content
pal_src = flat([32128, 40960, 28672], length=9)
dec3 = core.composite.Decode(core.composite.Encode(pal_src), dimensions=3)
fr = dec3.get_frame(4)
for pl, want in enumerate((32128, 40960, 28672)):
    vals = [fr[pl][r, x] for r in (40, 288, 535) for x in range(64, 656, 8)]
    worst = max(abs(v - want) for v in vals)
    assert worst <= 64, ('pal3d', pl, want, worst)

dec3n = core.composite.Decode(core.composite.Encode(bff, standard='ntsc'),
                              standard='ntsc', dimensions=3, transform=0)
fr = dec3n.get_frame(1)
for pl, want in enumerate((32128, 40960, 28672)):
    vals = [fr[pl][r, x] for r in (30, 240, 445) for x in range(64, 656, 8)]
    worst = max(abs(v - want) for v in vals)
    assert worst <= 96, ('ntsc3d', pl, want, worst)

# ---- eq: on by default; disabling changes chroma on textured content
tex = core.std.StackHorizontal([flat([30000, 40960, 28672], w=360),
                                flat([30000, 24576, 36864], w=360)])
enc_t = core.composite.Encode(tex)
assert bytes(core.composite.Decode(enc_t).get_frame(0)[1]) != \
       bytes(core.composite.Decode(enc_t, eq=0).get_frame(0)[1])

try:
    core.composite.Decode(core.std.BlankClip(format=vs.GRAY16, width=928, height=576),
                          dimensions=4)
except vs.Error as e:
    assert 'dimensions' in str(e)
else:
    assert False, 'dimensions=4 accepted'

# ---- dimensions=1 (crude) decode and Restore() end to end
crude = core.composite.Decode(enc_t, dimensions=1)
assert (crude.width, crude.height, crude.format.id) == (720, 576, vs.YUV444P16)

rest = core.composite.Restore(tex)
assert (rest.width, rest.height, rest.format.id) == (720, 576, vs.YUV444P16)
# refine only touches luma: chroma must match the plain round trip.
# Restore splices the out-of-raster edge columns from the source, so
# the comparisons exclude them.
plain = core.composite.Decode(core.composite.Encode(tex))
def _interior(clip, pl, n=0):
    return np.asarray(clip.get_frame(n)[pl])[:, 8:-8].tobytes()
assert _interior(rest, 1) == _interior(plain, 1)
assert _interior(rest, 0) != _interior(plain, 0)
# refine=0 Restore is exactly the plain round trip (interior)
rest0 = core.composite.Restore(tex, refine=0)
for pl in range(3):
    assert _interior(rest0, pl) == _interior(plain, pl)
# NTSC Restore with field-order detection
rest_n = core.composite.Restore(bff, standard='ntsc', dimensions=3)
assert (rest_n.width, rest_n.height) == (720, 480)
rest_n.get_frame(1)
# ---- Transform NTSC 3D: round-trips, param validation
bff9 = core.std.SetFrameProps(
    core.std.BlankClip(format=vs.YUV444P16, width=720, height=480, length=9,
                       fpsnum=30000, fpsden=1001, color=[32128, 40960, 28672]),
    _FieldBased=1)
ntf = core.composite.Decode(core.composite.Encode(bff9, standard='ntsc'),
                            standard='ntsc', dimensions=3, transform=1)
fr = ntf.get_frame(4)
for pl, want in enumerate((32128, 40960, 28672)):
    vals = [fr[pl][r, x] for r in (30, 240, 445) for x in range(64, 656, 8)]
    worst = max(abs(v - want) for v in vals)
    assert worst <= 96, ('ntsc-transform3d', pl, want, worst)
try:
    core.composite.Decode(core.composite.Encode(flat([126, 128, 128], fmt=vs.YUV420P8)),
                          transform=1, dimensions=3)
except vs.Error as e:
    assert 'ntsc' in str(e)
else:
    assert False, 'transform=1 accepted for pal'
try:
    core.composite.Decode(core.composite.Encode(bff, standard='ntsc'),
                          standard='ntsc', dimensions=2, transform=1)
except vs.Error as e:
    assert 'dimensions=3' in str(e)
else:
    assert False, 'transform without dimensions=3 accepted'

# ---- defaults are the measured best settings, and adapt to the path
import numpy as _np
_lut3 = list(_np.loadtxt('test/lut_pal_3d.txt'))
_lutn = list(_np.loadtxt('test/lut_ntsc.txt'))
d_def = core.composite.Decode(core.composite.Encode(pal_src))
d_exp = core.composite.Decode(core.composite.Encode(pal_src),
                              dimensions=3, lut=_lut3, eq=2)
for pl in range(3):
    assert bytes(d_def.get_frame(4)[pl]) == bytes(d_exp.get_frame(4)[pl])
n_def = core.composite.Decode(core.composite.Encode(bff9, standard='ntsc'),
                              standard='ntsc')
n_exp = core.composite.Decode(core.composite.Encode(bff9, standard='ntsc'),
                              standard='ntsc', dimensions=3, transform=2,
                              lut=_lutn, eq=2)
for pl in range(3):
    assert bytes(n_def.get_frame(4)[pl]) == bytes(n_exp.get_frame(4)[pl])
# an explicit level or threshold selects its own mode over the builtin lut
l_def = core.composite.Decode(core.composite.Encode(pal_src), level=1)
l_exp = core.composite.Decode(core.composite.Encode(pal_src),
                              dimensions=3, level=1, eq=2)
for pl in range(3):
    assert bytes(l_def.get_frame(4)[pl]) == bytes(l_exp.get_frame(4)[pl])
assert bytes(d_def.get_frame(4)[0]) != bytes(l_def.get_frame(4)[0])
t_def = core.composite.Decode(core.composite.Encode(pal_src), threshold=0.4)
t_exp = core.composite.Decode(core.composite.Encode(pal_src),
                              dimensions=3, level=0, eq=2, threshold=0.4)
for pl in range(3):
    assert bytes(t_def.get_frame(4)[pl]) == bytes(t_exp.get_frame(4)[pl])

# ---- precomb: flat chroma unchanged, vertical chroma detail differs
flat_pc = core.composite.Encode(flat([30000, 40960, 28672]))
flat_pc1 = core.composite.Encode(flat([30000, 40960, 28672]), precomb=1)
assert bytes(flat_pc.get_frame(0)[0]) == bytes(flat_pc1.get_frame(0)[0])
vstripes = core.std.StackVertical([flat([30000, 40960, 28672], h=288),
                                   flat([30000, 24576, 36864], h=288)])
assert bytes(core.composite.Encode(vstripes).get_frame(0)[0]) != \
       bytes(core.composite.Encode(vstripes, precomb=1).get_frame(0)[0])
core.composite.Restore(tex, precomb=1).get_frame(0)

# ---- per-bin thresholds: uniform list matches the scalar exactly
tl = core.composite.Decode(enc_t, dimensions=2, thresholds=[0.4] * 80)
ts = core.composite.Decode(enc_t, dimensions=2, threshold=0.4)
for pl in range(3):
    assert bytes(tl.get_frame(0)[pl]) == bytes(ts.get_frame(0)[pl])
assert bytes(core.composite.Decode(enc_t, dimensions=2, thresholds=[0.9] * 80).get_frame(0)[1]) != \
       bytes(ts.get_frame(0)[1])
for bad_kw, needle in ((dict(dimensions=2, thresholds=[0.4] * 79), '80'),
                       (dict(thresholds=[0.4] * 80, dimensions=3), '384'),
                       (dict(dimensions=2, thresholds=[1.5] * 80), '(0, 1]')):
    try:
        core.composite.Decode(enc_t, **bad_kw)
    except vs.Error as e:
        assert needle in str(e), (needle, str(e))
    else:
        assert False, f'expected error: {needle}'

try:
    core.composite.Restore(tex, refine=99)
except vs.Error as e:
    assert 'refine' in str(e)
else:
    assert False, 'refine=99 accepted'

# ---- level mode: amplitude limiting instead of the threshold test
lv = core.composite.Decode(enc_t, dimensions=2, level=1)
fr = lv.get_frame(0)
for pl, want in enumerate((30000, 40960, 28672)):
    vals = [fr[pl][r, x] for r in (100, 288, 475) for x in range(48, 312, 8)]
    worst = max(abs(v - want) for v in vals)
    assert worst <= 96, ('pal2d-level', pl, want, worst)
assert bytes(lv.get_frame(0)[0]) != bytes(ts.get_frame(0)[0])
core.composite.Decode(enc_t, level=1, dimensions=3).get_frame(0)
core.composite.Restore(tex, level=1).get_frame(0)
ntf_lv = core.composite.Decode(core.composite.Encode(bff9, standard='ntsc'),
                               standard='ntsc', dimensions=3, transform=1,
                               level=1)
fr = ntf_lv.get_frame(4)
for pl, want in enumerate((32128, 40960, 28672)):
    vals = [fr[pl][r, x] for r in (30, 240, 445) for x in range(64, 656, 8)]
    worst = max(abs(v - want) for v in vals)
    assert worst <= 96, ('ntsc-transform3d-level', pl, want, worst)
for bad_kw, needle in ((dict(level=1, dimensions=1), 'dimensions 2 or 3'),
                       (dict(level=1, standard='ntsc', dimensions=3, transform=0), 'transform separation'),
                       (dict(level=1, dimensions=2, thresholds=[0.4] * 80), 'level=0')):
    try:
        core.composite.Decode(core.composite.Encode(bff, standard='ntsc')
                              if bad_kw.get('standard') == 'ntsc' else enc_t,
                              **bad_kw)
    except vs.Error as e:
        assert needle in str(e), (needle, str(e))
    else:
        assert False, f'expected error: {needle}'

# ---- trained soft LUT: all-ones keeps everything; validation errors
lut1 = core.composite.Decode(enc_t, dimensions=2, lut=[1.0] * 1280)
fr = lut1.get_frame(0)
for pl, want in enumerate((30000, 40960, 28672)):
    vals = [fr[pl][r, x] for r in (100, 288, 475) for x in range(48, 312, 8)]
    worst = max(abs(v - want) for v in vals)
    assert worst <= 96, ('pal2d-lut', pl, want, worst)
core.composite.Decode(enc_t, lut=[1.0] * 6144, dimensions=3).get_frame(0)
core.composite.Restore(tex, dimensions=2, lut=[1.0] * 1280).get_frame(0)
for bad_kw, needle in ((dict(dimensions=2, lut=[1.0] * 100), '1280'),
                       (dict(lut=[1.0] * 100, dimensions=3), '6144'),
                       (dict(dimensions=2, lut=[1.0] * 1280, level=1), 'mutually exclusive'),
                       (dict(dimensions=2, lut=[1.0] * 1280, thresholds=[0.4] * 80),
                        'mutually exclusive'),
                       (dict(dimensions=2, lut=[2.0] * 1280), '[0, 1]'),
                       (dict(lut=[1.0] * 1280, dimensions=1), 'dimensions 2 or 3')):
    try:
        core.composite.Decode(enc_t, **bad_kw)
    except vs.Error as e:
        assert needle in str(e), (needle, str(e))
    else:
        assert False, f'expected error: {needle}'
ntf_lut = core.composite.Decode(core.composite.Encode(bff9, standard='ntsc'),
                                standard='ntsc', dimensions=3, transform=1,
                                lut=[1.0] * 12288)
ntf_lut.get_frame(4)
try:
    core.composite.Decode(core.composite.Encode(bff, standard='ntsc'),
                          standard='ntsc', dimensions=3, transform=0,
                          lut=[1.0] * 12288)
except vs.Error as e:
    assert 'transform separation' in str(e)
else:
    assert False, 'lut accepted for the ntsc comb'
# per-bin t0 values feed the ntsc shaped threshold (transform only)
core.composite.Decode(core.composite.Encode(bff9, standard='ntsc'),
                      standard='ntsc', dimensions=3, transform=1,
                      thresholds=[0.4] * 768).get_frame(4)
try:
    core.composite.Decode(core.composite.Encode(bff, standard='ntsc'),
                          standard='ntsc', dimensions=3, transform=0,
                          thresholds=[0.4] * 768)
except vs.Error as e:
    assert 'transform separation' in str(e)
else:
    assert False, 'thresholds accepted for the ntsc comb'

# ---- LF-luma evidence prior: flat color must survive, params validate
for dims in (2, 3):
    ev = core.composite.Decode(core.composite.Encode(pal_src), dimensions=dims,
                               evidence=1.0)
    fr = ev.get_frame(4)
    for pl, want in enumerate((32128, 40960, 28672)):
        vals = [fr[pl][r, x] for r in (40, 288, 535) for x in range(64, 656, 8)]
        worst = max(abs(v - want) for v in vals)
        assert worst <= 192, ('evidence-flat', dims, pl, want, worst)
core.composite.Restore(tex, evidence=1.0).get_frame(0)
for bad_kw, needle in ((dict(evidence=-1.0), '>= 0'),
                       (dict(evidence=1.0, dimensions=1), 'dimensions 2 or 3'),
                       (dict(evidence=1.0, standard='ntsc', dimensions=3,
                             transform=1), 'pal')):
    try:
        core.composite.Decode(core.composite.Encode(bff, standard='ntsc')
                              if bad_kw.get('standard') == 'ntsc' else enc_t,
                              **bad_kw)
    except vs.Error as e:
        assert needle in str(e), (needle, str(e))
    else:
        assert False, f'expected error: {needle}'

# ---- edge padding: out-of-raster columns replicate, never mirror
# (a blanking-black edge must not reflect bright interior picture)
edge_row = np.concatenate([np.full(8, 256), np.full(704, 30000),
                           np.full(8, 256)]).astype(np.uint16)
edge_arr = np.tile(edge_row, (576, 1))
eb = core.std.BlankClip(format=vs.YUV444P16, width=720, height=576, length=1,
                        fpsnum=25, fpsden=1)
def _efill(n, f):
    fout = f.copy()
    np.asarray(fout[0])[:] = edge_arr
    np.asarray(fout[1])[:] = 32768
    np.asarray(fout[2])[:] = 32768
    return fout
eclip = core.std.ModifyFrame(eb, eb, _efill)
efr = core.composite.Restore(eclip, refine=0, dimensions=2).get_frame(0)
ey = np.asarray(efr[0])[200]
assert ey[:4].max() < 4096 and ey[-4:].max() < 4096,     (ey[:4].tolist(), ey[-4:].tolist())

# ---- cti: sharpens coincident color edges, inert elsewhere
tex2 = core.std.StackHorizontal([flat([20000, 40960, 28672], w=360),
                                 flat([45000, 24576, 36864], w=360)])
enc2 = core.composite.Encode(tex2)
u_plain = np.asarray(core.composite.Decode(enc2, dimensions=2).get_frame(0)[1]).astype(np.int32)
u_cti = np.asarray(core.composite.Decode(enc2, dimensions=2, cti=1).get_frame(0)[1]).astype(np.int32)
want = np.where(np.arange(720) < 360, 40960, 24576)
band = slice(352, 368)
e_plain = np.abs(u_plain[100:476, band] - want[band]).mean()
e_cti = np.abs(u_cti[100:476, band] - want[band]).mean()
assert e_cti < e_plain * 0.7, (e_plain, e_cti)
# flat interior untouched
assert np.abs(u_cti[100:476, 40:300] - 40960).max() <= 96
try:
    core.composite.Decode(enc2, dimensions=1, cti=1)
except vs.Error as e:
    assert 'dimensions 2 or 3' in str(e)
else:
    assert False, 'cti accepted with dimensions=1'

# ---- leak-aware eq: works on PAL transforms, differs from eq=1
eq2 = core.composite.Decode(enc_t, dimensions=2, eq=2)
fr = eq2.get_frame(0)
for pl, want in enumerate((30000, 40960, 28672)):
    vals = [fr[pl][r, x] for r in (100, 288, 475) for x in range(48, 312, 8)]
    worst = max(abs(v - want) for v in vals)
    assert worst <= 96, ('pal2d-eq2', pl, want, worst)
core.composite.Decode(enc_t, eq=2, dimensions=3).get_frame(0)
core.composite.Restore(tex, eq=2).get_frame(0)
core.composite.Decode(core.composite.Encode(bff9, standard='ntsc'),
                      standard='ntsc', dimensions=3, transform=1,
                      eq=2).get_frame(4)

# ---- comb/transform hybrid
nhy = core.composite.Decode(core.composite.Encode(bff9, standard='ntsc'),
                            standard='ntsc', dimensions=3, transform=2)
fr = nhy.get_frame(4)
for pl, want in enumerate((32128, 40960, 28672)):
    vals = [fr[pl][r, x] for r in (30, 240, 445) for x in range(64, 656, 8)]
    worst = max(abs(v - want) for v in vals)
    assert worst <= 96, ('ntsc-hybrid', pl, want, worst)
try:
    core.composite.Decode(core.composite.Encode(bff, standard='ntsc'),
                          standard='ntsc', dimensions=3, transform=3)
except vs.Error as e:
    assert 'transform must be' in str(e)
else:
    assert False, 'transform=3 accepted'

for bad_kw, needle in ((dict(eq=3), 'eq must be'),
                       (dict(eq=2, dimensions=1), 'transform separation'),
                       (dict(eq=2, standard='ntsc', dimensions=3, transform=0),
                        'transform separation')):
    try:
        core.composite.Decode(core.composite.Encode(bff, standard='ntsc')
                              if bad_kw.get('standard') == 'ntsc' else enc_t,
                              **bad_kw)
    except vs.Error as e:
        assert needle in str(e), (needle, str(e))
    else:
        assert False, f'expected error: {needle}'

# ---- difficulty frame props: present only where their path is active,
# absent otherwise (the absence is part of the contract), values in range
CONF = ('CompositeSeparationConfidenceMean', 'CompositeSeparationConfidenceStdDev')
enc9 = core.composite.Encode(bff9, standard='ntsc')

# NTSC hybrid Restore (the default: eq=2, transform=2, refine=1) has all five
rprops = core.composite.Restore(bff9, standard='ntsc').get_frame(4).props
for k in CONF + ('CompositeMotionFraction', 'CompositeRefineResidual',
                 'CompositeRefineCorrection'):
    assert k in rprops, ('restore missing', k)
for k in CONF:
    assert 0.0 <= rprops[k] <= 1.0, (k, rprops[k])
assert 0.0 <= rprops['CompositeMotionFraction'] <= 1.0
assert rprops['CompositeRefineResidual'] >= 0.0 and rprops['CompositeRefineCorrection'] >= 0.0

# Decode (no refine anchor): confidence + motion fraction, but no refine props
dprops = core.composite.Decode(enc9, standard='ntsc').get_frame(4).props
assert all(k in dprops for k in CONF + ('CompositeMotionFraction',))
assert 'CompositeRefineResidual' not in dprops and 'CompositeRefineCorrection' not in dprops

# eq=1 has no confidence map -> no confidence props
eq1 = core.composite.Decode(enc9, standard='ntsc', dimensions=2, eq=1).get_frame(4).props
assert not any(k in eq1 for k in CONF), 'eq=1 should have no confidence props'

# non-hybrid NTSC (transform=1) has no motion router -> no motion fraction
tf1 = core.composite.Decode(enc9, standard='ntsc', transform=1).get_frame(4).props
assert 'CompositeMotionFraction' not in tf1, 'transform=1 should have no motion fraction'

# PAL has no motion router either
pprops = core.composite.Restore(flat([30000, 40960, 28672]), standard='pal').get_frame(0).props
assert 'CompositeMotionFraction' not in pprops, 'PAL should have no motion fraction'
assert all(k in pprops for k in CONF)

# refine=0 drops the refine props
r0 = core.composite.Restore(bff9, standard='ntsc', refine=0).get_frame(4).props
assert 'CompositeRefineResidual' not in r0 and 'CompositeRefineCorrection' not in r0

# ---- mask="motion": second output clip, return-type contract, polarity
# no mask -> a bare clip (the vnode[] registration must collapse a single
# append; a 1-element list would break every existing script)
assert isinstance(core.composite.Restore(bff9, standard='ntsc'), vs.VideoNode)
assert isinstance(core.composite.Decode(enc9, standard='ntsc'), vs.VideoNode)

# mask -> [picture, mask]
for out in (core.composite.Restore(bff9, standard='ntsc', mask='motion'),
            core.composite.Decode(enc9, standard='ntsc', mask='motion')):
    assert isinstance(out, list) and len(out) == 2, out
    pic, mask = out
    assert pic.format.id == vs.YUV444P16
    assert mask.format.id == vs.GRAY16
    assert (mask.width, mask.height) == (pic.width, pic.height)
    # the attached mask frame must not leak onto the picture output
    assert 'CompositeMask' not in pic.get_frame(4).props
    # mask is a probability of 16-bit range; still content -> low (this
    # flat clip is fully static, so the router sends it all to the comb)
    mrow = mask.get_frame(4)[0]
    assert set(mrow[100, x] for x in range(64, 656, 16)) <= {0}, 'static -> still (0)'

# mask errors off the hybrid path
for bad, needle in ((dict(standard='pal'), 'hybrid'),
                    (dict(standard='ntsc', transform=1), 'hybrid'),
                    (dict(standard='ntsc', dimensions=2), 'hybrid'),
                    (dict(standard='ntsc', mask='rainbow'), 'mask must be')):
    src = flat([30000, 40960, 28672]) if bad.get('standard') == 'pal' else bff9
    kw = {k: v for k, v in bad.items()}
    kw.setdefault('mask', 'motion')
    try:
        core.composite.Restore(src, **kw)
    except vs.Error as e:
        assert needle in str(e), (needle, str(e))
    else:
        assert False, f'expected mask error: {needle}'

# ---- width=0: raw 4xfsc raster, no horizontal resample
# geometry is the raster width at the input height (both NTSC heights, PAL)
n480 = core.composite.Decode(enc9, standard='ntsc', width=0)
assert (n480.width, n480.height) == (758, 480)
enc486 = core.composite.Encode(
    core.std.BlankClip(format=vs.YUV444P16, width=720, height=486, length=1,
                       color=[32128, 40960, 28672]), standard='ntsc')
n486 = core.composite.Decode(enc486, standard='ntsc', width=0)
assert (n486.width, n486.height) == (758, 486)
p576 = core.composite.Restore(flat([30000, 40960, 28672]), standard='pal', width=0)
assert (p576.width, p576.height) == (928, 576)
# a bare no-mask raw clip, and default width still resamples to 720
assert isinstance(n480, vs.VideoNode)
assert core.composite.Decode(enc9, standard='ntsc').width == 720
# a small width is still rejected (only 0 is the special value)
try:
    core.composite.Decode(enc9, standard='ntsc', width=5)
except vs.Error as e:
    assert 'width must be' in str(e)
else:
    assert False, 'width=5 accepted'

# the raw mask is crisp: exactly {0, 65535}, no bilinear intermediates
rawmask = core.composite.Restore(bff9, standard='ntsc', width=0, mask='motion')[1]
assert rawmask.format.id == vs.GRAY16 and (rawmask.width, rawmask.height) == (758, 480)
mvals = set(rawmask.get_frame(4)[0][100, x] for x in range(0, 758, 8))
assert mvals <= {0, 65535}, ('raw mask not binary', mvals)

# the documented BT.601 recipe reproduces Decode(width=720) byte-for-byte
def to_bt601(raw, standard, width=720):
    if standard == 'pal':
        rho, active0, anchor601 = 540000 / 709379, 182.0, 132.0
    else:
        rho, active0, anchor601 = 33 / 35, 130 + 57 / 90, 122.0
    pad = 24
    left = raw.std.Crop(right=raw.width - 1).resize.Point(width=pad)
    right = raw.std.Crop(left=raw.width - 1).resize.Point(width=pad)
    padded = core.std.StackHorizontal([left, raw, right])
    src_left = pad + anchor601 / rho - (active0 - 0.5) - 0.5 * (720 / width) / rho
    return padded.resize.Spline36(width=width, height=raw.height,
                                  src_left=src_left, src_width=720 / rho)

for enc, std, h in ((enc9, 'ntsc', 480), (core.composite.Encode(
        flat([30000, 40960, 28672]), standard='pal'), 'pal', 576)):
    ref = core.composite.Decode(enc, standard=std, width=720).get_frame(0)
    rec = to_bt601(core.composite.Decode(enc, standard=std, width=0), std).get_frame(0)
    for pl in range(3):
        a, b = ref[pl], rec[pl]
        assert bytes(a) == bytes(b), (std, 'recipe != Decode(720)', pl)

# ---- mask="confidence": soft mask on any eq=2 path (PAL and NTSC), the
# inverse of the confidence prop; a different gate from motion
tex9 = core.std.StackHorizontal([flat([30000, 40960, 28672], w=360, length=9),
                                 flat([30000, 24576, 36864], w=360, length=9)])
for src, std in ((tex9, 'pal'),
                 (core.std.SetFrameProps(core.std.StackHorizontal([
                     flat([30000, 40960, 28672], w=360, h=480, length=9),
                     flat([30000, 24576, 36864], w=360, h=480, length=9)]),
                     _FieldBased=1), 'ntsc')):
    out = core.composite.Restore(src, standard=std, mask='confidence')
    assert isinstance(out, list) and len(out) == 2
    pic, mask = out
    assert mask.format.id == vs.GRAY16
    assert (mask.width, mask.height) == (pic.width, pic.height)
    assert 'CompositeMask' not in pic.get_frame(0).props
    mp = np.asarray(mask.get_frame(0)[0]).astype(np.float64)
    assert mp.min() >= 0 and mp.max() <= 65535
    # the mask is 1 - conf, so its mean tracks 1 - the confidence prop
    conf = pic.get_frame(0).props['CompositeSeparationConfidenceMean']
    assert abs(mp.mean() / 65535.0 - (1 - conf)) < 0.02, (std, mp.mean() / 65535.0, conf)

# confidence needs eq=2 (a wider gate than motion — it works on PAL and on
# non-hybrid NTSC, but errors at eq=1)
for src, std in ((flat([30000, 40960, 28672]), 'pal'), (bff9, 'ntsc')):
    try:
        core.composite.Restore(src, standard=std, eq=1, dimensions=2,
                               mask='confidence')
    except vs.Error as e:
        assert 'eq=2' in str(e), (std, str(e))
    else:
        assert False, 'expected confidence eq=2 error'

print('test_composite: all tests passed')
