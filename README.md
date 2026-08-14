# vapoursynth-composite

Clean up cross-color and cross-luma artifacts in old PAL/NTSC footage,
for VapourSynth and AviSynth+.

If your source came off tape or a cheap decoder and has **rainbow
shimmer on fine detail** (cross-color) or **crawling dots along sharp
edges** (cross-luma / dot crawl), this plugin can take a lot of it back
out. It re-encodes the picture to the composite signal the bad decoder
would have seen, then decodes that signal properly with a comb / Transform
separator - so the artifacts the original decoder baked in never get
re-created.

You do not need to understand any of that to use it. Feed a clip to
`Restore` and get a cleaner clip back.

## Quick start

VapourSynth:

```python
import vapoursynth as vs
core = vs.core

clip = core.bs.VideoSource("tape.mkv")   # your NTSC/PAL source
out  = core.composite.Restore(clip, standard="ntsc")
out.set_output()
```

AviSynth+:

```
BSVideoSource("tape.mkv")
composite_Restore(standard="ntsc")
```

That is the whole thing. Everything below is tuning.

## What input it accepts

- **Format:** any constant-format YUV clip (`Restore` and `Encode`
  resample internally, so 4:2:0, 4:2:2, 8-bit, 10-bit - all fine).
- **Width:** any width. It is resampled to the composite raster and
  back; pick your output width with `width=` (`Restore` keeps the input
  width by default, `Decode` defaults to 720).
- **Height:** this is the one hard rule, because it fixes where the
  picture sits in the TV raster:
  - **PAL** - **576 lines**, exactly.
  - **NTSC** - **480 or 486 lines**. A 480-line clip is placed on the
    486-line raster automatically from its field order (`_FieldBased`):
    BFF/DV content at rows 4-483, TFF/RP 202 content at rows 5-484.
  - Anything else is an error, so resize/pad to 576 or 480/486 first.

Field order matters for NTSC 480-line input - set `_FieldBased`
correctly (most DV/tape captures are BFF). It selects the raster
placement, which tells the spatio-temporal separator which rows belong
to which field, and therefore which rows in the neighboring frames are
their temporal partners. Get it wrong on interlaced content and the
separator combs against the wrong field, costing luma and chroma
accuracy. The encode and decode read the same flag, so the subcarrier
phase itself stays consistent either way; what suffers is the temporal
pairing. The 2D path does not use neighboring frames and is unaffected.

Field pairing matters as well. The encoder derives each line's
subcarrier phase from the frame index and the line's parity in the
raster, so it assumes each input frame holds two fields that are
adjacent fields of the standard cadence. On a capture whose fields are
shifted by one, the two halves of a frame come from different instants,
and the re-encode stamps phase onto content that does not match it;
chroma then lands in luma as a stationary crosshatch. Such a clip looks
interlaced until it is field-matched.

This is not an argument for field matching first. Interlaced and
correctly telecined sources (a 2:3 pulldown DVD, broadcast capture)
already satisfy the assumption: their fields are adjacent fields of the
standard cadence, which is what the original decoder saw and what the
round trip has to reconstruct. Feed those in interlaced and run IVTC
(TFM, VFM, or similar) after `Restore` - matching first would decimate
fields, leaving the remaining frames out of cadence with each other and
creating the phase mismatch described above.

Field-match before `Restore` only when the pairing itself is wrong, as
on a capture whose fields are shifted by one. There the frames were
never valid interlaced frames, and matching repairs them into something
the encoder's assumption fits.

## Restore

```python
out = core.composite.Restore(clip, standard="ntsc")
```

The whole round trip in one call: resample to the composite raster,
re-encode, decode with a good separator, resample back to `width`. This
is the function you want for cleanup. It takes every `Decode` parameter
(see below) plus two of its own:

- `refine` - luma detail recovery, default `1`. A short Y-only loop that
  recovers luma detail the original decoder softened. Chroma is
  untouched. `0` disables it; higher values (2-4) push harder on clean,
  detailed sources. See *About `refine`* below - it is not a sharpener.
- `precomb` - passed to the internal encoder (see `Encode`); leave it
  off for cleanup.

Most users only ever set `standard` and maybe `width`. Reach for the
`Decode` parameters below only if the defaults leave something on the
table.

### About `refine` (it is not "fake" sharpening)

A sharpener guesses: it finds an edge and adds contrast around it,
inventing high-frequency content that was never in the signal. `refine`
does the opposite - it *solves* for the detail that the bad decoder
attenuated, using the fact that we know exactly how that decoder blurs.

The plugin already has a model of the crude decoder (it's what the round
trip is built on). `refine` runs a few steps of a constrained
deconvolution (a Y-only Landweber iteration): it proposes a sharper
luma, pushes it back through the *modeled* bad decoder, and checks
whether the result matches the luma you actually captured. It keeps only
the correction that makes the model reproduce your real footage. Nothing
is added that isn't required to explain the picture the decoder
produced - so the recovered detail is inferred from the signal and the
known blur, not painted on.

We measured this rather than assuming it. On real detailed footage the
extra high-frequency energy `refine=1` produces moves the image *closer*
to a clean reference (higher SSIMULACRA2, luma PSNR, and XPSNR), and it
survives a downstream sharpener - the signature of genuine detail, not
invented edges. It even reduces the error in flat regions there, the
opposite of grain amplification.

Two limits:

- **How much it helps depends on the source.** The gain scales with how
  closely the original decoder resembled the notch we model. Clean,
  detailed, fairly static material benefits most; on soft or heavily
  processed sources the correction is small.
- **It is not free on every source.** On grainy consumer tape
  (VHS/Hi8-class) some of the boosted high frequencies are amplified
  grain rather than detail, and on flat or graphic content (titles, test
  patterns, color bars) it can overreach. If your source is grainy and
  you follow this plugin with an aggressive denoiser, that denoiser
  removes most of the amplified grain anyway; if your source is
  graphic/flat, set `refine=0`.

`refine` lives inside `Restore` and cannot be applied later - it is
anchored to the pre-encode picture, which only exists during the round
trip. So the choice is simply whether to enable it in that first
`Restore` call. Because all of the cross-color / dot-crawl removal (the
plugin's main job) happens independently of `refine`, setting
`refine=0` costs nothing on artifact cleanup; it only forgoes the luma
detail recovery. Leave it at `1` for clean detailed sources; set it to
`0` for grainy tape into a denoiser, for graphic/flat content, or
whenever you want a conservative default that adds no high frequencies
of its own.

## Choosing a decoder mode

The defaults are the best general setting and you can stop here. The
numbers below are for when you want to tune.

**How to read the tables.** Every clip is measured as: clean source ->
simulated bad decoder -> this plugin -> compared back to the clean source
(`test/readme_tables.py`, built on `test/metrics.py`). *degraded* is the
artifact-ridden input (doing nothing); *transparency* is the clean
source run straight through encode/decode with no bad decoder - the PSNR
ceiling (for the lower-is-better columns it is just a reference, not a
floor). **chromaHF** is residual chroma high-frequency energy (rainbows /
dot patterns) and **flicker** is frame-to-frame chroma change (crawl);
for both, **lower is better**, and the `%` is how much of the artifact
each mode removed versus *degraded*. PSNR is fidelity to the clean
source in dB (higher is better).

Modes are compared through `Decode`; `Restore` adds a Y-only detail
recovery (`refine`) on top, so under `Restore` the luma PSNR is higher
and the chroma numbers here are unchanged. Each mode is measured at its
own defaults, so the comb rows use `eq=1` (the no-transform default) and
the transform/hybrid rows use `eq=2` - part of the comb's higher
chromaHF is that equalization choice, not the separator itself.

Corpora: **VQEG** rows are held-out clips (`src20-22`) the shipped
trained tables were *not* trained on - the generalization number.
**BT.802** is the NTSC restoration-target corpus the tables were tuned
on, split into **stills** (Rec. BT.802 scenes 1-13, single frame - every
temporal metric is trivially flat, so this is the easy case) and
**motion** (scenes 14+, real footage). The never-trained VQEG held-out
and the tuned-on BT.802-motion rows agree closely (chromaHF -73% vs
-75%; flicker -82% vs -73%, the same ballpark) - the evidence the
separation generalizes. **bars** is SMPTE EG-1 / EBU color
bars (flat color fidelity) and **zone** is a zone-plate sweep - a pure
cross-color torture test.

### NTSC

On real footage the defaults remove about **73-75% of the rainbow
(chromaHF)** and **73-82% of the dot-crawl (flicker)**. chromaHF /
flicker columns are the residual; **(-N%)** is how much was removed vs.
*degraded* (higher removed % is better). PSNR is dB vs. the clean source.

| corpus | mode | PSNR Y / U / V | chromaHF | flicker |
|---|---|---|---|---|
| **VQEG** (held-out) | degraded | 30.7 / 32.9 / 35.1 | 419 | 1356 |
| | **default** (3D hybrid) | 33.1 / 39.2 / 39.8 | **115 (-73%)** | **249 (-82%)** |
| | comb 3D (`transform=0`) | 32.8 / 38.6 / 39.6 | 233 (-44%) | 354 (-74%) |
| | transform 3D (`transform=1`) | 33.0 / 39.0 / 39.6 | 107 (-74%) | 261 (-81%) |
| | 2D comb (`dimensions=2`) | 31.7 / 36.0 / 37.7 | 334 (-20%) | 860 (-37%) |
| | *transparency* | 41.5 / 39.8 / 40.4 | 163 | 302 |
| **BT.802 motion** (scenes 14+) | degraded | 30.2 / 32.0 / 34.3 | 581 | 1629 |
| | **default** (3D hybrid) | 32.2 / 37.7 / 39.3 | **147 (-75%)** | **434 (-73%)** |
| | transform 3D | 32.1 / 38.2 / 39.8 | 138 (-76%) | 402 (-75%) |
| | comb 3D | 31.9 / 36.7 / 38.5 | 298 (-49%) | 548 (-66%) |
| | 2D comb | 31.3 / 35.5 / 37.7 | 369 (-37%) | 1042 (-36%) |
| | *transparency* | 40.8 / 38.6 / 40.4 | 213 | 520 |
| **BT.802 stills** (scenes 1-13) | degraded | 28.5 / 30.2 / 32.2 | 606 | 1924 |
| | **default** (3D hybrid) | 31.4 / 37.9 / 38.5 | 158 (-74%) | 136 (-93%) |
| | comb 3D | 31.4 / 39.2 / 39.8 | 180 (-70%) | 17 (-99%) |
| | 2D comb | 29.5 / 33.3 / 34.9 | 391 (-35%) | 1203 (-37%) |
| | *transparency* | 40.8 / 40.0 / 40.7 | 212 | 170 |
| **bars** (SMPTE EG-1) | degraded | 40.5 / 32.9 / 36.5 | 1186 | 110 |
| | **default** (3D hybrid) | 40.7 / 36.5 / 37.8 | 218 (-82%) | 217 (-) |
| | comb 3D | 42.6 / 37.9 / 39.5 | 218 (-82%) | 1 (-) |
| | *transparency* | 44.0 / 41.0 / 41.8 | 223 | 225 |
| **zone** (torture test) | degraded | 23.3 / 27.9 / 30.9 | 655 | 2377 |
| | **default** (3D hybrid) | 26.2 / 87.8 / 94.2 | 0 (-100%) | 0 (-100%) |
| | 2D comb | 24.9 / 34.3 / 37.2 | 432 (-34%) | 1265 (-47%) |
| | *transparency* | 68.1 / 90.3 / 96.3 | 0 | 0 |

On the **bars** rows the flicker column is round-trip residual, not real
crawl (a static frame has no motion - its only frame-to-frame change is
the subcarrier sequence), so no reduction % is meaningful there; read
flicker on the motion corpora. The comb nulls that residual on a static
input, which is why it reads near-zero on stills and bars - but that
does not hold on real footage, where the hybrid wins.

### PAL

On held-out footage the defaults remove about **43% of the rainbow** and
**26% of the dot-crawl**; `evidence=1.0` pushes both a little further.

| corpus | mode | PSNR Y / U / V | chromaHF | flicker |
|---|---|---|---|---|
| **VQEG** (held-out) | degraded | 35.4 / 37.6 / 39.2 | 222 | 1366 |
| | **default** (3D) | 35.8 / 38.2 / 38.0 | **127 (-43%)** | **1012 (-26%)** |
| | default `+evidence=1.0` | 35.7 / 38.0 / 37.6 | 122 (-45%) | 971 (-29%) |
| | trained 2D (`dimensions=2`) | 35.8 / 37.8 / 37.9 | 129 (-42%) | 1053 (-23%) |
| | level 2D (`level=1`) | 35.6 / 38.2 / 38.2 | 146 (-34%) | 1105 (-19%) |
| | threshold 2D (`threshold=0.4`) | 35.6 / 38.3 / 38.4 | 155 (-30%) | 1153 (-16%) |
| | *transparency* | 41.1 / 38.6 / 38.5 | 149 | 1085 |
| **bars** (EBU) | degraded | 46.8 / 36.8 / 39.1 | 430 | 191 |
| | **default** (3D) | 48.0 / 37.4 / 39.9 | 210 (-51%) | 89 (-53%) |
| | trained 2D | 49.3 / 37.5 / 39.9 | 212 (-51%) | 3 (-98%) |
| | *transparency* | 52.8 / 44.5 / 48.9 | 229 | 90 |
| **zone** (torture test) | degraded | 25.7 / 38.0 / 41.0 | 440 | 599 |
| | **default** (3D) | 26.2 / 76.3 / 79.9 | 1 (-100%) | 4 (-99%) |
| | trained 2D | 26.1 / 74.9 / 79.4 | 1 (-100%) | 6 (-99%) |
| | *transparency* | 56.7 / 69.3 / 72.6 | 2 | 7 |

Guidance:

- **Just use the defaults** (`dimensions=3`, trained tables, NTSC
  `transform=2`). They win or tie almost everywhere.
- **`dimensions=2`** is the fast path - no neighboring frames, several
  times less compute. Use it for stills, very short clips, or previews.
- **NTSC `transform=`** picks how the 3D path separates: `0` a comb
  (near-exact on fully static content), `1` a Transform separator
  (stronger on motion), `2` (default) routes between them per sample.
- **Artifact-heavy footage:** add `evidence=1.0` (PAL) to knock down
  residual chroma flicker in flat regions.
- **Graphics / titles / test patterns:** try `cti=1` (chroma transient
  improvement) to re-sharpen color edges; leave it off on natural
  footage, where it hurts.

## Decode

```python
out = core.composite.Decode(comp, standard="ntsc", width=720)
```

Decodes an already-composite GRAY16 clip (the output of `Encode`) back
to YUV444P16. `Restore` calls this internally; use it directly only if
you are working with composite signals yourself. Composite input must be
exactly `758x480`, `758x486` (NTSC) or `928x576` (PAL).

Parameters (all optional, all also available on `Restore`). Full types,
ranges and constraints are in *Parameter reference* below.

- `standard` - `"pal"` or `"ntsc"`, default `"pal"`. Must match the encode.
- `width` - output width, default `720`. Decoding happens internally on
  the 4x fsc raster and the result is resampled to `width` with a subpixel
  crop that lands the samples on the BT.601 grid (the exact inverse of
  `Encode`'s mapping), so a round trip is geometry-preserving. Set
  `width=0` to skip that final horizontal resample entirely and get the
  raw decode raster - **NTSC 758**, **PAL 928** wide (the SMPTE 244M /
  EBU 4fsc sampling), at the input height. The picture (and the `mask`
  output, which stays crisp) then come straight off the decode grid with
  no resize; these are non-square-pixel frames you resample yourself.
  Note that setting `width` to the raster value (e.g. `758`) is *not* the
  same - that still resamples; only `width=0` bypasses it.
- `dimensions` - `3` (default) spatio-temporal separation using
  neighboring frames; `2` fast 2D (spatial / line comb); `1` a crude
  notch reference (worst case).
- `transform` - NTSC `dimensions=3` only: `0` comb, `1` Transform, `2`
  motion-routed hybrid (default).
- `eq` - chroma equalization. `2` steers chroma bandwidth by the
  separator's own confidence; `1` a fixed inverse filter; `0` off. The
  default follows the path: `2` wherever a Transform separation is
  present, `1` otherwise (so the comb paths default to `1`).
- `evidence` - PAL only, default `0`. A low-frequency luma prior that
  attenuates chroma with no luma partner; `0.5-1` cuts flicker on
  artifact-heavy footage.
- `cti` - luma-guided chroma transient improvement, default off. For
  graphics-like sources only.
- `setup` - NTSC 7.5 IRE pedestal; must match the encode.
- `mask` - output a per-sample mask as a second clip: `"motion"` (the
  NTSC hybrid router) or `"confidence"` (the separation confidence). See
  *Masks* below.

Advanced separation controls (`threshold`, `thresholds`, `level`, `lut`)
override the built-in trained tables. **Passing any one of them turns the
trained table off** - including `threshold=0.4`, which is the same number
the fallback uses but selects a different mode; see *Parameter reference*
below. `level=1` is the robust untrained alternative to the trained
tables on synthetic extremes.

## Masks

`Decode` and `Restore` can output a per-sample mask as a second clip
alongside the picture, so you can postprocess selected regions
differently - the region a sample fell into, or how well it separated.
Two kinds are available via `mask=`.

The mask comes from the same decode as the picture (no second pass) and
is resampled to the output `width` with bilinear - a clean soft edge,
unlike the picture's sharper filter. For a crisp, un-resampled mask use
`width=0` (see `width` above); the mask then comes straight off the
decode raster.

How the two outputs are returned differs by host:

**VapourSynth** returns a two-element list, `[picture, mask]`:

```python
pic, mask = core.composite.Restore(clip, standard="ntsc", mask="motion")
alt = pic.some.AggressiveChromaCleanup()
out = core.std.MaskedMerge(pic, alt, mask, planes=[1, 2])
```

**AviSynth+** returns one `YUVA` clip with the mask as its alpha; pull it
out with `ExtractA`:

```
dec  = composite_Restore(clip, standard="ntsc", mask="motion")
mask = ExtractA(dec)
alt  = AggressiveChromaCleanup(dec)
Overlay(dec, alt, mask=mask)
```

### `mask="motion"`

On the NTSC hybrid path (the default: `dimensions=3`, `transform=2`) the
decoder routes each sample to the comb (still regions) or the Transform
separator (motion). `mask="motion"` exposes that per-pixel decision:
**white = motion**, black = still - matching the mvtools/mvutensils
convention, so `MaskedMerge` processes the moving regions. It works only
on the NTSC hybrid path (it errors elsewhere, since no other path has a
motion router).

### `mask="confidence"`

`mask="confidence"` exposes the separator's per-sample confidence - the
same signal behind the `CompositeSeparationConfidence*` properties -
available on **any `eq=2` path** (all Transform separations, PAL and
NTSC; it errors when `eq` is not 2). Unlike the binary motion mask it is
**soft** (a graded 0-max mask). Polarity is **white = least confident**,
i.e. the *inverse* of the confidence property (a `PlaneStats` mean of the
mask is ~ `1 - CompositeSeparationConfidenceMean`): white marks the
samples where chroma separation was most suspect - the ones you would
clean hardest.

## Working at the 4fsc raster (`width=0`)

`width=0` gives you the raw 4x fsc raster with no horizontal resample, so
you can run your own processing at that sampling and convert to BT.601
later without a double resize. When you *do* want BT.601 (720-wide,
square-ish pixels), reproduce exactly what the plugin does internally:
replicate-pad the edges, then a subpixel-crop Spline36. This is
byte-identical to `Decode(width=720)`:

```python
def to_bt601(raw, standard, width=720):
    # rho = 4fsc/13.5 MHz sample-rate ratio; active0 = active-window start
    # on the 4fsc raster; anchor601 = BT.601 first active luma sample
    if standard == "pal":
        rho, active0, anchor601 = 540000 / 709379, 182.0, 132.0
    else:
        rho, active0, anchor601 = 33 / 35, 130 + 57 / 90, 122.0
    pad = 24
    # edge-replicate `pad` columns each side (NOT black - a black step
    # would make Spline36 ring inward along the frame border)
    left  = raw.std.Crop(right=raw.width - 1).resize.Point(width=pad)
    right = raw.std.Crop(left=raw.width - 1).resize.Point(width=pad)
    padded = core.std.StackHorizontal([left, raw, right])
    src_left = pad + anchor601 / rho - (active0 - 0.5) - 0.5 * (720 / width) / rho
    return padded.resize.Spline36(width=width, height=raw.height,
                                  src_left=src_left, src_width=720 / rho)
```

Edges are filled by replication here; the plugin's own `Restore` instead
passes the few outermost columns through from the source (they sample
beyond the raster and were never reconstructed). Reproduce that only if
you specifically want byte-identical-to-`Restore` borders and still hold
the original source clip - for most processing, the replicated edge is
fine (and more consistent).

## Frame properties

`Decode` and `Restore` tag each output frame with a few read-only
diagnostics - how hard the decode was, per frame. They are *difficulty*
signals, not quality scores: there is no clean reference to score
against during restoration, so these report the decoder's own effort and
confidence, which correlate with where artifacts are likely to remain.
Read them in a script to log, plot, or gate later processing (for
example, denoise harder on low-confidence frames).

Each property appears **only when the path that produces it is active**,
so its presence is itself informative. In VapourSynth they are on
`frame.props`; in AviSynth+ read them with `propGetFloat`.

| Property | Present when | Meaning |
|---|---|---|
| `CompositeSeparationConfidenceMean` | `eq=2` (the default on transform paths) | Mean of the separator's per-sample confidence, ~0-1. Near 1 where chroma separated cleanly; low where luma leaked into chroma (rainbow-prone content). Lower = a harder frame. |
| `CompositeSeparationConfidenceStdDev` | `eq=2` | Spread of that confidence across the frame. High std means the trouble is localized (a few bad regions) rather than uniform. |
| `CompositeMotionFraction` | NTSC `dimensions=3`, `transform=2` (the default) | Fraction of samples the motion router judged to be in motion (and sent to the transform), 0-1. Near 0 = a nearly still frame (the comb handled it); near 1 = mostly motion. |
| `CompositeRefineResidual` | `Restore` with `refine>0` (the default) | Mean luma the crude-decoder model still cannot reproduce after refinement. High = the source's original decoder was unlike the model, so refine could only partly fit it. |
| `CompositeRefineCorrection` | `Restore` with `refine>0` | Mean amount `refine` moved the luma. Large = a lot of softened detail was recovered. (Equal to the residual at `refine=1`; they diverge at higher counts as the residual falls and the total correction grows.) |

All values are per frame and computed over the active picture. Absent
properties simply mean that path was not taken (for example, no
`CompositeMotionFraction` on PAL, or no refine properties at `refine=0`).

## Encode

```python
comp = core.composite.Encode(clip, standard="ntsc")
```

The forward direction: YUV to a composite GRAY16 clip on the 4x fsc
active raster (PAL 928x576, NTSC 758x480/486). Mostly useful for making
test composites; `Restore` does the encode for you.

- `standard` - `"pal"` or `"ntsc"`, default `"pal"`.
- `setup` - NTSC only: add the 7.5 IRE pedestal (default off).
- `precomb` - vertically low-pass U/V before modulation (Poynton's
  precombing). Helpful only when feeding a comb decoder a *clean*
  source; leave it off for cleanup, where it measurably hurts.

## Parameter reference

Which functions take what:

| | `Encode` | `Decode` | `Restore` |
|---|---|---|---|
| `standard`, `setup` | yes | yes | yes |
| `precomb` | yes | | yes |
| `width`, `dimensions`, `eq`, `transform`, `evidence`, `cti`, `mask` | | yes | yes |
| `threshold`, `thresholds`, `level`, `lut` | | yes | yes |
| `refine` | | | yes |

**Host differences.** The AviSynth+ names are `composite_Encode`,
`composite_Decode` and `composite_Restore`, and the parameters are the
same in the same order. AviSynth+ declares `setup`, `precomb` and `cti`
as booleans (`true`/`false`) where VapourSynth takes an int `0`/`1`;
`thresholds` and `lut` are float arrays in both.

### Common

**`standard`** - string, `"pal"` or `"ntsc"`. Default `"pal"`.

Selects the color standard: the subcarrier frequency, the raster, and
the separation geometry. It is **not** inferred from the clip, so an
NTSC source with `standard` unset is decoded as PAL and comes out wrong
- always pass it explicitly. On `Decode` it must match whatever encoded
the composite. Any other string is an error.

**`setup`** - int `0`/`1` (AviSynth+: bool). Default `0`.

Adds the NTSC 7.5 IRE setup pedestal, which lifts black off the blanking
level (SMPTE 170M). It must match between encode and decode - a
mismatch shifts black level. Accepted but **ignored on PAL**, which has
no pedestal, so it is not an error to leave it set on a PAL call.

**`width`** - int. `0`, or `16`-`8192`. Default `720` on `Decode`, the
**input width** on `Restore`.

Output width after the final horizontal resample. See *Working at the
4fsc raster* above for what `width=0` gives you and how to reproduce the
internal resample yourself. Note the two defaults differ: `Restore` is a
cleanup filter and preserves your clip's width, while `Decode` starts
from a fixed composite raster and targets BT.601 720. Any value outside
`16`-`8192` other than `0` is an error.

### Separation

**`dimensions`** - int, `1`, `2` or `3`. Default `3`.

How many dimensions the separator works in.

- `3` - spatio-temporal: uses neighboring frames as well as the current
  one. The best separation, the default, and the most compute.
- `2` - spatial only (line comb / 2D Transform). No neighboring frames,
  several times cheaper. Use for stills, very short clips, or previews.
- `1` - a crude notch filter: the worst case, kept as a reference point,
  not a fast mode. It rejects every advanced control:
  `transform`, `level`, `lut`, `thresholds`, `evidence` and `cti` all
  require `dimensions` of 2 or 3, and `eq=2` needs a Transform path, so
  `dimensions=1` accepts only `standard`, `width`, `setup` and `eq` 0/1.

**`transform`** - int, `0`, `1` or `2`. Default `2` on NTSC
`dimensions=3`, otherwise `0`.

Picks how the NTSC 3D path separates luma from chroma.

- `0` - a 3D comb. Near-exact on fully static content, weaker on motion.
- `1` - a Transform (frequency-domain) separator. Stronger on motion.
- `2` - a motion-routed hybrid: per sample, comb where still and
  Transform where moving. The default, and the best of the three on real
  footage.

NTSC only, and `dimensions=3` only - a nonzero `transform` on PAL is an
error (PAL always uses the Transform), as is a nonzero `transform` with
`dimensions` 1 or 2.

**`eq`** - int, `0`, `1` or `2`. Default `2` on any Transform path, `1`
otherwise.

Chroma equalization, which compensates the bandwidth the separation
cascade costs chroma.

- `0` - off.
- `1` - a fixed inverse filter. The default wherever there is no
  Transform separation (the NTSC comb paths, and `dimensions=1`).
- `2` - leak-aware: steers chroma bandwidth per sample by the
  separator's own confidence, narrowing where luma may have leaked in.
  The default wherever a Transform separation is present, meaning PAL
  `dimensions` 2 or 3, or NTSC with `transform` nonzero. Requesting
  `eq=2` anywhere else is an error, since there is no confidence map to
  steer with.

`mask="confidence"` requires `eq=2`, so forcing `eq=0` or `eq=1` on a
Transform path makes that mask an error rather than a no-op.

The mode tables above measure each mode at its own default, so the comb
rows are `eq=1` and the Transform/hybrid rows `eq=2`.

**`evidence`** - float, `>= 0`. Default `0` (off). PAL only.

A low-frequency luma prior, after US 7,872,689: each bin pair is scaled
by how much low-frequency luma energy sits at the pair's baseband
difference frequency. The scene statistic behind it is that genuine
chroma detail co-locates with low-frequency luma detail while
cross-color does not, so raising `evidence` attenuates chroma that has
no such luma partner. Useful band is roughly `0.5`-`1.0`;
`1.0` measurably cuts residual chroma flicker on artifact-heavy PAL
footage (see the PAL table above). There is no enforced upper bound -
values far above `1` are accepted and simply get more aggressive.
Requires PAL with `dimensions` 2 or 3; anything else is an error.

**`cti`** - int `0`/`1` (AviSynth+: bool). Default `0` (off).

Chroma transient improvement: resynthesizes chroma edges from the
coincident luma edge, re-sharpening color transitions the composite
bandwidth softened. Helps on graphics, titles and test patterns, where
color edges are meant to be hard. It **hurts on natural footage**, where
real chroma edges are not luma-aligned - leave it off there. Requires
`dimensions` 2 or 3.

### Trained tables and their overrides

On a Transform path the separator decides, per frequency bin, how much
of the bin is chroma. By default it uses a **built-in trained soft-gain
table** (a per-bin gain curve fitted on a corpus) - this is what the
`default` rows in the tables above measure.

**The four parameters below replace that table, and passing any one of
them switches it off.** That includes passing `threshold=0.4`: `0.4` is
the value the hard-threshold mode falls back to, but supplying it
explicitly moves you from the trained soft gain to the hard per-bin test.
The two are visibly different - compare the PAL `trained 2D` and
`threshold 2D (threshold=0.4)` rows above. If you want the trained
table, pass none of these four.

The four are not gated alike. `thresholds`, `lut` and `level` error out
on a path with no Transform separation (the NTSC comb paths), but the
scalar `threshold` is accepted there and does nothing, since a comb path
has no bins to threshold and no trained table to override.

**`threshold`** - float, in `(0, 1]`. Default `0.4` when the
hard-threshold mode is selected.

The bin-symmetry ratio the hard test compares against: a bin pair whose
energies are more lopsided than this is judged luma and discarded from
chroma. Lower keeps less chroma (less cross-color, more color loss);
higher keeps more. `0` or a value above `1` is an error.

**`thresholds`** - float array, each in `(0, 1]`. Default unset.

Per-bin thresholds, replacing the single scalar `threshold`. The length
must match the bin count of the path exactly:

| path | `thresholds` length |
|---|---|
| `dimensions=2` (PAL) | 80 |
| PAL `dimensions=3` | 384 |
| NTSC `dimensions=3` | 768 |

Requires `dimensions` 2 or 3, and on NTSC a Transform path. Mutually
exclusive with `level` and with `lut`.

**`lut`** - float array, each in `[0, 1]`. Default unset (the built-in
trained table is used instead).

A full trained soft-gain table of your own: per bin, a 16-knot gain
curve over the bin-symmetry ratio, interpolated linearly, **knots
innermost** (bin 0's 16 knots, then bin 1's, and so on). Length is the
bin count times 16:

| path | bins | `lut` length |
|---|---|---|
| `dimensions=2` (PAL) | 80 | 1280 |
| PAL `dimensions=3` | 384 | 6144 |
| NTSC `dimensions=3` | 768 | 12288 |

A wrong length is an error naming the expected one. Requires
`dimensions` 2 or 3, and on NTSC a Transform path. Mutually exclusive
with `level` and with `thresholds`. To generate one, see the comments in
`test/calibrate_thresholds.py`, which derives both the thresholds and
the soft LUT from a corpus.

**`level`** - int, `0` or `1`. Default `0`.

Selects amplitude limiting instead of the threshold test: within each
bin pair the larger magnitude is clamped down to the smaller, with no
threshold involved (`threshold` is then unused). It is untrained and
therefore robust - it makes no assumption fitted to a corpus - which
makes it the safer choice on synthetic extremes, at a real cost on
natural footage (see the PAL `level 2D` row above). Requires
`dimensions` 2 or 3, and on NTSC a Transform path. Mutually exclusive
with `thresholds` and `lut`.

### `Restore` only

**`refine`** - int, `0`-`16`. Default `1`.

Y-only detail recovery: iterations of the constrained deconvolution
described in *About `refine`* above. `0` disables it. `1` is the tuned
default; `2`-`4` push harder and are worth trying on clean, detailed
sources. Values up to `16` are accepted. Chroma is never touched, so
`refine` has no effect on the cross-color / dot-crawl numbers in the
tables above.

**`precomb`** - int `0`/`1` (AviSynth+: bool). Default `0`. Also on
`Encode`.

Vertically low-passes U/V before modulation (Poynton's precombing), so
line-alternating chroma nulls exactly in a comb decoder. That helps only
when you are *making* a clean composite for a comb decoder to consume -
which is what `Encode` is for. On `Restore` it applies to the internal
re-encode, where it measurably hurts, because the point there is to
reconstruct the signal the bad decoder actually saw. Leave it off for
cleanup.

### Masks and outputs

**`mask`** - string, `"motion"` or `"confidence"`. Default unset (no
mask).

Emits a per-sample mask alongside the picture; see *Masks* above for
what each one means and how to use it. `"motion"` requires the NTSC
hybrid path exactly (`standard="ntsc"`, `dimensions=3`, `transform=2`);
`"confidence"` requires `eq=2`. Any other string is an error.

`mask` changes the shape of the output:

- **VapourSynth** - without `mask` the filter returns a single node, as
  usual. With `mask` it returns a two-element list, `[picture, mask]`;
  indexing a single node will not work, and vice versa.
- **AviSynth+** - without `mask` the output is `YUV444P16`. With `mask`
  it is `YUVA444P16`, the mask in the alpha plane, which you pull out
  with `ExtractA`.

## Recipes

### Motion-compensated chroma cleanup

Some chroma residue is genuinely modulated color that no spectral
separator can touch - but it is phase-incoherent along motion
(rainbows rotate frame to frame while real color stays put). A
motion-compensated chroma degrain *after* the decode cancels exactly
that, with
[vapoursynth-mvutensils](https://pypi.org/project/vapoursynth-mvutensils/):

```python
dec = core.composite.Restore(clip, standard="ntsc")
sup = core.mvu.Super(dec, blksize=16, overlap=8, pel=2)
vec = core.mvu.AnalyseMany(sup, radius=2)
out = core.mvu.Degrain(dec, sup, vec, planes=[1, 2], thsad=[400, 1600])
```

Vectors come from the already-clean decoded luma; only chroma is
touched; where vectors fail it falls back to the unprocessed pixel.
Strength saturates around `thsad=[400, 1600]`, `radius=2`. Keep it after
the decode - the round trip must see the artifacts untouched.

### Clean test composites

```python
comp = core.composite.Encode(clip, standard="ntsc", precomb=1)
```

`precomb=1` nulls line-alternating chroma exactly - the friendly choice
when the consumer is a comb decoder. Keep it off for cleanup.

## Installing

**VapourSynth:** drop `composite.so`/`composite.dll` in your plugins
autoload dir (or `core.std.LoadPlugin(...)`).

**AviSynth+:** load the same module. The AviSynth frontend registers
`composite_Encode`, `composite_Decode`, `composite_Restore`, and needs
**avsresize** (`z_ConvertFormat`) loaded at runtime for the internal
resampling - install it alongside.

## Building

```sh
meson setup build
ninja -C build
meson test -C build
```

Requires Meson, a C99 compiler, FFTW3 (single precision), and
VapourSynth (V4 API) with the Python module available for header
discovery. Add `-Davisynth=true` to also build the AviSynth+ frontend
into the same module (the release wheels enable it).

End-to-end plugin tests (run by hand; each needs its host at runtime):

```sh
# VapourSynth
python test/test_composite.py build/composite.so
```

The AviSynth+ end-to-end test is a C harness that links libavisynth, so
it is an opt-in build target - point `-Davisynth_lib_dir` at the
directory holding your `libavisynth.so.N`:

```sh
meson setup build -Davisynth=true -Davisynth_lib_dir=$LIBAVS
ninja -C build
# run against the built module and a working avsresize:
./build/test_composite_avs build/composite.so /path/to/avsresize.so
```

It links libavisynth to build and needs a *working* avsresize
(`z_ConvertFormat`) at runtime for the resampling paths; without one it
still runs the argument-validation checks. Note some avsresize builds
export `AVS_linkage` as a global and segfault on load - that is an
avsresize build issue (relink it with `-Wl,-Bsymbolic`), not this plugin.

## How it works

Round-tripping Y'CbCr through a composite encode and a good decode
removes cross-luma and cross-color artifacts baked in by a bad hardware
decoder: the re-encode reconstructs the composite signal the bad decoder
saw, and a Transform/comb decode re-separates it properly. The signal
processing follows
[ld-decode](https://github.com/happycube/ld-decode)'s ld-chroma-encoder
and ld-chroma-decoder, after Clarke, *Colour encoding and decoding
techniques for line-locked sampled PAL and NTSC television signals*,
BBC RD 1986/2, with separation modes from GB 2365247 A (Easterbrook)
and US 7,872,689 (Weston), and decoder adaptivity after Faroudja
(*NTSC and Beyond*, 1988).

## License

GPL-3.0-or-later. See COPYING.
