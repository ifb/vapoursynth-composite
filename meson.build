project('composite', 'c',
    default_options: ['buildtype=release', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'c_std=c99'],
    license: 'GPL-3.0-or-later',
    license_files: 'COPYING',
    meson_version: '>=1.2.3',
    version: '2',
)

cc = meson.get_compiler('c')

# libm calls stay bit-exact without errno; keeping them errno-free lets
# the vectorizers treat lrintf and friends as pure and vectorize the
# quantization loops
add_project_arguments(cc.get_supported_arguments('-fno-math-errno'), language: 'c')

py = import('python').find_installation(pure: false)

incdir = include_directories(
    run_command(py, '-c', 'import vapoursynth as vs; print(vs.get_include())', check: true).stdout().strip(),
)

m_dep = cc.find_library('m', required: false)
# Static on macOS: a bundled libfftw3f.dylib carries Homebrew's own
# minimum-OS version, which is newer than the wheel's target, and the
# wheel repair step refuses to package it. Linking it in sidesteps both
# the version floor and the bundling.
fftw3f_dep = dependency('fftw3f', static: host_machine.system() == 'darwin')
threads_dep = dependency('threads')
core_deps = [m_dep, fftw3f_dep, threads_dep]

core_sources = files('src/subcarrier.c', 'src/encode.c', 'src/transform2d.c',
                     'src/transform3d.c', 'src/fft.c', 'src/cpu.c',
                     'src/fir_row.c', 'src/geom.c')

if host_machine.cpu_family() == 'x86_64'
    add_languages('nasm', native: false)
    nasm_args = [
        '-DARCH_X86_64=1', '-DPIC', '-Dprivate_prefix=comp',
        '-I' + meson.current_source_dir() / 'x86/',
    ]
    # Mach-O decorates C symbols with a leading underscore, which x86inc
    # applies only when PREFIX is defined
    if host_machine.system() == 'darwin'
        nasm_args += '-DPREFIX'
    endif
    add_project_arguments(nasm_args, language: 'nasm')
    core_sources += files('x86/cpu.asm', 'x86/pal_demod.asm', 'x86/fir_row.asm',
                          'x86/split3d.asm', 'x86/lut_gain.asm',
                          'x86/t3d_filter.asm', 'x86/comb2d.asm',
                          'x86/demod_quant.asm', 'x86/demod_rotate.asm',
                          'x86/encode_mod.asm', 'x86/fft.asm')
endif

# decode.c's float separation heuristics feed discrete candidate choices
# (ntsc_comb2d -> split3d compare), so FMA contraction there would change
# the 3D output and break the byte-identical SIMD contract. ISO C modes
# default this off, but pin it so the reference is stable under any -std
# (the STDC FP_CONTRACT pragma does not override -ffp-contract=fast on gcc).
decode_lib = static_library('composite_decode',
    files('src/decode.c'),
    c_args: cc.get_supported_arguments('-ffp-contract=off'),
    dependencies: core_deps,
    gnu_symbol_visibility: 'hidden',
    pic: true,
)

core = static_library('composite_core',
    core_sources,
    dependencies: core_deps,
    gnu_symbol_visibility: 'hidden',
    link_whole: decode_lib,
    pic: true,
)

# One module serves both hosts, as ffms2 does: each looks up only its own
# entry point (VapourSynthPluginInit2 / avisynth_c_plugin_init), so the
# AviSynth frontend is simply another source file in the same target. The
# AviSynth side needs the vendored, pinned avisynth_c.h (interface v12)
# and resamples through zimg via avsresize (z_ConvertFormat), a runtime
# dependency the user installs.
plugin_sources = files('src/composite.c', 'src/lut_tables.c')
plugin_incdirs = [incdir]

if get_option('avisynth')
    plugin_sources += files('src/composite_avs.c')
    plugin_incdirs += include_directories('third_party/avisynth')
endif

shared_module('composite',
    plugin_sources,
    dependencies: core_deps,
    gnu_symbol_visibility: 'hidden',
    include_directories: plugin_incdirs,
    install: true,
    install_dir: py.get_install_dir() / 'vapoursynth/plugins',
    link_with: core,
    name_prefix: '',
)

test('subcarrier', executable('test_subcarrier',
    files('test/test_subcarrier.c'),
    dependencies: core_deps,
    include_directories: include_directories('src'),
    link_with: core,
))

test('encode', executable('test_encode',
    files('test/test_encode.c'),
    dependencies: core_deps,
    include_directories: include_directories('src'),
    link_with: core,
))

test('transform2d', executable('test_transform2d',
    files('test/test_transform2d.c'),
    dependencies: core_deps + dependency('fftw3'),
    include_directories: include_directories('src'),
    link_with: core,
))

test('fft', executable('test_fft',
    files('test/test_fft.c'),
    dependencies: core_deps,
    include_directories: include_directories('src'),
    link_with: core,
))

test('decode', executable('test_decode',
    files('test/test_decode.c'),
    dependencies: core_deps,
    include_directories: include_directories('src'),
    link_with: core,
))

# AviSynth+ end-to-end test. It links an external libavisynth and, at
# runtime, drives the built plugin through a real avsresize — neither of
# which the repo vendors or CI provides — so it is an opt-in BUILD target
# (set -Davisynth_lib_dir=/path/to/libavisynth), not a `meson test` entry.
# Run it by hand like the VapourSynth test:
#   ./build/test_composite_avs build/composite.so /path/to/avsresize.so
avs_lib_dir = get_option('avisynth_lib_dir')
if avs_lib_dir != ''
    avisynth_dep = cc.find_library('avisynth', dirs: [avs_lib_dir])
    executable('test_composite_avs',
        files('test/test_composite_avs.c'),
        dependencies: avisynth_dep,
        include_directories: include_directories('third_party/avisynth'),
        build_rpath: avs_lib_dir,
    )
endif

checkasm_dep = dependency('checkasm', required: get_option('checkasm'), allow_fallback: true)
if checkasm_dep.found()
    checkasm_exe = executable('checkasm',
        files('test/checkasm/main.c', 'test/checkasm/pal_demod.c',
              'test/checkasm/fir_row.c', 'test/checkasm/split3d.c',
              'test/checkasm/comb2d.c', 'test/checkasm/demod_quant.c',
              'test/checkasm/demod_rotate.c', 'test/checkasm/encode_mod.c',
              'test/checkasm/lut_gain.c',
              'test/checkasm/t3d_filter.c', 'test/checkasm/fft.c'),
        dependencies: [checkasm_dep] + core_deps,
        include_directories: include_directories('src'),
        link_with: core,
    )
    test('checkasm', checkasm_exe)
endif
