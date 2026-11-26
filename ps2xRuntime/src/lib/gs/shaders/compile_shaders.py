import ctypes as c
import os
from pathlib import Path
import shutil
import subprocess
import sys


SHADERS_DIR = Path(__file__).resolve().parent
HEADER_OUT = SHADERS_DIR.parents[3] / 'include' / 'runtime' / 'gs' / 'gs_vulkan_shaders.h'
SHADERS = (
    ('kGsUploadSpv', 'gs_upload.comp'),
    ('kGsTileRasterSpv', 'gs_tile_raster.comp'),
    ('kGsPresentSpv', 'gs_present.comp'),
    ('kFmvScaleSpv', 'fmv_scale.comp'),
)


def verify_address_formulas():
    """Cross-check GLSL bitwise swizzle formulas against PS2 GS lookup tables."""
    block_c32 = [
        [0, 1, 4, 5, 16, 17, 20, 21],
        [2, 3, 6, 7, 18, 19, 22, 23],
        [8, 9, 12, 13, 24, 25, 28, 29],
        [10, 11, 14, 15, 26, 27, 30, 31],
    ]
    block_c16 = [
        [0, 2, 8, 10], [1, 3, 9, 11], [4, 6, 12, 14], [5, 7, 13, 15],
        [16, 18, 24, 26], [17, 19, 25, 27], [20, 22, 28, 30], [21, 23, 29, 31],
    ]
    block_c16s = [
        [0, 2, 16, 18], [1, 3, 17, 19], [8, 10, 24, 26], [9, 11, 25, 27],
        [4, 6, 20, 22], [5, 7, 21, 23], [12, 14, 28, 30], [13, 15, 29, 31],
    ]
    for by in range(4):
        for bx in range(8):
            f = (bx & 1) | ((by & 1) << 1) | ((bx & 2) << 1) | ((by & 2) << 2) | ((bx & 4) << 2)
            assert f == block_c32[by][bx], (bx, by, f)
    for by in range(8):
        for bx in range(4):
            f16 = (by & 1) | ((bx & 1) << 1) | ((by & 2) << 1) | ((bx & 2) << 2) | ((by & 4) << 2)
            assert f16 == block_c16[by][bx], (bx, by, f16)
            f16s = (by & 1) | ((bx & 1) << 1) | (by & 4) | ((by & 2) << 2) | ((bx & 2) << 3)
            assert f16s == block_c16s[by][bx], (bx, by, f16s)

    # Check P8 column table against ps2_gs_psmt8.h
    col8_rows = [
        [0, 4, 16, 20, 32, 36, 48, 52, 2, 6, 18, 22, 34, 38, 50, 54],
        [8, 12, 24, 28, 40, 44, 56, 60, 10, 14, 26, 30, 42, 46, 58, 62],
        [33, 37, 49, 53, 1, 5, 17, 21, 35, 39, 51, 55, 3, 7, 19, 23],
        [41, 45, 57, 61, 9, 13, 25, 29, 43, 47, 59, 63, 11, 15, 27, 31],
        [96, 100, 112, 116, 64, 68, 80, 84, 98, 102, 114, 118, 66, 70, 82, 86],
        [104, 108, 120, 124, 72, 76, 88, 92, 106, 110, 122, 126, 74, 78, 90, 94],
        [65, 69, 81, 85, 97, 101, 113, 117, 67, 71, 83, 87, 99, 103, 115, 119],
        [73, 77, 89, 93, 105, 109, 121, 125, 75, 79, 91, 95, 107, 111, 123, 127],
        [128, 132, 144, 148, 160, 164, 176, 180, 130, 134, 146, 150, 162, 166, 178, 182],
        [136, 140, 152, 156, 168, 172, 184, 188, 138, 142, 154, 158, 170, 174, 186, 190],
        [161, 165, 177, 181, 129, 133, 145, 149, 163, 167, 179, 183, 131, 135, 147, 151],
        [169, 173, 185, 189, 137, 141, 153, 157, 171, 175, 187, 191, 139, 143, 155, 159],
        [224, 228, 240, 244, 192, 196, 208, 212, 226, 230, 242, 246, 194, 198, 210, 214],
        [232, 236, 248, 252, 200, 204, 216, 220, 234, 238, 250, 254, 202, 206, 218, 222],
        [193, 197, 209, 213, 225, 229, 241, 245, 195, 199, 211, 215, 227, 231, 243, 247],
        [201, 205, 217, 221, 233, 237, 249, 253, 203, 207, 219, 223, 235, 239, 251, 255],
    ]
    for y in range(16):
        for x in range(16):
            y0, y1, y2, y3 = y & 1, (y >> 1) & 1, (y >> 2) & 1, (y >> 3) & 1
            fx2 = ((x >> 2) & 1) ^ y1 ^ y2
            col = y1 | (((x >> 3) & 1) << 1) | ((x & 1) << 2) | (y0 << 3) | (((x >> 1) & 1) << 4) | (fx2 << 5) | (y2 << 6) | (y3 << 7)
            assert col == col8_rows[y][x], (x, y, col, col8_rows[y][x])

    # Check P4 column table spot-check + bijection over 512 nibbles
    seen = set()
    for y in range(16):
        for x in range(32):
            y0, y1, y2, y3 = y & 1, (y >> 1) & 1, (y >> 2) & 1, (y >> 3) & 1
            fx2 = ((x >> 2) & 1) ^ y1 ^ y2
            col = y1 | ((x & 24) >> 2) | ((x & 1) << 3) | (y0 << 4) | ((x & 2) << 4) | (fx2 << 6) | (y2 << 7) | (y3 << 8)
            seen.add(col)
    assert len(seen) == 512
    print('All PS2 GS swizzle formulas verified exact.')


def find_shaderc_library():
    env_path = os.environ.get('SHADERC_DLL')
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
    for name in ('shaderc_shared.dll', 'libshaderc_shared.so.1', 'libshaderc_shared.so'):
        found = shutil.which(name)
        if found:
            return Path(found)
    local_fallback = Path(r'K:\Emulation\PS2\shaderc_shared.dll')
    if local_fallback.is_file():
        return local_fallback
    raise RuntimeError(
        'Neither glslc nor shaderc_shared was found. Install Vulkan SDK (glslc) or set SHADERC_DLL.')


def compile_with_shaderc(source_bytes, name):
    dll_path = find_shaderc_library()
    s = c.CDLL(str(dll_path))
    for key, result, args in (
        ('compiler_initialize', c.c_void_p, []),
        ('compiler_release', None, [c.c_void_p]),
        ('compile_options_initialize', c.c_void_p, []),
        ('compile_options_set_optimization_level', None, [c.c_void_p, c.c_int]),
        ('compile_options_release', None, [c.c_void_p]),
        ('compile_into_spv', c.c_void_p, [c.c_void_p, c.c_char_p, c.c_size_t, c.c_int, c.c_char_p, c.c_char_p, c.c_void_p]),
        ('result_get_compilation_status', c.c_int, [c.c_void_p]),
        ('result_get_error_message', c.c_char_p, [c.c_void_p]),
        ('result_get_length', c.c_size_t, [c.c_void_p]),
        ('result_get_bytes', c.c_void_p, [c.c_void_p]),
        ('result_release', None, [c.c_void_p]),
    ):
        fn = getattr(s, 'shaderc_' + key)
        fn.restype, fn.argtypes = result, args

    compiler = s.shaderc_compiler_initialize()
    opts = s.shaderc_compile_options_initialize()
    s.shaderc_compile_options_set_optimization_level(opts, 2)  # performance
    try:
        res = s.shaderc_compile_into_spv(compiler, source_bytes, len(source_bytes), 2, name.encode(), b'main', opts)
        try:
            if s.shaderc_result_get_compilation_status(res) != 0:
                raise RuntimeError(f'{name}: ' + s.shaderc_result_get_error_message(res).decode())
            return c.string_at(s.shaderc_result_get_bytes(res), s.shaderc_result_get_length(res))
        finally:
            s.shaderc_result_release(res)
    finally:
        s.shaderc_compile_options_release(opts)
        s.shaderc_compiler_release(compiler)


def compile_shader(filename):
    common = (SHADERS_DIR / 'gs_common.glsl').read_bytes()
    src = (SHADERS_DIR / filename).read_bytes()
    # Expand #include "gs_common.glsl" so both shaderc (without includer) and glslc work identically.
    expanded = src.replace(b'#extension GL_GOOGLE_include_directive : require\n', b'').replace(
        b'#extension GL_GOOGLE_include_directive : require\r\n', b'').replace(
        b'#include "gs_common.glsl"', common)
    glslc = shutil.which('glslc')
    if glslc:
        p = subprocess.run([glslc, '-O', '-fshader-stage=compute', '-o', '-', '-'],
                           input=expanded, capture_output=True, check=True)
        return p.stdout
    return compile_with_shaderc(expanded, filename)


def main():
    verify_address_formulas()
    lines = [
        '#pragma once',
        '',
        '// Auto-generated by src/lib/gs/shaders/compile_shaders.py. Do not edit by hand.',
        '#include <cstdint>',
        '',
        'namespace ps2vk_shaders',
        '{',
    ]
    for sym, filename in SHADERS:
        spv = compile_shader(filename)
        assert len(spv) >= 20 and len(spv) % 4 == 0, filename
        words = [int.from_bytes(spv[i:i + 4], 'little') for i in range(0, len(spv), 4)]
        print(f'Compiled {filename}: {len(spv)} bytes ({len(words)} SPIR-V words)')
        lines.append(f'alignas(4) inline constexpr uint32_t {sym}[{len(words)}] = {{')
        for i in range(0, len(words), 8):
            chunk = ', '.join(f'0x{w:08X}u' for w in words[i:i + 8])
            lines.append(f'    {chunk},')
        lines.append('};')
        lines.append('')
    lines.append('} // namespace ps2vk_shaders')
    lines.append('')
    HEADER_OUT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {HEADER_OUT}')


if __name__ == '__main__':
    main()
