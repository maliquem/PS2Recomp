#ifndef PS2_GS_COMMON_H
#define PS2_GS_COMMON_H

#include "runtime/gs/gs_types.h"
#include <cstdint>

namespace GSInternal
{
static inline uint32_t bitsPerPixel(uint8_t psm)
{
    switch (psm)
    {
    case GS_PSM_CT32:
    case GS_PSM_Z32:
        return 32;
    case GS_PSM_CT24:
    case GS_PSM_Z24:
        return 32;
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
    case GS_PSM_Z16:
    case GS_PSM_Z16S:
        return 16;
    case GS_PSM_T8:
    case GS_PSM_T8H:
        return 8;
    case GS_PSM_T4:
    case GS_PSM_T4HL:
    case GS_PSM_T4HH:
        return 4;
    default:
        return 32;
    }
}

static inline uint32_t fbStride(uint32_t fbw, uint8_t psm)
{
    uint32_t pixelsPerRow = fbw * 64u;
    return pixelsPerRow * (bitsPerPixel(psm) / 8u);
}

static inline uint32_t framePageBaseToBlock(uint32_t fbp)
{
    return fbp << 5u;
}

static inline int clampInt(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// Average of the texels (packed RGBA8) the two interlace fields show at one pixel. With texture alpha in use the
// colours are weighted by alpha: the same as blending each field over the background and averaging the results.
static inline uint32_t blendFieldTexels(uint32_t a, uint32_t b, bool alphaWeighted)
{
    const uint32_t aa = a >> 24, ab = b >> 24, ws = aa + ab;
    uint32_t out = 0u;
    for (uint32_t shift = 0u; shift < 24u; shift += 8u)
    {
        const uint32_t ca = (a >> shift) & 255u, cb = (b >> shift) & 255u;
        out |= ((alphaWeighted && ws != 0u) ? (ca * aa + cb * ab + (ws >> 1)) / ws : (ca + cb + 1u) >> 1) << shift;
    }
    return out | (((ws + 1u) >> 1) << 24);
}

static inline uint8_t clampU8(int v)
{
    if (v < 0) return 0;
    if (v > 255) return 255;
    return static_cast<uint8_t>(v);
}

// The GS line walk: Bresenham from v0 to v1 (offset removed), colour/z/fog stepped per pixel, iip=0 takes v1's colour.
// The texture is not sampled. Shared by the CPU backend (pixel by pixel) and the Vulkan backend (as runs).
template <typename Fn>
static inline void walkLine(const GSPrimitiveBatch &batch, Fn &&fn)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const int ofx = state.context.xyoffset.ofx >> 4;
    const int ofy = state.context.xyoffset.ofy >> 4;
    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    const int x1 = static_cast<int>(v1.x) - ofx;
    const int y1 = static_cast<int>(v1.y) - ofy;

    const int dx = x1 > x0 ? x1 - x0 : x0 - x1;
    const int dy = -(y1 > y0 ? y1 - y0 : y0 - y1);
    const int sx = (x0 < x1) ? 1 : -1;
    const int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;
    const int totalSteps = (dx > -dy ? dx : -dy) == 0 ? 1 : (dx > -dy ? dx : -dy);

    for (int step = 0;; ++step)
    {
        const float t = static_cast<float>(step) / static_cast<float>(totalSteps);
        uint8_t r = v1.r, g = v1.g, b = v1.b, a = v1.a;
        if (state.prim.iip)
        {
            r = clampU8(static_cast<int>(v0.r + (v1.r - v0.r) * t));
            g = clampU8(static_cast<int>(v0.g + (v1.g - v0.g) * t));
            b = clampU8(static_cast<int>(v0.b + (v1.b - v0.b) * t));
            a = clampU8(static_cast<int>(v0.a + (v1.a - v0.a) * t));
        }
        const double z = v0.z + (v1.z - v0.z) * t;
        const uint8_t fog = clampU8(static_cast<int>(v0.fog + (v1.fog - v0.fog) * t));
        fn(x0, y0, static_cast<uint32_t>(z), r, g, b, a, fog);

        if (x0 == x1 && y0 == y1)
            break;
        const int e2 = 2 * err;
        if (e2 >= dy)
        {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx)
        {
            err += dx;
            y0 += sy;
        }
    }
}

// A row or column of line pixels with the same colour, z and fog (inclusive bounds).
struct LineRun
{
    int x0, y0, x1, y1;
    uint32_t z;
    uint8_t r, g, b, a, fog;
};

// walkLine merged into runs; each pixel is still covered exactly once.
template <typename Fn>
static inline void lineRuns(const GSPrimitiveBatch &batch, Fn &&fn)
{
    LineRun run{};
    bool open = false;
    walkLine(batch, [&](int x, int y, uint32_t z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog)
    {
        if (open && z == run.z && r == run.r && g == run.g && b == run.b && a == run.a && fog == run.fog)
        {
            if (y == run.y0 && y == run.y1 && (x == run.x1 + 1 || x == run.x0 - 1))
            {
                run.x0 = x < run.x0 ? x : run.x0;
                run.x1 = x > run.x1 ? x : run.x1;
                return;
            }
            if (x == run.x0 && x == run.x1 && (y == run.y1 + 1 || y == run.y0 - 1))
            {
                run.y0 = y < run.y0 ? y : run.y0;
                run.y1 = y > run.y1 ? y : run.y1;
                return;
            }
        }
        if (open)
            fn(run);
        run = {x, y, x, y, z, r, g, b, a, fog};
        open = true;
    });
    if (open)
        fn(run);
}
}

#endif
