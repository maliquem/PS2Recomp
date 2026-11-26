#pragma once

#include <atomic>

// Movie (FMV) quality options (Shadow Hearts plan 4.1, study 08). The SDL host loads them on the main thread
// (PS2X_FMV* env vars win over the [video] fmv_* keys of the player TOML) and the pause menu changes them; the MPEG
// decoder (EE thread) and the presenter read them. The defaults below hold when nothing loads them (raylib host).
// PS2X_FMV=0 turns every step off: the output is today's, bit for bit.
namespace ps2fmv
{
    inline std::atomic<bool> enabled{true};
    // Step 1: bicubic chroma, accurate rounding, MPEG-2 chroma siting, BT.601 limited range in the YUV -> RGB.
    inline std::atomic<bool> csc{true};
    // Step 2: how the presenter scales a movie frame to the screen (fmv_scale.comp). Pixel = today's nearest blit.
    enum Scale : int
    {
        Pixel,
        Bilinear,
        Bicubic,
        Lanczos,
        ScaleCount
    };
    inline constexpr const char *kScaleNames[ScaleCount] = {"pixel", "bilinear", "bicubic", "lanczos"};
    inline std::atomic<int> scale{Lanczos};
    // Step 5: smooth 8-bit banding in flat gradients, with dither (needs a filter other than Pixel).
    inline std::atomic<bool> deband{false};
    // Step 3: FSR1 RCAS sharpening after the scale (off by default: 1.0 brought out blocks and grain in the photos).
    inline constexpr const char *kSharpenNames[] = {"off", "light", "medium"};
    inline constexpr float kSharpenAmounts[] = {0.0f, 0.25f, 0.5f};
    inline std::atomic<int> sharpen{0};

    inline bool cscOn() { return enabled.load(std::memory_order_relaxed) && csc.load(std::memory_order_relaxed); }
    inline int scaleFilter() { return enabled.load(std::memory_order_relaxed) ? scale.load(std::memory_order_relaxed) : Pixel; }
}
