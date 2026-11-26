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

    inline bool cscOn() { return enabled.load(std::memory_order_relaxed) && csc.load(std::memory_order_relaxed); }
}
