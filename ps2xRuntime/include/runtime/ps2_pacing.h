#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

// Host pacing of the guest clock (turbo / fast forward). The EE scheduler waits real time between
// VBlanks; turbo divides those waits. Guest cycle accounting is unchanged, so the guest sees the same
// schedule and only the wall clock between fields shrinks.
namespace ps2x::pacing
{
    // Guest speed in percent of real time: 100 = normal, 300 = 3x, 0 = unlimited.
    inline std::atomic<uint32_t> g_speedPercent{100u};

    inline std::chrono::nanoseconds scaleHostWait(std::chrono::nanoseconds wait) noexcept
    {
        const uint32_t percent = g_speedPercent.load(std::memory_order_relaxed);
        if (percent == 100u)
            return wait;
        if (percent == 0u)
            return std::chrono::nanoseconds(0);
        return wait * 100 / percent;
    }

    // Turbo frameskip level 2+ (GS::frameskipDropsDraw). The GS stops drawing into the back buffer of
    // frames the host will never show; the host only latches a frame whose drawing was kept.
    inline std::atomic<bool> g_frameskipDraws{false};        // EE: dropping allowed (turbo, level 2+, GPU path)
    inline std::atomic<bool> g_frameskipDisplayDrawn{true};  // GS: the frame on display was fully drawn
    inline std::atomic<bool> g_frameskipWant{false};         // host: needs a drawn frame
    inline std::atomic<bool> g_frameskipLatched{false};      // host: latched the drawn frame on display
    inline std::atomic<uint64_t> g_frameskipDropped{0};      // GS draws dropped / kept (turbo stats)
    inline std::atomic<uint64_t> g_frameskipSubmitted{0};
    inline std::atomic<uint64_t> g_frameskipSwaps{0};        // display swaps / frames entering Dropping
    inline std::atomic<uint64_t> g_frameskipDropFrames{0};
    inline std::atomic<int> g_frameskipReadsBack{0};         // 0 no, 1 back buffer textured, 2 read by a transfer
}
