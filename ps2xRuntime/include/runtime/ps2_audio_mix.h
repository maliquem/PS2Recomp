#pragma once

#include "runtime/ps2_audio_turbo.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

namespace ps2x::audio
{
    // Output buses. The SPU2 core tags each voice at key-on (game profile); a game with no profile
    // plays everything on the effects bus. All buses are pushed in step (silence included).
    enum : int
    {
        kBusEffects = 0, // also the default bus
        kBusMusic = 1,
        kBusVoice = 2,
        kBusFmv = 3,
        kBusCount = 4,
    };

    // Player volumes, linear (1 = as the game mixed it). Any thread.
    extern std::atomic<float> g_masterGain;
    extern std::array<std::atomic<float>, kBusCount> g_busGain;
    // "Music at normal speed in turbo": the music bus skips the time compressor and the core is asked
    // for less music instead (musicRatio).
    extern std::atomic<bool> g_musicWallClock;

    // True when the music bus runs on the wall clock at this speed (g_speedPercent: 100 = normal,
    // 0 = unlimited). shortcut: slow motion (<100) keeps the music on game time, it would need the
    // sequencer to repeat ticks; upgrade when a game ships slow motion as a feature.
    bool musicOnWallClock(uint32_t speedPercent);

    // Host side of the SPU2 output: one FIFO per bus, a prebuffer that refills after an underrun (one
    // clean gap instead of a crackle), the stream rate that keeps the fill level on target, and the
    // final mix (per-bus volume, float sum, limiter only above full scale). At volume 1 and below full
    // scale the mix is sample for sample the sum of the buses. Not thread safe: the caller locks.
    class OutputMixer
    {
    public:
        static constexpr size_t kPrebufferFrames = 2304u; // 48 ms

        OutputMixer();
        void reset();

        void push(int bus, const int16_t *interleaved, size_t frames);
        // Always writes `frames` frames. speedPercent as ps2x::pacing::g_speedPercent.
        void pull(int16_t *out, size_t frames, uint32_t speedPercent);

        // Playback rate for the output stream (SDL frequency ratio): 1 +-0.5 %, from the smoothed
        // fill level of the default bus. Exactly 1 in turbo (the compressor follows the input there).
        [[nodiscard]] double rateRatio() const { return m_rate; }
        // Music frames to render per game frame for the next chunk: exactly 1 unless the music runs
        // on the wall clock, then what keeps the music FIFO on target.
        [[nodiscard]] double musicRatio(uint32_t speedPercent) const;

        [[nodiscard]] size_t buffered(int bus) const;
        [[nodiscard]] uint64_t underruns() const { return m_underruns; }

    private:
        struct Bus
        {
            std::unique_ptr<TurboCompressor> fifo;
            bool playing = false; // false: holding silence until the prebuffer fills
            bool fadeIn = false;
            std::array<int16_t, 2> last{}; // last frame handed out, where a gap's ramp starts
        };

        std::array<Bus, kBusCount> m_bus;
        std::vector<int16_t> m_scratch;
        std::vector<float> m_sum;
        bool m_direct = true; // last pull was at normal speed
        double m_level = 0.0; // smoothed fill of the default bus, frames
        double m_rate = 1.0;
        float m_limiterGain = 1.0f;
        uint32_t m_limiterHold = 0u; // frames left before the gain may recover
        uint64_t m_underruns = 0;
        std::array<std::shared_ptr<std::FILE>, kBusCount> m_dump; // PS2X_AUDIO_DUMP=<prefix>: <prefix>bus<N>.raw
    };
}
