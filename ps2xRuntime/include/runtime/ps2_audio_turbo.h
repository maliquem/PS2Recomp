#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace ps2x::audio
{
    // Turbo time compressor for one SPU2 PCM stream (48 kHz stereo S16), ported from the BoF4 recomp
    // (psxrecomp audio_turbo.c). Input is pushed as the guest makes it; the audio callback pulls. While
    // `compress` is true the output plays the input faster at the same pitch (WSOLA: Hann grains with
    // 50% overlap, each placed where it best matches the previous grain's continuation, so no pops);
    // otherwise it is a plain FIFO, sample for sample the input. The ratio is measured, so any turbo
    // speed works, unlimited included. Not thread safe: the caller locks.
    class TurboCompressor
    {
    public:
        void reset();
        void push(const int16_t *interleaved, size_t frames);
        // Writes up to `frames` frames; returns how many (fewer when the input ran out).
        size_t pull(int16_t *out, size_t frames, bool compress);

    private:
        static constexpr int64_t kCap = 1 << 17;   // input ring: ~2.7 s
        static constexpr int64_t kMask = kCap - 1;
        static constexpr int kN = 1024;            // grain: 21 ms
        static constexpr int kHs = kN / 2;         // synthesis hop
        static constexpr int kDelta = 512;         // search: +-10.7 ms
        static constexpr int kLc = kHs;            // compared length: the overlap
        static constexpr int64_t kMargin = kN + kDelta + kHs;
        static constexpr int64_t kMeasure = 4800;  // ratio measured over 0.1 s of output
        static constexpr int64_t kDirectCap = 24000; // direct mode keeps at most 0.5 s (as the old ring)

        float sample(int64_t i, int c) const { return static_cast<float>(m_in[static_cast<size_t>(i & kMask)][c]) * (1.0f / 32768.0f); }
        float mono(int64_t i) const { return static_cast<float>(m_in[static_cast<size_t>(i & kMask)][0]) + m_in[static_cast<size_t>(i & kMask)][1]; }
        float corr(int64_t p, int64_t nat, int step) const;
        int64_t bestPos(int64_t lo, int64_t hi, int64_t nat) const;
        size_t takeFinished(int16_t *out, size_t room);
        void measure(size_t asked);

        std::array<std::array<int16_t, 2>, kCap> m_in{};
        int64_t m_end = 0;   // frames pushed (absolute)
        int64_t m_rd = 0;    // direct: next frame handed out
        bool m_compressing = false;
        int64_t m_prev = 0;  // compressing: last grain's input start
        std::array<std::array<float, 2>, kHs> m_pend{};    // its falling half, under the next grain
        std::array<std::array<int16_t, 2>, kHs> m_fin{};   // a finished hop not yet handed out
        int m_finN = 0, m_finOff = 0;
        double m_ratio = 1.0;
        int64_t m_inSince = 0, m_outSince = 0;
    };
}
