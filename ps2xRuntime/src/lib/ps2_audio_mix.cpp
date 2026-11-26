// Host side of the SPU2 output (runtime/ps2_audio_mix.h).

#include "runtime/ps2_audio_mix.h"
#include "ps2_user_config.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

namespace ps2x::audio
{
    std::atomic<float> g_masterGain{1.0f};
    std::array<std::atomic<float>, kBusCount> g_busGain{{{1.0f}, {1.0f}, {1.0f}, {1.0f}}};
    std::atomic<bool> g_musicWallClock{true};

    namespace
    {
        constexpr size_t kRamp = 64u;                // declick ramp around a gap: 1.3 ms
        constexpr double kLevelSmooth = 1.0 / 94.0;  // fill level averaged over ~0.5 s of pushes
        constexpr double kRateMax = 0.005;           // +-0.5 %: 9 cents, not heard as pitch
        constexpr double kRateSpan = 960.0;          // full correction at 20 ms off target
        constexpr float kLimiterRelease = 1.0f / 2400.0f; // 50 ms
        constexpr uint32_t kLimiterHold = 480u;           // 10 ms: covers the gap between peaks down to 50 Hz
        // Music FIFO target while it runs on the wall clock: the prebuffer plus one device period.
        constexpr double kMusicTarget = static_cast<double>(OutputMixer::kPrebufferFrames) + 1024.0;
    }

    namespace
    {
        constexpr const char *kVolumeKeys[1 + kBusCount] = {"audio.volume", "audio.effects", "audio.music", "audio.voice", "audio.video"};
        std::array<int, 1 + kBusCount> g_volumePercent{100, 100, 100, 100, 100};

        void applyVolume(int index, int percent)
        {
            percent = std::clamp(percent, 0, 100);
            g_volumePercent[static_cast<size_t>(index)] = percent;
            const float gain = static_cast<float>(percent * percent) / 10000.0f; // squared: 50 % sounds like half
            (index == 0 ? g_masterGain : g_busGain[static_cast<size_t>(index - 1)]).store(gain, std::memory_order_relaxed);
        }
    }

    void loadSettings()
    {
        for (int i = 0; i <= kBusCount; ++i)
            applyVolume(i, ps2cfg::getInt(kVolumeKeys[i], 100));
        g_musicWallClock.store(ps2cfg::envBool("PS2X_MUSIC_TURBO", "audio.music_normal_speed_in_turbo", true), std::memory_order_relaxed);
    }

    int volumePercent(int index)
    {
        return index >= 0 && index <= kBusCount ? g_volumePercent[static_cast<size_t>(index)] : 0;
    }

    void setVolumePercent(int index, int percent)
    {
        if (index < 0 || index > kBusCount)
            return;
        applyVolume(index, percent);
        ps2cfg::setInt(kVolumeKeys[index], g_volumePercent[static_cast<size_t>(index)]);
        ps2cfg::save();
    }

    void setMusicWallClock(bool on)
    {
        g_musicWallClock.store(on, std::memory_order_relaxed);
        ps2cfg::setBool("audio.music_normal_speed_in_turbo", on);
        ps2cfg::save();
    }

    bool musicOnWallClock(uint32_t speedPercent)
    {
        return g_musicWallClock.load(std::memory_order_relaxed) && (speedPercent == 0u || speedPercent > 100u);
    }

    OutputMixer::OutputMixer()
    {
        for (auto &b : m_bus)
            b.fifo = std::make_unique<TurboCompressor>();
        if (const char *prefix = std::getenv("PS2X_AUDIO_DUMP"); prefix && *prefix)
            for (int b = 0; b < kBusCount; ++b)
                m_dump[static_cast<size_t>(b)].reset(std::fopen((std::string(prefix) + "bus" + std::to_string(b) + ".raw").c_str(), "wb"),
                                                     [](std::FILE *f) { if (f) std::fclose(f); });
    }

    void OutputMixer::reset()
    {
        for (auto &b : m_bus)
        {
            b.fifo->reset();
            b.playing = b.fadeIn = false;
            b.last = {0, 0};
        }
        m_direct = true;
        m_level = 0.0;
        m_rate = 1.0;
        m_limiterGain = 1.0f;
        m_limiterHold = 0u;
    }

    size_t OutputMixer::buffered(int bus) const
    {
        return m_bus[static_cast<size_t>(bus)].fifo->buffered();
    }

    void OutputMixer::push(int bus, const int16_t *in, size_t frames)
    {
        if (bus < 0 || bus >= kBusCount)
            return;
        Bus &b = m_bus[static_cast<size_t>(bus)];
        b.fifo->push(in, frames);
        if (std::FILE *f = m_dump[static_cast<size_t>(bus)].get())
            std::fwrite(in, 4u, frames, f);
        if (bus != kBusEffects)
            return;
        const double level = static_cast<double>(b.fifo->buffered());
        if (!m_direct)
        {
            m_rate = 1.0;
            return;
        }
        if (!b.playing)
        {
            m_level = level; // filling
            m_rate = 1.0;
            return;
        }
        m_level += (level - m_level) * kLevelSmooth;
        m_rate = 1.0 + std::clamp((m_level - static_cast<double>(kPrebufferFrames)) / kRateSpan, -1.0, 1.0) * kRateMax;
    }

    double OutputMixer::musicRatio(uint32_t speedPercent) const
    {
        if (!musicOnWallClock(speedPercent))
            return 1.0;
        // Feed-forward from the speed asked for (none when unlimited) plus a quarter of the FIFO error
        // per chunk; never above 1, the sequencer cannot run ahead of the game.
        const double forward = speedPercent ? 100.0 / static_cast<double>(speedPercent) : 0.0;
        const double error = kMusicTarget - static_cast<double>(buffered(kBusMusic));
        return std::clamp(forward + 0.25 * error / 256.0, 0.0, 1.0);
    }

    void OutputMixer::pull(int16_t *out, size_t frames, uint32_t speedPercent)
    {
        m_scratch.resize(frames * 2u);
        m_sum.assign(frames * 2u, 0.0f);
        const bool turbo = speedPercent != 100u;
        for (int bi = 0; bi < kBusCount; ++bi)
        {
            Bus &b = m_bus[static_cast<size_t>(bi)];
            int16_t *s = m_scratch.data();
            size_t got = 0;
            if (turbo && !(bi == kBusMusic && musicOnWallClock(speedPercent)))
            {
                // The compressor follows the input by itself; leaving turbo goes straight on playing.
                got = b.fifo->pull(s, frames, true);
                b.playing = true;
                b.fadeIn = false;
            }
            else
            {
                if (!b.playing && b.fifo->buffered() >= kPrebufferFrames)
                    b.playing = b.fadeIn = true;
                if (b.playing)
                {
                    got = b.fifo->pull(s, frames, false);
                    if (b.fadeIn && got > 0u)
                    {
                        const size_t n = std::min(got, kRamp);
                        for (size_t i = 0; i < n; ++i)
                            for (size_t c = 0; c < 2u; ++c)
                                s[i * 2u + c] = static_cast<int16_t>(static_cast<int32_t>(s[i * 2u + c]) * static_cast<int32_t>(i) / static_cast<int32_t>(kRamp));
                        b.fadeIn = false;
                    }
                    if (got > 0u)
                        b.last = {s[(got - 1u) * 2u], s[(got - 1u) * 2u + 1u]};
                    if (got < frames)
                    {
                        // Ran dry: slide from the last sample to silence, then hold until the prebuffer refills.
                        const size_t n = std::min(frames - got, kRamp);
                        for (size_t i = 0; i < n; ++i)
                            for (size_t c = 0; c < 2u; ++c)
                                s[(got + i) * 2u + c] = static_cast<int16_t>(static_cast<int32_t>(b.last[c]) * static_cast<int32_t>(kRamp - 1u - i) / static_cast<int32_t>(kRamp));
                        got += n;
                        b.last = {0, 0};
                        b.playing = false;
                        if (bi == kBusEffects)
                            ++m_underruns;
                    }
                }
            }
            const float gain = g_busGain[static_cast<size_t>(bi)].load(std::memory_order_relaxed);
            for (size_t i = 0; i < got * 2u; ++i)
                m_sum[i] += static_cast<float>(s[i]) * gain;
        }
        if (m_direct == turbo)
        {
            // Speed changed: the fill level means something only in direct play, restart its average.
            m_direct = !turbo;
            m_level = static_cast<double>(buffered(kBusEffects));
        }

        // Final gain and limiter: instant attack to the gain that fits the peak, held 10 ms (so the gain
        // does not creep back between the peaks of a tone and flatten each one), then a slow release.
        // The gain stays exactly 1 while the mix fits, so nothing is touched below full scale.
        // shortcut: no look-ahead, the first peak of a loud passage is flattened once; add a 1 ms
        // delay line if that is ever heard.
        const float master = g_masterGain.load(std::memory_order_relaxed);
        for (size_t f = 0; f < frames; ++f)
        {
            const float l = m_sum[f * 2u] * master;
            const float r = m_sum[f * 2u + 1u] * master;
            const float peak = std::max(std::fabs(l), std::fabs(r));
            if (m_limiterHold != 0u)
                --m_limiterHold;
            else if (m_limiterGain < 1.0f)
                m_limiterGain = std::min(1.0f, m_limiterGain + (1.0f - m_limiterGain) * kLimiterRelease + 1e-6f);
            if (peak * m_limiterGain > 32767.0f)
            {
                m_limiterGain = 32767.0f / peak;
                m_limiterHold = kLimiterHold;
            }
            out[f * 2u] = static_cast<int16_t>(std::lrintf(std::clamp(l * m_limiterGain, -32768.0f, 32767.0f)));
            out[f * 2u + 1u] = static_cast<int16_t>(std::lrintf(std::clamp(r * m_limiterGain, -32768.0f, 32767.0f)));
        }
    }
}
