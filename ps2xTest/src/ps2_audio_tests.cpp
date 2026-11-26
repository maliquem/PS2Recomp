#include "MiniTest.h"
#include "runtime/ps2_audio_mix.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace
{
    using ps2x::audio::OutputMixer;

    std::vector<int16_t> tone(size_t frames, size_t &phase, double hz, double amp)
    {
        std::vector<int16_t> pcm(frames * 2u);
        for (size_t f = 0; f < frames; ++f, ++phase)
        {
            const auto v = static_cast<int16_t>(amp * std::sin(6.283185307 * hz * static_cast<double>(phase) / 48000.0));
            pcm[f * 2u] = v;
            pcm[f * 2u + 1u] = v;
        }
        return pcm;
    }

    int maxStep(const std::vector<int16_t> &pcm)
    {
        int m = 0;
        for (size_t i = 0; i + 2u < pcm.size(); i += 2u)
            m = std::max(m, std::abs(pcm[i + 2u] - pcm[i]));
        return m;
    }
}

void register_ps2_audio_tests()
{
    MiniTest::Case("PS2Audio", [](TestCase &tc)
    {
        tc.Run("OutputMixer_holds_the_level_through_slow_production_and_stalls", [](TestCase &t)
        {
            // The pattern measured in Shadow Hearts: 47,937 frames/s made in bursts of one video field
            // (the device wants 48,000), and two scene loads that stop production for 132 and 62 ms.
            auto m = std::make_unique<OutputMixer>();
            std::vector<int16_t> chunk(256u * 2u, 1000), out(1024u * 2u);
            const double field = 1.0 / 59.92, perField = 47937.0 / 59.92;
            double made = 0.0, nextField = 0.0, consumed = 0.0, minRate = 1.0, maxRate = 1.0, levelSum = 0.0;
            size_t levels = 0;
            for (int tick = 0; tick < 3000; ++tick) // 30 s in device periods of 10 ms
            {
                const double now = tick * 0.010;
                while (nextField <= now)
                {
                    const bool stalled = (nextField >= 5.0 && nextField < 5.132) || (nextField >= 12.0 && nextField < 12.062);
                    if (!stalled)
                        for (made += perField; made >= 256.0; made -= 256.0)
                            m->push(ps2x::audio::kBusEffects, chunk.data(), 256u);
                    nextField += field;
                }
                consumed += 480.0 * m->rateRatio();
                const auto frames = static_cast<size_t>(consumed);
                consumed -= static_cast<double>(frames);
                m->pull(out.data(), frames, 100u);
                minRate = std::min(minRate, m->rateRatio());
                maxRate = std::max(maxRate, m->rateRatio());
                if (now > 20.0)
                {
                    levelSum += static_cast<double>(m->buffered(ps2x::audio::kBusEffects));
                    ++levels;
                }
            }
            const double level = levelSum / static_cast<double>(levels);
            t.IsTrue(m->underruns() <= 2u, "only the two stalls longer than the prebuffer leave a gap (" + std::to_string(m->underruns()) + ")");
            t.IsTrue(minRate >= 0.995 - 1e-9 && maxRate <= 1.005 + 1e-9, "rate stays within +-0.5 %");
            t.IsTrue(level > 1200.0 && level < 3000.0, "fill level settles near the 48 ms target (" + std::to_string(level) + " frames)");
        });

        tc.Run("OutputMixer_without_rate_control_the_same_pattern_keeps_dropping", [](TestCase &t)
        {
            // Control for the test above: the same production against a fixed 48 kHz pull runs dry again and again.
            auto m = std::make_unique<OutputMixer>();
            std::vector<int16_t> chunk(256u * 2u, 1000), out(480u * 2u);
            const double field = 1.0 / 59.92, perField = 47937.0 / 59.92;
            double made = 0.0, nextField = 0.0;
            for (int tick = 0; tick < 12000; ++tick)
            {
                for (; nextField <= tick * 0.010; nextField += field)
                    for (made += perField; made >= 256.0; made -= 256.0)
                        m->push(ps2x::audio::kBusEffects, chunk.data(), 256u);
                m->pull(out.data(), 480u, 100u);
            }
            t.IsTrue(m->underruns() >= 2u, "a fixed-rate pull underruns (" + std::to_string(m->underruns()) + ")");
        });

        tc.Run("OutputMixer_mix_is_the_exact_sum_and_a_gap_does_not_click", [](TestCase &t)
        {
            auto m = std::make_unique<OutputMixer>();
            size_t pa = 0, pb = 0;
            const auto a = tone(9600u, pa, 440.0, 9000.0);
            const auto b = tone(9600u, pb, 1000.0, 7000.0);
            m->push(ps2x::audio::kBusEffects, a.data(), 9600u);
            m->push(ps2x::audio::kBusMusic, b.data(), 9600u);
            std::vector<int16_t> out(12000u * 2u);
            m->pull(out.data(), 12000u, 100u); // more than there is: runs dry at 9600
            bool exact = true;
            for (size_t i = 128u; i < 9600u * 2u; ++i) // past the 64-frame fade-in
                exact = exact && out[i] == a[i] + b[i];
            t.IsTrue(exact, "below full scale the mix is the sum of the buses, sample for sample");
            t.Equals(m->underruns(), uint64_t{1}, "running dry counts one underrun");
            // 440 Hz + 1 kHz at these levels move at most ~1440 per sample; a cut to silence would jump up to 16000.
            t.IsTrue(maxStep(out) < 1500, "the gap ramps to silence (max step " + std::to_string(maxStep(out)) + ")");
            bool silent = true;
            for (size_t i = (9600u + 64u) * 2u; i < out.size(); ++i)
                silent = silent && out[i] == 0;
            t.IsTrue(silent, "silence after the ramp");
        });

        tc.Run("OutputMixer_limiter_only_acts_above_full_scale", [](TestCase &t)
        {
            auto m = std::make_unique<OutputMixer>();
            size_t pa = 0, pb = 0;
            const auto a = tone(24000u, pa, 440.0, 30000.0);
            const auto b = tone(24000u, pb, 440.0, 30000.0); // in phase: the sum reaches 60000
            m->push(ps2x::audio::kBusEffects, a.data(), 24000u);
            m->push(ps2x::audio::kBusFmv, b.data(), 24000u);
            std::vector<int16_t> out(24000u * 2u);
            m->pull(out.data(), 24000u, 100u);
            int peak = 0, wraps = 0;
            for (size_t i = 2u; i < out.size(); i += 2u)
            {
                peak = std::max(peak, std::abs(static_cast<int>(out[i])));
                wraps += std::abs(out[i] - out[i - 2u]) > 8000 ? 1 : 0; // a wrap or a hard clip edge
            }
            t.IsTrue(peak > 30000, "the limited mix still uses the full scale (peak " + std::to_string(peak) + ")");
            t.Equals(wraps, 0, "no wrap-around or jump");
            // Flat tops are what a hard clip leaves; the limiter keeps the sine's shape.
            int flat = 0;
            for (size_t i = 4800u * 2u; i + 2u < out.size(); i += 2u)
                flat += (std::abs(static_cast<int>(out[i])) >= 32767 && out[i] == out[i + 2u]) ? 1 : 0;
            t.IsTrue(flat < 24, "no clipped plateaus (" + std::to_string(flat) + " repeated full-scale samples)");

            // Player volume: half on one bus is half of that bus in the mix.
            m->reset();
            ps2x::audio::g_busGain[ps2x::audio::kBusFmv].store(0.5f);
            pa = pb = 0;
            const auto c = tone(9600u, pa, 440.0, 8000.0);
            m->push(ps2x::audio::kBusFmv, c.data(), 9600u);
            m->push(ps2x::audio::kBusEffects, std::vector<int16_t>(9600u * 2u).data(), 9600u);
            m->pull(out.data(), 9600u, 100u);
            ps2x::audio::g_busGain[ps2x::audio::kBusFmv].store(1.0f);
            bool half = true;
            for (size_t i = 128u; i < 9600u * 2u; ++i)
                half = half && std::abs(out[i] - c[i] / 2) <= 1;
            t.IsTrue(half, "bus volume scales that bus");
        });

        tc.Run("OutputMixer_music_ratio_is_1_at_normal_speed_and_keeps_the_music_fifo_in_turbo", [](TestCase &t)
        {
            auto m = std::make_unique<OutputMixer>();
            t.Equals(m->musicRatio(100u), 1.0, "normal speed: the music clock is the game clock, exactly");
            t.Equals(m->musicRatio(50u), 1.0, "slow motion keeps the music on game time");
            for (const uint32_t speed : {300u, 0u}) // 3x asked for, and unlimited (really 3x here)
            {
                m->reset();
                size_t phase = 0;
                std::vector<int16_t> silence(256u * 2u), out(480u * 2u), played;
                double musicAcc = 0.0, made = 0.0;
                for (int tick = 0; tick < 1000; ++tick) // 10 s of device periods
                {
                    for (made += 480.0 * 3.0; made >= 256.0; made -= 256.0)
                    {
                        musicAcc += 256.0 * m->musicRatio(speed);
                        const auto frames = static_cast<size_t>(musicAcc);
                        musicAcc -= static_cast<double>(frames);
                        m->push(ps2x::audio::kBusEffects, silence.data(), 256u);
                        if (frames > 0u)
                            m->push(ps2x::audio::kBusMusic, tone(frames, phase, 440.0, 16384.0).data(), frames);
                    }
                    m->pull(out.data(), 480u, speed);
                    if (tick >= 100)
                        played.insert(played.end(), out.begin(), out.end());
                }
                const std::string at = " (speed " + std::to_string(speed) + ")";
                // 440 Hz at half scale moves at most ~943 per sample; a gap or a seam would jump far more.
                t.IsTrue(maxStep(played) < 1000, "the music plays on without gaps" + at + ", max step " + std::to_string(maxStep(played)));
                int peak = 0;
                for (size_t i = played.size() / 2u; i < played.size(); ++i)
                    peak = std::max(peak, std::abs(static_cast<int>(played[i])));
                t.IsTrue(peak > 16000, "and it is the music, not silence" + at);
                // 10 s of game at 3x made 30 s of game audio; the music made about 10 s.
                t.IsTrue(phase > 9u * 48000u && phase < 11u * 48000u, "music made at wall-clock rate" + at + ": " + std::to_string(phase) + " frames");
            }
        });

        tc.Run("MusicTickGate_runs_one_music_tick_per_tick_of_music_samples", [](TestCase &t)
        {
            // A 240 Hz sequencer (200 samples a tick). Game time advances 200 samples per tick; the
            // music bus advances 200 / speed of them.
            for (const double speed : {1.0, 2.0, 3.0, 2.5})
            {
                ps2x::audio::MusicTickGate gate;
                uint32_t game = 0xFFFFFF00u; // the counters wrap in the middle of the run
                double music = 4294967000.0;
                gate.lastGame = game;
                gate.lastMusic = static_cast<uint32_t>(static_cast<uint64_t>(music));
                int ran = 0;
                const int ticks = 2400;
                for (int i = 0; i < ticks; ++i)
                {
                    game += 200u;
                    music += 200.0 / speed;
                    ran += gate.skipTick(game, static_cast<uint32_t>(static_cast<uint64_t>(music)), 200) ? 0 : 1;
                }
                const std::string at = " at " + std::to_string(speed) + "x";
                if (speed == 1.0)
                    t.IsTrue(ran == ticks, "normal speed never skips" + at);
                t.IsTrue(std::abs(ran - ticks / speed) <= 2.0, "music ticks follow the music bus" + at + ": " + std::to_string(ran));
            }
            ps2x::audio::MusicTickGate gate;
            (void)gate.skipTick(1000u, 1000u, 200);
            (void)gate.skipTick(90000u, 2000u, 200); // a loaded state: the counters jump apart
            t.IsTrue(gate.debt <= 400, "a jump costs at most two ticks");
        });
    });
}
