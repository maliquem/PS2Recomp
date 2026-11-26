#include "MiniTest.h"
#include "runtime/ps2_audio_turbo.h"
#include "runtime/ps2_memory.h"
#include "ps2x/state_archive.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace
{
    std::vector<int16_t> sine(size_t frames, size_t &phase)
    {
        std::vector<int16_t> pcm(frames * 2u);
        for (size_t f = 0; f < frames; ++f, ++phase)
        {
            const auto v = static_cast<int16_t>(16384.0 * std::sin(6.283185307 * 440.0 * static_cast<double>(phase) / 48000.0));
            pcm[f * 2u] = v;
            pcm[f * 2u + 1u] = v;
        }
        return pcm;
    }
}

void register_ps2_savestate_tests()
{
    MiniTest::Case("PS2SaveState", [](TestCase &tc)
    {
        tc.Run("TurboCompressor_direct_is_fifo", [](TestCase &t)
        {
            auto c = std::make_unique<ps2x::audio::TurboCompressor>();
            size_t phase = 0;
            const auto in = sine(1000, phase);
            c->push(in.data(), 1000);
            std::vector<int16_t> out(2000);
            t.Equals(c->pull(out.data(), 1000, false), size_t{1000}, "direct pull returns what was pushed");
            t.IsTrue(out == in, "direct pull is sample for sample the input");
        });

        tc.Run("TurboCompressor_3x_keeps_pitch_without_pops", [](TestCase &t)
        {
            // Guest makes 3 blocks per callback block; output length tracks wall clock and stays smooth.
            auto c = std::make_unique<ps2x::audio::TurboCompressor>();
            size_t phase = 0;
            std::vector<int16_t> out;
            std::vector<int16_t> block(512u * 2u);
            for (int i = 0; i < 400; ++i)
            {
                const auto in = sine(512u * 3u, phase);
                c->push(in.data(), 512u * 3u);
                const size_t got = c->pull(block.data(), 512u, true);
                out.insert(out.end(), block.begin(), block.begin() + static_cast<std::ptrdiff_t>(got * 2u));
            }
            t.IsTrue(out.size() / 2u > 400u * 512u * 9u / 10u, "compressed output keeps up with the callback");
            // 440 Hz at half scale moves at most ~943 per sample; a seam would jump far more.
            int maxStep = 0;
            for (size_t i = 2u * 2048u; i + 2u < out.size(); i += 2u)
                maxStep = std::max(maxStep, std::abs(out[i + 2u] - out[i]));
            t.IsTrue(maxStep < 1300, "no discontinuity between grains (max step " + std::to_string(maxStep) + ")");
        });

        tc.Run("TurboCompressor_leaving_turbo_drops_backlog_without_pops", [](TestCase &t)
        {
            auto c = std::make_unique<ps2x::audio::TurboCompressor>();
            size_t phase = 0;
            std::vector<int16_t> out;
            std::vector<int16_t> block(512u * 2u);
            auto pullInto = [&](bool compress)
            {
                const size_t got = c->pull(block.data(), 512u, compress);
                out.insert(out.end(), block.begin(), block.begin() + static_cast<std::ptrdiff_t>(got * 2u));
            };
            for (int i = 0; i < 200; ++i)
            {
                const auto in = sine(512u * 8u, phase);
                c->push(in.data(), 512u * 8u);
                pullInto(true);
            }
            const auto burst = sine(16000u, phase); // turbo ends with a backlog
            c->push(burst.data(), 16000u);
            for (int i = 0; i < 50; ++i)
            {
                const auto in = sine(512u, phase);
                c->push(in.data(), 512u);
                pullInto(false);
            }
            std::vector<int16_t> rest(48000u * 2u);
            const size_t left = c->pull(rest.data(), 48000u, false);
            t.IsTrue(left < 4096u, "1x play does not lag behind after turbo (" + std::to_string(left) + " frames held)");
            int maxStep = 0;
            for (size_t i = 2u * 2048u; i + 2u < out.size(); i += 2u)
                maxStep = std::max(maxStep, std::abs(out[i + 2u] - out[i]));
            t.IsTrue(maxStep < 1300, "no discontinuity across the switch (max step " + std::to_string(maxStep) + ")");
        });

        tc.Run("PS2Memory_state_round_trip", [](TestCase &t)
        {
            auto mem = std::make_unique<PS2Memory>();
            t.IsTrue(mem->initialize(), "PS2Memory initialize");
            mem->write32(0x00001000u, 0xDEADBEEFu);
            std::vector<uint8_t> blob;
            {
                ps2x::state::Archive ar(blob);
                mem->stateIo(ar);
            }
            mem->write32(0x00001000u, 0x12345678u);
            ps2x::state::Archive in(blob.data(), blob.size());
            mem->stateIo(in);
            t.Equals(in.remaining(), size_t{0}, "load reads the whole blob");
            t.Equals(mem->read32(0x00001000u), 0xDEADBEEFu, "load restores RDRAM");
        });
    });
}
