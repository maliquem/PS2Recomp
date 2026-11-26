#include "MiniTest.h"
#include "runtime/ps2_audio_turbo.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_pacing.h"
#include "runtime/ps2_savestate.h"
#include "runtime/gs/gs_frontend.h"
#include "ps2x/state_archive.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
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

        tc.Run("Archive_unordered_map_bytes_do_not_depend_on_insertion_order", [](TestCase &t)
        {
            // Save, load, save again must give the same bytes, or a loaded state hashes differently.
            std::unordered_map<uint32_t, uint32_t> a, b;
            for (uint32_t i = 0; i < 200; ++i)
                a[i * 2654435761u] = i;
            for (uint32_t i = 200; i-- > 0;)
                b[i * 2654435761u] = i;
            std::vector<uint8_t> blobA, blobB, blobC;
            {
                ps2x::state::Archive ar(blobA);
                ar(a);
            }
            {
                ps2x::state::Archive ar(blobB);
                ar(b);
            }
            t.IsTrue(blobA == blobB, "same contents, same bytes");
            std::unordered_map<uint32_t, uint32_t> loaded;
            ps2x::state::Archive in(blobA.data(), blobA.size());
            in(loaded);
            t.IsTrue(loaded == a, "round trip keeps the contents");
            {
                ps2x::state::Archive ar(blobC);
                ar(loaded);
            }
            t.IsTrue(blobA == blobC, "a loaded map saves to the same bytes");
        });

        tc.Run("Archive_rejects_count_past_the_end", [](TestCase &t)
        {
            std::vector<uint32_t> v(4, 7u);
            std::vector<uint8_t> blob;
            {
                ps2x::state::Archive ar(blob);
                ar(v);
            }
            blob[0] = 0xFF; // count far larger than the bytes left
            std::vector<uint32_t> out;
            bool threw = false;
            try
            {
                ps2x::state::Archive in(blob.data(), blob.size());
                in(out);
            }
            catch (const std::runtime_error &)
            {
                threw = true;
            }
            t.IsTrue(threw, "a damaged count throws instead of allocating");
        });

        tc.Run("Frameskip_drops_only_back_buffer_and_stops_on_readback", [](TestCase &t)
        {
            using namespace ps2x::pacing;
            constexpr uint32_t kA = 0u, kB = 70u; // two display buffers (FBP units)
            constexpr uint64_t kFbw = 10ull << 9; // DISPFB FBW: 640 pixels
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GSRegisters regs{};
            regs.display1 = 223ull << 44; // 224 rows: a 640x224 buffer is 70 pages, so B sits right after A
            regs.smode2 = 3u;              // Shadow Hearts' game mode: the buffer is DH+1 rows
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), &regs);
            auto draw = [&](uint32_t fbp, bool textured = false, uint32_t tbp = 0u)
            {
                GSPrimitiveBatch batch{};
                batch.state.context.frame.fbp = fbp;
                batch.state.context.frame.fbw = 10u;
                batch.state.prim.tme = textured;
                batch.state.context.tex0.tbp0 = tbp;
                return gs.frameskipDropsDraw(batch);
            };

            g_frameskipWant.store(false);
            g_frameskipLatched.store(false);
            g_frameskipDraws.store(false);
            t.IsFalse(draw(kA), "off: nothing dropped");
            t.IsTrue(g_frameskipDisplayDrawn.load(), "off: the display counts as drawn");

            g_frameskipDraws.store(true);
            regs.dispfb1 = kA | kFbw;
            t.IsFalse(draw(kB), "first frame is drawn");
            g_frameskipLatched.store(true); // the host took the frame on display
            regs.dispfb1 = kB | kFbw;              // swap: A is the back buffer now
            t.IsTrue(draw(kA), "after the latch, back-buffer draws are dropped");
            t.IsFalse(draw(kB + 200u), "other render targets are drawn");
            regs.dispfb1 = kA | kFbw; // swap with a frame that lost draws
            t.IsTrue(draw(kB), "still dropping until the host asks");
            t.IsFalse(g_frameskipDisplayDrawn.load(), "a frame that lost draws is not latched");

            t.IsTrue(draw(kB, true, (kB + 70u) * 32u), "a texture right after the back buffer is not a read of it");
            t.IsFalse(draw(kB, true, (kB + 69u) * 32u), "a texture in the last page of the back buffer is a read of it");
            t.IsFalse(draw(kB + 200u, true, kB * 32u), "texturing from the back buffer is drawn");
            t.IsFalse(draw(kB), "once the game reads the back buffer, nothing more is dropped");

            g_frameskipDraws.store(false);
            t.IsFalse(draw(kA), "turbo off: drawn");
            regs.dispfb1 = kB | kFbw;
            t.IsFalse(draw(kA), "turbo off: drawn after the swap");
            t.IsFalse(draw(kA), "turbo off: drawn");
            t.IsFalse(g_frameskipDisplayDrawn.load(), "turbo off: the frame that lost draws stays unlatched over more draws");
            t.IsFalse(g_frameskipDisplayDrawn.load(), "turbo off: the frame that lost draws is still not latched");
            regs.dispfb1 = kA | kFbw;
            t.IsFalse(draw(kB), "turbo off: drawn");
            t.IsTrue(g_frameskipDisplayDrawn.load(), "turbo off: the next full frame is latched");
            t.IsFalse(draw(kB), "turbo off: back to idle");
            g_frameskipDropped.store(0);
            g_frameskipSubmitted.store(0);
        });

        tc.Run("StateFile_round_trip_and_rejections", [](TestCase &t)
        {
            namespace ss = ps2x::savestate;
            const auto dir = std::filesystem::temp_directory_path() / "ps2x_savestate_test";
            std::filesystem::create_directories(dir);
            const auto path = dir / "GAME.slot1.ps2xstate";
            const std::vector<uint8_t> payload(1000, 0x5A);
            ss::writeStateFile(path, payload, "GAME", 42u, 7u);
            t.IsTrue(ss::readStateFile(path, "GAME", 42u) == payload, "a written slot reads back");
            t.IsFalse(std::filesystem::exists(path.string() + ".tmp"), "no temporary file left");

            auto rejects = [&](const std::filesystem::path &file, const std::string &game, uint64_t exe, const std::string &why)
            {
                std::string message;
                try
                {
                    (void)ss::readStateFile(file, game, exe);
                }
                catch (const std::runtime_error &e)
                {
                    message = e.what();
                }
                t.IsTrue(message.find(why) != std::string::npos, "rejected with '" + why + "' (got '" + message + "')");
            };
            rejects(dir / "missing.ps2xstate", "GAME", 42u, "no save state");
            rejects(path, "OTHER", 42u, "another game");
            rejects(path, "GAME", 43u, "different build");

            auto damaged = [&](const char *name, auto change)
            {
                const auto copy = dir / name;
                std::filesystem::copy_file(path, copy, std::filesystem::copy_options::overwrite_existing);
                std::fstream f(copy, std::ios::in | std::ios::out | std::ios::binary);
                change(f);
                f.close();
                return copy;
            };
            rejects(damaged("magic.ps2xstate", [](std::fstream &f) { f.seekp(0); f.put('X'); }), "GAME", 42u, "not a save state");
            rejects(damaged("version.ps2xstate", [](std::fstream &f) { f.seekp(8); f.put(9); }), "GAME", 42u, "format");
            rejects(damaged("payload.ps2xstate", [](std::fstream &f) { f.seekp(-1, std::ios::end); f.put(0); }), "GAME", 42u,
                    "checksum");
            const auto cut = damaged("cut.ps2xstate", [](std::fstream &) {});
            std::filesystem::resize_file(cut, std::filesystem::file_size(cut) - 10u);
            rejects(cut, "GAME", 42u, "corrupt");
            std::error_code ec;
            std::filesystem::remove_all(dir, ec);
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
