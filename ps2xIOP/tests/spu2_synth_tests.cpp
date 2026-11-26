// Tests of the SPU2 core (emulator/core/spu2_synth.h) against values worked out from the hardware
// rules: the interpolation table PCSX2 ships, envelope times, the mixer chain and the bus split.
#include "emulator/core/spu2_synth.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{
    using ps2x::iop::detail::Spu2Synth;

    int g_failed = 0;

    void check(bool ok, const std::string &what)
    {
        if (!ok)
        {
            ++g_failed;
            std::cout << "FAIL: " << what << "\n";
        }
    }

    void w(Spu2Synth &s, uint32_t offset, uint16_t value) { s.write16(0x1F900000u + offset, value); }
    uint16_t r(const Spu2Synth &s, uint32_t offset) { return s.read16(0x1F900000u + offset); }

    constexpr uint32_t kVoiceRam = 0x3000u; // halfword address of the test sample

    // 16 bytes of ADPCM at kVoiceRam (+8 halfwords per block), filter 0, shift 0: nibble n decodes to n << 12.
    void upload(Spu2Synth &s, uint32_t block, uint8_t flags, const std::vector<uint8_t> &data14)
    {
        uint8_t bytes[16] = {0x00, flags};
        std::copy(data14.begin(), data14.end(), bytes + 2);
        w(s, 0x1A8u, 0u);
        w(s, 0x1AAu, static_cast<uint16_t>(kVoiceRam + block * 8u));
        s.dmaWrite(0, bytes, 16u);
    }

    // Master volumes of both cores at full, as a game's sound driver leaves them.
    std::unique_ptr<Spu2Synth> makeSynth()
    {
        auto s = std::make_unique<Spu2Synth>();
        for (const uint32_t reg : {0x760u, 0x762u, 0x788u, 0x78Au})
            w(*s, reg, 0x3FFFu);
        return s;
    }

    // Voice `v` of `core`: full volume, pitch 1:1, attack `attackShift` (linear), no decay, held
    // sustain, linear release with shift 10 (16 per sample).
    void setVoice(Spu2Synth &s, uint32_t core, uint32_t v, uint32_t attackShift)
    {
        const uint32_t base = core * 0x400u + v * 0x10u;
        w(s, base + 0x0u, 0x3FFFu);
        w(s, base + 0x2u, 0x3FFFu);
        w(s, base + 0x4u, 0x1000u);
        w(s, base + 0x6u, static_cast<uint16_t>((attackShift << 10u) | 0x00FFu));
        w(s, base + 0x8u, static_cast<uint16_t>((0x1Fu << 8u) | (3u << 6u) | 10u));
        w(s, core * 0x400u + 0x1C0u + v * 12u, 0u);
        w(s, core * 0x400u + 0x1C2u + v * 12u, static_cast<uint16_t>(kVoiceRam));
    }

    void keyOn(Spu2Synth &s, uint32_t core, uint32_t mask) { w(s, core * 0x400u + 0x1A0u, static_cast<uint16_t>(mask)); }
    void keyOff(Spu2Synth &s, uint32_t core, uint32_t mask) { w(s, core * 0x400u + 0x1A4u, static_cast<uint16_t>(mask)); }

    int mul15(int a, int b) { return (a * b) >> 15; }
    // What a core 0 voice sample becomes on its way out: envelope and voice volume at full, core 0
    // master, core 1 external input volume, core 1 master.
    int chain0(int v) { return mul15(mul15(mul15(mul15(mul15(v, 0x7FFF), 0x7FFE), 0x7FFE), 0x7FFF), 0x7FFE); }

    const std::vector<uint8_t> kFlat(14u, 0x77u); // every sample 0x7000

    void testInterpolationTable()
    {
        // Rows of PCSX2's table (interpTable in SPU2/Mixer.cpp).
        const int expected[5][5] = {{0, 0x12C7, 0x59B3, 0x1307, -1}, {1, 0x1288, 0x59B2, 0x1347, -1}, {15, 0x0F46, 0x593A, 0x1700, -1},
                                    {16, 0x0F0F, 0x592A, 0x1747, 0}, {21, 0x0E00, 0x58CB, 0x18B5, 0}};
        for (const auto &e : expected)
        {
            const auto row = Spu2Synth::interpolationRow(static_cast<uint32_t>(e[0]));
            check(row[0] == e[1] && row[1] == e[2] && row[2] == e[3] && row[3] == e[4], "interpolation row " + std::to_string(e[0]));
        }
        const auto last = Spu2Synth::interpolationRow(255u), first = Spu2Synth::interpolationRow(0u);
        check(last[0] == first[3] && last[1] == first[2] && last[2] == first[1] && last[3] == first[0], "the table is symmetric");
    }

    void testEnvelope()
    {
        auto s = makeSynth();
        upload(*s, 0u, 0x07u, kFlat); // one block that loops on itself
        setVoice(*s, 0u, 0u, 10u);    // attack: +14 per sample
        keyOn(*s, 0u, 1u);
        s->render(1000u);
        check(r(*s, 0x00Au) == 14000u, "linear attack, shift 10: 14 per sample (ENVX " + std::to_string(r(*s, 0x00Au)) + ")");
        s->render(1340u);
        check(r(*s, 0x00Au) == 32760u, "attack still short of the top after 2340 samples");
        s->render(1u);
        check(r(*s, 0x00Au) == 0x7FFFu, "attack reaches 0x7FFF on sample 2341");
        s->render(5000u);
        check(r(*s, 0x00Au) == 0x7FFFu, "sustain holds");
        check((r(*s, 0x340u) & 1u) != 0u, "ENDX set when the block with the end flag was played");
        keyOff(*s, 0u, 1u);
        s->render(2047u);
        check(r(*s, 0x00Au) == 15u, "linear release, shift 10: 16 per sample (ENVX " + std::to_string(r(*s, 0x00Au)) + ")");
        check(s->render(1u) == false && r(*s, 0x00Au) == 0u, "release ends in silence and the voice stops");

        // Written in the same sample: the later write wins.
        keyOn(*s, 0u, 1u);
        keyOff(*s, 0u, 1u);
        check(s->render(4u) == false, "key-on then key-off: the note never starts");
        keyOff(*s, 0u, 1u);
        keyOn(*s, 0u, 1u);
        check(s->render(4u) == true, "key-off then key-on: the note starts");
    }

    void testVoiceThroughTheMixer()
    {
        // An impulse through the interpolator at pitch 1:1 comes out as the four weights of row 0,
        // newest sample last, then through the volume chain of both cores.
        auto s = makeSynth();
        std::vector<uint8_t> impulse(14u, 0x00u);
        impulse[6] = 0x07u; // sample 12 = 0x7000
        upload(*s, 0u, 0x04u, impulse);
        upload(*s, 1u, 0x03u, std::vector<uint8_t>(14u, 0u));
        setVoice(*s, 0u, 0u, 0u); // attack done in three samples
        keyOn(*s, 0u, 1u);
        s->render(64u);
        const auto &out = s->output(0);
        check(out.size() == 128u && s->output(1).size() == 128u && s->output(3).size() == 128u, "every bus gets every frame");
        const size_t peak = static_cast<size_t>(std::max_element(out.begin(), out.end()) - out.begin()) / 2u;
        const auto row = Spu2Synth::interpolationRow(0u);
        bool exact = peak >= 2u && peak + 1u < 64u;
        for (int i = 0; exact && i < 4; ++i)
        {
            const int expected = chain0((row[static_cast<size_t>(3 - i)] * 0x7000) >> 15);
            const size_t frame = peak - 2u + static_cast<size_t>(i);
            exact = out[frame * 2u] == expected && out[frame * 2u + 1u] == expected;
        }
        check(exact, "impulse response is the interpolation row through the volume chain, both channels");
        bool silent = true;
        for (const int bus : {1, 2, 3})
            for (const int16_t v : s->output(bus))
                silent = silent && v == 0;
        check(silent, "untagged voices leave the other buses silent");

        // Two voices at full scale add up past 16 bits: the mix saturates, it does not wrap.
        s = makeSynth();
        upload(*s, 0u, 0x07u, kFlat);
        setVoice(*s, 0u, 0u, 0u);
        setVoice(*s, 0u, 1u, 0u);
        keyOn(*s, 0u, 3u);
        s->render(64u);
        const auto &loud = s->output(0);
        const int one = chain0(((row[0] + row[1] + row[2]) * 0x7000 >> 15) - 1);
        bool saturated = true;
        for (size_t i = 32u; i < loud.size(); ++i)
            saturated = saturated && loud[i] > one && loud[i] > 32000;
        check(saturated, "two full-scale voices saturate at the top (" + std::to_string(loud.back()) + ")");
    }

    void testLoopLength()
    {
        // A two-block loop is exactly 56 samples long: one sample more or less is a note out of tune
        // (the synth this one replaced looped 2 samples short).
        auto s = makeSynth();
        std::vector<uint8_t> impulse(14u, 0x00u);
        impulse[0] = 0x07u;
        upload(*s, 0u, 0x04u, impulse);
        upload(*s, 1u, 0x03u, std::vector<uint8_t>(14u, 0u));
        setVoice(*s, 0u, 0u, 0u);
        keyOn(*s, 0u, 1u);
        s->render(600u);
        const auto &out = s->output(0);
        std::vector<size_t> peaks;
        for (size_t f = 8u; f < 599u; ++f)
            if (out[f * 2u] > 8000 && out[f * 2u] > out[f * 2u - 2u] && out[f * 2u] >= out[f * 2u + 2u])
                peaks.push_back(f);
        bool ok = peaks.size() >= 9u;
        for (size_t i = 1u; ok && i < peaks.size(); ++i)
            ok = peaks[i] - peaks[i - 1u] == 56u;
        check(ok, "a two-block loop repeats every 56 samples (" + std::to_string(peaks.size()) + " peaks)");
    }

    void testInputBlock()
    {
        auto s = makeSynth();
        int16_t block[512];
        std::fill(block, block + 256, int16_t{1000});
        std::fill(block + 256, block + 512, int16_t{-2000});
        w(*s, 0x5B0u, 2u);    // AutoDMA on core 1
        w(*s, 0x794u, 0x4000u); // BVOL left of core 1: half
        check(s->isAutoDmaEnabled(1) && !s->isAutoDmaEnabled(0), "AutoDMA enable bit per core");
        s->setInputBus(1, 3);
        s->setInputBlock(1, reinterpret_cast<const uint8_t *>(block));
        s->render(300u);
        const auto &fmv = s->output(3);
        bool ok = fmv.size() == 600u;
        for (size_t f = 0; ok && f < 300u; ++f)
        {
            const int l = f < 256u ? mul15(mul15(1000, 0x4000), 0x7FFE) : 0;
            const int rr = f < 256u ? mul15(mul15(-2000, 0x7FFF), 0x7FFE) : 0;
            ok = fmv[f * 2u] == l && fmv[f * 2u + 1u] == rr;
        }
        check(ok, "the input block plays for 256 samples through BVOL and MVOL, on its bus");
        bool silent = true;
        for (const int16_t v : s->output(0))
            silent = silent && v == 0;
        check(silent, "and not on the default bus");
    }

    // The reverb the game sets up at its title screen (the hall preset), on core 0.
    void setHall(Spu2Synth &s)
    {
        const uint16_t addr[22][2] = {{0x2E4, 0x38C}, {0x2E8, 0x2A4}, {0x2EC, 0x37EC}, {0x2F0, 0x2D60}, {0x2F4, 0x3424}, {0x2F8, 0x28F0},
                                      {0x2FC, 0x2F64}, {0x300, 0x25CC}, {0x304, 0x2D64}, {0x308, 0x2368}, {0x30C, 0x2364}, {0x310, 0x17A4},
                                      {0x314, 0x1FB0}, {0x318, 0x12C0}, {0x31C, 0x1BBC}, {0x320, 0xF48}, {0x324, 0x17A8}, {0x328, 0xC74},
                                      {0x32C, 0xC70}, {0x330, 0x8E0}, {0x334, 0x550}, {0x338, 0x2A8}};
        for (const auto &a : addr)
        {
            w(s, a[0], 0u);
            w(s, a[0] + 2u, a[1]);
        }
        const uint16_t vol[10] = {0x6F60, 0x4FA8, 0xBCE0, 0x4510, 0xBEF0, 0xA680, 0x5680, 0x52C0, 0x8000, 0x8000};
        for (uint32_t i = 0; i < 10u; ++i)
            w(s, 0x774u + i * 2u, vol[i]);
        w(s, 0x2E0u, 0xEu); // work area 0xEC810..0xEFFFF
        w(s, 0x2E2u, 0xC810u);
        w(s, 0x33Cu, 0xEu);
        w(s, 0x764u, 0x2800u); // EVOL
        w(s, 0x766u, 0x2800u);
        w(s, 0x19Au, 0xC080u); // reverb on
    }

    std::vector<int16_t> playNote(Spu2Synth &s, int bus, bool reverbOn, size_t frames)
    {
        upload(s, 0u, 0x07u, kFlat);
        setHall(s);
        s.setReverbEnabled(reverbOn);
        for (uint32_t v = 0; v < 2u; ++v)
        {
            setVoice(s, 0u, v, 0u);
            w(s, v * 0x10u, 0x1000u); // an eighth each: the two voices together stay far from saturating
            w(s, v * 0x10u + 2u, 0x1000u);
            s.setVoiceBus(0, v, v == 0u ? 0 : bus);
        }
        keyOn(s, 0u, 3u);
        s.render(2400u);
        keyOff(s, 0u, 3u);
        s.render(frames - 2400u);
        std::vector<int16_t> mix(frames * 2u);
        for (size_t i = 0; i < mix.size(); ++i)
        {
            int sum = 0;
            for (int b = 0; b < Spu2Synth::kBusCount; ++b)
                sum += s.output(b)[i];
            mix[i] = static_cast<int16_t>(std::clamp(sum, -32768, 32767));
        }
        return mix;
    }

    void testReverbAndBuses()
    {
        const auto run = [](int bus, bool reverbOn) { return playNote(*makeSynth(), bus, reverbOn, 48000u); };
        const auto dry = run(0, false), wet = run(0, true), split = run(2, true);

        const auto energy = [](const std::vector<int16_t> &pcm, size_t from, size_t to)
        {
            double e = 0.0;
            for (size_t i = from * 2u; i < to * 2u; ++i)
                e += static_cast<double>(pcm[i]) * pcm[i];
            return e;
        };
        check(energy(dry, 6000u, 48000u) == 0.0, "without reverb the note ends with its release");
        const double early = energy(wet, 6000u, 24000u), late = energy(wet, 24000u, 48000u);
        check(early > 1e6, "the reverb leaves a tail after the note (" + std::to_string(early) + ")");
        check(late < early * 0.5, "and the tail decays (" + std::to_string(late) + ")");
        bool same = true;
        for (size_t i = 0; i < 100u * 2u; ++i)
            same = same && dry[i] == wet[i];
        check(same, "the first reflections take time: the start of the note is dry");

        // One voice moved to another bus: the buses add up to what the single mix was.
        int worst = 0;
        for (size_t i = 0; i < wet.size(); ++i)
            worst = std::max(worst, std::abs(wet[i] - split[i]));
        check(worst <= 8, "sum of the buses equals the single-bus mix within rounding (worst " + std::to_string(worst) + ")");
    }

    void testMusicClock()
    {
        const auto run = [](uint32_t ratio)
        {
            auto s = makeSynth();
            upload(*s, 0u, 0x07u, kFlat);
            setVoice(*s, 0u, 0u, 10u);
            s->setVoiceBus(0, 0u, Spu2Synth::kBusMusic);
            s->setMusicRatio(ratio);
            keyOn(*s, 0u, 1u);
            s->render(4000u);
            check(s->output(0).size() == 8000u, "game-time buses always get every frame");
            return s->output(Spu2Synth::kBusMusic);
        };
        const auto full = run(0x10000u), half = run(0x8000u), third = run(0x5555u);
        check(full.size() == 8000u, "ratio 1: the music bus is in step with the rest");
        check(half.size() == 4000u, "ratio 1/2: half the music frames (" + std::to_string(half.size() / 2u) + ")");
        check(third.size() / 2u >= 1332u && third.size() / 2u <= 1334u, "ratio 1/3: a third of the music frames");
        check(std::equal(half.begin(), half.end(), full.begin()) && std::equal(third.begin(), third.end(), full.begin()),
              "and they are the same music, only made later");
    }
}

int main()
{
    testInterpolationTable();
    testEnvelope();
    testVoiceThroughTheMixer();
    testLoopLength();
    testInputBlock();
    testReverbAndBuses();
    testMusicClock();
    if (g_failed != 0)
    {
        std::cout << g_failed << " spu2 synth checks failed\n";
        return 1;
    }
    std::cout << "spu2 synth tests passed\n";
    return 0;
}
