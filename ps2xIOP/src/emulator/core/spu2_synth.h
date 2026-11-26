#pragma once

#include "ps2x/iop/iop_types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

namespace ps2x::state
{
    class Archive;
}

namespace ps2x::iop::detail
{
    // PS2 SPU2 core: 2 MiB of sound RAM, 2 cores x 24 ADPCM voices (4-tap Gaussian interpolation,
    // ADSR, volume slides, noise, pitch modulation), the per-core mixer (VMIX/MMIX gates, core 0 into
    // core 1 through AVOL), reverb and the AutoDMA PCM input.
    //
    // The voice path, envelopes, mixer and reverb are a port of PCSX2's SPU2 (pcsx2/SPU2/Mixer.cpp,
    // ADSR.cpp, Reverb.cpp, spu2sys.cpp; PCSX2 Dev Team, GPL-3.0+), kept sample for sample.
    //
    // What is not hardware: the output leaves on separate buses (effects 0, music 1, voice 2, FMV 3;
    // the numbers of ps2x::audio in runtime/ps2_audio_mix.h). A voice takes its bus at key-on
    // (setVoiceBus), the PCM input has one per core (setInputBus), and each bus has its own reverb
    // and its own trip through the core 0 -> core 1 chain. With nothing tagged everything is on bus 0
    // and the core is the plain hardware. The music bus can run slower than the rest (setMusicRatio):
    // its voices, reverb and output then advance only on some of the sample ticks.
    //
    // shortcut: the PCM input always runs on game time, so it must not be put on the music bus while
    // the music ratio is below 1; give stream music a game-time bus of its own if a game needs it.
    // shortcut: the SPU2 interrupt (IRQA) is tested where voices read sound RAM, not on the reverb
    // work area, the capture areas or DMA transfers; add those if a game waits on such an address.
    // shortcut: STATX always reports the transfer as done, DMA being instant here.
    class Spu2Synth
    {
    public:
        static constexpr uint32_t kRamBytes = 2u * 1024u * 1024u;
        static constexpr uint32_t kRamHalfwords = kRamBytes / 2u;
        static constexpr uint32_t kRamHalfwordMask = kRamHalfwords - 1u;
        static constexpr uint32_t kVoicesPerCore = 24u;
        static constexpr uint32_t kCoreCount = 2u;
        static constexpr int kBusCount = kSpu2BusCount;
        static constexpr int kBusMusic = kSpu2MusicBus;
        static constexpr uint32_t kInputFrames = 256u; // one AutoDMA block: 256 left then 256 right samples

        Spu2Synth();

        void reset();

        [[nodiscard]] uint16_t read16(uint32_t physAddr) const;
        void write16(uint32_t physAddr, uint16_t value);

        void dmaWrite(int core, const uint8_t *src, uint32_t byteSize);
        void dmaRead(int core, uint8_t *dst, uint32_t byteSize);

        [[nodiscard]] bool isAutoDmaEnabled(int core) const noexcept;
        // AutoDMA: the core's PCM input for its next 256 samples (1024 bytes: 256 left, 256 right).
        void setInputBlock(int core, const uint8_t *block1024);

        // Bus the voice plays on from its next key-on, and the bus of a core's PCM input (a setting
        // of the host: reset() keeps it).
        void setVoiceBus(int core, uint32_t voice, int bus);
        void setInputBus(int core, int bus);
        // Samples the bus has advanced since reset (wraps).
        [[nodiscard]] uint32_t busSamples(int bus) const noexcept { return m_busTicks[static_cast<size_t>(bus)]; }
        // Music-bus ticks per sample tick, 16.16 fixed point, at most 1.0 (0x10000 = the hardware).
        void setMusicRatio(uint32_t ratioQ16) noexcept { m_musicRatio = ratioQ16 < 0x10000u ? ratioQ16 : 0x10000u; }

        // Runs the core for `frames` samples (48 kHz) and appends each bus's interleaved stereo to
        // output(bus); the music bus gets fewer frames while its ratio is below 1.
        // Returns true if a voice or the PCM input was playing at the end.
        bool render(size_t frames);
        [[nodiscard]] const std::vector<int16_t> &output(int bus) const { return m_out[static_cast<size_t>(bus)]; }
        void clearOutput();

        // True once per SPU2 interrupt request (IOP interrupt 9).
        [[nodiscard]] bool takeIrq() noexcept
        {
            const bool irq = m_irq;
            m_irq = false;
            return irq;
        }

        // For listening tests and trace comparisons; the game cannot turn this off.
        void setReverbEnabled(bool enabled) noexcept { m_reverbEnabled = enabled; }

        // Row `phase` (0..255) of the interpolation table: the weights of four consecutive samples.
        [[nodiscard]] static std::array<int16_t, 4> interpolationRow(uint32_t phase);

        // Save state: saves or restores the guest-visible state (same executable only).
        void stateIo(ps2x::state::Archive &ar);

    private:
        struct Slide // VOLL/VOLR and MVOLL/MVOLR: a fixed volume or a hardware slide
        {
            uint16_t reg = 0u;
            int32_t value = 0;
            uint32_t counter = 0u;
        };

        struct Voice
        {
            Slide vol[2];
            uint16_t pitch = 0x3FFFu;
            uint16_t adsr1 = 0u;
            uint16_t adsr2 = 0u;
            uint32_t ssa = 0x2800u; // halfword addresses in sound RAM
            uint32_t lsa = 0x2800u;
            uint32_t nax = 0x2801u;
            bool loopMode = false; // the game wrote LSAX: block loop-start flags no longer move it
            uint8_t loopFlags = 0u;

            uint8_t phase = 0u; // 0 stopped, 1 attack, 2 decay, 3 sustain, 4 release
            int32_t env = 0;    // 0..0x7FFF
            uint32_t envCounter = 0u;

            int32_t prev1 = 0;
            int32_t prev2 = 0;
            int32_t sp = 0; // 4.12 position between decoded samples
            uint32_t decRead = 0u;
            uint32_t decWrite = 0u;
            int16_t fifo[32] = {};
            int16_t block[28] = {};
            bool blockValid = false;
            int32_t outX = 0; // last output after the envelope, the pitch modulator of the next voice

            uint8_t bus = 0u;
            uint8_t nextBus = 0u;
        };

        struct Reverb // one per core and bus
        {
            int16_t down[2][128] = {};
            int16_t up[2][128] = {};
            uint32_t pos = 0u;
            uint32_t areaPos = 0u; // where the work area's offsets count from; one step per 24 kHz sample
        };

        struct Core
        {
            std::array<Voice, kVoicesPerCore> voices{};
            Slide mvol[2];
            uint32_t tsa = 0u; // DMA/PIO transfer halfword address
            uint32_t endx = 0x00FFFFFFu;
            uint32_t keyOn = 0u; // written, applied at the next sample
            uint32_t keyOff = 0u;
            uint32_t noiseCnt = 0u;
            uint32_t noiseOut = 0u;
            int16_t input[kInputFrames * 2u] = {};
            uint32_t inputLeft = 0u;
            uint8_t inputBus = 0u;
            std::array<Reverb, kBusCount> reverb{};
        };

        struct Config // registers decoded once per render call
        {
            uint32_t pmon = 0u, non = 0u, vmix[4] = {}; // vmix: dry L, dry R, wet L, wet R
            uint32_t mmix = 0u, irqa = 0u, esa = 0u, eea = 0u;
            bool irqEnable = false, fxEnable = false;
            uint32_t noiseClk = 0u;
            int32_t evol[2] = {}, avol[2] = {}, bvol[2] = {};
            struct Side
            {
                uint32_t sameSrc, sameDst, diffSrc, diffDst, comb[4], apf1Src, apf1Dst, apf2Src, apf2Dst;
                int32_t inCoef;
            } side[2] = {};
            int32_t iirVol = 0, combVol[4] = {}, wallVol = 0, apf1Vol = 0, apf2Vol = 0;
        };

        [[nodiscard]] uint16_t &raw(uint32_t core, uint32_t local) { return m_rawRegs[(core * 0x400u + local) >> 1u]; }
        [[nodiscard]] uint16_t raw(uint32_t core, uint32_t local) const { return m_rawRegs[(core * 0x400u + local) >> 1u]; }
        [[nodiscard]] int16_t ramRead(uint32_t halfword) const noexcept;
        void ramWrite(uint32_t halfword, int16_t value) noexcept;

        void loadConfig();
        void tick();
        void stepVoice(uint32_t core, uint32_t index, int32_t &outL, int32_t &outR);
        void incrementNax(Voice &voice);
        void raiseIrq(uint32_t core);
        void reverb(uint32_t core, int bus, const int32_t in[2], int32_t out[2]);
        void clearPrivateReverb(uint32_t core);

        std::vector<uint8_t> m_ram;
        std::array<uint16_t, 0x400u> m_rawRegs{}; // every register as written; live values override on read
        std::array<Core, kCoreCount> m_cores{};
        std::array<Config, kCoreCount> m_cfg{};
        // Reverb work areas of buses 1..3 (bus 0 uses the game's area in sound RAM), index core * 4 + bus.
        std::array<std::vector<int16_t>, kCoreCount * kBusCount> m_reverbRam;
        std::array<std::vector<int16_t>, kBusCount> m_out;
        std::array<uint32_t, kBusCount> m_busTicks{};
        uint32_t m_busUsed = 1u; // buses that ever carried something; the others stay silent for free
        uint32_t m_outPos = 0u;  // position in the 0x200-sample capture areas
        uint32_t m_musicRatio = 0x10000u;
        uint32_t m_musicAcc = 0u;
        bool m_irq = false;
        bool m_reverbEnabled = true;

        // PS2X_SPU2_TRACE=<file>: every register write, DMA and render call goes to the file, for the
        // offline replayer (tests/spu2_trace_replay.cpp). Format: one tag byte per event, then
        // W u16 offset, u16 value | D u8 core, u32 bytes, data | d u8 core, u32 bytes (DMA read) |
        // R u32 frames, u64 microseconds since the first event | A u8 core, 8 unused bytes, 1024 bytes
        // (PCM input block for the samples that follow).
        // shortcut: the trace only replays from a power-on synth, start it at boot (not after a save state).
        void trace(char tag, const void *a, size_t aSize, const void *b = nullptr, size_t bSize = 0u);
        std::shared_ptr<std::FILE> m_trace;
    };
}
