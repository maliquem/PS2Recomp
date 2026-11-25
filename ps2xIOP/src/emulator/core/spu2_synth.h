#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ps2x::iop::detail
{
    // Minimal hardware-accurate PS2 SPU2 core: 2 MiB SPU2 RAM, 2 cores x 24 ADPCM voices,
    // ADSR envelopes, voice/core volume mixing, and DMA4/DMA7 SPU2 RAM transfers.
    // shortcut: omits SPU2 reverb, noise generator, pitch modulation, and hardware volume sweeps;
    // upgrade if a game relies on reverb tails or FM/noise synth voices.
    class Spu2Synth
    {
    public:
        static constexpr uint32_t kRamBytes = 2u * 1024u * 1024u;
        static constexpr uint32_t kRamHalfwords = kRamBytes / 2u;
        static constexpr uint32_t kRamHalfwordMask = kRamHalfwords - 1u;
        static constexpr uint32_t kVoicesPerCore = 24u;
        static constexpr uint32_t kCoreCount = 2u;

        Spu2Synth();

        void reset();

        [[nodiscard]] uint16_t read16(uint32_t physAddr) const;
        void write16(uint32_t physAddr, uint16_t value);

        void dmaWrite(int core, const uint8_t *src, uint32_t byteSize);
        void dmaRead(int core, uint8_t *dst, uint32_t byteSize);

        [[nodiscard]] bool isAutoDmaEnabled(int core) const noexcept;
        void getAutoDmaVolume(int core, int32_t &outLeft, int32_t &outRight) const noexcept;
        [[nodiscard]] bool hasActiveVoices() const noexcept;

        // Renders interleaved 48 kHz 16-bit stereo samples for all 48 ADPCM voices.
        // Returns true if any voice was active during the rendered block.
        bool render(int16_t *interleavedStereo, size_t frameCount);

    private:
        enum class AdsrPhase : uint8_t
        {
            Off = 0,
            Attack,
            Decay,
            Sustain,
            Release,
        };

        struct Voice
        {
            uint16_t volL = 0u;
            uint16_t volR = 0u;
            uint16_t pitch = 0u;
            uint16_t adsr1 = 0u;
            uint16_t adsr2 = 0u;
            uint32_t ssa = 0u; // halfword address in SPU2 RAM
            uint32_t lsa = 0u; // halfword loop start address
            uint32_t nax = 0u; // current halfword address
            bool lsaWritten = false;

            AdsrPhase phase = AdsrPhase::Off;
            int32_t env = 0; // 0..0x7FFF
            uint32_t envCounter = 0u;

            int32_t old1 = 0;
            int32_t old2 = 0;
            int32_t prevSample = 0;
            int32_t curSample = 0;
            uint32_t pitchCounter = 0u;
            uint32_t sampleIndex = 28u;
            uint8_t blockFlags = 0u;
            bool pendingStop = false;
            std::array<int16_t, 28> blockSamples{};
        };

        struct Core
        {
            std::array<Voice, kVoicesPerCore> voices{};
            uint16_t attr = 0u;
            uint16_t statx = 0x0080u;
            uint32_t tsa = 0u;  // DMA/PIO transfer halfword address
            uint32_t endx = 0u; // 24-bit voice end flags
            uint16_t mvolL = 0x3FFFu;
            uint16_t mvolR = 0x3FFFu;
            uint16_t avolL = 0u;
            uint16_t avolR = 0u;
            uint16_t bvolL = 0u;
            uint16_t bvolR = 0u;
            uint16_t evolL = 0u;
            uint16_t evolR = 0u;
            bool avolWritten = false;
            bool bvolWritten = false;
            bool mvolWritten = false;
        };

        void keyOn(Core &core, uint32_t mask);
        void keyOff(Core &core, uint32_t mask);
        void decodeBlock(Voice &voice, Core &core, uint32_t voiceIndex);
        void stepAdsr(Voice &voice);
        [[nodiscard]] static int32_t decodeVolume15(uint16_t reg) noexcept;

        std::vector<uint8_t> m_ram;
        std::array<uint16_t, 0x400u> m_rawRegs{};
        std::array<Core, kCoreCount> m_cores{};
        uint16_t m_admas = 0u; // Shared AutoDMA control register (0x1F9001B0): bit 0 = Core 0, bit 1 = Core 1
    };
}
