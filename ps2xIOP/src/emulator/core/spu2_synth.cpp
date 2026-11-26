// PS2 SPU2 core (spu2_synth.h). The voice path, ADSR, volume slides, noise, mixer and reverb are
// ported from PCSX2's SPU2 (pcsx2/SPU2/Mixer.cpp, ADSR.cpp, Reverb.cpp, ReverbResample.cpp,
// spu2sys.cpp; Copyright PCSX2 Dev Team, GPL-3.0+).

#include "spu2_synth.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "ps2x/state_archive.h"

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kSpu2Base = 0x1F900000u;
        constexpr uint32_t kIrqInfo = 0x7C2u; // bit 2 = core 0 asked for an interrupt, bit 3 = core 1
        constexpr int32_t kAdpcmFilter[16][2] = {{0, 0}, {60, 0}, {115, -52}, {98, -55}, {122, -60}};

        enum : uint8_t { kStopped = 0, kAttack, kDecay, kSustain, kRelease };

        // Per-core register offsets (core 1 at +0x400).
        enum : uint32_t
        {
            kPmon = 0x180u, kNon = 0x184u, kVmixL = 0x188u, kVmixEL = 0x18Cu, kVmixR = 0x190u, kVmixER = 0x194u,
            kMmix = 0x198u, kAttr = 0x19Au, kIrqa = 0x19Cu, kKon = 0x1A0u, kKoff = 0x1A4u, kTsa = 0x1A8u,
            kPioData = 0x1ACu, kAdmas = 0x1B0u, kVoiceAddr = 0x1C0u, kEsa = 0x2E0u, kApf1Size = 0x2E4u,
            kApf2Size = 0x2E8u, kSameLDst = 0x2ECu, kSameRDst = 0x2F0u, kComb1LSrc = 0x2F4u, kComb1RSrc = 0x2F8u,
            kComb2LSrc = 0x2FCu, kComb2RSrc = 0x300u, kSameLSrc = 0x304u, kSameRSrc = 0x308u, kDiffLDst = 0x30Cu,
            kDiffRDst = 0x310u, kComb3LSrc = 0x314u, kComb3RSrc = 0x318u, kComb4LSrc = 0x31Cu, kComb4RSrc = 0x320u,
            kDiffLSrc = 0x324u, kDiffRSrc = 0x328u, kApf1LDst = 0x32Cu, kApf1RDst = 0x330u, kApf2LDst = 0x334u,
            kApf2RDst = 0x338u, kEea = 0x33Cu, kEndx = 0x340u, kStatx = 0x344u,
        };
        // The volume block: core 0 at 0x760, core 1 at 0x788.
        enum : uint32_t
        {
            kVolBlock = 0x760u, kVolBlockSize = 0x28u,
            kMvol = 0x00u, kEvol = 0x04u, kAvol = 0x08u, kBvol = 0x0Cu, kMvolx = 0x10u, kIirVol = 0x14u,
            kComb1Vol = 0x16u, kWallVol = 0x1Eu, kApf1Vol = 0x20u, kApf2Vol = 0x22u, kInCoefL = 0x24u, kInCoefR = 0x26u,
        };

        inline int32_t clamp16(int64_t v) { return static_cast<int32_t>(std::clamp<int64_t>(v, -0x8000, 0x7FFF)); }
        inline int32_t mul15(int32_t a, int32_t b) { return (a * b) >> 15; } // both within 17 bits

        // The 4-tap interpolation table of the SPU, generated as PCSX2 does (row = phase / 16 of the
        // 4.12 position, the four weights apply to consecutive samples and add up to 0x7F80).
        struct InterpolationTable
        {
            int16_t row[256][4];
            InterpolationTable()
            {
                constexpr double pi = 3.14159265358979323846;
                double table[512];
                double sum = 0.0;
                for (int n = 0; n < 512; ++n)
                {
                    const double k = 0.5 + n;
                    const double s = std::sin(pi * k * 2.048 / 1024.0);
                    const double t = (std::cos(pi * k * 2.000 / 1023.0) - 1.0) * 0.50;
                    const double u = (std::cos(pi * k * 4.000 / 1023.0) - 1.0) * 0.08;
                    table[511 - n] = s * (t + u + 1.0) / k;
                    sum += table[511 - n];
                }
                const double scale = 0x7F80 * 128 / sum;
                for (double &v : table)
                    v *= scale;
                for (int phase = 0; phase < 256; ++phase)
                {
                    const double diff = (table[phase] + table[phase + 256] + table[511 - phase] + table[255 - phase] - 0x7F80) / 4.0;
                    row[255 - phase][0] = static_cast<int16_t>(std::lround(table[phase] - diff));
                    row[255 - phase][1] = static_cast<int16_t>(std::lround(table[phase + 256] - diff));
                    row[255 - phase][2] = static_cast<int16_t>(std::lround(table[511 - phase] - diff));
                    row[255 - phase][3] = static_cast<int16_t>(std::lround(table[255 - phase] - diff));
                }
            }
        };
        const InterpolationTable g_interpolation;

        // Reverb runs at 24 kHz: 39-tap half-band filter going in, the same one at twice the gain coming out.
        constexpr int kReverbTaps = 39;
        constexpr int32_t kReverbDown[kReverbTaps] = {
            -1, 0, 2, 0, -10, 0, 35, 0, -103, 0, 266, 0, -616, 0, 1332, 0, -2960, 0, 10246, 16384,
            10246, 0, -2960, 0, 1332, 0, -616, 0, 266, 0, -103, 0, 35, 0, -10, 0, 2, 0, -1};

        int32_t reverbFilter(const int16_t *line, uint32_t pos, int gain)
        {
            const uint32_t index = (pos - kReverbTaps) & 63u;
            int64_t out = 0;
            for (int i = 0; i < kReverbTaps; ++i)
                out += line[index + static_cast<uint32_t>(i)] * std::min(kReverbDown[i] * gain, 0x7FFF);
            return clamp16(out >> 15);
        }

        void slideSet(auto &slide, uint16_t reg)
        {
            slide.reg = reg;
            if ((reg & 0x8000u) == 0u)
                slide.value = static_cast<int16_t>(static_cast<uint16_t>(reg << 1u));
        }

        // One sample of a hardware volume slide (register bit 15 set).
        void slideUpdate(auto &slide)
        {
            if ((slide.reg & 0x8000u) == 0u)
                return;
            const int32_t stepBits = slide.reg & 3;
            const int32_t shift = (slide.reg >> 2) & 0x1F;
            const bool phase = (slide.reg & 0x1000u) != 0u;
            const bool decr = (slide.reg & 0x2000u) != 0u;
            const bool exp = (slide.reg & 0x4000u) != 0u;
            const int32_t step = decr ? ~(7 - stepBits) : 7 - stepBits;
            uint32_t counterInc = 0x8000u >> std::max(0, shift - 11);
            int32_t levelInc = step * (1 << std::max(0, 11 - shift));
            if (exp)
            {
                if (!decr && slide.value > 0x6000)
                    counterInc >>= 2;
                if (decr)
                    levelInc = static_cast<int16_t>((levelInc * slide.value) >> 15);
            }
            if (stepBits != 3 && shift != 0x1F)
                counterInc = std::max(1u, counterInc);
            slide.counter += counterInc;
            if (!(exp && decr))
                levelInc = phase ? -levelInc : levelInc;
            if (slide.counter < 0x8000u)
                return;
            slide.counter = 0u;
            int32_t low = -0x8000, high = 0x7FFF;
            if (decr)
            {
                low = (phase && !exp) ? -0x8000 : 0;
                high = (phase && !exp) ? 0 : 0x7FFF;
            }
            slide.value = std::clamp(slide.value + levelInc, low, high);
        }

        // One sample of the envelope. False when the voice has run out (release reached silence).
        bool stepAdsr(auto &voice)
        {
            const uint32_t reg = static_cast<uint32_t>(voice.adsr1) | (static_cast<uint32_t>(voice.adsr2) << 16u);
            bool decr = true, exp = true;
            int32_t shift = 0, step = -8, target = 0;
            switch (voice.phase)
            {
            case kAttack:
                decr = false;
                exp = ((reg >> 15) & 1u) != 0u;
                shift = (reg >> 10) & 0x1F;
                step = 7 - static_cast<int32_t>((reg >> 8) & 3u);
                target = 0x7FFF;
                break;
            case kDecay:
                shift = (reg >> 4) & 0xF;
                target = static_cast<int32_t>(((reg & 0xFu) + 1u) << 11);
                break;
            case kSustain:
                decr = ((reg >> 30) & 1u) != 0u;
                exp = ((reg >> 31) & 1u) != 0u;
                shift = (reg >> 24) & 0x1F;
                step = 7 - static_cast<int32_t>((reg >> 22) & 3u);
                if (decr)
                    step = ~step;
                break;
            case kRelease:
                exp = ((reg >> 21) & 1u) != 0u;
                shift = (reg >> 16) & 0x1F;
                break;
            default:
                return false;
            }
            uint32_t counterInc = 0x8000u >> std::max(0, shift - 11);
            int32_t levelInc = step * (1 << std::max(0, 11 - shift));
            if (exp)
            {
                if (!decr && voice.env > 0x6000)
                    counterInc >>= 2;
                if (decr)
                    levelInc = static_cast<int16_t>((levelInc * voice.env) >> 15);
            }
            voice.envCounter += std::max(1u, counterInc);
            if (voice.envCounter >= 0x8000u)
            {
                voice.envCounter = 0u;
                voice.env = std::clamp(voice.env + levelInc, 0, 0x7FFF);
            }
            if (voice.phase == kSustain)
                return voice.env != 0;
            if ((!decr && voice.env >= target) || (decr && voice.env <= target))
                ++voice.phase;
            return voice.phase <= kRelease;
        }

        void stopVoice(auto &voice)
        {
            voice.env = 0;
            voice.phase = kStopped;
        }

        // Decodes the 28 samples of one ADPCM block (16 bytes: shift/filter, flags, 14 data bytes).
        void decodeBlock(auto &voice, const uint8_t *bytes)
        {
            const int32_t shift = (bytes[0] & 0xF) + 16;
            const int32_t *filter = kAdpcmFilter[bytes[0] >> 4];
            int32_t prev1 = voice.prev1, prev2 = voice.prev2;
            int16_t *out = voice.block;
            for (int i = 0; i < 14; ++i)
            {
                const uint32_t byte = bytes[2 + i];
                const int32_t low = static_cast<int32_t>((byte << 28) & 0xF0000000u) >> shift;
                const int32_t pcm = clamp16(low + ((filter[0] * prev1 + filter[1] * prev2 + 32) >> 6));
                const int32_t high = static_cast<int32_t>((byte << 24) & 0xF0000000u) >> shift;
                const int32_t pcm2 = clamp16(high + ((filter[0] * pcm + filter[1] * prev1 + 32) >> 6));
                *out++ = static_cast<int16_t>(pcm);
                *out++ = static_cast<int16_t>(pcm2);
                prev2 = pcm;
                prev1 = pcm2;
            }
            voice.prev1 = prev1;
            voice.prev2 = prev2;
            voice.blockValid = true;
        }

        void updateNoise(auto &core, uint32_t clock)
        {
            static constexpr uint8_t kAdd[64] = {
                1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0,
                0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1};
            static constexpr uint32_t kFreqAdd[5] = {0, 84, 140, 180, 210};
            const uint32_t level = (0x8000u >> (clock >> 2)) << 16;
            core.noiseCnt += 0x10000u + kFreqAdd[clock & 3u];
            if ((core.noiseCnt & 0xFFFFu) >= 210u)
                core.noiseCnt += 0x10000u - kFreqAdd[clock & 3u];
            if (core.noiseCnt >= level)
            {
                core.noiseCnt %= level;
                core.noiseOut = (core.noiseOut << 1) | kAdd[(core.noiseOut >> 10) & 63u];
            }
        }
    }

    std::array<int16_t, 4> Spu2Synth::interpolationRow(uint32_t phase)
    {
        const int16_t *r = g_interpolation.row[phase & 0xFFu];
        return {r[0], r[1], r[2], r[3]};
    }

    Spu2Synth::Spu2Synth()
        : m_ram(kRamBytes, 0u)
    {
        if (const char *path = std::getenv("PS2X_SPU2_TRACE"); path && *path)
            m_trace.reset(std::fopen(path, "wb"), [](std::FILE *f) { if (f) std::fclose(f); });
        reset();
    }

    void Spu2Synth::trace(char tag, const void *a, size_t aSize, const void *b, size_t bSize)
    {
        if (!m_trace)
            return;
        std::fputc(tag, m_trace.get());
        std::fwrite(a, 1u, aSize, m_trace.get());
        if (b)
            std::fwrite(b, 1u, bSize, m_trace.get());
    }

    void Spu2Synth::reset()
    {
        std::fill(m_ram.begin(), m_ram.end(), uint8_t{0});
        m_rawRegs.fill(0u);
        for (uint32_t c = 0; c < kCoreCount; ++c)
        {
            const uint8_t inputBus = m_cores[c].inputBus;
            m_cores[c] = Core{};
            m_cores[c].inputBus = inputBus;
            // Power-on values (PCSX2 V_Core::Init): every gate open except the external input,
            // an empty reverb area at the top of the core's megabyte, input volumes at full.
            raw(c, kMmix) = c ? 0xFFCu : 0xFF0u;
            for (const uint32_t reg : {kVmixL, kVmixEL, kVmixR, kVmixER})
            {
                raw(c, reg) = 0xFFFFu;
                raw(c, reg + 2u) = 0x00FFu;
            }
            raw(c, kEsa) = raw(c, kEea) = c ? 0xFu : 0xEu;
            raw(c, kEsa + 2u) = 0xFFF8u;
            raw(c, kIrqa + 2u) = 0x0800u;
            const uint32_t vol = (kVolBlock + c * kVolBlockSize) >> 1u;
            m_rawRegs[vol + (kAvol >> 1u)] = m_rawRegs[vol + (kAvol >> 1u) + 1u] = 0x7FFFu;
            m_rawRegs[vol + (kBvol >> 1u)] = m_rawRegs[vol + (kBvol >> 1u) + 1u] = 0x7FFFu;
        }
        for (auto &area : m_reverbRam)
            area.clear();
        clearOutput();
        m_busTicks.fill(0u);
        m_busUsed = 1u;
        m_outPos = 0u;
        m_musicAcc = 0u;
        m_irq = false;
    }

    void Spu2Synth::clearOutput()
    {
        for (auto &out : m_out)
            out.clear();
    }

    int16_t Spu2Synth::ramRead(uint32_t halfword) const noexcept
    {
        int16_t value;
        std::memcpy(&value, m_ram.data() + (halfword & kRamHalfwordMask) * 2u, 2u);
        return value;
    }

    void Spu2Synth::ramWrite(uint32_t halfword, int16_t value) noexcept
    {
        std::memcpy(m_ram.data() + (halfword & kRamHalfwordMask) * 2u, &value, 2u);
    }

    uint16_t Spu2Synth::read16(uint32_t physAddr) const
    {
        if (physAddr < kSpu2Base || physAddr >= kSpu2Base + 0x800u)
            return 0u;

        const uint32_t offset = (physAddr - kSpu2Base) & 0x7FEu;
        if (offset >= kVolBlock)
        {
            if (offset < kVolBlock + 2u * kVolBlockSize)
            {
                const uint32_t c = (offset - kVolBlock) / kVolBlockSize;
                const uint32_t local = (offset - kVolBlock) % kVolBlockSize;
                if (local == kMvolx || local == kMvolx + 2u)
                    return static_cast<uint16_t>(m_cores[c].mvol[(local - kMvolx) >> 1u].value);
            }
            return m_rawRegs[offset >> 1u];
        }

        const uint32_t c = offset >> 10u;
        const uint32_t local = offset & 0x3FEu;
        const Core &core = m_cores[c];
        if (local < 0x180u)
        {
            const Voice &voice = core.voices[local >> 4u];
            switch (local & 0xEu)
            {
            case 0xAu: return static_cast<uint16_t>(voice.env);
            case 0xCu: return static_cast<uint16_t>(voice.vol[0].value);
            case 0xEu: return static_cast<uint16_t>(voice.vol[1].value);
            default: return m_rawRegs[offset >> 1u];
            }
        }
        if (local >= kVoiceAddr && local < kEsa)
        {
            const Voice &voice = core.voices[(local - kVoiceAddr) / 12u];
            const uint32_t sub = (local - kVoiceAddr) % 12u;
            const uint32_t address = sub < 4u ? voice.ssa : sub < 8u ? voice.lsa : voice.nax;
            return static_cast<uint16_t>((sub & 2u) ? address : address >> 16u);
        }
        switch (local)
        {
        case kTsa: return static_cast<uint16_t>(core.tsa >> 16u);
        case kTsa + 2u: return static_cast<uint16_t>(core.tsa);
        case kEndx: return static_cast<uint16_t>(core.endx);
        case kEndx + 2u: return static_cast<uint16_t>(core.endx >> 16u);
        case kStatx: return static_cast<uint16_t>((raw(c, kAttr) & 0x003Fu) | 0x0080u);
        default: return m_rawRegs[offset >> 1u];
        }
    }

    void Spu2Synth::write16(uint32_t physAddr, uint16_t value)
    {
        if (physAddr < kSpu2Base || physAddr >= kSpu2Base + 0x800u)
            return;

        const uint32_t offset = (physAddr - kSpu2Base) & 0x7FEu;
        const uint16_t old = m_rawRegs[offset >> 1u];
        m_rawRegs[offset >> 1u] = value;
        if (m_trace)
        {
            const uint16_t event[2] = {static_cast<uint16_t>(offset), value};
            trace('W', event, sizeof(event));
        }

        if (offset >= kVolBlock)
        {
            if (offset < kVolBlock + 2u * kVolBlockSize)
            {
                const uint32_t local = (offset - kVolBlock) % kVolBlockSize;
                if (local == kMvol || local == kMvol + 2u)
                    slideSet(m_cores[(offset - kVolBlock) / kVolBlockSize].mvol[local >> 1u], value);
            }
            return;
        }

        const uint32_t c = offset >> 10u;
        const uint32_t local = offset & 0x3FEu;
        Core &core = m_cores[c];
        if (local < 0x180u)
        {
            Voice &voice = core.voices[local >> 4u];
            switch (local & 0xEu)
            {
            case 0x0u: slideSet(voice.vol[0], value); return;
            case 0x2u: slideSet(voice.vol[1], value); return;
            case 0x4u: voice.pitch = value; return;
            case 0x6u: voice.adsr1 = value; return;
            case 0x8u: voice.adsr2 = value; return;
            case 0xAu: voice.env = value & 0x7FFF; return;
            default: return;
            }
        }
        if (local >= kVoiceAddr && local < kEsa)
        {
            Voice &voice = core.voices[(local - kVoiceAddr) / 12u];
            const uint32_t sub = (local - kVoiceAddr) % 12u;
            uint32_t &address = sub < 4u ? voice.ssa : sub < 8u ? voice.lsa : voice.nax;
            // Start and loop addresses are block aligned; the play address is never on a block header.
            const uint32_t low = sub < 8u ? 0xFFF8u : 0xFFFFu;
            address = (sub & 2u) ? (address & 0xF0000u) | (value & low)
                                 : ((static_cast<uint32_t>(value) & 0xFu) << 16u) | (address & low);
            if (sub >= 8u)
                address |= 1u;
            else if (sub >= 4u)
                voice.loopMode = true;
            return;
        }

        switch (local)
        {
        case kAttr:
            if ((value & 0x40u) == 0u)
                m_rawRegs[kIrqInfo >> 1u] &= static_cast<uint16_t>(~(4u << c));
            if ((old & 0x80u) == 0u && (value & 0x80u) != 0u)
                clearPrivateReverb(c); // reverb switched on: no tail from the last time it ran
            return;
        // A key-off after a key-on of the same voice in the same sample cancels it, as the write order says.
        case kKon: core.keyOn |= value; return;
        case kKon + 2u: core.keyOn |= (static_cast<uint32_t>(value) & 0xFFu) << 16u; return;
        case kKoff: core.keyOff |= value; core.keyOn &= ~static_cast<uint32_t>(value); return;
        case kKoff + 2u:
            core.keyOff |= (static_cast<uint32_t>(value) & 0xFFu) << 16u;
            core.keyOn &= ~((static_cast<uint32_t>(value) & 0xFFu) << 16u);
            return;
        case kTsa: core.tsa = (core.tsa & 0x0000FFFFu) | ((static_cast<uint32_t>(value) & 0xFu) << 16u); return;
        case kTsa + 2u: core.tsa = (core.tsa & 0x000F0000u) | value; return;
        case kPioData:
            ramWrite(core.tsa, static_cast<int16_t>(value));
            core.tsa = (core.tsa + 1u) & kRamHalfwordMask;
            return;
        case kAdmas:
            if ((value & 3u) == 0u)
                core.inputLeft = 0u;
            return;
        case kEndx: core.endx &= 0x00FF0000u; return;
        case kEndx + 2u: core.endx &= 0x0000FFFFu; return;
        default: return;
        }
    }

    void Spu2Synth::dmaWrite(int coreIndex, const uint8_t *src, uint32_t byteSize)
    {
        if (!src || byteSize == 0u || coreIndex < 0 || coreIndex >= static_cast<int>(kCoreCount))
            return;

#pragma pack(push, 1)
        const struct { uint8_t core; uint32_t bytes; } dmaEvent{static_cast<uint8_t>(coreIndex), byteSize};
#pragma pack(pop)
        trace('D', &dmaEvent, sizeof(dmaEvent), src, byteSize);
        Core &core = m_cores[static_cast<size_t>(coreIndex)];
        const uint32_t firstHalfword = core.tsa & kRamHalfwordMask;
        uint32_t dstByte = firstHalfword * 2u;
        uint32_t remaining = byteSize;
        uint32_t srcOffset = 0u;
        while (remaining > 0u)
        {
            const uint32_t chunk = std::min(remaining, kRamBytes - dstByte);
            std::memcpy(m_ram.data() + dstByte, src + srcOffset, chunk);
            srcOffset += chunk;
            remaining -= chunk;
            dstByte = (dstByte + chunk) & (kRamBytes - 1u);
        }
        core.tsa = (core.tsa + (byteSize / 2u)) & kRamHalfwordMask;
        // A transfer into a reverb work area is the game clearing it: clear the other buses' copies too.
        for (uint32_t c = 0; c < kCoreCount; ++c)
        {
            const uint32_t start = (static_cast<uint32_t>(raw(c, kEsa)) << 16u | raw(c, kEsa + 2u)) & kRamHalfwordMask;
            const uint32_t end = (static_cast<uint32_t>(raw(c, kEea)) << 16u | 0xFFFFu) & kRamHalfwordMask;
            if (firstHalfword <= end && firstHalfword + byteSize / 2u > start)
                clearPrivateReverb(c);
        }
    }

    void Spu2Synth::dmaRead(int coreIndex, uint8_t *dst, uint32_t byteSize)
    {
        if (!dst || byteSize == 0u || coreIndex < 0 || coreIndex >= static_cast<int>(kCoreCount))
            return;

#pragma pack(push, 1)
        const struct { uint8_t core; uint32_t bytes; } dmaEvent{static_cast<uint8_t>(coreIndex), byteSize};
#pragma pack(pop)
        trace('d', &dmaEvent, sizeof(dmaEvent));
        Core &core = m_cores[static_cast<size_t>(coreIndex)];
        uint32_t srcByte = (core.tsa & kRamHalfwordMask) * 2u;
        uint32_t remaining = byteSize;
        uint32_t dstOffset = 0u;
        while (remaining > 0u)
        {
            const uint32_t chunk = std::min(remaining, kRamBytes - srcByte);
            std::memcpy(dst + dstOffset, m_ram.data() + srcByte, chunk);
            dstOffset += chunk;
            remaining -= chunk;
            srcByte = (srcByte + chunk) & (kRamBytes - 1u);
        }
        core.tsa = (core.tsa + (byteSize / 2u)) & kRamHalfwordMask;
    }

    bool Spu2Synth::isAutoDmaEnabled(int coreIndex) const noexcept
    {
        if (coreIndex < 0 || coreIndex >= static_cast<int>(kCoreCount))
            return false;
        return (raw(static_cast<uint32_t>(coreIndex), kAdmas) & (static_cast<uint32_t>(coreIndex) + 1u)) != 0u;
    }

    void Spu2Synth::setInputBlock(int coreIndex, const uint8_t *block1024)
    {
        if (!block1024 || coreIndex < 0 || coreIndex >= static_cast<int>(kCoreCount))
            return;
        if (m_trace)
        {
            const uint8_t head[9] = {static_cast<uint8_t>(coreIndex)};
            trace('A', head, sizeof(head), block1024, 1024u);
        }
        Core &core = m_cores[static_cast<size_t>(coreIndex)];
        std::memcpy(core.input, block1024, sizeof(core.input));
        core.inputLeft = kInputFrames;
        m_busUsed |= 1u << core.inputBus;
    }

    void Spu2Synth::setVoiceBus(int coreIndex, uint32_t voice, int bus)
    {
        if (coreIndex >= 0 && coreIndex < static_cast<int>(kCoreCount) && voice < kVoicesPerCore && bus >= 0 && bus < kBusCount)
            m_cores[static_cast<size_t>(coreIndex)].voices[voice].nextBus = static_cast<uint8_t>(bus);
    }

    void Spu2Synth::setInputBus(int coreIndex, int bus)
    {
        if (coreIndex >= 0 && coreIndex < static_cast<int>(kCoreCount) && bus >= 0 && bus < kBusCount)
            m_cores[static_cast<size_t>(coreIndex)].inputBus = static_cast<uint8_t>(bus);
    }

    void Spu2Synth::clearPrivateReverb(uint32_t core)
    {
        for (int bus = 1; bus < kBusCount; ++bus)
            m_reverbRam[core * kBusCount + static_cast<uint32_t>(bus)].clear();
    }

    void Spu2Synth::raiseIrq(uint32_t core)
    {
        uint16_t &info = m_rawRegs[kIrqInfo >> 1u];
        if ((info & (4u << core)) != 0u)
            return;
        info |= static_cast<uint16_t>(4u << core);
        m_irq = true;
    }

    void Spu2Synth::loadConfig()
    {
        for (uint32_t c = 0; c < kCoreCount; ++c)
        {
            Config &cfg = m_cfg[c];
            const auto low32 = [&](uint32_t reg) { return static_cast<uint32_t>(raw(c, reg)) | (static_cast<uint32_t>(raw(c, reg + 2u)) << 16u); };
            const auto high32 = [&](uint32_t reg) { return (static_cast<uint32_t>(raw(c, reg)) << 16u) | raw(c, reg + 2u); };
            const auto vol = [&](uint32_t reg) { return static_cast<int32_t>(static_cast<int16_t>(m_rawRegs[(kVolBlock + c * kVolBlockSize + reg) >> 1u])); };
            cfg.pmon = low32(kPmon) & 0x00FFFFFEu; // voice 0 has nothing to be modulated by
            cfg.non = low32(kNon);
            cfg.vmix[0] = low32(kVmixL);
            cfg.vmix[1] = low32(kVmixR);
            cfg.vmix[2] = low32(kVmixEL);
            cfg.vmix[3] = low32(kVmixER);
            cfg.mmix = raw(c, kMmix) & (c ? 0xFFFu : 0xFF0u);
            const uint32_t attr = raw(c, kAttr);
            cfg.irqEnable = (attr & 0x40u) != 0u;
            cfg.fxEnable = (attr & 0x80u) != 0u;
            cfg.noiseClk = (attr >> 8u) & 0x3Fu;
            cfg.irqa = high32(kIrqa) & 0xFFFFFu;
            cfg.esa = high32(kEsa);
            cfg.eea = (static_cast<uint32_t>(raw(c, kEea)) << 16u) | 0xFFFFu;
            for (uint32_t ch = 0; ch < 2u; ++ch)
            {
                cfg.evol[ch] = vol(kEvol + ch * 2u);
                cfg.avol[ch] = vol(kAvol + ch * 2u);
                cfg.bvol[ch] = vol(kBvol + ch * 2u);
            }
            cfg.iirVol = vol(kIirVol);
            for (uint32_t i = 0; i < 4u; ++i)
                cfg.combVol[i] = vol(kComb1Vol + i * 2u);
            cfg.wallVol = vol(kWallVol);
            cfg.apf1Vol = vol(kApf1Vol);
            cfg.apf2Vol = vol(kApf2Vol);
            // Left on even samples, right on odd ones. The cross feed (diff) reads the other side.
            cfg.side[0] = {high32(kSameLSrc), high32(kSameLDst), high32(kDiffRSrc), high32(kDiffLDst),
                           {high32(kComb1LSrc), high32(kComb2LSrc), high32(kComb3LSrc), high32(kComb4LSrc)},
                           high32(kApf1LDst) - high32(kApf1Size), high32(kApf1LDst),
                           high32(kApf2LDst) - high32(kApf2Size), high32(kApf2LDst), vol(kInCoefL)};
            cfg.side[1] = {high32(kSameRSrc), high32(kSameRDst), high32(kDiffLSrc), high32(kDiffRDst),
                           {high32(kComb1RSrc), high32(kComb2RSrc), high32(kComb3RSrc), high32(kComb4RSrc)},
                           high32(kApf1RDst) - high32(kApf1Size), high32(kApf1RDst),
                           high32(kApf2RDst) - high32(kApf2Size), high32(kApf2RDst), vol(kInCoefR)};
        }
    }

    void Spu2Synth::incrementNax(Voice &voice)
    {
        for (uint32_t c = 0; c < kCoreCount; ++c)
            if (m_cfg[c].irqEnable && voice.nax == m_cfg[c].irqa)
                raiseIrq(c);
        voice.nax = (voice.nax + 1u) & kRamHalfwordMask;
    }

    // One sample of one voice (PCSX2 MixVoice). A stopped voice keeps walking through sound RAM, so
    // its end flags and interrupts still happen.
    void Spu2Synth::stepVoice(uint32_t c, uint32_t index, int32_t &outL, int32_t &outR)
    {
        Core &core = m_cores[c];
        Voice &voice = core.voices[index];
        slideUpdate(voice.vol[0]);
        slideUpdate(voice.vol[1]);

        const uint32_t block = voice.nax & 0xFFFF8u;
        for (uint32_t i = 0; i < kCoreCount; ++i)
            if (m_cfg[i].irqEnable && m_cfg[i].irqa == block)
                raiseIrq(i);
        voice.loopFlags = static_cast<uint8_t>(static_cast<uint16_t>(ramRead(block)) >> 8u); // 1 end, 2 repeat, 4 loop start
        if ((voice.loopFlags & 4u) != 0u && !voice.loopMode)
            voice.lsa = block;

        if (static_cast<int32_t>(voice.decWrite - voice.decRead) <= 12)
        {
            if (voice.phase != kStopped)
            {
                if (!voice.blockValid)
                    decodeBlock(voice, m_ram.data() + block * 2u);
                const uint32_t word = voice.nax & 7u;
                if (word != 0u)
                    for (uint32_t i = 0; i < 4u; ++i)
                        voice.fifo[(voice.decWrite + i) & 31u] = voice.block[(word - 1u) * 4u + i];
            }
            voice.decWrite += 4u;
            incrementNax(voice);
            if ((voice.nax & 7u) == 0u)
            {
                if ((voice.loopFlags & 1u) != 0u)
                {
                    core.endx |= 1u << index;
                    voice.nax = voice.lsa;
                    if ((voice.loopFlags & 2u) == 0u)
                        stopVoice(voice);
                }
                incrementNax(voice);
                voice.blockValid = false;
            }
        }

        int32_t value = 0;
        if (voice.phase != kStopped)
        {
            if ((m_cfg[c].non & (1u << index)) != 0u)
            {
                value = static_cast<int16_t>(core.noiseOut);
            }
            else
            {
                const int16_t *w = g_interpolation.row[(voice.sp & 0x0FF0) >> 4];
                for (uint32_t i = 0; i < 4u; ++i)
                    value += (w[i] * voice.fifo[(voice.decRead + i) & 31u]) >> 15;
            }
            if (!stepAdsr(voice))
                stopVoice(voice);
            value = mul15(value, voice.env);
        }
        voice.outX = value;
        outL = mul15(value, voice.vol[0].value);
        outR = mul15(value, voice.vol[1].value);

        int32_t pitch = voice.pitch;
        if ((m_cfg[c].pmon & (1u << index)) != 0u)
            pitch = static_cast<int32_t>(std::clamp<int64_t>((int64_t{pitch} * (32768 + core.voices[index - 1u].outX)) >> 15, 0, 0x3FFF));
        voice.sp += std::min(pitch, 0x3FFF);
        voice.decRead += static_cast<uint32_t>(voice.sp >> 12);
        voice.sp &= 0xFFF;

        // Voices 1 and 3 are captured into sound RAM, after the envelope.
        if (index == 1u)
            ramWrite((c ? 0xC00u : 0x400u) + m_outPos, static_cast<int16_t>(value));
        else if (index == 3u)
            ramWrite((c ? 0xE00u : 0x600u) + m_outPos, static_cast<int16_t>(value));
    }

    // One sample of one core's reverb for one bus (PCSX2 DoReverb): the effect runs at half rate, the
    // left channel on even samples and the right on odd ones, resampled on the way in and out.
    void Spu2Synth::reverb(uint32_t c, int bus, const int32_t in[2], int32_t out[2])
    {
        out[0] = out[1] = 0;
        const Config &cfg = m_cfg[c];
        if (!m_reverbEnabled || cfg.esa >= cfg.eea)
            return;
        Reverb &rv = m_cores[c].reverb[static_cast<size_t>(bus)];
        const uint32_t start = cfg.esa & 0x3FFFFFu;
        const uint32_t end = (cfg.eea & 0x3FFFFFu) | 0xFFFFu;
        const uint32_t size = end - start + 1u;
        int16_t *area = nullptr;
        if (bus != 0)
        {
            auto &buf = m_reverbRam[c * kBusCount + static_cast<uint32_t>(bus)];
            if (buf.size() != size)
                buf.assign(size, 0);
            area = buf.data();
        }
        const uint32_t r = m_busTicks[static_cast<size_t>(bus)] & 1u;
        const int64_t base = rv.areaPos % size; // the game may have shrunk the area
        if (r != 0u)
            rv.areaPos = static_cast<uint32_t>((base + 1) % size);
        // Offsets below the position (sameDst - 1, a destination minus its size) arrive wrapped: signed.
        const auto index = [&](uint32_t offset) -> uint32_t
        {
            const int64_t rel = (base + static_cast<int32_t>(offset)) % static_cast<int64_t>(size);
            return static_cast<uint32_t>(rel < 0 ? rel + size : rel);
        };
        const auto get = [&](uint32_t offset) -> int32_t
        {
            const uint32_t rel = index(offset);
            return area ? area[rel] : ramRead(rel + start);
        };
        const auto set = [&](uint32_t offset, int64_t value)
        {
            const uint32_t rel = index(offset);
            if (area)
                area[rel] = static_cast<int16_t>(clamp16(value));
            else
                ramWrite(rel + start, static_cast<int16_t>(clamp16(value)));
        };
        const auto mul = [](int64_t a, int64_t b) { return (a * b) >> 15; };

        for (uint32_t ch = 0; ch < 2u; ++ch)
            rv.down[ch][rv.pos] = rv.down[ch][rv.pos | 64u] = static_cast<int16_t>(clamp16(in[ch]));

        const Config::Side &s = cfg.side[r];
        const int64_t input = mul(s.inCoef, reverbFilter(rv.down[r], rv.pos, 1));
        const int64_t samePrev = get(s.sameDst - 1u), diffPrev = get(s.diffDst - 1u);
        const int64_t same = mul(cfg.iirVol, input + mul(cfg.wallVol, get(s.sameSrc)) - samePrev) + samePrev;
        const int64_t diff = mul(cfg.iirVol, input + mul(cfg.wallVol, get(s.diffSrc)) - diffPrev) + diffPrev;
        int64_t wet = 0;
        for (uint32_t i = 0; i < 4u; ++i)
            wet += mul(cfg.combVol[i], get(s.comb[i]));
        const int64_t apf1Src = get(s.apf1Src);
        const int64_t apf1 = wet - mul(cfg.apf1Vol, apf1Src);
        wet = apf1Src + mul(cfg.apf1Vol, apf1);
        const int64_t apf2Src = get(s.apf2Src);
        const int64_t apf2 = wet - mul(cfg.apf2Vol, apf2Src);
        wet = apf2Src + mul(cfg.apf2Vol, apf2);
        if (cfg.fxEnable)
        {
            set(s.sameDst, same);
            set(s.diffDst, diff);
            set(s.apf1Dst, apf1);
            set(s.apf2Dst, apf2);
        }

        rv.up[r][rv.pos] = rv.up[r][rv.pos | 64u] = static_cast<int16_t>(clamp16(wet));
        rv.up[r ^ 1u][rv.pos] = rv.up[r ^ 1u][rv.pos | 64u] = 0;
        rv.pos = (rv.pos + 1u) & 63u;
        out[0] = reverbFilter(rv.up[0], rv.pos, 2);
        out[1] = reverbFilter(rv.up[1], rv.pos, 2);
    }

    // One 48 kHz sample of both cores (PCSX2 spu2Mix / MixCore), with the mix kept apart per bus.
    void Spu2Synth::tick()
    {
        bool musicTick = true;
        if (m_musicRatio < 0x10000u)
        {
            m_musicAcc += m_musicRatio;
            musicTick = m_musicAcc >= 0x10000u;
            if (musicTick)
                m_musicAcc -= 0x10000u;
        }
        const auto runs = [&](int bus) { return bus != kBusMusic || musicTick; };

        int32_t ext[kBusCount][2] = {}; // core 0's output on its way into core 1
        for (uint32_t c = 0; c < kCoreCount; ++c)
        {
            Core &core = m_cores[c];
            const Config &cfg = m_cfg[c];

            if ((core.keyOff | core.keyOn) != 0u)
            {
                for (uint32_t v = 0; v < kVoicesPerCore; ++v)
                {
                    Voice &voice = core.voices[v];
                    if ((core.keyOff >> v) & 1u)
                    {
                        if (voice.phase != kStopped)
                        {
                            voice.phase = kRelease;
                            voice.envCounter = 0u;
                        }
                    }
                    if ((core.keyOn >> v) & 1u)
                    {
                        voice.phase = kAttack;
                        voice.env = 0;
                        voice.envCounter = 0u;
                        voice.loopMode = false;
                        voice.loopFlags = 0u;
                        voice.sp = 0;
                        voice.nax = voice.ssa | 1u;
                        voice.prev1 = voice.prev2 = 0;
                        voice.blockValid = false;
                        voice.decRead = voice.decWrite = 0u;
                        voice.bus = voice.nextBus;
                        m_busUsed |= 1u << voice.bus;
                    }
                }
                core.endx &= ~core.keyOn;
                core.keyOn = core.keyOff = 0u;
            }

            slideUpdate(core.mvol[0]);
            slideUpdate(core.mvol[1]);
            updateNoise(core, cfg.noiseClk);

            int32_t dry[kBusCount][2] = {}, wet[kBusCount][2] = {};
            for (uint32_t v = 0; v < kVoicesPerCore; ++v)
            {
                const uint32_t bus = core.voices[v].bus;
                if (!runs(static_cast<int>(bus)))
                    continue;
                int32_t l = 0, r = 0;
                stepVoice(c, v, l, r);
                const uint32_t bit = 1u << v;
                dry[bus][0] += (cfg.vmix[0] & bit) ? l : 0;
                dry[bus][1] += (cfg.vmix[1] & bit) ? r : 0;
                wet[bus][0] += (cfg.vmix[2] & bit) ? l : 0;
                wet[bus][1] += (cfg.vmix[3] & bit) ? r : 0;
            }

            int32_t input[2] = {0, 0};
            if (core.inputLeft != 0u)
            {
                const uint32_t pos = kInputFrames - core.inputLeft--;
                input[0] = mul15(core.input[pos], cfg.bvol[0]);
                input[1] = mul15(core.input[kInputFrames + pos], cfg.bvol[1]);
            }

            int32_t capture[3][2] = {}; // dry voices, wet voices, core output: what the capture areas get
            for (int bus = 0; bus < kBusCount; ++bus)
            {
                if ((m_busUsed & (1u << bus)) == 0u || !runs(bus))
                    continue;
                int32_t send[2], out[2];
                for (uint32_t ch = 0; ch < 2u; ++ch)
                {
                    // MMIX, right channel bits: 0x001 wet ext, 0x004 dry ext, 0x010 wet input,
                    // 0x040 dry input, 0x100 wet voices, 0x400 dry voices; the left ones are the next bit up.
                    const uint32_t gates = cfg.mmix >> (1u - ch);
                    const int32_t d = clamp16(dry[bus][ch]), w = clamp16(wet[bus][ch]);
                    const int32_t in = bus == core.inputBus ? input[ch] : 0;
                    capture[0][ch] += d;
                    capture[1][ch] += w;
                    out[ch] = ((gates & 0x040u) ? in : 0) + ((gates & 0x400u) ? d : 0) + ((gates & 0x004u) ? ext[bus][ch] : 0);
                    send[ch] = ((gates & 0x010u) ? in : 0) + ((gates & 0x100u) ? w : 0) + ((gates & 0x001u) ? ext[bus][ch] : 0);
                }
                int32_t echo[2];
                reverb(c, bus, send, echo);
                for (uint32_t ch = 0; ch < 2u; ++ch)
                {
                    const int32_t mixed = mul15(clamp16(out[ch] + mul15(echo[ch], cfg.evol[ch])), core.mvol[ch].value);
                    if (c == 0u)
                    {
                        capture[2][ch] += mixed;
                        ext[bus][ch] = mul15(mixed, m_cfg[1].avol[ch]);
                    }
                    else
                    {
                        m_out[static_cast<size_t>(bus)].push_back(static_cast<int16_t>(clamp16(mixed)));
                    }
                }
            }

            const uint32_t area = (c ? 0x1800u : 0x1000u) + m_outPos;
            for (uint32_t ch = 0; ch < 2u; ++ch)
            {
                ramWrite(area + ch * 0x200u, static_cast<int16_t>(clamp16(capture[0][ch])));
                ramWrite(area + 0x400u + ch * 0x200u, static_cast<int16_t>(clamp16(capture[1][ch])));
                if (c == 0u)
                    ramWrite(0x800u + ch * 0x200u + m_outPos, static_cast<int16_t>(clamp16(capture[2][ch])));
            }
        }

        m_outPos = (m_outPos + 1u) & 0x1FFu;
        for (int bus = 0; bus < kBusCount; ++bus)
        {
            if (!runs(bus))
                continue;
            ++m_busTicks[static_cast<size_t>(bus)];
            if ((m_busUsed & (1u << bus)) == 0u)
                m_out[static_cast<size_t>(bus)].insert(m_out[static_cast<size_t>(bus)].end(), 2u, int16_t{0});
        }
    }

    bool Spu2Synth::render(size_t frames)
    {
        if (frames != 0u)
        {
            if (m_trace)
            {
                static const auto traceStart = std::chrono::steady_clock::now();
#pragma pack(push, 1)
                const struct { uint32_t frames; uint64_t micros; } event{
                    static_cast<uint32_t>(frames),
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - traceStart).count())};
#pragma pack(pop)
                trace('R', &event, sizeof(event));
            }
            loadConfig();
            for (size_t f = 0; f < frames; ++f)
                tick();
        }
        for (const Core &core : m_cores)
        {
            if (core.inputLeft != 0u)
                return true;
            for (const Voice &voice : core.voices)
                if (voice.phase != kStopped)
                    return true;
        }
        return false;
    }

    void Spu2Synth::stateIo(ps2x::state::Archive &ar)
    {
        ar(m_ram, m_rawRegs, m_cores, m_busTicks, m_busUsed, m_outPos, m_musicAcc, m_irq);
        for (auto &area : m_reverbRam)
            ar(area);
        if (ar.loading())
            clearOutput(); // the chunk in progress belongs to the state being replaced
    }
}
