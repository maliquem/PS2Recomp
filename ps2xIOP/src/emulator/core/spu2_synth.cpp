#include "spu2_synth.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include "ps2x/state_archive.h"

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kSpu2Base = 0x1F900000u;
        constexpr int32_t kAdpcmFilter0[5] = {0, 60, 115, 98, 122};
        constexpr int32_t kAdpcmFilter1[5] = {0, 0, -52, -55, -60};

        void stepEnvelopeRate(int32_t &env, uint32_t &counter, uint32_t rate, bool decrease, bool exponential)
        {
            if (rate >= 0x7Fu)
                return;

            const uint32_t shift = rate >> 2u;
            uint32_t step = decrease ? static_cast<uint32_t>(~7 + static_cast<int32_t>(rate & 3u))
                                     : (7u - (rate & 3u));
            int32_t signedStep = decrease ? -static_cast<int32_t>(8u - (rate & 3u))
                                          : static_cast<int32_t>(step);

            uint32_t period = 1u;
            if (shift <= 11u)
            {
                signedStep <<= (11u - shift);
            }
            else
            {
                period = 1u << std::min<uint32_t>(shift - 11u, 20u);
            }

            if (exponential)
            {
                if (!decrease)
                {
                    if (env > 0x6000)
                        period <<= 2u;
                }
                else
                {
                    signedStep = static_cast<int32_t>((static_cast<int64_t>(signedStep) * env) >> 15);
                    if (signedStep == 0 && env > 0)
                        signedStep = -1;
                }
            }

            if (++counter < period)
                return;
            counter = 0u;
            env = std::clamp(env + signedStep, 0, 0x7FFF);
        }
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

    void Spu2Synth::traceAutoDma(int core, int32_t volL, int32_t volR, const uint8_t *block1024)
    {
        if (!m_trace)
            return;
#pragma pack(push, 1)
        const struct { uint8_t core; int32_t volL, volR; } head{static_cast<uint8_t>(core), volL, volR};
#pragma pack(pop)
        trace('A', &head, sizeof(head), block1024, 1024u);
    }

    void Spu2Synth::reset()
    {
        std::fill(m_ram.begin(), m_ram.end(), uint8_t{0});
        m_rawRegs.fill(0u);
        for (Core &core : m_cores)
        {
            core = Core{};
        }
        m_admas = 0u;
    }

    int32_t Spu2Synth::decodeVolume15(uint16_t reg) noexcept
    {
        if ((reg & 0x8000u) != 0u)
        {
            // Sweep mode fallback: preserve audibility at moderate gain.
            const uint32_t rate = reg & 0x7Fu;
            const bool decrease = (reg & 0x2000u) != 0u;
            return decrease ? 0 : static_cast<int32_t>((127u - rate) << 8u);
        }
        return static_cast<int16_t>(static_cast<uint16_t>(reg << 1u));
    }

    uint16_t Spu2Synth::read16(uint32_t physAddr) const
    {
        if (physAddr < kSpu2Base || physAddr >= kSpu2Base + 0x800u)
            return 0u;

        const uint32_t offset = (physAddr - kSpu2Base) & 0x7FEu;

        // Shared core master/external volume block (0x760..0x7AF)
        if (offset >= 0x760u && offset < 0x7B0u)
        {
            const uint32_t coreIndex = (offset >= 0x788u) ? 1u : 0u;
            const uint32_t local = offset - (coreIndex != 0u ? 0x788u : 0x760u);
            const Core &core = m_cores[coreIndex];
            switch (local)
            {
            case 0x00u: return core.mvolL;
            case 0x02u: return core.mvolR;
            case 0x04u: return core.evolL;
            case 0x06u: return core.evolR;
            case 0x08u: return core.avolL;
            case 0x0Au: return core.avolR;
            case 0x0Cu: return core.bvolL;
            case 0x0Eu: return core.bvolR;
            case 0x10u: return static_cast<uint16_t>(decodeVolume15(core.mvolL) >> 1);
            case 0x12u: return static_cast<uint16_t>(decodeVolume15(core.mvolR) >> 1);
            default: return m_rawRegs[offset >> 1u];
            }
        }

        const uint32_t coreIndex = (offset >= 0x400u) ? 1u : 0u;
        const uint32_t local = offset & 0x3FEu;
        const Core &core = m_cores[coreIndex];

        if (local < 0x180u)
        {
            const uint32_t v = local >> 4u;
            const uint32_t reg = local & 0x0Eu;
            const Voice &voice = core.voices[v];
            switch (reg)
            {
            case 0x00u: return voice.volL;
            case 0x02u: return voice.volR;
            case 0x04u: return voice.pitch;
            case 0x06u: return voice.adsr1;
            case 0x08u: return voice.adsr2;
            case 0x0Au: return static_cast<uint16_t>(voice.env);
            case 0x0Cu: return static_cast<uint16_t>(decodeVolume15(voice.volL) >> 1);
            case 0x0Eu: return static_cast<uint16_t>(decodeVolume15(voice.volR) >> 1);
            default: break;
            }
        }

        if (local >= 0x1C0u && local < 0x2E0u)
        {
            const uint32_t rel = local - 0x1C0u;
            const uint32_t v = rel / 12u;
            const uint32_t sub = rel % 12u;
            const Voice &voice = core.voices[v];
            switch (sub)
            {
            case 0x00u: return static_cast<uint16_t>((voice.ssa >> 16u) & 0x000Fu);
            case 0x02u: return static_cast<uint16_t>(voice.ssa & 0xFFFFu);
            case 0x04u: return static_cast<uint16_t>((voice.lsa >> 16u) & 0x000Fu);
            case 0x06u: return static_cast<uint16_t>(voice.lsa & 0xFFFFu);
            case 0x08u: return static_cast<uint16_t>((voice.nax >> 16u) & 0x000Fu);
            case 0x0Au: return static_cast<uint16_t>(voice.nax & 0xFFFFu);
            default: break;
            }
        }

        switch (local)
        {
        case 0x19Au: return core.attr;
        case 0x1A8u: return static_cast<uint16_t>((core.tsa >> 16u) & 0x000Fu);
        case 0x1AAu: return static_cast<uint16_t>(core.tsa & 0xFFFFu);
        case 0x1B0u: return m_admas;
        case 0x340u: return static_cast<uint16_t>(core.endx & 0xFFFFu);
        case 0x342u: return static_cast<uint16_t>((core.endx >> 16u) & 0x00FFu);
        case 0x344u: return static_cast<uint16_t>((core.attr & 0x003Fu) | 0x0080u);
        default: return m_rawRegs[offset >> 1u];
        }
    }

    void Spu2Synth::write16(uint32_t physAddr, uint16_t value)
    {
        if (physAddr < kSpu2Base || physAddr >= kSpu2Base + 0x800u)
            return;

        const uint32_t offset = (physAddr - kSpu2Base) & 0x7FEu;
        m_rawRegs[offset >> 1u] = value;
        if (m_trace)
        {
            const uint16_t event[2] = {static_cast<uint16_t>(offset), value};
            trace('W', event, sizeof(event));
        }

        if (offset >= 0x760u && offset < 0x7B0u)
        {
            const uint32_t coreIndex = (offset >= 0x788u) ? 1u : 0u;
            const uint32_t local = offset - (coreIndex != 0u ? 0x788u : 0x760u);
            Core &core = m_cores[coreIndex];
            switch (local)
            {
            case 0x00u: core.mvolL = value; core.mvolWritten = true; return;
            case 0x02u: core.mvolR = value; core.mvolWritten = true; return;
            case 0x04u: core.evolL = value; return;
            case 0x06u: core.evolR = value; return;
            case 0x08u: core.avolL = value; core.avolWritten = true; return;
            case 0x0Au: core.avolR = value; core.avolWritten = true; return;
            case 0x0Cu: core.bvolL = value; core.bvolWritten = true; return;
            case 0x0Eu: core.bvolR = value; core.bvolWritten = true; return;
            default: return;
            }
        }

        const uint32_t coreIndex = (offset >= 0x400u) ? 1u : 0u;
        const uint32_t local = offset & 0x3FEu;
        Core &core = m_cores[coreIndex];

        if (local < 0x180u)
        {
            const uint32_t v = local >> 4u;
            const uint32_t reg = local & 0x0Eu;
            Voice &voice = core.voices[v];
            switch (reg)
            {
            case 0x00u: voice.volL = value; return;
            case 0x02u: voice.volR = value; return;
            case 0x04u: voice.pitch = value; return;
            case 0x06u: voice.adsr1 = value; return;
            case 0x08u: voice.adsr2 = value; return;
            case 0x0Au: voice.env = value & 0x7FFF; return;
            default: return;
            }
        }

        if (local >= 0x1C0u && local < 0x2E0u)
        {
            const uint32_t rel = local - 0x1C0u;
            const uint32_t v = rel / 12u;
            const uint32_t sub = rel % 12u;
            Voice &voice = core.voices[v];
            switch (sub)
            {
            case 0x00u:
                voice.ssa = (voice.ssa & 0x0000FFFFu) | ((static_cast<uint32_t>(value) & 0x000Fu) << 16u);
                return;
            case 0x02u:
                voice.ssa = (voice.ssa & 0x000F0000u) | static_cast<uint32_t>(value);
                return;
            case 0x04u:
                voice.lsa = (voice.lsa & 0x0000FFFFu) | ((static_cast<uint32_t>(value) & 0x000Fu) << 16u);
                voice.lsaWritten = true;
                return;
            case 0x06u:
                voice.lsa = (voice.lsa & 0x000F0000u) | static_cast<uint32_t>(value);
                voice.lsaWritten = true;
                return;
            default:
                return;
            }
        }

        switch (local)
        {
        case 0x19Au:
            core.attr = value;
            core.statx = static_cast<uint16_t>((value & 0x003Fu) | 0x0080u);
            return;
        case 0x1A0u:
            keyOn(core, static_cast<uint32_t>(value));
            return;
        case 0x1A2u:
            keyOn(core, (static_cast<uint32_t>(value) & 0x00FFu) << 16u);
            return;
        case 0x1A4u:
            keyOff(core, static_cast<uint32_t>(value));
            return;
        case 0x1A6u:
            keyOff(core, (static_cast<uint32_t>(value) & 0x00FFu) << 16u);
            return;
        case 0x1A8u:
            core.tsa = (core.tsa & 0x0000FFFFu) | ((static_cast<uint32_t>(value) & 0x000Fu) << 16u);
            return;
        case 0x1AAu:
            core.tsa = (core.tsa & 0x000F0000u) | static_cast<uint32_t>(value);
            return;
        case 0x1ACu:
        {
            const uint32_t byteAddr = (core.tsa & kRamHalfwordMask) * 2u;
            m_ram[byteAddr] = static_cast<uint8_t>(value & 0xFFu);
            m_ram[byteAddr + 1u] = static_cast<uint8_t>(value >> 8u);
            core.tsa = (core.tsa + 1u) & kRamHalfwordMask;
            return;
        }
        case 0x1B0u:
            m_admas = value;
            return;
        case 0x340u:
            core.endx = 0u;
            return;
        case 0x342u:
            core.endx = 0u;
            return;
        default:
            return;
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
        uint32_t dstByte = (core.tsa & kRamHalfwordMask) * 2u;
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
        core.statx = static_cast<uint16_t>((core.attr & 0x003Fu) | 0x0080u);
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
        core.statx = static_cast<uint16_t>((core.attr & 0x003Fu) | 0x0080u);
    }

    bool Spu2Synth::isAutoDmaEnabled(int coreIndex) const noexcept
    {
        if (coreIndex < 0 || coreIndex >= static_cast<int>(kCoreCount))
            return false;
        const uint16_t coreMask = static_cast<uint16_t>(1u << static_cast<uint32_t>(coreIndex));
        return (m_admas & coreMask) != 0u || (m_admas & 4u) != 0u;
    }

    void Spu2Synth::getAutoDmaVolume(int coreIndex, int32_t &outLeft, int32_t &outRight) const noexcept
    {
        if (coreIndex < 0 || coreIndex >= static_cast<int>(kCoreCount))
        {
            outLeft = 0;
            outRight = 0;
            return;
        }
        const Core &core = m_cores[static_cast<size_t>(coreIndex)];
        const int32_t mL = core.mvolWritten ? decodeVolume15(core.mvolL) : 0x7FFE;
        const int32_t mR = core.mvolWritten ? decodeVolume15(core.mvolR) : 0x7FFE;
        // BVOL/AVOL are signed 16-bit gains (0x7FFF = full), unlike 15-bit sweep-flagged MVOL/VOLL.
        int32_t aL = 0x7FFF;
        int32_t aR = 0x7FFF;
        if (core.bvolWritten)
        {
            aL = static_cast<int16_t>(core.bvolL);
            aR = static_cast<int16_t>(core.bvolR);
        }
        else if (core.avolWritten)
        {
            aL = static_cast<int16_t>(core.avolL);
            aR = static_cast<int16_t>(core.avolR);
        }
        outLeft = static_cast<int32_t>((static_cast<int64_t>(aL) * mL) >> 15);
        outRight = static_cast<int32_t>((static_cast<int64_t>(aR) * mR) >> 15);
    }

    bool Spu2Synth::hasActiveVoices() const noexcept
    {
        for (const Core &core : m_cores)
        {
            for (const Voice &voice : core.voices)
            {
                if (voice.phase != AdsrPhase::Off)
                    return true;
            }
        }
        return false;
    }

    void Spu2Synth::keyOn(Core &core, uint32_t mask)
    {
        for (uint32_t v = 0u; v < kVoicesPerCore; ++v)
        {
            if ((mask & (1u << v)) == 0u)
                continue;
            Voice &voice = core.voices[v];
            voice.nax = voice.ssa & ~7u;
            voice.old1 = 0;
            voice.old2 = 0;
            voice.prevSample = 0;
            voice.curSample = 0;
            voice.pitchCounter = 0u;
            voice.sampleIndex = 28u;
            voice.blockFlags = 0u;
            voice.pendingStop = false;
            voice.env = 0;
            voice.envCounter = 0u;
            voice.phase = AdsrPhase::Attack;
            voice.lsaWritten = false;
            core.endx &= ~(1u << v);
        }
    }

    void Spu2Synth::keyOff(Core &core, uint32_t mask)
    {
        for (uint32_t v = 0u; v < kVoicesPerCore; ++v)
        {
            if ((mask & (1u << v)) == 0u)
                continue;
            Voice &voice = core.voices[v];
            if (voice.phase != AdsrPhase::Off)
            {
                voice.phase = AdsrPhase::Release;
                voice.envCounter = 0u;
            }
        }
    }

    void Spu2Synth::decodeBlock(Voice &voice, Core &core, uint32_t voiceIndex)
    {
        const uint32_t blockAddr = (voice.nax & ~7u) & kRamHalfwordMask;
        const uint32_t byteAddr = blockAddr * 2u;
        const uint8_t *block = m_ram.data() + byteAddr;

        uint32_t shift = block[0] & 0x0Fu;
        if (shift > 12u)
            shift = 9u;
        uint32_t filter = (block[0] >> 4u) & 0x07u;
        if (filter > 4u)
            filter = 4u;

        const uint8_t flags = block[1];
        voice.blockFlags = flags;
        if ((flags & 0x04u) != 0u && !voice.lsaWritten)
        {
            voice.lsa = blockAddr;
        }

        const int32_t f0 = kAdpcmFilter0[filter];
        const int32_t f1 = kAdpcmFilter1[filter];
        int32_t old1 = voice.old1;
        int32_t old2 = voice.old2;

        for (uint32_t i = 0u; i < 14u; ++i)
        {
            const uint8_t packed = block[2u + i];
            const int32_t nibbles[2] = {
                static_cast<int16_t>(static_cast<uint16_t>(packed & 0x0Fu) << 12u),
                static_cast<int16_t>(static_cast<uint16_t>(packed & 0xF0u) << 8u),
            };
            for (uint32_t n = 0u; n < 2u; ++n)
            {
                int32_t sample = (nibbles[n] >> shift) + ((old1 * f0 + old2 * f1 + 32) >> 6);
                sample = std::clamp(sample, -32768, 32767);
                voice.blockSamples[i * 2u + n] = static_cast<int16_t>(sample);
                old2 = old1;
                old1 = sample;
            }
        }

        voice.old1 = old1;
        voice.old2 = old2;
        voice.sampleIndex = 0u;

        if ((flags & 0x01u) != 0u)
        {
            core.endx |= (1u << voiceIndex);
            voice.nax = voice.lsa & ~7u;
            if ((flags & 0x02u) == 0u)
            {
                voice.pendingStop = true;
            }
        }
        else
        {
            voice.nax = (blockAddr + 8u) & kRamHalfwordMask;
        }
    }

    void Spu2Synth::stepAdsr(Voice &voice)
    {
        switch (voice.phase)
        {
        case AdsrPhase::Off:
            voice.env = 0;
            return;
        case AdsrPhase::Attack:
        {
            const uint32_t rate = (voice.adsr1 >> 8u) & 0x7Fu;
            const bool expMode = (voice.adsr1 & 0x8000u) != 0u;
            stepEnvelopeRate(voice.env, voice.envCounter, rate, false, expMode);
            if (voice.env >= 0x7FFF)
            {
                voice.env = 0x7FFF;
                voice.phase = AdsrPhase::Decay;
                voice.envCounter = 0u;
            }
            return;
        }
        case AdsrPhase::Decay:
        {
            const uint32_t decayShift = (voice.adsr1 >> 4u) & 0x0Fu;
            const uint32_t sl = voice.adsr1 & 0x0Fu;
            const int32_t sustainLevel = (sl == 15u) ? 0x7FFF : static_cast<int32_t>((sl + 1u) << 11u);
            stepEnvelopeRate(voice.env, voice.envCounter, decayShift << 2u, true, true);
            if (voice.env <= sustainLevel)
            {
                voice.phase = AdsrPhase::Sustain;
                voice.envCounter = 0u;
            }
            return;
        }
        case AdsrPhase::Sustain:
        {
            const uint32_t rate = (voice.adsr2 >> 6u) & 0x7Fu;
            const bool decrease = (voice.adsr2 & 0x4000u) != 0u;
            const bool expMode = (voice.adsr2 & 0x8000u) != 0u;
            stepEnvelopeRate(voice.env, voice.envCounter, rate, decrease, expMode);
            return;
        }
        case AdsrPhase::Release:
        {
            const uint32_t relShift = voice.adsr2 & 0x1Fu;
            const bool expMode = (voice.adsr2 & 0x0020u) != 0u;
            stepEnvelopeRate(voice.env, voice.envCounter, relShift << 2u, true, expMode);
            if (voice.env <= 0)
            {
                voice.env = 0;
                voice.phase = AdsrPhase::Off;
            }
            return;
        }
        }
    }

    bool Spu2Synth::render(int16_t *interleavedStereo, size_t frameCount)
    {
        if (!interleavedStereo || frameCount == 0u)
            return false;

        if (m_trace)
        {
            using Clock = std::chrono::steady_clock;
            static const Clock::time_point start = Clock::now();
#pragma pack(push, 1)
            const struct { uint32_t frames; uint64_t micros; } event{static_cast<uint32_t>(frameCount),
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count())};
#pragma pack(pop)
            trace('R', &event, sizeof(event));
            std::fflush(m_trace.get()); // the test scripts kill the process
        }

        std::memset(interleavedStereo, 0, frameCount * 2u * sizeof(int16_t));
        bool anyActive = false;

        for (size_t f = 0u; f < frameCount; ++f)
        {
            int32_t mixL = 0;
            int32_t mixR = 0;

            for (uint32_t c = 0u; c < kCoreCount; ++c)
            {
                Core &core = m_cores[c];
                int32_t coreL = 0;
                int32_t coreR = 0;

                for (uint32_t v = 0u; v < kVoicesPerCore; ++v)
                {
                    Voice &voice = core.voices[v];
                    if (voice.phase == AdsrPhase::Off)
                        continue;

                    anyActive = true;
                    if (voice.sampleIndex >= 28u)
                    {
                        decodeBlock(voice, core, v);
                        if (voice.phase == AdsrPhase::Off)
                            continue;
                        voice.prevSample = voice.curSample;
                        voice.curSample = voice.blockSamples[0];
                        voice.sampleIndex = 1u;
                    }

                    const uint32_t frac = voice.pitchCounter & 0x0FFFu;
                    const int32_t interp = voice.prevSample +
                                           (((voice.curSample - voice.prevSample) * static_cast<int32_t>(frac)) >> 12);
                    const int32_t envSample = (interp * voice.env) >> 15;

                    const int32_t vL = decodeVolume15(voice.volL);
                    const int32_t vR = decodeVolume15(voice.volR);
                    coreL += (envSample * vL) >> 15;
                    coreR += (envSample * vR) >> 15;

                    stepAdsr(voice);

                    const uint32_t step = std::min<uint32_t>(voice.pitch, 0x3FFFu);
                    voice.pitchCounter += step;
                    while (voice.pitchCounter >= 0x1000u)
                    {
                        voice.pitchCounter -= 0x1000u;
                        if (voice.sampleIndex >= 28u)
                        {
                            if (voice.pendingStop)
                            {
                                voice.env = 0;
                                voice.phase = AdsrPhase::Off;
                                voice.pendingStop = false;
                                break;
                            }
                            decodeBlock(voice, core, v);
                            if (voice.phase == AdsrPhase::Off)
                                break;
                        }
                        voice.prevSample = voice.curSample;
                        voice.curSample = voice.blockSamples[voice.sampleIndex++];
                    }
                }

                const int32_t mL = core.mvolWritten ? decodeVolume15(core.mvolL) : 0x7FFE;
                const int32_t mR = core.mvolWritten ? decodeVolume15(core.mvolR) : 0x7FFE;
                mixL += (coreL * mL) >> 15;
                mixR += (coreR * mR) >> 15;
            }

            interleavedStereo[f * 2u] = static_cast<int16_t>(std::clamp(mixL, -32768, 32767));
            interleavedStereo[f * 2u + 1u] = static_cast<int16_t>(std::clamp(mixR, -32768, 32767));
        }

        return anyActive;
    }

    void Spu2Synth::stateIo(ps2x::state::Archive &ar)
    {
        ar(m_ram, m_rawRegs, m_cores, m_admas);
    }
}
