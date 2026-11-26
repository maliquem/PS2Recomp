#include "runtime/ps2_audio.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_audio_mix.h"
#include "runtime/ps2_pacing.h"
#include "ps2_host_backend.h"
#if !PS2X_HOST_RAYLIB
#include <SDL3/SDL.h>
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace
{
    // SPU2 output: the IOP core pushes one PCM stream per bus, the device callback pulls the mix
    // (runtime/ps2_audio_mix.h).
    std::mutex g_spu2RingMutex;
    ps2x::audio::OutputMixer g_spu2Mixer;
    double g_spu2Rate = 1.0; // frequency ratio the SDL stream is at (under the backend mutex)

#if !defined(PLATFORM_VITA)
    void spu2AudioCallback(void *bufferData, unsigned int frames)
    {
        std::lock_guard<std::mutex> lock(g_spu2RingMutex);
        g_spu2Mixer.pull(static_cast<int16_t *>(bufferData), frames,
                         ps2x::pacing::g_speedPercent.load(std::memory_order_relaxed));
    }
#endif

#if !PS2X_HOST_RAYLIB
    // SDL pulls the SPU2 stream from the audio thread; feed it from the rings.
    void SDLCALL spu2StreamCallback(void *, SDL_AudioStream *stream, int additionalAmount, int)
    {
        static thread_local std::vector<int16_t> mix;
        const unsigned int frames = static_cast<unsigned int>(std::max(additionalAmount, 0)) / 4u;
        if (frames == 0u)
            return;
        mix.resize(static_cast<size_t>(frames) * 2u);
        spu2AudioCallback(mix.data(), frames);
        SDL_PutAudioStreamData(stream, mix.data(), static_cast<int>(frames * 4u));
    }
#endif

#if PS2X_HOST_RAYLIB
    std::vector<uint8_t> buildWavFromPcm(const int16_t *pcm, size_t sampleCount, uint32_t sampleRate)
    {
        const uint32_t dataSize = static_cast<uint32_t>(sampleCount * 2);
        const uint32_t fileSize = 36 + dataSize;
        std::vector<uint8_t> wav(8 + fileSize);

        uint8_t *p = wav.data();
        p[0] = 'R';
        p[1] = 'I';
        p[2] = 'F';
        p[3] = 'F';
        p[4] = static_cast<uint8_t>(fileSize);
        p[5] = static_cast<uint8_t>(fileSize >> 8);
        p[6] = static_cast<uint8_t>(fileSize >> 16);
        p[7] = static_cast<uint8_t>(fileSize >> 24);
        p[8] = 'W';
        p[9] = 'A';
        p[10] = 'V';
        p[11] = 'E';
        p[12] = 'f';
        p[13] = 'm';
        p[14] = 't';
        p[15] = ' ';
        p[16] = 16;
        p[17] = 0;
        p[18] = 0;
        p[19] = 0;
        p[20] = 1;
        p[21] = 0;
        p[22] = 1;
        p[23] = 0;
        p[24] = static_cast<uint8_t>(sampleRate);
        p[25] = static_cast<uint8_t>(sampleRate >> 8);
        p[26] = static_cast<uint8_t>(sampleRate >> 16);
        p[27] = static_cast<uint8_t>(sampleRate >> 24);
        const uint32_t byteRate = sampleRate * 2;
        p[28] = static_cast<uint8_t>(byteRate);
        p[29] = static_cast<uint8_t>(byteRate >> 8);
        p[30] = static_cast<uint8_t>(byteRate >> 16);
        p[31] = static_cast<uint8_t>(byteRate >> 24);
        p[32] = 2;
        p[33] = 0;
        p[34] = 16;
        p[35] = 0;
        p[36] = 'd';
        p[37] = 'a';
        p[38] = 't';
        p[39] = 'a';
        p[40] = static_cast<uint8_t>(dataSize);
        p[41] = static_cast<uint8_t>(dataSize >> 8);
        p[42] = static_cast<uint8_t>(dataSize >> 16);
        p[43] = static_cast<uint8_t>(dataSize >> 24);
        std::memcpy(p + 44, pcm, dataSize);
        return wav;
    }
#endif

#if !defined(PLATFORM_VITA)
#if PS2X_HOST_RAYLIB
    using HostSound = Sound;
    bool soundPlaying(const HostSound &s) { return IsSoundPlaying(s); }
    void releaseSound(HostSound &s)
    {
        StopSound(s);
        UnloadSound(s);
    }
    bool startSound(HostSound &out, const std::vector<int16_t> &pcm, uint32_t sampleRate, float pitch, float volume)
    {
        std::vector<uint8_t> wav = buildWavFromPcm(pcm.data(), pcm.size(), sampleRate);
        Wave wave = LoadWaveFromMemory(".wav", wav.data(), static_cast<int>(wav.size()));
        if (wave.frameCount <= 0)
            return false;
        out = LoadSoundFromWave(wave);
        UnloadWave(wave);
        SetSoundPitch(out, pitch);
        SetSoundVolume(out, volume);
        PlaySound(out);
        return true;
    }
#else
    // One SDL stream per sound: mono S16 at the sample's rate, pitch as frequency ratio, mixed by the device.
    using HostSound = SDL_AudioStream *;
    bool soundPlaying(const HostSound &s)
    {
        return s && (SDL_GetAudioStreamQueued(s) > 0 || SDL_GetAudioStreamAvailable(s) > 0);
    }
    void releaseSound(HostSound &s)
    {
        SDL_DestroyAudioStream(s);
        s = nullptr;
    }
    bool startSound(HostSound &out, const std::vector<int16_t> &pcm, uint32_t sampleRate, float pitch, float volume)
    {
        const SDL_AudioSpec srcSpec{SDL_AUDIO_S16, 1, static_cast<int>(sampleRate)};
        const SDL_AudioSpec dstSpec{SDL_AUDIO_S16, 2, 48000};
        out = SDL_CreateAudioStream(&srcSpec, &dstSpec);
        if (!out)
            return false;
        SDL_SetAudioStreamFrequencyRatio(out, std::clamp(pitch, 0.01f, 100.0f));
        SDL_SetAudioStreamGain(out, volume);
        SDL_PutAudioStreamData(out, pcm.data(), static_cast<int>(pcm.size() * sizeof(int16_t)));
        SDL_FlushAudioStream(out);
        if (!SDL_BindAudioStream(ps2host::audioDevice(), out))
        {
            releaseSound(out);
            return false;
        }
        return true;
    }
#endif
#endif
}

namespace ps2_vag
{
    bool decode(const uint8_t *data, uint32_t sizeBytes,
                std::vector<int16_t> &outPcm, uint32_t &outSampleRate);
}

struct PS2AudioBackend::Impl
{
#if !defined(PLATFORM_VITA)
    struct TrackedSound
    {
        HostSound snd;
        uint32_t sampleKey;
    };
    std::vector<TrackedSound> activeSounds;
#if PS2X_HOST_RAYLIB
    AudioStream spu2Stream{};
#else
    SDL_AudioStream *spu2Stream = nullptr;
#endif
    bool spu2StreamStarted = false;
#endif
};

PS2AudioBackend::PS2AudioBackend() : m_impl(std::make_unique<Impl>())
{
}

PS2AudioBackend::~PS2AudioBackend()
{
    if (m_impl)
        stopAll();
}

void PS2AudioBackend::onVagTransfer(const uint8_t *rdram, uint32_t srcAddr, uint32_t sizeBytes)
{
    if (!rdram || sizeBytes < 48)
        return;

    const uint32_t physAddr = srcAddr & PS2_RAM_MASK;
    if (physAddr + sizeBytes > PS2_RAM_SIZE)
        return;

    std::vector<int16_t> pcm;
    uint32_t sampleRate = 44100;
    if (!ps2_vag::decode(rdram + physAddr, sizeBytes, pcm, sampleRate))
        return;

    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample sample;
    sample.pcm = std::move(pcm);
    sample.sampleRate = sampleRate;
    m_sampleBank[physAddr] = std::move(sample);
    m_mostRecentSampleKey = physAddr;
}

void PS2AudioBackend::onVagTransferFromBuffer(const uint8_t *data, uint32_t sizeBytes, uint32_t keyAddr)
{
    if (!data || sizeBytes < 48)
        return;

    std::vector<int16_t> pcm;
    uint32_t sampleRate = 44100;
    if (!ps2_vag::decode(data, sizeBytes, pcm, sampleRate))
        return;

    const uint32_t physAddr = keyAddr & PS2_RAM_MASK;
    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample sample;
    sample.pcm = std::move(pcm);
    sample.sampleRate = sampleRate;
    m_sampleBank[physAddr] = sample;
    m_mostRecentSampleKey = physAddr;
    m_loadOrderSamples.push_back(std::move(sample));
    m_loadOrderSampleKeys.push_back(physAddr);
    constexpr size_t kMaxLoadOrderSamples = 32;
    if (m_loadOrderSamples.size() > kMaxLoadOrderSamples)
    {
        m_loadOrderSamples.erase(m_loadOrderSamples.begin());
        m_loadOrderSampleKeys.erase(m_loadOrderSampleKeys.begin());
    }
}

namespace
{
    constexpr uint32_t LIBSD_CMD_SET_VOICE = 0x8010u;
}

void PS2AudioBackend::onSoundCommand(uint32_t sid, uint32_t rpcNum,
                                     const uint8_t *sendBuf, uint32_t sendSize,
                                     uint8_t *recvBuf, uint32_t recvSize)
{
    if (sid != 0x80000701u)
        return;

    if ((rpcNum == LIBSD_CMD_SET_VOICE || (rpcNum & 0xFF00u) == 0x8100u) &&
        sendBuf && sendSize >= 20)
    {
        uint32_t sampleAddr = 0;
        uint32_t voiceIndex = 0xFFFFFFFFu;
        for (int vo = 4; vo >= 0 && voiceIndex == 0xFFFFFFFFu; vo -= 4)
        {
            if (vo < static_cast<int>(sendSize))
            {
                uint32_t v = 0;
                std::memcpy(&v, sendBuf + vo, sizeof(v));
                if (v < 24u)
                    voiceIndex = v;
            }
        }

        constexpr uint32_t kMinPlausibleAddr = 0x1000u;
        for (int off = 12; off <= 24 && sampleAddr == 0; off += 4)
        {
            if (sendSize >= static_cast<uint32_t>(off + 4))
            {
                uint32_t cand = 0;
                std::memcpy(&cand, sendBuf + off, sizeof(cand));
                if (cand >= kMinPlausibleAddr && (cand <= PS2_RAM_MASK || (cand & ~PS2_RAM_MASK) == 0))
                    sampleAddr = cand;
            }
        }
        if (sampleAddr == 0)
            sampleAddr = m_mostRecentSampleKey;

        float pitch = 1.0f;
        if (sendSize >= 12)
        {
            uint16_t pitchHalf = 0;
            std::memcpy(&pitchHalf, sendBuf + 8, sizeof(pitchHalf));
            if (pitchHalf != 0)
                pitch = 4096.0f / static_cast<float>(pitchHalf);
        }
        play(sampleAddr, pitch, 1.0f, voiceIndex);
    }
}

void PS2AudioBackend::play(uint32_t sampleAddr, float pitch, float volume, uint32_t voiceIndex)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample *sampleToPlay = nullptr;
    uint32_t sampleKey = 0;

    auto it = m_sampleBank.find(sampleAddr & PS2_RAM_MASK);
    if (it != m_sampleBank.end())
    {
        sampleToPlay = &it->second;
        sampleKey = it->first;
    }
    else if (voiceIndex != 0xFFFFFFFFu &&
             voiceIndex < m_loadOrderSamples.size() &&
             voiceIndex < m_loadOrderSampleKeys.size())
    {
        sampleToPlay = &m_loadOrderSamples[voiceIndex];
        sampleKey = m_loadOrderSampleKeys[voiceIndex];
    }
    else
    {
        it = m_sampleBank.find(m_mostRecentSampleKey);
        if (it == m_sampleBank.end())
            return;
        sampleToPlay = &it->second;
        sampleKey = it->first;
    }
    if (!sampleToPlay || sampleToPlay->pcm.empty())
        return;

    const bool isBgm = (sampleToPlay->pcm.size() > static_cast<size_t>(sampleToPlay->sampleRate * 5));
    playDecodedSample(sampleKey, *sampleToPlay, pitch, volume, isBgm);
}

void PS2AudioBackend::pruneFinishedSounds()
{
#if defined(PLATFORM_VITA)
    return;
#else
    auto &sounds = m_impl->activeSounds;
    auto it = sounds.begin();
    while (it != sounds.end())
    {
        if (!soundPlaying(it->snd))
        {
            releaseSound(it->snd);
            it = sounds.erase(it);
        }
        else
        {
            ++it;
        }
    }
#endif
}

void PS2AudioBackend::playDecodedSample(uint32_t sampleKey, DecodedSample &sample, float pitch, float volume,
                                        bool isBgm)
{
#if defined(PLATFORM_VITA)
    (void)sampleKey;
    (void)sample;
    (void)pitch;
    (void)volume;
    (void)isBgm;
    return;
#else
    if (!m_audioReady || sample.pcm.empty())
        return;

    pruneFinishedSounds();

    for (const auto &t : m_impl->activeSounds)
    {
        if (t.sampleKey == sampleKey && soundPlaying(t.snd))
            return;
    }

    auto &sounds = m_impl->activeSounds;
    if (isBgm)
    {
        for (auto it = sounds.begin(); it != sounds.end();)
        {
            if (soundPlaying(it->snd))
            {
                releaseSound(it->snd);
                it = sounds.erase(it);
            }
            else
                ++it;
        }
    }

    constexpr int kMaxConcurrentSounds = 4;
    while (static_cast<int>(sounds.size()) >= kMaxConcurrentSounds)
    {
        releaseSound(sounds.front().snd);
        sounds.erase(sounds.begin());
    }

    HostSound snd{};
    if (startSound(snd, sample.pcm, sample.sampleRate, pitch, volume))
        m_impl->activeSounds.push_back({snd, sampleKey});
#endif
}

void PS2AudioBackend::stop(uint32_t voiceId)
{
    (void)voiceId;
}

void PS2AudioBackend::submitSpu2Pcm(int bus, const int16_t *interleavedStereoPcm, size_t frameCount)
{
#if defined(PLATFORM_VITA)
    (void)bus;
    (void)interleavedStereoPcm;
    (void)frameCount;
    return;
#else
    if (!m_audioReady || !interleavedStereoPcm || frameCount == 0u)
        return;

    double rate = 1.0;
    {
        std::lock_guard<std::mutex> lock(g_spu2RingMutex);
        g_spu2Mixer.push(bus, interleavedStereoPcm, frameCount);
        rate = g_spu2Mixer.rateRatio();
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_impl->spu2StreamStarted)
    {
#if PS2X_HOST_RAYLIB
        m_impl->spu2Stream = LoadAudioStream(48000u, 16u, 2u);
        if (m_impl->spu2Stream.buffer != nullptr)
        {
            SetAudioStreamCallback(m_impl->spu2Stream, &spu2AudioCallback);
            PlayAudioStream(m_impl->spu2Stream);
            m_impl->spu2StreamStarted = true;
        }
#else
        const SDL_AudioSpec spec{SDL_AUDIO_S16, 2, 48000};
        m_impl->spu2Stream = SDL_CreateAudioStream(&spec, &spec);
        if (m_impl->spu2Stream)
        {
            SDL_SetAudioStreamGetCallback(m_impl->spu2Stream, &spu2StreamCallback, nullptr);
            if (SDL_BindAudioStream(ps2host::audioDevice(), m_impl->spu2Stream))
                m_impl->spu2StreamStarted = true;
            else
            {
                SDL_DestroyAudioStream(m_impl->spu2Stream);
                m_impl->spu2Stream = nullptr;
            }
        }
        g_spu2Rate = 1.0;
#endif
    }
#if !PS2X_HOST_RAYLIB
    // The game makes audio a little off the device's 48 kHz (measured -0.13 %): follow it by playing
    // that much slower or faster. shortcut: the raylib host has no rate control, it keeps the gaps.
    if (m_impl->spu2Stream && std::abs(rate - g_spu2Rate) > 1e-5)
    {
        SDL_SetAudioStreamFrequencyRatio(m_impl->spu2Stream, static_cast<float>(rate));
        g_spu2Rate = rate;
    }
#endif
#endif
}

double PS2AudioBackend::spu2MusicRatio()
{
    std::lock_guard<std::mutex> lock(g_spu2RingMutex);
    return g_spu2Mixer.musicRatio(ps2x::pacing::g_speedPercent.load(std::memory_order_relaxed));
}

void PS2AudioBackend::stopAll()
{
    std::lock_guard<std::mutex> lock(m_mutex);
#if defined(PLATFORM_VITA)
    return;
#else
    for (auto &t : m_impl->activeSounds)
        releaseSound(t.snd);
    m_impl->activeSounds.clear();
    if (m_impl->spu2StreamStarted)
    {
#if PS2X_HOST_RAYLIB
        StopAudioStream(m_impl->spu2Stream);
        UnloadAudioStream(m_impl->spu2Stream);
        m_impl->spu2Stream = AudioStream{};
#else
        SDL_DestroyAudioStream(m_impl->spu2Stream);
        m_impl->spu2Stream = nullptr;
#endif
        m_impl->spu2StreamStarted = false;
    }
    {
        std::lock_guard<std::mutex> ringLock(g_spu2RingMutex);
        g_spu2Mixer.reset();
    }
#endif
}
