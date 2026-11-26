// Renders a PS2X_SPU2_TRACE file through Spu2Synth into a 48 kHz 16-bit stereo WAV, without the game:
// spu2_trace_replay <trace> <out.wav> [--no-reverb]. Same input, so two synth versions can be compared.
#include "emulator/core/spu2_synth.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace
{
    template <typename T>
    bool get(std::FILE *f, T &value)
    {
        return std::fread(&value, sizeof(value), 1u, f) == 1u;
    }

    void put32(std::FILE *f, uint32_t value)
    {
        std::fwrite(&value, 4u, 1u, f);
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: spu2_trace_replay <trace> <out.wav> [--no-reverb]\n");
        return 2;
    }
    std::FILE *in = std::fopen(argv[1], "rb");
    std::FILE *out = std::fopen(argv[2], "wb");
    if (!in || !out)
    {
        std::fprintf(stderr, "cannot open files\n");
        return 2;
    }
    const char header[44] = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ', 16, 0, 0, 0, 1, 0, 2, 0,
                             char(0x80), char(0xBB), 0, 0, 0, char(0xEE), 2, 0, 4, 0, 16, 0, 'd', 'a', 't', 'a', 0, 0, 0, 0};
    std::fwrite(header, 1u, sizeof(header), out);

    using ps2x::iop::detail::Spu2Synth;
    const auto synth = std::make_unique<Spu2Synth>();
    synth->setReverbEnabled(argc < 4 || std::strcmp(argv[3], "--no-reverb") != 0);
    std::vector<uint8_t> data;
    std::vector<int16_t> mix;
    uint32_t frames = 0u;

    for (int tag; (tag = std::fgetc(in)) != EOF;)
    {
        bool ok = true;
        if (tag == 'W')
        {
            uint16_t event[2];
            if ((ok = get(in, event)))
                synth->write16(0x1F900000u + event[0], event[1]);
        }
        else if (tag == 'D' || tag == 'd')
        {
            uint8_t core;
            uint32_t bytes;
            if ((ok = get(in, core) && get(in, bytes) && bytes <= (64u << 20)))
            {
                data.resize(bytes);
                if (tag == 'D')
                {
                    if ((ok = std::fread(data.data(), 1u, bytes, in) == bytes))
                        synth->dmaWrite(core, data.data(), bytes);
                }
                else
                    synth->dmaRead(core, data.data(), bytes);
            }
        }
        else if (tag == 'R')
        {
            uint32_t count;
            uint64_t micros;
            if ((ok = get(in, count) && get(in, micros) && count <= 48000u))
            {
                synth->render(count);
                mix.assign(count * 2u, 0);
                for (int bus = 0; bus < Spu2Synth::kBusCount; ++bus)
                {
                    const auto &pcm = synth->output(bus);
                    for (size_t i = 0u; i < std::min(mix.size(), pcm.size()); ++i)
                        mix[i] = static_cast<int16_t>(std::clamp(mix[i] + pcm[i], -32768, 32767));
                }
                synth->clearOutput();
                std::fwrite(mix.data(), sizeof(int16_t), mix.size(), out);
                frames += count;
            }
        }
        else if (tag == 'A')
        {
            // The PCM input of the samples that follow (traces of the older synth put it after the
            // chunk it was mixed into: those replay 256 samples late, which nothing can hear).
            uint8_t core, unused[8], block[1024];
            if ((ok = get(in, core) && get(in, unused) && get(in, block)))
                synth->setInputBlock(core, block);
        }
        else
            ok = false;
        if (!ok)
        {
            std::fprintf(stderr, "trace cut or bad at byte %ld (tag %d); writing what was rendered\n", std::ftell(in), tag);
            break;
        }
    }

    std::fseek(out, 4, SEEK_SET);
    put32(out, 36u + frames * 4u);
    std::fseek(out, 40, SEEK_SET);
    put32(out, frames * 4u);
    std::fclose(out);
    std::fclose(in);
    std::printf("%u frames (%.2f s)\n", frames, frames / 48000.0);
    return 0;
}
