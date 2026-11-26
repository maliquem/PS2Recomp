// Renders a PS2X_SPU2_TRACE file through Spu2Synth into a 48 kHz 16-bit stereo WAV, without the game:
// spu2_trace_replay <trace> <out.wav>. Same input, so two synth versions can be compared sample by sample.
#include "emulator/core/spu2_synth.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
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
        std::fprintf(stderr, "usage: spu2_trace_replay <trace> <out.wav>\n");
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

    ps2x::iop::detail::Spu2Synth synth;
    std::vector<int16_t> chunk; // last rendered chunk: AutoDMA blocks that follow it are mixed in before it is written
    std::vector<uint8_t> data;
    uint32_t frames = 0u;
    auto flush = [&]
    {
        std::fwrite(chunk.data(), sizeof(int16_t), chunk.size(), out);
        frames += static_cast<uint32_t>(chunk.size() / 2u);
        chunk.clear();
    };

    for (int tag; (tag = std::fgetc(in)) != EOF;)
    {
        bool ok = true;
        if (tag == 'W')
        {
            uint16_t event[2];
            if ((ok = get(in, event)))
                synth.write16(0x1F900000u + event[0], event[1]);
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
                        synth.dmaWrite(core, data.data(), bytes);
                }
                else
                    synth.dmaRead(core, data.data(), bytes);
            }
        }
        else if (tag == 'R')
        {
            uint32_t count;
            uint64_t micros;
            if ((ok = get(in, count) && get(in, micros) && count <= 48000u))
            {
                flush();
                chunk.assign(count * 2u, 0);
                synth.render(chunk.data(), count);
            }
        }
        else if (tag == 'A')
        {
            uint8_t core;
            int32_t volL, volR;
            int16_t block[512];
            if ((ok = get(in, core) && get(in, volL) && get(in, volR) && get(in, block)))
            {
                // Same mix as IopEmulator::advanceSpu2Audio: left in the first half of the block, right in the second.
                for (size_t i = 0u; i < std::min<size_t>(256u, chunk.size() / 2u); ++i)
                {
                    chunk[i * 2u] = static_cast<int16_t>(std::clamp(chunk[i * 2u] + ((block[i] * volL) >> 15), -32768, 32767));
                    chunk[i * 2u + 1u] = static_cast<int16_t>(std::clamp(chunk[i * 2u + 1u] + ((block[256u + i] * volR) >> 15), -32768, 32767));
                }
            }
        }
        else
            ok = false;
        if (!ok)
        {
            std::fprintf(stderr, "trace cut or bad at byte %ld (tag %d); writing what was rendered\n", std::ftell(in), tag);
            break;
        }
    }
    flush();

    std::fseek(out, 4, SEEK_SET);
    put32(out, 36u + frames * 4u);
    std::fseek(out, 40, SEEK_SET);
    put32(out, frames * 4u);
    std::fclose(out);
    std::fclose(in);
    std::printf("%u frames (%.2f s)\n", frames, frames / 48000.0);
    return 0;
}
