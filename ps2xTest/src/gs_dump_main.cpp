// ps2x_gsdump: replays a PCSX2 GS dump through our GS and writes the frame presented at each VSync.
//   ps2x_gsdump <dump.gs> <out_dir> [cpu|vulkan|both]
// Writes <out_dir>/<stem>.<backend>.<NNN>.ppm (RGB, one per VSync with a frame on display) and
// <out_dir>/<stem>.<backend>.json (frames, warnings). tools/gs-dump/compare.py reads both.
// Exit: 0 done, 1 bad arguments or unreadable dump, 2 Vulkan asked for and not available.

#include "gs_dump_replay.h"

#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/gs_vulkan_backend.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace
{
    std::string jsonString(const std::string &s)
    {
        std::string out = "\"";
        for (const char ch : s)
        {
            const unsigned char c = static_cast<unsigned char>(ch);
            if (c == '"' || c == '\\')
            {
                out += '\\';
                out += ch;
            }
            else if (c < 0x20u)
            {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            }
            else
                out += ch;
        }
        return out + "\"";
    }

    bool writePpm(const fs::path &path, const gsdump::Frame &f)
    {
        std::ofstream out(path, std::ios::binary);
        out << "P6\n" << f.width << " " << f.height << "\n255\n";
        std::string row(static_cast<size_t>(f.width) * 3u, '\0');
        for (uint32_t y = 0; y < f.height; ++y)
        {
            const uint8_t *src = f.rgba.data() + static_cast<size_t>(y) * f.width * 4u;
            for (uint32_t x = 0; x < f.width; ++x)
                for (int c = 0; c < 3; ++c)
                    row[x * 3u + c] = static_cast<char>(src[x * 4u + c]);
            out.write(row.data(), static_cast<std::streamsize>(row.size()));
        }
        return static_cast<bool>(out);
    }

    // Returns false when the output could not be written.
    bool writeResult(const fs::path &outDir, const std::string &stem, const char *backendName, const gsdump::Result &r)
    {
        std::ofstream js(outDir / (stem + "." + backendName + ".json"));
        js << "{\n  \"backend\": " << jsonString(backendName) << ",\n  \"ok\": " << (r.ok ? "true" : "false")
           << ",\n  \"error\": " << jsonString(r.error) << ",\n  \"serial\": " << jsonString(r.serial)
           << ",\n  \"state_version\": " << r.stateVersion << ",\n  \"transfers\": " << r.transfers
           << ",\n  \"read_fifos\": " << r.readFifos << ",\n  \"warnings\": [";
        for (size_t i = 0; i < r.warnings.size(); ++i)
            js << (i ? ", " : "") << jsonString(r.warnings[i]);
        js << "],\n  \"frames\": [";
        bool ok = true;
        for (size_t i = 0; i < r.frames.size(); ++i)
        {
            const gsdump::Frame &f = r.frames[i];
            char name[32];
            std::snprintf(name, sizeof(name), ".%03zu.ppm", i);
            const std::string file = f.rgba.empty() ? std::string() : stem + "." + backendName + name;
            if (!file.empty())
                ok = writePpm(outDir / file, f) && ok;
            js << (i ? "," : "") << "\n    {\"vsync\": " << i << ", \"file\": " << jsonString(file)
               << ", \"width\": " << f.width << ", \"height\": " << f.height << ", \"display_fbp\": " << f.displayFbp
               << ", \"source_fbp\": " << f.sourceFbp << ", \"display_psm\": " << f.displayPsm
               << ", \"used_preferred\": " << (f.usedPreferred ? "true" : "false")
               << ", \"field\": " << (f.field ? "true" : "false") << "}";
        }
        js << "\n  ]\n}\n";
        return ok && static_cast<bool>(js);
    }
}

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 4)
    {
        std::cerr << "usage: ps2x_gsdump <dump.gs> <out_dir> [cpu|vulkan|both]\n";
        return 1;
    }
    const fs::path dumpPath = fs::u8path(argv[1]);
    const fs::path outDir = fs::u8path(argv[2]);
    const std::string which = (argc == 4) ? argv[3] : "both";
    if (which != "cpu" && which != "vulkan" && which != "both")
    {
        std::cerr << "backend must be cpu, vulkan or both\n";
        return 1;
    }

    std::vector<uint8_t> file;
    if (!gsdump::readFile(dumpPath, file))
    {
        std::cerr << "cannot read " << dumpPath.string() << "\n";
        return 1;
    }
    std::error_code ec;
    fs::create_directories(outDir, ec);
    const std::string stem = dumpPath.stem().string();

    int exitCode = 0;
    const auto run = [&](const char *name, std::unique_ptr<GSRasterBackend> backend)
    {
        const gsdump::Result r = gsdump::replay(file, std::move(backend));
        if (!writeResult(outDir, stem, name, r))
        {
            std::cerr << "cannot write the output in " << outDir.string() << "\n";
            exitCode = 1;
        }
        std::cout << stem << " [" << name << "]: " << (r.ok ? "ok" : "ERROR " + r.error) << ", " << r.frames.size()
                  << " vsyncs, " << r.transfers << " transfers, " << r.warnings.size() << " warnings\n";
        if (!r.ok)
            exitCode = 1;
    };

    if (which != "vulkan")
        run("cpu", std::make_unique<GSCpuBackend>());
    if (which != "cpu")
    {
        std::string reason;
        auto vk = GSVulkanBackend::CreateIfAvailable(&reason);
        if (vk)
            run("vulkan", std::move(vk));
        else
        {
            std::cerr << "Vulkan backend not available: " << reason << "\n";
            if (exitCode == 0)
                exitCode = 2;
        }
    }
    return exitCode;
}
