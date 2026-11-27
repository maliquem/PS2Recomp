#ifndef PS2X_GS_DUMP_REPLAY_H
#define PS2X_GS_DUMP_REPLAY_H

// Replays a PCSX2 GS dump (.gs, uncompressed) through our GS, the same way the runtime feeds it
// (GS::processGIFPacket), and returns the frame presented at each VSync. Used by ps2x_gsdump (batch test
// against PCSX2 software-renderer screenshots) and by the synthetic-dump unit test.
// Format: PCSX2 GSDump.h / GSDump.cpp and GSState::Freeze (GPL-3.0+, PCSX2 Dev Team), state versions 7 to 9.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

class GSRasterBackend;

namespace gsdump
{
    // One entry per VSync packet; empty rgba when nothing was on display.
    struct Frame
    {
        std::vector<uint8_t> rgba; // width * height * 4, tightly packed
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t displayFbp = 0;
        uint32_t sourceFbp = 0;
        uint32_t displayPsm = 0;    // DISPFB PSM of the shown circuit (16-bit frames get a wider tolerance)
        bool usedPreferred = false; // our presentation heuristic showed another buffer than DISPFB
        bool field = false;         // SMODE2 INT=1 FFMD=1: one field, half the height the TV shows
    };

    struct Result
    {
        bool ok = false;
        std::string error;
        std::vector<std::string> warnings;
        std::string serial;
        uint32_t stateVersion = 0;
        uint32_t transfers = 0;
        uint32_t readFifos = 0;
        std::vector<Frame> frames;
    };

    // backend null = CPU backend.
    Result replay(const std::vector<uint8_t> &file, std::unique_ptr<GSRasterBackend> backend);
    bool readFile(const std::filesystem::path &path, std::vector<uint8_t> &out);
}

#endif
