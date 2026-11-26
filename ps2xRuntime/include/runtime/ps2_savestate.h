#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

class PS2Runtime;

// Save states, turbo and turbo frameskip (Kernel/SaveState.cpp). Options come from PS2X_* environment
// variables; hotkeys are read on the main thread, the machine is saved/loaded on the EE thread at the
// scheduler safe point.
namespace ps2x::savestate
{
    // Before the game thread starts: reads the options and registers the safe-point hook.
    void install(PS2Runtime &runtime);
    // Main thread, once per host frame after pumpEvents: hotkeys, turbo, messages.
    void onHostFrame(PS2Runtime &runtime);
    // Frameskip level 1+: true when this host frame should not be presented.
    bool skipPresent();

    // Slot files: a header (game, build, checksum) then the machine. Reading throws with a message for the
    // player when the file is missing, damaged or from another game or build.
    void writeStateFile(const std::filesystem::path &path, const std::vector<uint8_t> &payload,
                        const std::string &gameId, uint64_t exeHash, uint64_t vsyncTick);
    std::vector<uint8_t> readStateFile(const std::filesystem::path &path, const std::string &gameId, uint64_t exeHash);
}
