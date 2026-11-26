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

    // Auto frameskip, once a second in turbo: the level (0..3) for the next second from this second's speed.
    // While the target is missed it tries the next level up when that one has no fresh measurement and otherwise
    // keeps the fastest fresh one (a higher level is not always faster; a stale lower level that was within 3% is
    // measured again). While the target is met (the pacer caps
    // the speed at the target, so "met" is the only sign of spare speed) it tries the next level down when that
    // one has no fresh measurement or met the target too. A measurement goes stale after kAutoStaleSeconds, so a
    // noisy second or a scene change is measured again.
    inline constexpr int kAutoStaleSeconds = 10;
    struct AutoFrameskip
    {
        int level = 0;
        double vps[4]{};                                                          // smoothed speed at each level
        int age[4]{kAutoStaleSeconds + 1, kAutoStaleSeconds + 1, kAutoStaleSeconds + 1, kAutoStaleSeconds + 1}; // seconds since measured
    };
    int autoFrameskipStep(AutoFrameskip &a, double vps, double target);

    // Slot files: a header (game, build, checksum) then the machine. Reading throws with a message for the
    // player when the file is missing, damaged or from another game or build.
    void writeStateFile(const std::filesystem::path &path, const std::vector<uint8_t> &payload,
                        const std::string &gameId, uint64_t exeHash, uint64_t vsyncTick);
    std::vector<uint8_t> readStateFile(const std::filesystem::path &path, const std::string &gameId, uint64_t exeHash);
}
