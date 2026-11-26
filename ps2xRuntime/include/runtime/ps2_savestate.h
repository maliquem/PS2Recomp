#pragma once

#include <cstdint>

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
}
