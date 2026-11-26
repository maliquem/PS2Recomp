#pragma once

// Player input on SDL3 (main thread unless noted): keyboard + every SDL gamepad (hotplug, SDL's mapping
// database plus gamecontrollerdb.txt next to the exe), rebindable and saved in ps2x_player.toml.
// Port 0 = keyboard + first controller, port 1 = second controller. The EE side reads ps2host::padInput.

#include <SDL3/SDL.h>

#include <string>

namespace ps2input
{
    constexpr int kPorts = 2;
    constexpr int kButtons = 16;    // PS2 buttons: actions 0..15, bindable on keyboard and controller
    constexpr int kActions = 24;    // + 8 stick directions (16..23), keyboard only
    constexpr int kKeysPerAction = 2;

    const char *actionName(int action, bool pt);

    // Before SDL_INIT_GAMEPAD (hints for the test simulation).
    void preInit();
    // After the window exists: bindings, gamecontrollerdb.txt, simulation file.
    void init(SDL_Window *window);
    void shutdown();
    // Before SDL_PollEvent: PS2X_INPUT_SIM=<file> turns held tokens into SDL events (tests).
    void pollSimulation();
    bool simulating(); // PS2X_INPUT_SIM is set
    void handleEvent(const SDL_Event &e);
    // Once per loop after the events: builds what the EE reads, applies rumble.
    void update(bool menuOpen);

    // Bindings. Setters save the player file.
    SDL_Scancode keyBinding(int action, int slot);
    void setKeyBinding(int action, int slot, SDL_Scancode sc); // SDL_SCANCODE_UNKNOWN clears
    std::string padBinding(int button);                         // SDL gamepad name ("a", "lefttrigger"); "" = none
    void setPadBinding(int button, const std::string &name);
    void resetKeyboard();
    void resetGamepad();
    std::string keyLabel(SDL_Scancode sc);
    std::string padLabel(const std::string &name, bool pt);
    // SDL name for a controller button or trigger event, "" when it cannot be bound.
    std::string padSourceFromEvent(const SDL_Event &e);

    bool rumbleEnabled();
    void setRumbleEnabled(bool on);
    int pressureRampMs();
    void setPressureRampMs(int ms);
}
