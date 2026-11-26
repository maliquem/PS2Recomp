#pragma once

// Desktop host layer on SDL3: window, Vulkan swapchain presenter, input, audio device, ImGui.
// Kept free of SDL/Vulkan includes because Kernel/Stubs/Common.h pulls it into most of the runtime.

#include <cstdint>

class GS;
struct GSGpuFrame;

// shortcut: raylib-named input shim over SDL3 so Pad.cpp/ps2_pad.cpp keep today's mapping unchanged;
// replace with the real input layer (rebind, rumble, per-player config) in Etapa F.
// Key values are SDL scancodes (checked in ps2_host_sdl.cpp).
enum HostKey : int
{
    KEY_A = 4,
    KEY_C = 6,
    KEY_D = 7,
    KEY_E = 8,
    KEY_Q = 20,
    KEY_S = 22,
    KEY_V = 25,
    KEY_W = 26,
    KEY_X = 27,
    KEY_Z = 29,
    KEY_ONE = 30,
    KEY_THREE = 32,
    KEY_ENTER = 40,
    KEY_ESCAPE = 41,
    KEY_TAB = 43,
    KEY_SPACE = 44,
    KEY_GRAVE = 53,
    KEY_F1 = 58,
    KEY_F3 = 60,
    KEY_F5 = 62,
    KEY_F6 = 63,
    KEY_F7 = 64,
    KEY_F8 = 65,
    KEY_F9 = 66,
    KEY_F10 = 67,
    KEY_RIGHT = 79,
    KEY_LEFT = 80,
    KEY_DOWN = 81,
    KEY_UP = 82,
    KEY_KP_1 = 89,
    KEY_KP_0 = 98,
    KEY_LEFT_CONTROL = 224,
    KEY_LEFT_SHIFT = 225,
    KEY_RIGHT_CONTROL = 228,
    KEY_RIGHT_SHIFT = 229,
};

enum HostGamepadButton : int
{
    GAMEPAD_BUTTON_LEFT_FACE_UP,
    GAMEPAD_BUTTON_LEFT_FACE_RIGHT,
    GAMEPAD_BUTTON_LEFT_FACE_DOWN,
    GAMEPAD_BUTTON_LEFT_FACE_LEFT,
    GAMEPAD_BUTTON_RIGHT_FACE_UP,
    GAMEPAD_BUTTON_RIGHT_FACE_RIGHT,
    GAMEPAD_BUTTON_RIGHT_FACE_DOWN,
    GAMEPAD_BUTTON_RIGHT_FACE_LEFT,
    GAMEPAD_BUTTON_LEFT_TRIGGER_1,
    GAMEPAD_BUTTON_LEFT_TRIGGER_2,
    GAMEPAD_BUTTON_RIGHT_TRIGGER_1,
    GAMEPAD_BUTTON_RIGHT_TRIGGER_2,
    GAMEPAD_BUTTON_MIDDLE_LEFT,
    GAMEPAD_BUTTON_MIDDLE_RIGHT,
    GAMEPAD_BUTTON_LEFT_THUMB,
    GAMEPAD_BUTTON_RIGHT_THUMB,
    GAMEPAD_BUTTON_COUNT
};

enum HostGamepadAxis : int
{
    GAMEPAD_AXIS_LEFT_X,
    GAMEPAD_AXIS_LEFT_Y,
    GAMEPAD_AXIS_RIGHT_X,
    GAMEPAD_AXIS_RIGHT_Y,
    GAMEPAD_AXIS_COUNT
};

// Thread-safe: the main thread snapshots SDL state in ps2host::pumpEvents, the EE thread reads it.
bool IsWindowReady();
bool IsKeyDown(int key);
bool IsKeyPressed(int key); // pressed since the previous pumpEvents
bool IsGamepadAvailable(int gamepad);
bool IsGamepadButtonDown(int gamepad, int button);
float GetGamepadAxisMovement(int gamepad, int axis);

namespace ps2host
{
    // Before the GS backend exists: starts SDL and hands the surface instance extensions to the Vulkan GS.
    bool preInit();
    // Window (hidden with PS2X_HIDDEN_WINDOW), audio device and presenter. Presents from the GS backend's
    // Vulkan device when it has one, else through an SDL_Renderer upload path.
    bool openWindow(const char *title, int width, int height, GS &gs);
    void shutdown();

    bool audioReady();
    unsigned int audioDevice(); // SDL_AudioDeviceID, 0 when closed

    // True when presentFrame takes a GSGpuFrame (no readback); false means pass CPU pixels.
    bool gpuPresent();
    // Main thread, once per loop: waits until the previous present stopped reading the GS output buffer.
    void beginFrame();
    // False when the user asked to close the window.
    bool pumpEvents();
    // `fresh` = a new field was latched; otherwise the last one is drawn again.
    void presentGpuFrame(const GSGpuFrame &frame, bool fresh);
    void presentPixels(const uint8_t *rgba, uint32_t rowPixels, uint32_t width, uint32_t height, bool fresh);
    // 60 Hz host loop pacing, as raylib SetTargetFPS(60) did.
    void waitNextFrame();

    // Shows "<title> - <suffix>"; null or empty restores the title.
    void setWindowTitleSuffix(const char *suffix);

    // Dev-tools ImGui drawn over the game in the presenter's pass.
    bool imguiInit();
    void imguiShutdown();
    bool imguiBeginFrame();
    void imguiEndFrame();
}
