#pragma once

// Vita and Android still run on raylib; the desktop host is SDL3 + Vulkan (ps2_host_sdl.h).
#if defined(PLATFORM_VITA) || defined(__ANDROID__)
#define PS2X_HOST_RAYLIB 1
#include "raylib.h"
#else
#define PS2X_HOST_RAYLIB 0
#include "ps2_host_sdl.h"
#endif
