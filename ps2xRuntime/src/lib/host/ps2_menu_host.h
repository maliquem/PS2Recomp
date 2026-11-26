#pragma once

// Host side of the pause menu (ps2_menu.cpp), driven by ps2_host_sdl.cpp on the main thread.

#include <SDL3/SDL.h>

namespace ps2menu::host
{
    void init(SDL_Window *window);
    void shutdown();      // before the ImGui backend shuts down
    void freeTextures();  // after it: drops the ImGui texture records
    // True when the event belongs to the menu (open, or the open shortcut) and must not reach the game or ImGui.
    bool handleEvent(const SDL_Event &e);
    void update();        // once per loop after the events: controller repeat, capture timeout
    void render();        // inside an ImGui frame
    bool quitRequested();
}
