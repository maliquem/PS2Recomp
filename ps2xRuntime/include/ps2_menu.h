#pragma once

// Pause menu (RmlUi, drawn over the game after the scene). Opened by Esc on the keyboard, Guide (or Back+Start)
// on a controller. Built-in sections: Controller, Keyboard, Display, Audio, System.
//
// Other parts of the runtime add their own sections and options from code, on the main thread, before or after
// the window opens (for example right after ps2host::openWindow):
//
//     ps2menu::addSection("savestate", {"Save states", "Estados salvos"});
//     ps2menu::addAction("savestate", {"Save to slot 1", "Salvar no espaço 1"}, [] { ... });
//     ps2menu::addToggle("display", {"Bloom", "Bloom"}, [] { return bloomOn; }, [](bool v) { bloomOn = v; });
//
// Sections appear as tabs in the order they are added, after the built-in ones; options are appended to an existing
// section (built-in or not). Getters run each time the menu redraws; setters run on the main thread when the player
// changes the value, and own persistence (ps2cfg::set* + ps2cfg::save for values kept in ps2x_player.toml).

#include <functional>
#include <string>
#include <vector>

namespace ps2menu
{
    struct Text
    {
        std::string en, pt; // pt empty = use en
    };

    void addSection(const std::string &id, Text title);
    void addToggle(const std::string &section, Text label, std::function<bool()> get, std::function<void(bool)> set);
    // Left/right (or click) cycles through `choices`; get/set use the index.
    void addChoice(const std::string &section, Text label, std::vector<Text> choices, std::function<int()> get,
                   std::function<void(int)> set);
    void addAction(const std::string &section, Text label, std::function<void()> run);
    // Read-only line (placeholders, status).
    void addInfo(const std::string &section, Text label, Text value);

    bool isOpen();
    void close();
}
