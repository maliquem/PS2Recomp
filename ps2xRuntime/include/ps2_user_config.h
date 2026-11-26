#pragma once

#include <string>
#include <vector>

// Player settings: ps2x_player.toml next to the exe (or the file named by PS2X_CONFIG).
// Keys are "section.name". Read lazily on first use; main thread only.
// Keys this runtime does not know are kept on save, so later stages can add their own.
namespace ps2cfg
{

    bool getBool(const char *key, bool def);
    int getInt(const char *key, int def);
    std::string getString(const char *key, const std::string &def);
    // Missing key -> `def`; a single string counts as a one-item list.
    std::vector<std::string> getStringList(const char *key, const std::vector<std::string> &def);

    void setBool(const char *key, bool value);
    void setInt(const char *key, int value);
    void setString(const char *key, const std::string &value);
    void setStringList(const char *key, const std::vector<std::string> &value);

    // Writes the whole table (atomic replace). False on I/O error.
    bool save();
    // First visible run: writes a commented file with the defaults so the player can find and edit it.
    void writeTemplateIfMissing();

    // An explicit, non-empty environment variable wins over the file ("0" = false).
    bool envBool(const char *env, const char *key, bool def);
    std::string envString(const char *env, const char *key, const std::string &def);
}
