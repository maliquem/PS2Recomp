#include "ps2_user_config.h"

#include <SDL3/SDL_filesystem.h>
#include <toml.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>

namespace
{
    namespace fs = std::filesystem;
    using Value = toml::ordered_value;

    constexpr const char *kTemplate = R"(# PS2Recomp player settings. Delete this file to get the defaults back.
# An environment variable set explicitly (PS2X_VSYNC, PS2X_SHOW_FPS, PS2X_GS_BACKEND, PS2X_FMV*) wins over this file.
# The pause menu (Esc, or Guide / Back+Start on a controller) changes most of these and saves them here.

[display]
# Sync to the display refresh (no tearing). Runs with --hidden-window ignore it.
vsync = true
# FPS/VPS counter in the corner (F3 toggles it).
show_fps = false

[video]
# Movie (FMV) enhancements; false shows the movies exactly as before (PS2X_FMV=0 does the same).
fmv = true
# Precise colour conversion of the movies (sharper colour edges, no colour shift).
fmv_csc = true

[gs]
# "vulkan" (default) or "cpu".
backend = "vulkan"

[ui]
# Menu language: "auto" (system), "en" or "pt".
language = "auto"

[input]
# Controller vibration (DualShock 2 motors).
rumble = true
# Pressure-sensitive buttons: 0 = full pressure at once; > 0 = milliseconds from a light to a full press.
# L2/R2 on a controller with analog triggers always use the trigger travel.
pressure_ramp_ms = 0
# Key and button bindings live in [input.keyboard] and [input.gamepad]; the menu writes them on the first rebind.
)";

    Value g_root{toml::ordered_table{}};
    bool g_loaded = false;
    bool g_noSave = false; // the player's file is left alone (test run, or a file that could not be read)
    fs::path g_path;

    fs::path utf8Path(const char *s)
    {
#if defined(__cpp_char8_t)
        return fs::path(reinterpret_cast<const char8_t *>(s));
#else
        return fs::u8path(s);
#endif
    }

    const fs::path &filePath()
    {
        if (g_path.empty())
        {
            if (const char *env = std::getenv("PS2X_CONFIG"); env && *env)
                g_path = fs::path(env);
            else
            {
                const char *base = SDL_GetBasePath(); // UTF-8, ends with a separator; owned by SDL
                g_path = utf8Path(base ? base : "") / "ps2x_player.toml";
            }
        }
        return g_path;
    }

    bool parseInto(const std::string &text, const std::string &name)
    {
        auto result = toml::try_parse_str<toml::ordered_type_config>(text, toml::spec::default_version());
        if (result.is_err())
        {
            for (const auto &err : result.unwrap_err())
                std::fprintf(stderr, "[config] %s: %s\n", name.c_str(), toml::format_error(err).c_str());
            return false;
        }
        Value parsed = result.unwrap();
        if (!parsed.is_table())
            return false;
        g_root = std::move(parsed);
        return true;
    }

    void ensureLoaded()
    {
        if (g_loaded)
            return;
        g_loaded = true;
        const char *hidden = std::getenv("PS2X_HIDDEN_WINDOW");
        const char *chosen = std::getenv("PS2X_CONFIG");
        if (hidden && *hidden && !(chosen && *chosen))
        {
            g_noSave = true; // hidden test runs use the defaults unless PS2X_CONFIG names a file
            return;
        }
        const fs::path &p = filePath();
        std::ifstream in(p, std::ios::binary);
        if (!in)
        {
            std::error_code ec;
            if (fs::exists(p, ec) || ec)
            {
                g_noSave = true; // there but unreadable: saving one key would drop the rest
                std::fprintf(stderr, "[config] cannot read %s; changes will not be saved\n", p.string().c_str());
            }
            return; // defaults
        }
        std::stringstream text;
        text << in.rdbuf();
        in.close();
        if (parseInto(text.str(), p.string()))
        {
            std::fprintf(stderr, "[config] loaded %s\n", p.string().c_str());
            return;
        }
        // Keep the broken file for the player instead of overwriting it on the next save.
        fs::path bad = p;
        bad += ".bad";
        std::error_code ec;
        fs::remove(bad, ec);
        fs::rename(p, bad, ec);
        std::fprintf(stderr, "[config] %s is not valid TOML; using defaults (kept as %s)\n", p.string().c_str(),
                     ec ? "-" : bad.string().c_str());
    }

    const Value *find(const char *key)
    {
        ensureLoaded();
        const Value *cur = &g_root;
        std::string k = key;
        size_t start = 0;
        for (;;)
        {
            const size_t dot = k.find('.', start);
            const std::string part = k.substr(start, dot == std::string::npos ? std::string::npos : dot - start);
            if (!cur->is_table() || !cur->contains(part))
                return nullptr;
            cur = &cur->at(part);
            if (dot == std::string::npos)
                return cur;
            start = dot + 1;
        }
    }

    Value &slot(const char *key)
    {
        ensureLoaded();
        Value *cur = &g_root;
        std::string k = key;
        size_t start = 0;
        for (;;)
        {
            const size_t dot = k.find('.', start);
            const std::string part = k.substr(start, dot == std::string::npos ? std::string::npos : dot - start);
            if (dot == std::string::npos)
                return (*cur)[part];
            if (!cur->contains(part) || !cur->at(part).is_table())
                (*cur)[part] = toml::ordered_table{};
            cur = &cur->at(part);
            start = dot + 1;
        }
    }

    void typeWarning(const char *key, const char *want)
    {
        std::fprintf(stderr, "[config] %s: expected %s, using the default\n", key, want);
    }

    bool envSet(const char *env)
    {
        const char *v = env ? std::getenv(env) : nullptr;
        return v && *v;
    }
}

namespace ps2cfg
{

    bool getBool(const char *key, bool def)
    {
        const Value *v = find(key);
        if (!v)
            return def;
        if (v->is_boolean())
            return v->as_boolean();
        typeWarning(key, "true/false");
        return def;
    }

    int getInt(const char *key, int def)
    {
        const Value *v = find(key);
        if (!v)
            return def;
        if (v->is_integer())
        {
            const auto i = v->as_integer();
            return i < INT32_MIN ? INT32_MIN : i > INT32_MAX ? INT32_MAX : static_cast<int>(i);
        }
        typeWarning(key, "an integer");
        return def;
    }

    std::string getString(const char *key, const std::string &def)
    {
        const Value *v = find(key);
        if (!v)
            return def;
        if (v->is_string())
            return v->as_string();
        typeWarning(key, "a string");
        return def;
    }

    std::vector<std::string> getStringList(const char *key, const std::vector<std::string> &def)
    {
        const Value *v = find(key);
        if (!v)
            return def;
        if (v->is_string())
            return {v->as_string()};
        if (v->is_array())
        {
            std::vector<std::string> out;
            for (const Value &item : v->as_array())
            {
                if (!item.is_string())
                {
                    typeWarning(key, "a list of strings");
                    return def;
                }
                out.push_back(item.as_string());
            }
            return out;
        }
        typeWarning(key, "a list of strings");
        return def;
    }

    void setBool(const char *key, bool value) { slot(key) = value; }
    void setInt(const char *key, int value) { slot(key) = static_cast<std::int64_t>(value); }
    void setString(const char *key, const std::string &value) { slot(key) = value; }

    void setStringList(const char *key, const std::vector<std::string> &value)
    {
        toml::ordered_array arr;
        for (const std::string &s : value)
            arr.push_back(Value(s));
        slot(key) = std::move(arr);
    }

    bool save()
    {
        ensureLoaded();
        if (g_noSave)
            return false;
        const fs::path &p = filePath();
        fs::path tmp = p;
        tmp += ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out)
            {
                std::fprintf(stderr, "[config] cannot write %s\n", tmp.string().c_str());
                return false;
            }
            out << toml::format(g_root);
            out.close(); // flush first, so a full disk shows up before the rename
            if (!out)
            {
                std::fprintf(stderr, "[config] cannot write %s\n", tmp.string().c_str());
                return false;
            }
        }
        std::error_code ec;
        fs::rename(tmp, p, ec); // replaces the old file
        if (ec)
        {
            std::fprintf(stderr, "[config] cannot replace %s: %s\n", p.string().c_str(), ec.message().c_str());
            return false;
        }
        return true;
    }

    void writeTemplateIfMissing()
    {
        ensureLoaded();
        const fs::path &p = filePath();
        std::error_code ec;
        if (fs::exists(p, ec) || ec)
            return;
        if (g_root.is_table() && !g_root.as_table().empty())
            return; // something was already set this run; save() will write it
        {
            std::ofstream out(p, std::ios::binary);
            if (!out)
                return;
            out << kTemplate;
        }
        parseInto(kTemplate, "template"); // keep its comments for later saves
    }

    bool envBool(const char *env, const char *key, bool def)
    {
        if (envSet(env))
            return std::strcmp(std::getenv(env), "0") != 0;
        return getBool(key, def);
    }

    std::string envString(const char *env, const char *key, const std::string &def)
    {
        if (envSet(env))
            return std::getenv(env);
        return getString(key, def);
    }
}
