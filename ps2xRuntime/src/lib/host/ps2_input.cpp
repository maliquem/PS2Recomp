#include "ps2_input.h"
#include "ps2_host_sdl.h"
#include "ps2_user_config.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <vector>

namespace
{
    struct ButtonDef
    {
        const char *id, *en, *pt;
        uint16_t mask;      // libpad bit
        int pressure;       // index in PadInput::pressure, -1 = not pressure sensitive
        const char *pad;    // default SDL gamepad source
        const char *key[2]; // default scancode names
    };

    // Actions 0..15 are PS2 buttons, 16..23 stick directions (keyboard only).
    const ButtonDef kDefs[ps2input::kActions] = {
        {"up", "Up", "Cima", 0x0010, 2, "dpup", {"Up", "W"}},
        {"down", "Down", "Baixo", 0x0040, 3, "dpdown", {"Down", "S"}},
        {"left", "Left", "Esquerda", 0x0080, 1, "dpleft", {"Left", "A"}},
        {"right", "Right", "Direita", 0x0020, 0, "dpright", {"Right", "D"}},
        {"cross", "Cross", "Xis", 0x4000, 6, "a", {"X", "Space"}},
        {"circle", "Circle", "Círculo", 0x2000, 5, "b", {"C", "Backspace"}},
        {"square", "Square", "Quadrado", 0x8000, 7, "x", {"Z", "Keypad 0"}},
        {"triangle", "Triangle", "Triângulo", 0x1000, 4, "y", {"V", "Keypad 1"}},
        {"l1", "L1", "L1", 0x0400, 8, "leftshoulder", {"Q", ""}},
        {"r1", "R1", "R1", 0x0800, 10, "rightshoulder", {"E", ""}},
        {"l2", "L2", "L2", 0x0100, 9, "lefttrigger", {"Left Shift", ""}},
        {"r2", "R2", "R2", 0x0200, 11, "righttrigger", {"Right Shift", ""}},
        {"l3", "L3", "L3", 0x0002, -1, "leftstick", {"", ""}},
        {"r3", "R3", "R3", 0x0004, -1, "rightstick", {"", ""}},
        {"select", "Select", "Select", 0x0001, -1, "back", {"Tab", ""}},
        {"start", "Start", "Start", 0x0008, -1, "start", {"Return", ""}},
        {"lstick_up", "Left stick up", "Analógico esq. cima", 0, -1, "", {"", ""}},
        {"lstick_down", "Left stick down", "Analógico esq. baixo", 0, -1, "", {"", ""}},
        {"lstick_left", "Left stick left", "Analógico esq. esquerda", 0, -1, "", {"", ""}},
        {"lstick_right", "Left stick right", "Analógico esq. direita", 0, -1, "", {"", ""}},
        {"rstick_up", "Right stick up", "Analógico dir. cima", 0, -1, "", {"", ""}},
        {"rstick_down", "Right stick down", "Analógico dir. baixo", 0, -1, "", {"", ""}},
        {"rstick_left", "Right stick left", "Analógico dir. esquerda", 0, -1, "", {"", ""}},
        {"rstick_right", "Right stick right", "Analógico dir. direita", 0, -1, "", {"", ""}},
    };

    // Past this travel an analog trigger counts as pressed; its pressure follows the travel.
    constexpr Sint16 kTriggerPress = 8192;
    constexpr uint64_t kRumbleRefreshNs = 500000000ull;
    constexpr Uint32 kRumbleMs = 1000;

    struct PadSource
    {
        std::string name;
        int button = -1; // SDL_GamepadButton
        int axis = -1;   // SDL_GamepadAxis (triggers)
    };

    SDL_Window *g_window = nullptr;
    bool g_keyDown[SDL_SCANCODE_COUNT]{};
    std::vector<SDL_Gamepad *> g_pads; // connection order: g_pads[p] drives port p
    SDL_Scancode g_keys[ps2input::kActions][ps2input::kKeysPerAction]{};
    PadSource g_padSrc[ps2input::kButtons];
    bool g_rumbleOn = true;
    int g_rampMs = 0;

    uint64_t g_pressStart[ps2input::kPorts][12]{};
    uint16_t g_suppress[ps2input::kPorts]{};

    std::mutex g_publishMutex;
    ps2host::PadInput g_published[ps2input::kPorts];

    std::atomic<uint16_t> g_rumbleWanted[ps2input::kPorts]{}; // small << 8 | large
    uint16_t g_rumbleSent[ps2input::kPorts]{};
    uint64_t g_rumbleAt[ps2input::kPorts]{};
    SDL_Gamepad *g_rumblePad[ps2input::kPorts]{};

    // Test simulation (PS2X_INPUT_SIM).
    std::string g_simPath;
    uint64_t g_simNextNs = 0;
    std::set<std::string> g_simHeld;
    SDL_Joystick *g_simJoy = nullptr;
    float g_simMouseX = 0, g_simMouseY = 0;

    std::string configKey(const char *section, int action) { return std::string("input.") + section + "." + kDefs[action].id; }

    PadSource parsePad(const std::string &name)
    {
        PadSource s;
        s.name = name;
        if (name.empty())
            return s;
        const SDL_GamepadButton b = SDL_GetGamepadButtonFromString(name.c_str());
        if (b != SDL_GAMEPAD_BUTTON_INVALID && b != SDL_GAMEPAD_BUTTON_GUIDE)
        {
            s.button = b;
            return s;
        }
        const SDL_GamepadAxis a = SDL_GetGamepadAxisFromString(name.c_str());
        if (a == SDL_GAMEPAD_AXIS_LEFT_TRIGGER || a == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER)
        {
            s.axis = a;
            return s;
        }
        std::fprintf(stderr, "[input] unknown controller input '%s' ignored\n", name.c_str());
        s.name.clear();
        return s;
    }

    void loadBindings()
    {
        for (int a = 0; a < ps2input::kActions; ++a)
        {
            std::vector<std::string> def;
            for (const char *k : kDefs[a].key)
                if (*k)
                    def.emplace_back(k);
            const std::vector<std::string> names = ps2cfg::getStringList(configKey("keyboard", a).c_str(), def);
            int slot = 0;
            for (int s = 0; s < ps2input::kKeysPerAction; ++s)
                g_keys[a][s] = SDL_SCANCODE_UNKNOWN;
            for (const std::string &n : names)
            {
                if (slot >= ps2input::kKeysPerAction)
                    break;
                const SDL_Scancode sc = SDL_GetScancodeFromName(n.c_str());
                if (sc == SDL_SCANCODE_UNKNOWN)
                {
                    std::fprintf(stderr, "[input] unknown key '%s' for %s ignored\n", n.c_str(), kDefs[a].id);
                    continue;
                }
                g_keys[a][slot++] = sc;
            }
        }
        for (int b = 0; b < ps2input::kButtons; ++b)
            g_padSrc[b] = parsePad(ps2cfg::getString(configKey("gamepad", b).c_str(), kDefs[b].pad));
        g_rumbleOn = ps2cfg::getBool("input.rumble", true);
        g_rampMs = std::clamp(ps2cfg::getInt("input.pressure_ramp_ms", 0), 0, 2000);
    }

    void saveKeys(int action)
    {
        std::vector<std::string> names;
        for (SDL_Scancode sc : g_keys[action])
            if (sc != SDL_SCANCODE_UNKNOWN)
                names.emplace_back(SDL_GetScancodeName(sc));
        ps2cfg::setStringList(configKey("keyboard", action).c_str(), names);
    }

    uint8_t axisByte(Sint16 v) { return static_cast<uint8_t>((static_cast<int>(v) + 32768) >> 8); }

    bool keyHeld(int action)
    {
        // Enter is Start: Alt+Enter toggles fullscreen and must not press Start too.
        const bool alt = g_keyDown[SDL_SCANCODE_LALT] || g_keyDown[SDL_SCANCODE_RALT];
        for (SDL_Scancode sc : g_keys[action])
            if (sc != SDL_SCANCODE_UNKNOWN && g_keyDown[sc] && !(alt && sc == SDL_SCANCODE_RETURN))
                return true;
        return false;
    }

    // Keyboard stick direction pair overrides the controller stick.
    uint8_t stickByte(uint8_t fromPad, bool keyboard, int negAction, int posAction)
    {
        if (!keyboard)
            return fromPad;
        const bool neg = keyHeld(negAction), pos = keyHeld(posAction);
        if (neg != pos)
            return neg ? 0x00 : 0xFF;
        return fromPad;
    }

    ps2host::PadInput buildPort(int port, uint64_t now, uint16_t &rawHeld)
    {
        ps2host::PadInput in;
        SDL_Gamepad *pad = port < static_cast<int>(g_pads.size()) ? g_pads[port] : nullptr;
        const bool keyboard = port == 0;
        int analog[ps2input::kButtons];
        for (int b = 0; b < ps2input::kButtons; ++b)
        {
            analog[b] = -1;
            bool down = keyboard && keyHeld(b);
            const PadSource &src = g_padSrc[b];
            if (pad && src.button >= 0 && SDL_GetGamepadButton(pad, static_cast<SDL_GamepadButton>(src.button)))
                down = true;
            if (pad && src.axis >= 0)
            {
                const Sint16 v = SDL_GetGamepadAxis(pad, static_cast<SDL_GamepadAxis>(src.axis));
                if (v > kTriggerPress)
                {
                    down = true;
                    analog[b] = std::clamp(v * 255 / 32767, 1, 255);
                }
            }
            if (down)
                in.held |= kDefs[b].mask;
        }
        if (pad)
        {
            in.lx = axisByte(SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_LEFTX));
            in.ly = axisByte(SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_LEFTY));
            in.rx = axisByte(SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_RIGHTX));
            in.ry = axisByte(SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_RIGHTY));
        }
        in.ly = stickByte(in.ly, keyboard, 16, 17);
        in.lx = stickByte(in.lx, keyboard, 18, 19);
        in.ry = stickByte(in.ry, keyboard, 20, 21);
        in.rx = stickByte(in.rx, keyboard, 22, 23);

        rawHeld = in.held;
        in.held &= static_cast<uint16_t>(~g_suppress[port]);
        for (int b = 0; b < ps2input::kButtons; ++b)
        {
            const int p = kDefs[b].pressure;
            if (p < 0)
                continue;
            if (!(in.held & kDefs[b].mask))
            {
                g_pressStart[port][p] = 0;
                continue;
            }
            if (analog[b] >= 0)
                in.pressure[p] = static_cast<uint8_t>(analog[b]);
            else if (g_rampMs > 0)
            {
                if (!g_pressStart[port][p])
                    g_pressStart[port][p] = now;
                const uint64_t ramp = static_cast<uint64_t>(g_rampMs) * 1000000ull;
                const uint64_t t = std::min(now - g_pressStart[port][p], ramp);
                in.pressure[p] = static_cast<uint8_t>(32 + t * 223 / ramp); // a light press still registers
            }
        }
        return in;
    }

    void applyRumble(bool menuOpen, uint64_t now)
    {
        for (int p = 0; p < ps2input::kPorts; ++p)
        {
            SDL_Gamepad *pad = p < static_cast<int>(g_pads.size()) ? g_pads[p] : nullptr;
            if (pad != g_rumblePad[p])
            {
                g_rumblePad[p] = pad;
                g_rumbleSent[p] = 0;
            }
            if (!pad)
                continue;
            const uint16_t want = (menuOpen || !g_rumbleOn) ? 0 : g_rumbleWanted[p].load(std::memory_order_relaxed);
            if (want == g_rumbleSent[p] && (!want || now - g_rumbleAt[p] < kRumbleRefreshNs))
                continue;
            const Uint16 low = static_cast<Uint16>((want & 0xFF) * 257);   // large motor
            const Uint16 high = (want >> 8) ? 0xFFFF : 0;                    // small motor (on/off)
            SDL_RumbleGamepad(pad, low, high, want ? kRumbleMs : 0);
            g_rumbleSent[p] = want;
            g_rumbleAt[p] = now;
        }
    }

    // --- test simulation -------------------------------------------------------------------------

    bool simRumble(void *, Uint16 low, Uint16 high)
    {
        std::fprintf(stderr, "[input] rumble low=%u high=%u\n", static_cast<unsigned>(low), static_cast<unsigned>(high));
        return true;
    }

    SDL_Joystick *simJoystick()
    {
        if (g_simJoy)
            return g_simJoy;
        SDL_VirtualJoystickDesc desc;
        SDL_INIT_INTERFACE(&desc);
        desc.type = SDL_JOYSTICK_TYPE_GAMEPAD;
        desc.naxes = SDL_GAMEPAD_AXIS_COUNT;
        desc.nbuttons = SDL_GAMEPAD_BUTTON_COUNT;
        desc.name = "PS2X simulated pad";
        desc.Rumble = simRumble;
        const SDL_JoystickID id = SDL_AttachVirtualJoystick(&desc);
        g_simJoy = id ? SDL_OpenJoystick(id) : nullptr;
        if (!g_simJoy)
            std::fprintf(stderr, "[input] simulated pad failed: %s\n", SDL_GetError());
        return g_simJoy;
    }

    void simToken(const std::string &tok, bool down)
    {
        const size_t colon = tok.find(':');
        const std::string kind = tok.substr(0, colon);
        const std::string arg = colon == std::string::npos ? std::string() : tok.substr(colon + 1);
        SDL_Event ev{};
        if (kind == "key")
        {
            const SDL_Scancode sc = SDL_GetScancodeFromName(arg.c_str());
            if (sc == SDL_SCANCODE_UNKNOWN)
                return;
            ev.type = down ? SDL_EVENT_KEY_DOWN : SDL_EVENT_KEY_UP;
            ev.key.timestamp = SDL_GetTicksNS();
            ev.key.windowID = SDL_GetWindowID(g_window);
            ev.key.scancode = sc;
            ev.key.key = SDL_GetKeyFromScancode(sc, SDL_KMOD_NONE, false);
            ev.key.down = down;
            SDL_PushEvent(&ev);
        }
        else if (kind == "pad")
        {
            SDL_Joystick *joy = simJoystick();
            if (!joy)
                return;
            const size_t eq = arg.find('=');
            const std::string name = arg.substr(0, eq);
            const float value = eq == std::string::npos ? 1.0f : std::strtof(arg.c_str() + eq + 1, nullptr);
            const SDL_GamepadButton b = SDL_GetGamepadButtonFromString(name.c_str());
            if (b != SDL_GAMEPAD_BUTTON_INVALID)
                SDL_SetJoystickVirtualButton(joy, b, down);
            else if (const SDL_GamepadAxis a = SDL_GetGamepadAxisFromString(name.c_str()); a != SDL_GAMEPAD_AXIS_INVALID)
                SDL_SetJoystickVirtualAxis(joy, a, down ? static_cast<Sint16>(std::clamp(value, -1.0f, 1.0f) * 32767) : 0);
        }
        else if (kind == "mouse" && down)
        {
            std::sscanf(arg.c_str(), "%f,%f", &g_simMouseX, &g_simMouseY);
            ev.type = SDL_EVENT_MOUSE_MOTION;
            ev.motion.windowID = SDL_GetWindowID(g_window);
            ev.motion.x = g_simMouseX;
            ev.motion.y = g_simMouseY;
            SDL_PushEvent(&ev);
        }
        else if (kind == "click" || kind == "rclick")
        {
            ev.type = down ? SDL_EVENT_MOUSE_BUTTON_DOWN : SDL_EVENT_MOUSE_BUTTON_UP;
            ev.button.windowID = SDL_GetWindowID(g_window);
            ev.button.button = kind == "click" ? SDL_BUTTON_LEFT : SDL_BUTTON_RIGHT;
            ev.button.down = down;
            ev.button.clicks = 1;
            ev.button.x = g_simMouseX;
            ev.button.y = g_simMouseY;
            SDL_PushEvent(&ev);
        }
        else if (kind == "wheel" && down)
        {
            ev.type = SDL_EVENT_MOUSE_WHEEL;
            ev.wheel.windowID = SDL_GetWindowID(g_window);
            ev.wheel.y = std::strtof(arg.c_str(), nullptr);
            ev.wheel.mouse_x = g_simMouseX;
            ev.wheel.mouse_y = g_simMouseY;
            SDL_PushEvent(&ev);
        }
    }
}

namespace ps2input
{
    const char *actionName(int action, bool pt) { return pt ? kDefs[action].pt : kDefs[action].en; }

    void preInit()
    {
        if (const char *sim = std::getenv("PS2X_INPUT_SIM"); sim && *sim)
        {
            g_simPath = sim;
            // Hidden test windows never get focus; SDL drops controller input without it.
            SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
        }
    }

    void init(SDL_Window *window)
    {
        g_window = window;
        const char *base = SDL_GetBasePath();
        const std::string db = std::string(base ? base : "") + "gamecontrollerdb.txt";
        if (SDL_GetPathInfo(db.c_str(), nullptr))
        {
            const int n = SDL_AddGamepadMappingsFromFile(db.c_str());
            std::fprintf(stderr, "[input] %s: %d mappings\n", db.c_str(), n);
        }
        loadBindings();
        if (!g_simPath.empty())
            std::fprintf(stderr, "[input] simulation from %s\n", g_simPath.c_str());
    }

    void shutdown()
    {
        for (SDL_Gamepad *pad : g_pads)
        {
            SDL_RumbleGamepad(pad, 0, 0, 0);
            SDL_CloseGamepad(pad);
        }
        g_pads.clear();
        if (g_simJoy)
        {
            const SDL_JoystickID id = SDL_GetJoystickID(g_simJoy);
            SDL_CloseJoystick(g_simJoy);
            SDL_DetachVirtualJoystick(id);
            g_simJoy = nullptr;
        }
    }

    bool simulating() { return !g_simPath.empty(); }

    void pollSimulation()
    {
        if (g_simPath.empty())
            return;
        const uint64_t now = SDL_GetTicksNS();
        if (now < g_simNextNs)
            return;
        g_simNextNs = now + 30000000ull;
        std::set<std::string> held;
        std::ifstream in(g_simPath);
        for (std::string tok; in >> tok;)
            held.insert(tok);
        for (const std::string &t : g_simHeld)
            if (!held.count(t))
                simToken(t, false);
        for (const std::string &t : held)
            if (!g_simHeld.count(t))
                simToken(t, true);
        g_simHeld = std::move(held);
    }

    void handleEvent(const SDL_Event &e)
    {
        switch (e.type)
        {
        case SDL_EVENT_KEY_DOWN:
        case SDL_EVENT_KEY_UP:
            if (e.key.scancode < SDL_SCANCODE_COUNT)
                g_keyDown[e.key.scancode] = e.key.down;
            break;
        case SDL_EVENT_WINDOW_FOCUS_LOST:
            std::fill(std::begin(g_keyDown), std::end(g_keyDown), false);
            break;
        case SDL_EVENT_GAMEPAD_ADDED:
            for (SDL_Gamepad *pad : g_pads)
                if (SDL_GetGamepadID(pad) == e.gdevice.which)
                    return;
            if (SDL_Gamepad *pad = SDL_OpenGamepad(e.gdevice.which))
            {
                g_pads.push_back(pad);
                const char *name = SDL_GetGamepadName(pad);
                std::fprintf(stderr, "[input] controller %zu: %s\n", g_pads.size(), name ? name : "?");
            }
            break;
        case SDL_EVENT_GAMEPAD_REMOVED:
            for (auto it = g_pads.begin(); it != g_pads.end(); ++it)
                if (SDL_GetGamepadID(*it) == e.gdevice.which)
                {
                    SDL_CloseGamepad(*it);
                    g_pads.erase(it);
                    std::fprintf(stderr, "[input] controller removed (%zu left)\n", g_pads.size());
                    break;
                }
            break;
        default:
            break;
        }
    }

    void update(bool menuOpen)
    {
        const uint64_t now = SDL_GetTicksNS();
        ps2host::PadInput ports[kPorts];
        for (int p = 0; p < kPorts; ++p)
        {
            uint16_t raw = 0;
            ports[p] = buildPort(p, now, raw);
            // Buttons held when the menu closes (Start, Enter...) stay hidden from the game until released.
            g_suppress[p] = menuOpen ? raw : static_cast<uint16_t>(g_suppress[p] & raw);
            if (menuOpen)
                ports[p] = ps2host::PadInput{};
        }
        {
            std::lock_guard<std::mutex> lock(g_publishMutex);
            for (int p = 0; p < kPorts; ++p)
                g_published[p] = ports[p];
        }
        applyRumble(menuOpen, now);
    }

    SDL_Scancode keyBinding(int action, int slot) { return g_keys[action][slot]; }

    void setKeyBinding(int action, int slot, SDL_Scancode sc)
    {
        if (sc != SDL_SCANCODE_UNKNOWN) // one key drives one action
            for (int a = 0; a < kActions; ++a)
                for (int s = 0; s < kKeysPerAction; ++s)
                    if (g_keys[a][s] == sc && !(a == action && s == slot))
                    {
                        g_keys[a][s] = SDL_SCANCODE_UNKNOWN;
                        saveKeys(a);
                    }
        g_keys[action][slot] = sc;
        saveKeys(action);
        ps2cfg::save();
    }

    std::string padBinding(int button) { return g_padSrc[button].name; }

    void setPadBinding(int button, const std::string &name)
    {
        g_padSrc[button] = parsePad(name);
        ps2cfg::setString(configKey("gamepad", button).c_str(), g_padSrc[button].name);
        ps2cfg::save();
    }

    void resetKeyboard()
    {
        for (int a = 0; a < kActions; ++a)
        {
            for (int s = 0; s < kKeysPerAction; ++s)
                g_keys[a][s] = *kDefs[a].key[s] ? SDL_GetScancodeFromName(kDefs[a].key[s]) : SDL_SCANCODE_UNKNOWN;
            saveKeys(a);
        }
        ps2cfg::save();
    }

    void resetGamepad()
    {
        for (int b = 0; b < kButtons; ++b)
        {
            g_padSrc[b] = parsePad(kDefs[b].pad);
            ps2cfg::setString(configKey("gamepad", b).c_str(), g_padSrc[b].name);
        }
        ps2cfg::save();
    }

    std::string keyLabel(SDL_Scancode sc)
    {
        const char *n = sc != SDL_SCANCODE_UNKNOWN ? SDL_GetScancodeName(sc) : "";
        return (n && *n) ? n : "-";
    }

    std::string padLabel(const std::string &name, bool pt)
    {
        static const struct
        {
            const char *sdl, *en, *pt;
        } kLabels[] = {
            {"a", "South (A / Cross)", "Sul (A / Xis)"},
            {"b", "East (B / Circle)", "Leste (B / Círculo)"},
            {"x", "West (X / Square)", "Oeste (X / Quadrado)"},
            {"y", "North (Y / Triangle)", "Norte (Y / Triângulo)"},
            {"dpup", "D-pad Up", "Direcional cima"},
            {"dpdown", "D-pad Down", "Direcional baixo"},
            {"dpleft", "D-pad Left", "Direcional esquerda"},
            {"dpright", "D-pad Right", "Direcional direita"},
            {"leftshoulder", "LB / L1", "LB / L1"},
            {"rightshoulder", "RB / R1", "RB / R1"},
            {"lefttrigger", "LT / L2", "LT / L2"},
            {"righttrigger", "RT / R2", "RT / R2"},
            {"leftstick", "Left stick click", "Clique analógico esq."},
            {"rightstick", "Right stick click", "Clique analógico dir."},
            {"back", "Back / Select", "Back / Select"},
            {"start", "Start", "Start"},
            {"misc1", "Misc", "Extra"},
            {"touchpad", "Touchpad", "Touchpad"},
            {"paddle1", "Paddle 1", "Paddle 1"},
            {"paddle2", "Paddle 2", "Paddle 2"},
            {"paddle3", "Paddle 3", "Paddle 3"},
            {"paddle4", "Paddle 4", "Paddle 4"},
        };
        if (name.empty())
            return "-";
        for (const auto &l : kLabels)
            if (name == l.sdl)
                return pt ? l.pt : l.en;
        return name;
    }

    std::string padSourceFromEvent(const SDL_Event &e)
    {
        if (e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN && e.gbutton.button != SDL_GAMEPAD_BUTTON_GUIDE)
        {
            const char *n = SDL_GetGamepadStringForButton(static_cast<SDL_GamepadButton>(e.gbutton.button));
            return n ? n : "";
        }
        if (e.type == SDL_EVENT_GAMEPAD_AXIS_MOTION && e.gaxis.value > 16384 &&
            (e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER || e.gaxis.axis == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER))
            return SDL_GetGamepadStringForAxis(static_cast<SDL_GamepadAxis>(e.gaxis.axis));
        return "";
    }

    bool rumbleEnabled() { return g_rumbleOn; }

    void setRumbleEnabled(bool on)
    {
        g_rumbleOn = on;
        ps2cfg::setBool("input.rumble", on);
        ps2cfg::save();
    }

    int pressureRampMs() { return g_rampMs; }

    void setPressureRampMs(int ms)
    {
        g_rampMs = std::clamp(ms, 0, 2000);
        ps2cfg::setInt("input.pressure_ramp_ms", g_rampMs);
        ps2cfg::save();
    }
}

namespace ps2host
{
    PadInput padInput(int port)
    {
        if (port < 0 || port >= ps2input::kPorts)
            return PadInput{};
        std::lock_guard<std::mutex> lock(g_publishMutex);
        return g_published[port];
    }

    void setRumble(int port, uint8_t small, uint8_t large)
    {
        if (port >= 0 && port < ps2input::kPorts)
            g_rumbleWanted[port].store(static_cast<uint16_t>((small ? 1u << 8 : 0u) | large), std::memory_order_relaxed);
    }
}
