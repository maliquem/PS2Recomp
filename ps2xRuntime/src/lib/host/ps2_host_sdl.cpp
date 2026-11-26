#include "ps2_host_sdl.h"
#include "ps2_input.h"
#include "ps2_menu_host.h"
#include "ps2_menu.h"
#include "ps2_user_config.h"
#include "ps2_vk_presenter.h"

#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/gs_vulkan_backend.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

static_assert(KEY_A == SDL_SCANCODE_A && KEY_C == SDL_SCANCODE_C && KEY_D == SDL_SCANCODE_D && KEY_E == SDL_SCANCODE_E &&
              KEY_Q == SDL_SCANCODE_Q && KEY_S == SDL_SCANCODE_S && KEY_V == SDL_SCANCODE_V && KEY_W == SDL_SCANCODE_W &&
              KEY_X == SDL_SCANCODE_X && KEY_Z == SDL_SCANCODE_Z && KEY_ONE == SDL_SCANCODE_1 && KEY_THREE == SDL_SCANCODE_3 &&
              KEY_ENTER == SDL_SCANCODE_RETURN && KEY_ESCAPE == SDL_SCANCODE_ESCAPE && KEY_TAB == SDL_SCANCODE_TAB &&
              KEY_SPACE == SDL_SCANCODE_SPACE && KEY_F1 == SDL_SCANCODE_F1 && KEY_F3 == SDL_SCANCODE_F3 &&
              KEY_RIGHT == SDL_SCANCODE_RIGHT &&
              KEY_LEFT == SDL_SCANCODE_LEFT && KEY_DOWN == SDL_SCANCODE_DOWN && KEY_UP == SDL_SCANCODE_UP &&
              KEY_KP_1 == SDL_SCANCODE_KP_1 && KEY_KP_0 == SDL_SCANCODE_KP_0 && KEY_LEFT_CONTROL == SDL_SCANCODE_LCTRL &&
              KEY_LEFT_SHIFT == SDL_SCANCODE_LSHIFT && KEY_RIGHT_CONTROL == SDL_SCANCODE_RCTRL &&
              KEY_RIGHT_SHIFT == SDL_SCANCODE_RSHIFT,
              "HostKey values must be SDL scancodes");

namespace
{
    constexpr uint64_t kFramePeriodNs = 1000000000ull / 60ull;

    std::atomic<bool> g_windowReady{false};
    std::atomic<uint8_t> g_keys[SDL_SCANCODE_COUNT]{};
    std::atomic<uint8_t> g_pressed[SDL_SCANCODE_COUNT]{};

    SDL_Window *g_window = nullptr;
    SDL_AudioDeviceID g_audio = 0;
    std::unique_ptr<VkPresenter> g_presenter;

    // CPU path (PS2X_GS_BACKEND=cpu or no usable Vulkan swapchain): SDL_Renderer streaming texture.
    SDL_Renderer *g_renderer = nullptr;
    SDL_Texture *g_texture = nullptr;
    int g_textureW = 0, g_textureH = 0;

    bool g_imguiContext = false;
    bool g_imguiFrameBegun = false;
    bool g_imguiFrameReady = false;
    bool g_imguiDrewLastFrame = false; // ImGui only drains its input queue inside NewFrame
    bool g_vsync = false;

    // F3 counter: presents per second (FPS) and EE VBlanks per second (VPS).
    bool g_showRate = false;
    uint64_t g_rateStartNs = 0, g_rateTick = 0;
    unsigned int g_rateFrames = 0;
    float g_fps = 0.0f, g_vps = 0.0f;
    uint64_t g_nextFrameNs = 0;
    std::string g_baseTitle;

    bool envFlag(const char *name)
    {
        const char *v = std::getenv(name);
        return v && *v && std::strcmp(v, "0") != 0;
    }

    void toggleFullscreen(bool exclusive)
    {
        if (SDL_GetWindowFlags(g_window) & SDL_WINDOW_FULLSCREEN)
        {
            SDL_SetWindowFullscreen(g_window, false);
            return;
        }
        const SDL_DisplayMode *mode = exclusive ? SDL_GetDesktopDisplayMode(SDL_GetDisplayForWindow(g_window)) : nullptr;
        SDL_SetWindowFullscreenMode(g_window, mode); // null = borderless desktop
        SDL_SetWindowFullscreen(g_window, true);
    }

    bool openRendererFallback(bool vsync)
    {
        g_renderer = SDL_CreateRenderer(g_window, nullptr);
        if (!g_renderer)
        {
            std::fprintf(stderr, "[host] SDL_CreateRenderer failed: %s\n", SDL_GetError());
            return false;
        }
        SDL_SetRenderVSync(g_renderer, vsync ? 1 : 0);
        return true;
    }
}

bool IsWindowReady() { return g_windowReady.load(std::memory_order_relaxed); }

bool IsKeyDown(int key)
{
    return key > 0 && key < SDL_SCANCODE_COUNT && g_keys[key].load(std::memory_order_relaxed) != 0;
}

bool IsKeyPressed(int key)
{
    return key > 0 && key < SDL_SCANCODE_COUNT && g_pressed[key].load(std::memory_order_relaxed) != 0;
}

// Controllers are read through ps2host::padInput (bindings, both ports); the raylib-named gamepad shim reports none.
bool IsGamepadAvailable(int) { return false; }
bool IsGamepadButtonDown(int, int) { return false; }
float GetGamepadAxisMovement(int, int) { return 0.0f; }

namespace ps2host
{
    bool preInit()
    {
        if (!SDL_Init(SDL_INIT_VIDEO))
        {
            std::fprintf(stderr, "[host] SDL_Init failed: %s\n", SDL_GetError());
            return false;
        }
        // Without audio or gamepads the game still runs (muted / keyboard only).
        if (!SDL_InitSubSystem(SDL_INIT_AUDIO))
            std::fprintf(stderr, "[host] no audio subsystem: %s\n", SDL_GetError());
        ps2input::preInit();
        if (!SDL_InitSubSystem(SDL_INIT_GAMEPAD))
            std::fprintf(stderr, "[host] no gamepad subsystem: %s\n", SDL_GetError());
        if (!SDL_Vulkan_LoadLibrary(nullptr))
        {
            std::fprintf(stderr, "[host] no Vulkan loader for the window: %s\n", SDL_GetError());
            return true; // the GS backend reports its own Vulkan failure; we present through SDL_Renderer
        }
        Uint32 count = 0;
        const char *const *exts = SDL_Vulkan_GetInstanceExtensions(&count);
        std::vector<std::string> names;
        for (Uint32 i = 0; exts && i < count; ++i)
            names.emplace_back(exts[i]);
        GSVulkanBackend::SetHostInstanceExtensions(std::move(names));
        return true;
    }

    bool openWindow(const char *title, int width, int height, GS &gs)
    {
        const bool hidden = envFlag("PS2X_HIDDEN_WINDOW");
        if (!hidden)
            ps2cfg::writeTemplateIfMissing(); // test runs leave the player's folder alone
        // Visible windows follow the player file (default on); hidden runs are measurements and do not sync.
        // An explicit PS2X_VSYNC wins either way (PS2X_VSYNC=1 paces hidden parity runs).
        const char *vsyncEnv = std::getenv("PS2X_VSYNC");
        const bool vsync = (vsyncEnv && *vsyncEnv) ? envFlag("PS2X_VSYNC") : (!hidden && ps2cfg::getBool("display.vsync", true));
        g_vsync = vsync;
        g_showRate = ps2cfg::envBool("PS2X_SHOW_FPS", "display.show_fps", false);
        g_baseTitle = title;
        const GSHostGpuContext *ctx = gs.hostGpuContext();
        const bool tryVulkan = ctx && ctx->swapchainEnabled;

        const SDL_WindowFlags flags = SDL_WINDOW_RESIZABLE | (hidden ? SDL_WINDOW_HIDDEN : 0);
        g_window = SDL_CreateWindow(title, width, height, flags | (tryVulkan ? SDL_WINDOW_VULKAN : 0));
        if (!g_window)
        {
            std::fprintf(stderr, "[host] SDL_CreateWindow failed: %s\n", SDL_GetError());
            return false;
        }
        if (tryVulkan)
        {
            g_presenter = std::make_unique<VkPresenter>();
            std::string error;
            if (!g_presenter->init(g_window, *ctx, vsync, error))
            {
                std::fprintf(stderr, "[host] Vulkan presenter unavailable (%s); using SDL_Renderer\n", error.c_str());
                g_presenter.reset();
                SDL_DestroyWindow(g_window);
                g_window = SDL_CreateWindow(title, width, height, flags);
                if (!g_window)
                    return false;
            }
        }
        if (!g_presenter && !openRendererFallback(vsync))
            return false;

        const SDL_AudioSpec spec{SDL_AUDIO_S16, 2, 48000};
        g_audio = SDL_OpenAudioDevice(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec);
        if (!g_audio)
            std::fprintf(stderr, "[host] no audio device: %s\n", SDL_GetError());

        ps2input::init(g_window);
        // The pause menu draws through ImGui, so the host owns the ImGui context (with or without the debug panel).
        if (imguiInit())
            ps2menu::host::init(g_window);

        g_windowReady.store(true, std::memory_order_relaxed);
        g_nextFrameNs = SDL_GetTicksNS();
        return true;
    }

    void shutdown()
    {
        g_windowReady.store(false, std::memory_order_relaxed);
        ps2menu::host::shutdown();
        imguiShutdown();
        ps2menu::host::freeTextures();
        if (g_presenter)
            g_presenter->shutdown();
        g_presenter.reset();
        if (g_texture)
            SDL_DestroyTexture(g_texture);
        if (g_renderer)
            SDL_DestroyRenderer(g_renderer);
        g_texture = nullptr;
        g_renderer = nullptr;
        if (g_audio)
            SDL_CloseAudioDevice(g_audio);
        g_audio = 0;
        ps2input::shutdown();
        if (g_window)
            SDL_DestroyWindow(g_window);
        g_window = nullptr;
        SDL_Quit();
    }

    bool audioReady() { return g_audio != 0; }
    unsigned int audioDevice() { return g_audio; }
    bool gpuPresent() { return g_presenter != nullptr; }

    void beginFrame()
    {
        if (g_presenter)
            g_presenter->waitFrame();
    }

    bool pumpEvents()
    {
        bool keepRunning = true;
        for (auto &p : g_pressed)
            p.store(0, std::memory_order_relaxed);
        const bool forwardToImgui = g_imguiContext && g_imguiDrewLastFrame;
        g_imguiDrewLastFrame = false;
        ps2input::pollSimulation();
        SDL_Event e;
        while (SDL_PollEvent(&e))
        {
            ps2input::handleEvent(e);
            switch (e.type)
            {
            case SDL_EVENT_QUIT:
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
                keepRunning = false;
                break;
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
            case SDL_EVENT_WINDOW_RESTORED:
                if (g_presenter)
                    g_presenter->requestResize();
                break;
            case SDL_EVENT_WINDOW_FOCUS_LOST:
                for (auto &k : g_keys)
                    k.store(0, std::memory_order_relaxed);
                break;
            case SDL_EVENT_KEY_DOWN:
                if (e.key.repeat)
                    break;
                if (e.key.scancode == SDL_SCANCODE_F11)
                    toggleFullscreen(false);
                else if (e.key.scancode == SDL_SCANCODE_RETURN && (e.key.mod & SDL_KMOD_ALT))
                {
                    toggleFullscreen(true);
                    continue; // Enter is Start: Alt+Enter must not reach the game or the menu
                }
                else if (e.key.scancode == SDL_SCANCODE_F3)
                    setShowFps(!g_showRate);
                break;
            default:
                break;
            }
            if (ps2menu::host::handleEvent(e))
                continue; // the menu took it (open, or its shortcut)
            if (forwardToImgui)
                ImGui_ImplSDL3_ProcessEvent(&e);
            if ((e.type == SDL_EVENT_KEY_DOWN || e.type == SDL_EVENT_KEY_UP) && e.key.scancode < SDL_SCANCODE_COUNT)
            {
                g_keys[e.key.scancode].store(e.key.down ? 1 : 0, std::memory_order_relaxed);
                if (e.key.down && !e.key.repeat)
                    g_pressed[e.key.scancode].store(1, std::memory_order_relaxed);
            }
        }
        const bool menuOpen = ps2menu::isOpen();
        if (menuOpen)
            for (auto &k : g_keys)
                k.store(0, std::memory_order_relaxed);
        ps2input::update(menuOpen);
        ps2menu::host::update();
        return keepRunning && !ps2menu::host::quitRequested();
    }

    void presentGpuFrame(const GSGpuFrame &frame, bool fresh)
    {
        ImDrawData *ui = g_imguiFrameReady ? ImGui::GetDrawData() : nullptr;
        g_imguiFrameReady = false;
        if (g_presenter)
            g_presenter->present(fresh ? &frame : nullptr, ui);
    }

    void presentPixels(const uint8_t *rgba, uint32_t rowPixels, uint32_t width, uint32_t height, bool fresh)
    {
        ImDrawData *ui = g_imguiFrameReady ? ImGui::GetDrawData() : nullptr;
        g_imguiFrameReady = false;
        if (!g_renderer)
            return;
        if (fresh && rgba && width && height)
        {
            if (!g_texture || g_textureW != static_cast<int>(width) || g_textureH != static_cast<int>(height))
            {
                if (g_texture)
                    SDL_DestroyTexture(g_texture);
                g_texture = SDL_CreateTexture(g_renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING,
                                              static_cast<int>(width), static_cast<int>(height));
                SDL_SetTextureScaleMode(g_texture, SDL_SCALEMODE_NEAREST);
                g_textureW = static_cast<int>(width);
                g_textureH = static_cast<int>(height);
            }
            SDL_UpdateTexture(g_texture, nullptr, rgba, static_cast<int>(rowPixels * 4u));
        }
        else if (fresh && g_texture)
        {
            SDL_DestroyTexture(g_texture); // a latched tick without a field shows black
            g_texture = nullptr;
        }
        SDL_SetRenderDrawColor(g_renderer, 0, 0, 0, 255);
        SDL_RenderClear(g_renderer);
        if (g_texture)
        {
            int w = 0, h = 0;
            SDL_GetCurrentRenderOutputSize(g_renderer, &w, &h);
            const OutputRect r = letterbox43(static_cast<float>(w), static_cast<float>(h));
            const SDL_FRect dst{r.x, r.y, r.w, r.h};
            SDL_RenderTexture(g_renderer, g_texture, nullptr, &dst);
        }
        if (ui)
            ImGui_ImplSDLRenderer3_RenderDrawData(ui, g_renderer);
        SDL_RenderPresent(g_renderer);
    }

    void waitNextFrame()
    {
        uint64_t now = SDL_GetTicksNS();
        if (g_nextFrameNs > now)
        {
            SDL_DelayPrecise(g_nextFrameNs - now);
            now = g_nextFrameNs;
        }
        // Keep the 60 Hz cadence after a slightly late frame; restart it after a long stall.
        g_nextFrameNs = (now > g_nextFrameNs + kFramePeriodNs) ? now + kFramePeriodNs : g_nextFrameNs + kFramePeriodNs;
    }

    void setWindowTitleSuffix(const char *suffix)
    {
        if (g_window)
            SDL_SetWindowTitle(g_window, (suffix && *suffix) ? (g_baseTitle + " - " + suffix).c_str() : g_baseTitle.c_str());
    }

    bool imguiInit()
    {
        if (g_imguiContext || !g_window)
            return g_imguiContext;
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        bool ok = false;
        if (g_presenter)
            ok = ImGui_ImplSDL3_InitForVulkan(g_window) && g_presenter->imguiInit();
        else if (g_renderer)
            ok = ImGui_ImplSDL3_InitForSDLRenderer(g_window, g_renderer) && ImGui_ImplSDLRenderer3_Init(g_renderer);
        if (!ok)
        {
            std::fprintf(stderr, "[host] ImGui backend init failed\n");
            ImGui::DestroyContext();
            return false;
        }
        g_imguiContext = true;
        return true;
    }

    void imguiShutdown()
    {
        if (!g_imguiContext)
            return;
        if (g_presenter)
            g_presenter->imguiShutdown();
        else
            ImGui_ImplSDLRenderer3_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
        g_imguiContext = false;
        g_imguiFrameBegun = false;
        g_imguiFrameReady = false;
    }

    bool imguiBeginFrame()
    {
        if (!g_imguiContext)
            return false;
        if (g_imguiFrameBegun)
            return true;
        if (g_presenter)
            g_presenter->imguiNewFrame();
        else
            ImGui_ImplSDLRenderer3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        g_imguiFrameBegun = true;
        return true;
    }

    void imguiEndFrame() {} // the frame is rendered once in drawOverlay, after the counter and the menu

    void drawOverlay(uint64_t vsyncTick)
    {
        if (g_showRate)
        {
            const uint64_t now = SDL_GetTicksNS();
            if (g_rateStartNs == 0)
            {
                g_rateStartNs = now;
                g_rateTick = vsyncTick;
                g_rateFrames = 0;
            }
            else if (++g_rateFrames, now - g_rateStartNs >= 1000000000ull)
            {
                // Called once per host present, so its call rate is the screen FPS.
                const double secs = static_cast<double>(now - g_rateStartNs) / 1e9;
                g_fps = static_cast<float>(g_rateFrames / secs);
                g_vps = static_cast<float>(static_cast<double>(vsyncTick - g_rateTick) / secs);
                g_rateStartNs = now;
                g_rateTick = vsyncTick;
                g_rateFrames = 0;
                char text[64];
                std::snprintf(text, sizeof(text), "%.0f FPS / %.0f VPS", g_fps, g_vps);
                setWindowTitleSuffix(text);
            }
        }
        const bool menu = ps2menu::isOpen();
        if (!g_imguiFrameBegun && !g_showRate && !menu)
            return; // nothing over the game: no ImGui frame at all
        if (!imguiBeginFrame())
            return;
        if (g_showRate)
        {
            ImGui::SetNextWindowPos(ImVec2(8.0f, 8.0f));
            ImGui::SetNextWindowBgAlpha(0.6f);
            ImGui::Begin("##rate", nullptr,
                         ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs);
            ImGui::Text("%.0f FPS  %.0f VPS", g_fps, g_vps);
            ImGui::End();
        }
        if (menu)
            ps2menu::host::render();
        ImGui::Render();
        g_imguiFrameBegun = false;
        g_imguiFrameReady = true;
        g_imguiDrewLastFrame = true;
    }

    bool showFps() { return g_showRate; }

    void setShowFps(bool on)
    {
        if (on == g_showRate)
            return;
        g_showRate = on;
        g_rateStartNs = 0;
        if (!on)
            setWindowTitleSuffix(nullptr);
        ps2cfg::setBool("display.show_fps", on);
        ps2cfg::save();
    }

    bool vsync() { return g_vsync; }

    void setVsync(bool on)
    {
        g_vsync = on;
        if (g_presenter)
            g_presenter->setVsync(on);
        else if (g_renderer)
            SDL_SetRenderVSync(g_renderer, on ? 1 : 0);
    }
}
