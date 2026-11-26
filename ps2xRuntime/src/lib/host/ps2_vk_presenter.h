#pragma once

#include "runtime/gs/gs_types.h"

#include <memory>
#include <string>

struct SDL_Window;
struct ImDrawData;

// The 4:3 output rect centered in a w x h target; both presenters draw the field into it.
// The GS output is meant for a 4:3 TV; C2 turns this into the widescreen/scale setting.
struct OutputRect
{
    float x, y, w, h;
};
inline OutputRect letterbox43(float w, float h)
{
    float dw = w, dh = w * 3.0f / 4.0f;
    if (dh > h)
    {
        dh = h;
        dw = h * 4.0f / 3.0f;
    }
    return {(w - dw) * 0.5f, (h - dh) * 0.5f, dw, dh};
}

// PS2X_PRESENT_DUMP_REQ: the prefix a test wrote in the request file, empty when nothing is asked. An empty or
// unreadable file is deleted here; a valid one stays until the caller has taken the dump and deletes it.
std::string readDumpRequest(const char *path);

// Presents the GS output buffer through a Vulkan swapchain on the GS backend's device: buffer -> scene
// image -> (post-process passes, none yet) -> letterboxed blit -> ImGui pass -> present. No CPU readback.
class VkPresenter
{
public:
    VkPresenter();
    ~VkPresenter();

    bool init(SDL_Window *window, const GSHostGpuContext &ctx, bool vsync, std::string &error);
    void shutdown();

    // Waits until the last present stopped reading the GS buffer; call before the GS writes it again.
    void waitFrame();
    // `frame` null or not fresh = draw the last scene again. `imgui` may be null.
    void present(const GSGpuFrame *frame, ImDrawData *imgui);
    void requestResize() { m_resizePending = true; }

    bool imguiInit();
    void imguiNewFrame();
    void imguiShutdown();

private:
    struct Impl;
    std::unique_ptr<Impl> m;
    bool m_resizePending = false;
};
