// Save state for the GS: frontend registers, GIF arbiter queue and each backend's VRAM, CLUT and
// transfer state. Kept out of the backend sources so it merges independently of render work.
// The Vulkan backend and GIF arbiter parts are in gs_vulkan_state.cpp (the gs_cache tests build without Vulkan).

#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "ps2x/state_archive.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_pacing.h"

#include <stdexcept>

namespace
{
    constexpr uint32_t kVramBytes = GSMem::MEMORY_SIZE;
}

void GSRasterBackend::stateIo(ps2x::state::Archive &)
{
    throw std::runtime_error("GS backend does not support save states");
}

void GSCpuBackend::stateIo(ps2x::state::Archive &ar)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || m_vramSize < kVramBytes)
        throw std::runtime_error("GS CPU backend has no VRAM");
    ar.raw(m_vram, kVramBytes);
    ar(m_clut, m_clutCbp, m_transfer, m_transferState, m_localToHostBuffer, m_localToHostReadPos);
    if (ar.loading())
    {
        m_clutLutValid = false;
        m_texturePageCache.Invalidate();
    }
}

void GS::stateIo(ps2x::state::Archive &ar)
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    ar(m_ctx, m_prim, m_primRegister, m_prmodeRegister,
       m_curR, m_curG, m_curB, m_curA, m_curQ, m_curS, m_curT, m_curU, m_curV, m_curFog, m_fogR, m_fogG, m_fogB,
       m_prmodecont, m_pabe, m_scanmsk, m_dimx, m_dthe, m_colclamp, m_texa, m_texclut,
       m_bitbltbuf, m_trxpos, m_trxreg, m_trxdir, m_vtxQueue, m_vtxCount, m_vtxIndex,
       m_lastDisplayBaseBytes, m_preferredDisplaySourceFrame, m_preferredDisplayDestFbp, m_hasPreferredDisplaySource);
    {
        std::lock_guard<std::mutex> backendLock(m_backendLifetimeMutex);
        if (!m_backend)
            throw std::runtime_error("GS has no backend");
        m_backend->stateIo(ar);
    }
    if (ar.loading())
        snapshotVRAM();
}

// ---------------------------------------------------------------------------------------------
// Turbo frameskip level 2+: drop the drawing of frames the host will never show.
//
// A buffer swap is a change of the displayed FBP. At each swap the GS decides whether the frame that
// starts in the back buffer (the buffer that just left the display) is drawn. Only draws into that
// back buffer are dropped: render targets, the front buffer, CLUT and transfers always run. The host
// latches only while the displayed frame was drawn, so a dropped frame is never shown, and when turbo
// ends drops stop at the next swap.
// ---------------------------------------------------------------------------------------------

namespace
{
    enum class SkipPhase
    {
        Drawing,    // every frame drawn; waiting for the host to latch the one on display
        Dropping,   // back-buffer draws dropped until the host asks for a frame
        DrawWanted, // the frame being drawn is the one the host asked for
    };

    struct SkipState
    {
        bool known = false;
        uint32_t displayFbp = 0;
        uint32_t backFbp = 0;
        bool dropCurrent = false;
        bool hadDrops = false; // the frame in the back buffer lost draws
        SkipPhase phase = SkipPhase::Drawing;
    };

    SkipState g_skip; // guarded by GS::m_stateMutex (vertexKick runs under it)
}

bool GS::frameskipDropsDraw(const GSPrimitiveBatch &batch) const
{
    using namespace ps2x::pacing;
    if (!m_privRegs)
        return false;
    const bool enabled = g_frameskipDraws.load(std::memory_order_relaxed);
    if (!enabled)
    {
        // Turbo ended: draw everything now; a frame that already lost draws is still not latched.
        g_skip.dropCurrent = false;
        g_skip.phase = SkipPhase::Drawing;
        if (!g_skip.hadDrops)
        {
            g_skip.known = false;
            return false;
        }
    }

    const uint64_t pmode = m_privRegs->pmode;
    const uint64_t dispfb = (pmode & 2u) ? m_privRegs->dispfb2 : m_privRegs->dispfb1;
    const uint32_t displayFbp = static_cast<uint32_t>(dispfb & 0x1FFu);
    if (!g_skip.known)
    {
        g_skip.known = true;
        g_skip.displayFbp = displayFbp;
        g_skip.backFbp = displayFbp;
        g_skip.dropCurrent = false;
        g_skip.phase = SkipPhase::Drawing;
    }
    else if (displayFbp != g_skip.displayFbp)
    {
        // The frame that just went on display is complete; the old display buffer becomes the back buffer.
        g_frameskipDisplayDrawn.store(!g_skip.hadDrops, std::memory_order_release);
        g_skip.backFbp = g_skip.displayFbp;
        g_skip.displayFbp = displayFbp;
        if (enabled)
        {
            switch (g_skip.phase)
            {
            case SkipPhase::Drawing:
                // Keep drawing until the host took the drawn frame now on display.
                if (g_frameskipLatched.exchange(false, std::memory_order_acq_rel))
                    g_skip.phase = SkipPhase::Dropping;
                break;
            case SkipPhase::Dropping:
                if (g_frameskipWant.exchange(false, std::memory_order_acq_rel))
                    g_skip.phase = SkipPhase::DrawWanted;
                break;
            case SkipPhase::DrawWanted:
                // The wanted frame is on display now; draw on until the host has latched it.
                g_skip.phase = SkipPhase::Drawing;
                break;
            }
            g_skip.dropCurrent = g_skip.phase == SkipPhase::Dropping;
        }
        g_skip.hadDrops = false;
    }

    const bool drop = g_skip.dropCurrent && batch.state.context.frame.fbp == g_skip.backFbp;
    g_skip.hadDrops = g_skip.hadDrops || drop;
    if (enabled)
        (drop ? g_frameskipDropped : g_frameskipSubmitted).fetch_add(1u, std::memory_order_relaxed);
    return drop;
}
