// Save state for the GS: frontend registers, GIF arbiter queue and each backend's VRAM, CLUT and
// transfer state. Kept out of the backend sources so it merges independently of render work.
// The Vulkan backend and GIF arbiter parts are in gs_vulkan_state.cpp (the gs_cache tests build without Vulkan).

#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "ps2x/state_archive.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_pacing.h"

#include <algorithm>
#include <iostream>
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
    {
        m_image24 = m_trxdir == 0u && (m_bitbltbuf.dpsm == GS_PSM_CT24 || m_bitbltbuf.dpsm == GS_PSM_Z24);
        m_imageCarryLen = 0u;
        snapshotVRAM();
    }
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
        bool readsBack = false; // the game reads the back buffer (texture or transfer): never drop into it again
        uint32_t backRows = 512; // rows of the displayed frame buffer (DISPLAY height)
        uint32_t backFbw = 10;   // width of the displayed frame buffer (DISPFB FBW, 64-pixel units)
        SkipPhase phase = SkipPhase::Drawing;
    };

    SkipState g_skip; // guarded by GS::m_stateMutex (vertexKick and TRXDIR run under it)

    // Rows of the displayed frame buffer: DISPLAY height over the vertical magnification. With SMODE2 INT=1, FFMD=0
    // DISPLAY holds one field of a full frame buffer twice as tall (the FMV player); with FFMD=1 (Shadow Hearts'
    // game mode, DH 223) the buffer is DH+1 rows, as the Present reads it.
    uint32_t displayRows(const GSRegisters &regs)
    {
        const uint64_t display = (regs.pmode & 2u) ? regs.display2 : regs.display1;
        uint32_t rows = static_cast<uint32_t>(((display >> 44) & 0x7FFu) + 1u) / static_cast<uint32_t>(((display >> 27) & 3u) + 1u);
        if ((regs.smode2 & 3u) == 1u)
            rows *= 2u;
        return std::clamp<uint32_t>(rows, 1u, 2048u);
    }

    // Once per run: which read stopped the drops (turbo stats on stdout). The read is only seen when it happens, so
    // the draws dropped before it in that frame are lost; keeping readsBack for the whole run limits that to once.
    void noteReadsBack(int why, uint32_t blockAddress)
    {
        if (!g_skip.readsBack)
            std::cout << "[frameskip] back buffer read (" << (why == 1 ? "texture" : "transfer") << "): block " << blockAddress
                      << ", back fbp " << g_skip.backFbp << ", fbw " << g_skip.backFbw << ", rows " << g_skip.backRows << std::endl;
        g_skip.readsBack = true;
        ps2x::pacing::g_frameskipReadsBack.store(why, std::memory_order_relaxed);
    }

    // Counts PSMCT32 pages (32 rows): a 16-bit buffer is covered twice over, which only stops more drops.
    // shortcut: only reads of the back buffer are caught; a game that reads the buffer on display (feedback, screen
    // capture) after a swap of a frame that lost draws would see them missing. Widen it if a game does that.
    bool readsBackBuffer(uint32_t blockAddress)
    {
        const uint32_t start = g_skip.backFbp * 32u;
        const uint32_t pages = std::max<uint32_t>(g_skip.backFbw, 1u) * ((g_skip.backRows + 31u) / 32u);
        return g_skip.known && blockAddress >= start && blockAddress < start + pages * 32u;
    }
}

void GS::frameskipNoteTransfer(const GSTransferCommand &command) const
{
    // Directions 1 (local to host) and 2 (local to local) read GS memory.
    if ((command.direction == 1u || command.direction == 2u) && readsBackBuffer(command.bitbltbuf.sbp))
        noteReadsBack(2, command.bitbltbuf.sbp);
}

bool GS::frameskipDropsDraw(const GSPrimitiveBatch &batch) const
{
    using namespace ps2x::pacing;
    if (!m_privRegs)
        return false;
    const bool enabled = g_frameskipDraws.load(std::memory_order_relaxed);
    if (!enabled)
    {
        // Turbo ended: draw everything now. Keep tracking swaps while a frame that lost draws is in the back buffer
        // or on display; the swap after it (or the host's fallback) marks the display drawn again.
        g_skip.dropCurrent = false;
        g_skip.phase = SkipPhase::Drawing;
        if (!g_skip.hadDrops && g_frameskipDisplayDrawn.load(std::memory_order_acquire))
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
        g_skip.backRows = displayRows(*m_privRegs);
        g_skip.backFbw = static_cast<uint32_t>((dispfb >> 9) & 0x3Fu);
        g_skip.dropCurrent = false;
        g_skip.phase = SkipPhase::Drawing;
    }
    else if (displayFbp != g_skip.displayFbp)
    {
        // The frame that just went on display is complete; the old display buffer becomes the back buffer.
        g_frameskipDisplayDrawn.store(!g_skip.hadDrops, std::memory_order_release);
        g_skip.backFbp = g_skip.displayFbp;
        g_skip.displayFbp = displayFbp;
        g_skip.backRows = displayRows(*m_privRegs);
        g_skip.backFbw = static_cast<uint32_t>((dispfb >> 9) & 0x3Fu);
        if (enabled)
        {
            g_frameskipSwaps.fetch_add(1u, std::memory_order_relaxed);
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
            if (g_skip.dropCurrent)
                g_frameskipDropFrames.fetch_add(1u, std::memory_order_relaxed);
        }
        g_skip.hadDrops = false;
    }

    const GSContext &context = batch.state.context;
    if (batch.state.prim.tme && readsBackBuffer(context.tex0.tbp0))
        noteReadsBack(1, context.tex0.tbp0);
    const bool drop = g_skip.dropCurrent && !g_skip.readsBack && context.frame.fbp == g_skip.backFbp;
    g_skip.hadDrops = g_skip.hadDrops || drop;
    if (enabled)
        (drop ? g_frameskipDropped : g_frameskipSubmitted).fetch_add(1u, std::memory_order_relaxed);
    return drop;
}
