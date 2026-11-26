// Save state for the GS Vulkan backend (VRAM, CLUT read back from the GPU) and the GIF arbiter queue.
// Separate from gs_backend_state.cpp so the CPU-only gs_cache test library links without Vulkan.

#include "runtime/gs/gs_vulkan_backend.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "ps2x/state_archive.h"

#include <array>
#include <cstring>
#include <stdexcept>

namespace
{
    constexpr uint32_t kVramBytes = GSMem::MEMORY_SIZE;
    constexpr uint32_t kClutBytes = 512u * sizeof(uint16_t);
}

void GSVulkanBackend::readClutUnlocked(uint8_t *out) const
{
    // The CLUT lives only on the GPU: copy it into the (idle) readback buffer and wait.
    if (!beginCmdUnlocked(m_cur.cmdBuf))
        throw std::runtime_error("GS Vulkan: CLUT readback failed");
    const ps2vk::VkBufferCopy copy{0u, 0u, kClutBytes};
    m_vk.vkCmdCopyBuffer(m_cur.cmdBuf, m_clutGpu.buffer, m_vramReadback.buffer, 1u, &copy);
    ps2vk::VkBufferMemoryBarrier toHost{};
    toHost.sType = ps2vk::VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    toHost.srcAccessMask = ps2vk::VK_ACCESS_TRANSFER_WRITE_BIT;
    toHost.dstAccessMask = ps2vk::VK_ACCESS_HOST_READ_BIT;
    toHost.srcQueueFamilyIndex = ps2vk::VK_QUEUE_FAMILY_IGNORED;
    toHost.dstQueueFamilyIndex = ps2vk::VK_QUEUE_FAMILY_IGNORED;
    toHost.buffer = m_vramReadback.buffer;
    toHost.offset = 0u;
    toHost.size = kClutBytes;
    m_vk.vkCmdPipelineBarrier(m_cur.cmdBuf, ps2vk::VK_PIPELINE_STAGE_TRANSFER_BIT, ps2vk::VK_PIPELINE_STAGE_HOST_BIT,
                              0u, 0u, nullptr, 1u, &toHost, 0u, nullptr);
    if (m_vk.vkEndCommandBuffer(m_cur.cmdBuf) != ps2vk::VK_SUCCESS)
        throw std::runtime_error("GS Vulkan: CLUT readback failed");
    ps2vk::VkSubmitInfo submitInfo{};
    submitInfo.sType = ps2vk::VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1u;
    submitInfo.pCommandBuffers = &m_cur.cmdBuf;
    if (m_vk.vkQueueSubmit(m_queue, 1u, &submitInfo, m_cur.fence) != ps2vk::VK_SUCCESS ||
        m_vk.vkWaitForFences(m_device, 1u, &m_cur.fence, ps2vk::VK_TRUE, UINT64_MAX) != ps2vk::VK_SUCCESS)
        throw std::runtime_error("GS Vulkan: CLUT readback failed");
    m_vk.vkResetFences(m_device, 1u, &m_cur.fence);
    std::memcpy(out, m_vramReadback.mapped, kClutBytes);
}

void GSVulkanBackend::stateIo(ps2x::state::Archive &ar)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || m_vramSize < kVramBytes)
        throw std::runtime_error("GS Vulkan backend has no VRAM");

    std::array<uint8_t, kClutBytes> clut{};
    if (!ar.loading())
    {
        // Everything recorded so far lands in m_vram, then the GPU is idle for the CLUT copy.
        syncGpuToCpuUnlocked();
        waitForInFlightUnlocked();
        readClutUnlocked(clut.data());
    }

    // The CPU fallback shares m_vram, so its io carries the VRAM bytes and its local->host state.
    m_cpuFallback.stateIo(ar);
    ar(clut, m_clutCbp, m_transfer, m_transferState);

    if (ar.loading())
    {
        if (!m_cmds.empty() || !m_batchDraws.empty() || !m_deferredUploads.empty() || m_activeTransferContiguous)
            submitPendingUnlocked(false, true);
        waitForInFlightUnlocked();
        m_cmds.clear();
        m_batchDraws.clear();
        m_deferredUploads.clear();
        m_activeTransferContiguous = false;
        m_activeTransferStagingOff = 0u;
        m_activeTransferStagingBytes = 0u;
        m_batchWritePages.fill(0ull);
        m_batchReadPages.fill(0ull);
        m_batchFrameKey = 0xFFFFFFFFu;
        m_batchReadsClut = false;
        m_batchEstimatedIndices = 0u;
        m_gpuVramDirtyForCpu = false;
        m_cpuVramDirtyForGpu = true;
        syncCpuToGpuUnlocked();
        std::memcpy(m_cur.uploadStaging.mapped, clut.data(), kClutBytes);
        submitBufferCopyUnlocked(m_cur.uploadStaging.buffer, m_clutGpu.buffer, kClutBytes);
    }
}

void GifArbiter::stateIo(ps2x::state::Archive &ar)
{
    ar(m_queue, m_arena);
}
