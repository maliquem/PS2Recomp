#pragma once

#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/gs_vulkan_loader.h"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class GSVulkanBackend final : public GSRasterBackend
{
public:
    GSVulkanBackend();
    ~GSVulkanBackend() override;

    GSVulkanBackend(const GSVulkanBackend &) = delete;
    GSVulkanBackend &operator=(const GSVulkanBackend &) = delete;

    // Attempts to load Vulkan dynamically and initialize compute pipelines.
    // Returns nullptr if no hardware Vulkan compute device with shaderFloat64 is available.
    static std::unique_ptr<GSVulkanBackend> CreateIfAvailable(std::string *outReason = nullptr);

    const std::string &deviceName() const { return m_deviceName; }

    // Instance extensions a windowed host needs for its surface (from SDL). Set before CreateIfAvailable;
    // when non-empty the device also enables VK_KHR_swapchain so the host can present from it.
    static void SetHostInstanceExtensions(std::vector<std::string> extensions);

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;

    void Submit(const GSPrimitiveBatch &batch) override;
    void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override;
    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;

    void Flush() override;
    void TextureFlush() override;
    void Sync(GSSyncReason reason) override;
    PresentationFrame Present(const GSPresentationRequest &request) override;
    bool PresentGpu(const GSPresentationRequest &request, GSGpuFrame &out) override;
    const GSHostGpuContext *HostGpuContext() const override;

    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;

    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;

private:
    struct GpuBuffer
    {
        ps2vk::VkBuffer buffer = nullptr;
        ps2vk::VkDeviceMemory memory = nullptr;
        void *mapped = nullptr;
        ps2vk::VkDeviceSize size = 0;
    };

    struct TransferPush
    {
        uint32_t mode;
        uint32_t dbp, dbw, dpsm;
        uint32_t dsax, dsay, rrw, rrh;
        uint32_t startPixel, pixelCount, stagingByteOffset, dir;
        uint32_t sbp, sbw, spsm, ssax;
        uint32_t ssay, csm, csa, texclutPacked;
    };

    struct GpuSpriteDraw
    {
        int32_t drawX0, drawY0, drawW, drawH;
        int32_t unclippedX0, unclippedY0, uBaseFixed, uStepFixed;
        float v0f, v1f, invH;
        uint32_t zVal;
        uint32_t rgba;
        uint32_t fogAndColor;
        uint32_t frameInfo;
        uint32_t fbmsk;
        uint32_t depthInfo;
        uint32_t testLo;
        uint32_t alphaLo;
        uint32_t alphaFix;
        uint32_t tbp0;
        uint32_t texMeta;
        uint32_t texDim;
        uint32_t clampLo;
        uint32_t clampHi;
        uint32_t texaPacked;
        uint32_t stagingTexOff;
        uint32_t stagingTexRect;
    };

    struct alignas(16) GpuTriangleDraw
    {
        int32_t bounds[4];
        int32_t vx[4];
        int32_t vy[4];
        float s[4];
        float t[4];
        float q[4];
        float uv01[4];
        float uv2_pad[4];
        double z0, z1, z2, invArea;
        uint32_t rgba[4];
        uint32_t fogColor;
        uint32_t frameInfo;
        uint32_t fbmsk;
        uint32_t depthInfo;
        uint32_t testLo;
        uint32_t alphaLo;
        uint32_t alphaFix;
        uint32_t tbp0;
        uint32_t texMeta;
        uint32_t texDim;
        uint32_t clampLo;
        uint32_t clampHi;
        uint32_t texaPacked;
        uint32_t pad0, pad1, pad2;
    };

    struct GpuTileHeader
    {
        uint32_t tileX;
        uint32_t tileY;
        uint32_t offset;
        uint32_t count;
    };

    struct PresentCircuitPush
    {
        uint32_t width, height, baseBlock, pitch, originX, originY, psm;
    };

    struct PresentPush
    {
        PresentCircuitPush crt1, crt2;
        uint32_t pmode, background, field, parity;
    };

    enum class CmdKind : uint8_t
    {
        Upload,
        TileBatch,
    };

    struct PendingCmd
    {
        CmdKind kind = CmdKind::Upload;
        bool barrierBefore = false;
        uint32_t firstTile = 0;
        uint32_t tileCount = 0;
        uint32_t groupsX = 0;
        TransferPush transfer{};
    };

    struct BinnedDraw
    {
        uint32_t encodedIdx = 0;
        uint16_t tx0 = 0, ty0 = 0, tx1 = 0, ty1 = 0;
    };

    struct DeferredUpload
    {
        uint32_t dbp = 0, dbw = 0, dpsm = 0;
        uint32_t dsax = 0, dsay = 0, rrw = 0, rrh = 0;
        uint32_t stagingByteOffset = 0;
        uint32_t totalPixels = 0;
        uint32_t firstPage = 0, lastPage = 0;
    };

    using PageMask = std::array<uint64_t, 8>;

    // Per-submission resources. Two slots let the EE record into one while the GPU runs the other.
    struct FrameSlot
    {
        GpuBuffer uploadStaging{};
        GpuBuffer spriteBuf{};
        GpuBuffer triangleBuf{};
        GpuBuffer tileHeaderBuf{};
        GpuBuffer tileIndexBuf{};
        ps2vk::VkCommandBuffer cmdBuf = nullptr;
        ps2vk::VkFence fence = nullptr;
        ps2vk::VkDescriptorSet uploadDescSet = nullptr;
        ps2vk::VkDescriptorSet tileDescSet = nullptr;
    };

    bool initVulkan(std::string *outReason);
    void destroyVulkan();
    bool createBuffer(GpuBuffer &buf, ps2vk::VkDeviceSize size, ps2vk::VkFlags usage,
                      ps2vk::VkFlags requiredProps, ps2vk::VkFlags preferredProps = 0);
    void destroyBuffer(GpuBuffer &buf);
    bool createComputePipeline(const uint32_t *spv, size_t wordCount, uint32_t pushBytes,
                               ps2vk::VkDescriptorSetLayout descLayout,
                               ps2vk::VkPipelineLayout &outLayout, ps2vk::VkPipeline &outPipeline);

    void waitForInFlightUnlocked() const;
    bool beginCmdUnlocked(ps2vk::VkCommandBuffer cmd) const;
    void submitBufferCopyUnlocked(ps2vk::VkBuffer src, ps2vk::VkBuffer dst, ps2vk::VkDeviceSize size) const;
    void submitPendingUnlocked(bool copyVramToReadback, bool waitForCompletion) const;
    bool submitPresentUnlocked(const PresentPush &presentPush, uint32_t presentHeight) const;
    void syncGpuToCpuUnlocked() const;
    void syncCpuToGpuUnlocked();
    PresentationFrame presentImpl(const GSPresentationRequest &request, GSGpuFrame *gpuOut);
    PresentationFrame presentCpuFallbackUnlocked(const GSPresentationRequest &request, GSGpuFrame *gpuOut);

    void flushPartialContiguousTransferUnlocked() const;
    void closeTileBatchUnlocked() const;
    void materializeDeferredUploadsUnlocked(uint32_t firstPage, uint32_t lastPage, bool all) const;
    void emitUploadCmdUnlocked(const TransferPush &tr, uint32_t groupsX, uint32_t firstPage, uint32_t lastPage) const;

    void recordSpriteUnlocked(const GSPrimitiveBatch &batch);
    void recordTriangleUnlocked(const GSPrimitiveBatch &batch);

    static void setPageRange(PageMask &mask, uint32_t firstPage, uint32_t lastPage);
    static bool masksOverlap(const PageMask &a, const PageMask &b);
    static bool maskOverlapsRange(const PageMask &mask, uint32_t firstPage, uint32_t lastPage);
    static void mergeMasks(PageMask &dst, const PageMask &src);

    mutable std::mutex m_mutex;
    // Serializes Present() so the UI thread can wait on m_presentFence without holding m_mutex.
    // Lock order: m_presentMutex, then m_mutex.
    std::mutex m_presentMutex;
    ps2vk::Dispatch m_vk{};
    ps2vk::VkInstance m_instance = nullptr;
    ps2vk::VkPhysicalDevice m_physicalDevice = nullptr;
    ps2vk::VkDevice m_device = nullptr;
    ps2vk::VkQueue m_queue = nullptr;
    uint32_t m_queueFamily = 0;
    GSHostGpuContext m_hostContext{};
    ps2vk::VkPhysicalDeviceMemoryProperties m_memProps{};
    std::string m_deviceName;

    ps2vk::VkDescriptorSetLayout m_descLayout3 = nullptr;
    ps2vk::VkDescriptorSetLayout m_descLayout7 = nullptr;
    ps2vk::VkDescriptorPool m_descPool = nullptr;
    ps2vk::VkDescriptorSet m_presentDescSet = nullptr;

    ps2vk::VkPipelineLayout m_uploadLayout = nullptr;
    ps2vk::VkPipeline m_uploadPipeline = nullptr;
    ps2vk::VkPipelineLayout m_tileLayout = nullptr;
    ps2vk::VkPipeline m_tilePipeline = nullptr;
    ps2vk::VkPipelineLayout m_presentLayout = nullptr;
    ps2vk::VkPipeline m_presentPipeline = nullptr;

    ps2vk::VkCommandPool m_cmdPool = nullptr;
    // m_cur is always free for recording; m_spare is the previous submission, in flight while m_gpuInFlight.
    mutable FrameSlot m_cur{};
    mutable FrameSlot m_spare{};
    mutable bool m_gpuInFlight = false;
    ps2vk::VkCommandBuffer m_presentCmdBuf = nullptr;
    ps2vk::VkFence m_presentFence = nullptr;

    GpuBuffer m_vramGpu{};
    GpuBuffer m_clutGpu{};
    GpuBuffer m_presentOut{};
    GpuBuffer m_vramReadback{};

    uint8_t *m_vram = nullptr;
    uint32_t m_vramSize = 0;
    mutable bool m_gpuVramDirtyForCpu = false;
    mutable bool m_cpuVramDirtyForGpu = false;
    mutable GSCpuBackend m_cpuFallback;

    std::array<uint32_t, 2> m_clutCbp{};
    GSTransferCommand m_transfer{};
    GSTransferSnapshot m_transferState{};
    mutable bool m_activeTransferContiguous = false;
    mutable uint32_t m_activeTransferStagingOff = 0;
    mutable uint32_t m_activeTransferStagingBytes = 0;

    mutable std::vector<PendingCmd> m_cmds;
    mutable std::vector<BinnedDraw> m_batchDraws;
    mutable std::vector<DeferredUpload> m_deferredUploads;
    mutable std::vector<uint32_t> m_tileCounts;
    mutable std::vector<uint32_t> m_tileCursor;
    mutable uint32_t m_stagingOffset = 0;
    mutable uint32_t m_spriteCount = 0;
    mutable uint32_t m_triangleCount = 0;
    mutable uint32_t m_tileHeaderCount = 0;
    mutable uint32_t m_tileIndexCount = 0;

    // Exact GPU-level and batch-level page hazard tracking (RAW + WAR + WAW).
    mutable PageMask m_gpuWritePages{};
    mutable PageMask m_gpuReadPages{};
    mutable bool m_gpuClutDirty = false;

    mutable PageMask m_batchWritePages{};
    mutable PageMask m_batchReadPages{};
    mutable uint32_t m_batchFrameKey = 0xFFFFFFFFu;
    mutable bool m_batchReadsClut = false;
    mutable uint32_t m_batchEstimatedIndices = 0;
};
