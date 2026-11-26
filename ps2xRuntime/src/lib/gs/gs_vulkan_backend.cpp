#include "runtime/gs/gs_vulkan_backend.h"
#include "runtime/gs/gs_vulkan_shaders.h"
#include "runtime/gs/ps2_gs_common.h"
#include "runtime/gs/ps2_gs_memory.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace
{
constexpr uint32_t kVramBytes = GSMem::MEMORY_SIZE;
constexpr uint32_t kClutBytes = 512u * sizeof(uint16_t);
constexpr uint32_t kStagingBytes = 16u * 1024u * 1024u;
constexpr uint32_t kMaxSprites = 32768u;
constexpr uint32_t kMaxTriangles = 32768u;
constexpr uint32_t kMaxTileHeaders = 65536u;
constexpr uint32_t kMaxTileIndices = 1048576u; // 4 MiB
constexpr uint32_t kTileShift = 4u;            // 16x16 tiles
constexpr uint32_t kTileSize = 1u << kTileShift;
constexpr uint32_t kTileGridW = 64u;           // up to 1024 px wide
constexpr uint32_t kTileGridH = 64u;           // up to 1024 px tall
constexpr uint32_t kTileGridSize = kTileGridW * kTileGridH;
constexpr uint32_t kHostFrameWidth = 640u;
constexpr uint32_t kHostFrameHeight = 512u;
constexpr uint32_t kDefaultDisplayWidth = 640u;
constexpr uint32_t kDefaultDisplayHeight = 448u;
constexpr uint32_t kPresentBytes = kHostFrameWidth * kHostFrameHeight * 4u;

float fabsQ(float q)
{
    return (std::fabs(q) > 1.0e-8f) ? q : 1.0f;
}

bool isFourBitIndexedPsm(uint8_t psm)
{
    return psm == GS_PSM_T4 || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH;
}

bool isEightBitIndexedPsm(uint8_t psm)
{
    return psm == GS_PSM_T8 || psm == GS_PSM_T8H;
}

bool isIndexedPsm(uint8_t psm)
{
    return isFourBitIndexedPsm(psm) || isEightBitIndexedPsm(psm);
}

uint32_t packTexa(const GSTexaReg &texa)
{
    return static_cast<uint32_t>(texa.ta0) |
           (static_cast<uint32_t>(texa.ta1) << 8u) |
           (texa.aem ? (1u << 16u) : 0u);
}

uint32_t packTexMeta(const GSTex0Reg &tex)
{
    return (static_cast<uint32_t>(tex.tbw) & 0x3Fu) |
           ((static_cast<uint32_t>(tex.psm) & 0x3Fu) << 6u) |
           ((static_cast<uint32_t>(tex.tfx) & 0x3u) << 12u) |
           ((static_cast<uint32_t>(tex.tcc) & 0x1u) << 14u) |
           ((static_cast<uint32_t>(tex.csm) & 0x1u) << 15u) |
           ((static_cast<uint32_t>(tex.csa) & 0x1Fu) << 16u) |
           ((static_cast<uint32_t>(tex.cpsm) & 0x3Fu) << 21u);
}

uint32_t bytesForPixels(uint8_t psm, uint32_t pixels)
{
    switch (psm)
    {
    case GS_PSM_CT32:
    case GS_PSM_Z32:
        return pixels * 4u;
    case GS_PSM_CT24:
    case GS_PSM_Z24:
        return pixels * 3u;
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
    case GS_PSM_Z16:
    case GS_PSM_Z16S:
        return pixels * 2u;
    case GS_PSM_T8:
    case GS_PSM_T8H:
        return pixels;
    case GS_PSM_T4:
    case GS_PSM_T4HL:
    case GS_PSM_T4HH:
        return (pixels + 1u) / 2u;
    default:
        return 0u;
    }
}

void computePsmPageRange(uint32_t psm, uint32_t baseBlock, uint32_t bw,
                         uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1,
                         uint32_t &firstPage, uint32_t &lastPage)
{
    const uint32_t basePage = baseBlock >> 5u;
    uint32_t pageWShift = 6u;
    uint32_t pageHShift = 5u;
    uint32_t pagesPerRow = std::max<uint32_t>(bw, 1u);
    if (psm == GS_PSM_CT16 || psm == GS_PSM_CT16S || psm == GS_PSM_Z16 || psm == GS_PSM_Z16S)
    {
        pageWShift = 6u;
        pageHShift = 6u;
    }
    else if (psm == GS_PSM_T8)
    {
        pageWShift = 7u;
        pageHShift = 6u;
        pagesPerRow = std::max<uint32_t>(bw >> 1u, 1u);
    }
    else if (psm == GS_PSM_T4)
    {
        pageWShift = 7u;
        pageHShift = 7u;
        pagesPerRow = std::max<uint32_t>(bw >> 1u, 1u);
    }
    firstPage = basePage + (y0 >> pageHShift) * pagesPerRow + (x0 >> pageWShift);
    lastPage = basePage + (y1 >> pageHShift) * pagesPerRow + (x1 >> pageWShift) + ((baseBlock & 31u) != 0u ? 1u : 0u);
}

void decodeDisplaySize(uint64_t display64, uint32_t &outWidth, uint32_t &outHeight)
{
    const uint32_t dw = static_cast<uint32_t>((display64 >> 32u) & 0x0FFFu);
    const uint32_t dh = static_cast<uint32_t>((display64 >> 44u) & 0x07FFu);
    const uint32_t magh = static_cast<uint32_t>((display64 >> 23u) & 0x0Fu);

    outWidth = (dw + 1u) / (magh + 1u);
    outHeight = dh + 1u;
    if (outWidth < 64u || outHeight < 64u)
    {
        outWidth = kDefaultDisplayWidth;
        outHeight = kDefaultDisplayHeight;
    }
    outWidth = std::min<uint32_t>(outWidth, kHostFrameWidth);
    outHeight = std::min<uint32_t>(outHeight, kHostFrameHeight);
}

GSFrameReg decodeDisplayFrame(uint64_t dispfb64)
{
    GSFrameReg frame{};
    frame.fbp = static_cast<uint32_t>(dispfb64 & 0x1FFu);
    frame.fbw = static_cast<uint32_t>((dispfb64 >> 9u) & 0x3Fu);
    frame.psm = static_cast<uint8_t>((dispfb64 >> 15u) & 0x1Fu);
    return frame;
}

bool hasDisplaySetup(uint64_t display64, const GSFrameReg &frame)
{
    const uint32_t dw = static_cast<uint32_t>((display64 >> 32u) & 0x0FFFu);
    const uint32_t dh = static_cast<uint32_t>((display64 >> 44u) & 0x07FFu);
    const uint32_t magh = static_cast<uint32_t>((display64 >> 23u) & 0x0Fu);
    return frame.fbw != 0u || dw != 0u || dh != 0u || magh != 0u;
}

bool isPresentablePsm(uint8_t psm)
{
    return psm == GS_PSM_CT32 || psm == GS_PSM_CT24 || psm == GS_PSM_CT16 || psm == GS_PSM_CT16S;
}

std::vector<std::string> &hostInstanceExtensions()
{
    static std::vector<std::string> s_extensions;
    return s_extensions;
}

template <typename Handle>
uint64_t handleBits(Handle h)
{
    if constexpr (std::is_pointer_v<Handle>)
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(h));
    else
        return static_cast<uint64_t>(h);
}

bool loadVulkanLibrary(ps2vk::Dispatch &vk, std::string *outReason)
{
#if defined(_WIN32)
    HMODULE lib = LoadLibraryA("vulkan-1.dll");
    if (!lib)
    {
        if (outReason) *outReason = "vulkan-1.dll not found";
        return false;
    }
    vk.library = lib;
    vk.vkGetInstanceProcAddr = reinterpret_cast<ps2vk::PFN_vkGetInstanceProcAddr>(
        GetProcAddress(lib, "vkGetInstanceProcAddr"));
#else
    void *lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib)
        lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib)
    {
        if (outReason) *outReason = "libvulkan.so.1 not found";
        return false;
    }
    vk.library = lib;
    vk.vkGetInstanceProcAddr = reinterpret_cast<ps2vk::PFN_vkGetInstanceProcAddr>(
        dlsym(lib, "vkGetInstanceProcAddr"));
#endif
    if (!vk.vkGetInstanceProcAddr)
    {
        if (outReason) *outReason = "vkGetInstanceProcAddr missing";
        return false;
    }
    vk.vkCreateInstance = reinterpret_cast<ps2vk::PFN_vkCreateInstance>(
        vk.vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));
    return vk.vkCreateInstance != nullptr;
}

void unloadVulkanLibrary(ps2vk::Dispatch &vk)
{
    if (!vk.library)
        return;
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(vk.library));
#else
    dlclose(vk.library);
#endif
    vk = {};
}
} // namespace

GSVulkanBackend::GSVulkanBackend()
{
    static_assert(sizeof(TransferPush) == 80u);
    static_assert(sizeof(GpuSpriteDraw) == 112u);
    static_assert(sizeof(GpuTriangleDraw) == 240u);
    static_assert(offsetof(GpuTriangleDraw, z0) == 128u);
    static_assert(offsetof(GpuTriangleDraw, rgba) == 160u);
    static_assert(sizeof(GpuTileHeader) == 16u);
    static_assert(sizeof(PresentPush) == 72u);
    m_cmds.reserve(512u);
    m_batchDraws.reserve(4096u);
    m_deferredUploads.reserve(16u);
    m_tileCounts.assign(kTileGridSize, 0u);
    m_tileCursor.assign(kTileGridSize, 0u);
}

GSVulkanBackend::~GSVulkanBackend()
{
    destroyVulkan();
}

void GSVulkanBackend::SetHostInstanceExtensions(std::vector<std::string> extensions)
{
    hostInstanceExtensions() = std::move(extensions);
}

std::unique_ptr<GSVulkanBackend> GSVulkanBackend::CreateIfAvailable(std::string *outReason)
{
    auto backend = std::make_unique<GSVulkanBackend>();
    if (!backend->initVulkan(outReason))
        return nullptr;
    return backend;
}

void GSVulkanBackend::setPageRange(PageMask &mask, uint32_t firstPage, uint32_t lastPage)
{
    if (lastPage < firstPage)
        return;
    const uint32_t span = lastPage - firstPage + 1u;
    if (span >= 512u)
    {
        mask.fill(~0ull);
        return;
    }
    for (uint32_t p = firstPage; p <= lastPage; ++p)
    {
        const uint32_t wrapped = p & 511u;
        mask[wrapped >> 6u] |= (1ull << (wrapped & 63u));
    }
}

bool GSVulkanBackend::masksOverlap(const PageMask &a, const PageMask &b)
{
    for (size_t i = 0u; i < a.size(); ++i)
        if ((a[i] & b[i]) != 0ull)
            return true;
    return false;
}

bool GSVulkanBackend::maskOverlapsRange(const PageMask &mask, uint32_t firstPage, uint32_t lastPage)
{
    if (lastPage < firstPage)
        return false;
    const uint32_t span = lastPage - firstPage + 1u;
    if (span >= 512u)
    {
        for (uint64_t w : mask)
            if (w != 0ull)
                return true;
        return false;
    }
    for (uint32_t p = firstPage; p <= lastPage; ++p)
    {
        const uint32_t wrapped = p & 511u;
        if ((mask[wrapped >> 6u] & (1ull << (wrapped & 63u))) != 0ull)
            return true;
    }
    return false;
}

void GSVulkanBackend::mergeMasks(PageMask &dst, const PageMask &src)
{
    for (size_t i = 0u; i < dst.size(); ++i)
        dst[i] |= src[i];
}

bool GSVulkanBackend::initVulkan(std::string *outReason)
{
    if (!loadVulkanLibrary(m_vk, outReason))
        return false;

    ps2vk::VkApplicationInfo app{};
    app.sType = ps2vk::VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "PS2Recomp GS Vulkan";
    app.apiVersion = ps2vk::VK_API_VERSION_1_0;

    std::vector<const char *> instExts;
    for (const std::string &ext : hostInstanceExtensions())
        instExts.push_back(ext.c_str());

    ps2vk::VkInstanceCreateInfo instInfo{};
    instInfo.sType = ps2vk::VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &app;
    instInfo.enabledExtensionCount = static_cast<uint32_t>(instExts.size());
    instInfo.ppEnabledExtensionNames = instExts.empty() ? nullptr : instExts.data();

    bool hostSurfaceExts = !instExts.empty();
    if (m_vk.vkCreateInstance(&instInfo, nullptr, &m_instance) != ps2vk::VK_SUCCESS)
    {
        // The GS still runs without surface extensions; the host then uses its CPU-upload presenter.
        hostSurfaceExts = false;
        instInfo.enabledExtensionCount = 0u;
        instInfo.ppEnabledExtensionNames = nullptr;
        if (m_vk.vkCreateInstance(&instInfo, nullptr, &m_instance) != ps2vk::VK_SUCCESS)
        {
            if (outReason) *outReason = "vkCreateInstance failed";
            destroyVulkan();
            return false;
        }
    }

#define LOAD_INST(fn) \
    m_vk.fn = reinterpret_cast<ps2vk::PFN_##fn>(m_vk.vkGetInstanceProcAddr(m_instance, #fn)); \
    if (!m_vk.fn) { if (outReason) *outReason = "Missing " #fn; destroyVulkan(); return false; }

    LOAD_INST(vkDestroyInstance);
    LOAD_INST(vkEnumeratePhysicalDevices);
    LOAD_INST(vkGetPhysicalDeviceProperties);
    LOAD_INST(vkGetPhysicalDeviceFeatures);
    LOAD_INST(vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD_INST(vkGetPhysicalDeviceMemoryProperties);
    LOAD_INST(vkCreateDevice);
    LOAD_INST(vkGetDeviceProcAddr);
#undef LOAD_INST

    uint32_t devCount = 0u;
    if (m_vk.vkEnumeratePhysicalDevices(m_instance, &devCount, nullptr) != ps2vk::VK_SUCCESS || devCount == 0u)
    {
        if (outReason) *outReason = "No Vulkan physical devices";
        destroyVulkan();
        return false;
    }
    std::vector<ps2vk::VkPhysicalDevice> devices(devCount);
    if (m_vk.vkEnumeratePhysicalDevices(m_instance, &devCount, devices.data()) != ps2vk::VK_SUCCESS)
    {
        if (outReason) *outReason = "vkEnumeratePhysicalDevices failed";
        destroyVulkan();
        return false;
    }

    ps2vk::VkPhysicalDevice selected = nullptr;
    uint32_t selectedFamily = 0u;
    uint32_t selectedQueueCount = 1u;
    ps2vk::VkPhysicalDeviceProperties selectedProps{};
    for (ps2vk::VkPhysicalDevice cand : devices)
    {
        ps2vk::VkPhysicalDeviceProperties props{};
        m_vk.vkGetPhysicalDeviceProperties(cand, &props);
        if (props.deviceType == ps2vk::VK_PHYSICAL_DEVICE_TYPE_CPU)
            continue;
        ps2vk::VkPhysicalDeviceFeatures feats{};
        m_vk.vkGetPhysicalDeviceFeatures(cand, &feats);
        if (!feats.shaderFloat64)
            continue;

        uint32_t famCount = 0u;
        m_vk.vkGetPhysicalDeviceQueueFamilyProperties(cand, &famCount, nullptr);
        std::vector<ps2vk::VkQueueFamilyProperties> fams(famCount);
        m_vk.vkGetPhysicalDeviceQueueFamilyProperties(cand, &famCount, fams.data());
        for (uint32_t i = 0u; i < famCount; ++i)
        {
            if ((fams[i].queueFlags & ps2vk::VK_QUEUE_COMPUTE_BIT) == 0u)
                continue;
            if (!selected ||
                (selectedProps.deviceType != ps2vk::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
                 props.deviceType == ps2vk::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU))
            {
                selected = cand;
                selectedFamily = i;
                selectedQueueCount = fams[i].queueCount;
                selectedProps = props;
            }
            break;
        }
    }

    if (!selected)
    {
        if (outReason) *outReason = "No hardware Vulkan compute device with shaderFloat64";
        destroyVulkan();
        return false;
    }

    m_physicalDevice = selected;
    m_queueFamily = selectedFamily;
    m_deviceName = selectedProps.deviceName;
    m_vk.vkGetPhysicalDeviceMemoryProperties(selected, &m_memProps);

    // A second queue in the family lets the host present without taking the GS lock.
    const float prios[2] = {1.0f, 1.0f};
    const uint32_t queueCount = (hostSurfaceExts && selectedQueueCount >= 2u) ? 2u : 1u;
    ps2vk::VkDeviceQueueCreateInfo qInfo{};
    qInfo.sType = ps2vk::VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qInfo.queueFamilyIndex = m_queueFamily;
    qInfo.queueCount = queueCount;
    qInfo.pQueuePriorities = prios;

    ps2vk::VkPhysicalDeviceFeatures enabledFeats{};
    enabledFeats.shaderFloat64 = ps2vk::VK_TRUE;

    ps2vk::VkDeviceCreateInfo devInfo{};
    devInfo.sType = ps2vk::VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount = 1u;
    devInfo.pQueueCreateInfos = &qInfo;
    devInfo.pEnabledFeatures = &enabledFeats;

    const char *const swapchainExt = "VK_KHR_swapchain";
    bool swapchainEnabled = false;
    if (hostSurfaceExts)
    {
        devInfo.enabledExtensionCount = 1u;
        devInfo.ppEnabledExtensionNames = &swapchainExt;
        swapchainEnabled = m_vk.vkCreateDevice(selected, &devInfo, nullptr, &m_device) == ps2vk::VK_SUCCESS;
        devInfo.enabledExtensionCount = 0u;
        devInfo.ppEnabledExtensionNames = nullptr;
    }
    if (!swapchainEnabled && m_vk.vkCreateDevice(selected, &devInfo, nullptr, &m_device) != ps2vk::VK_SUCCESS)
    {
        if (outReason) *outReason = "vkCreateDevice failed";
        destroyVulkan();
        return false;
    }

#define LOAD_DEV(fn) \
    m_vk.fn = reinterpret_cast<ps2vk::PFN_##fn>(m_vk.vkGetDeviceProcAddr(m_device, #fn)); \
    if (!m_vk.fn) { if (outReason) *outReason = "Missing device " #fn; destroyVulkan(); return false; }

    LOAD_DEV(vkDestroyDevice);
    LOAD_DEV(vkGetDeviceQueue);
    LOAD_DEV(vkDeviceWaitIdle);
    LOAD_DEV(vkQueueSubmit);
    LOAD_DEV(vkCreateBuffer);
    LOAD_DEV(vkDestroyBuffer);
    LOAD_DEV(vkGetBufferMemoryRequirements);
    LOAD_DEV(vkAllocateMemory);
    LOAD_DEV(vkFreeMemory);
    LOAD_DEV(vkBindBufferMemory);
    LOAD_DEV(vkMapMemory);
    LOAD_DEV(vkUnmapMemory);
    LOAD_DEV(vkCreateDescriptorSetLayout);
    LOAD_DEV(vkDestroyDescriptorSetLayout);
    LOAD_DEV(vkCreatePipelineLayout);
    LOAD_DEV(vkDestroyPipelineLayout);
    LOAD_DEV(vkCreateShaderModule);
    LOAD_DEV(vkDestroyShaderModule);
    LOAD_DEV(vkCreateComputePipelines);
    LOAD_DEV(vkDestroyPipeline);
    LOAD_DEV(vkCreateDescriptorPool);
    LOAD_DEV(vkDestroyDescriptorPool);
    LOAD_DEV(vkAllocateDescriptorSets);
    LOAD_DEV(vkUpdateDescriptorSets);
    LOAD_DEV(vkCreateCommandPool);
    LOAD_DEV(vkDestroyCommandPool);
    LOAD_DEV(vkResetCommandPool);
    LOAD_DEV(vkAllocateCommandBuffers);
    LOAD_DEV(vkBeginCommandBuffer);
    LOAD_DEV(vkEndCommandBuffer);
    LOAD_DEV(vkCmdBindPipeline);
    LOAD_DEV(vkCmdBindDescriptorSets);
    LOAD_DEV(vkCmdPushConstants);
    LOAD_DEV(vkCmdDispatch);
    LOAD_DEV(vkCmdPipelineBarrier);
    LOAD_DEV(vkCmdCopyBuffer);
    LOAD_DEV(vkCreateFence);
    LOAD_DEV(vkDestroyFence);
    LOAD_DEV(vkWaitForFences);
    LOAD_DEV(vkResetFences);
#undef LOAD_DEV

    m_vk.vkGetDeviceQueue(m_device, m_queueFamily, 0u, &m_queue);

    m_hostContext.instance = m_instance;
    m_hostContext.physicalDevice = m_physicalDevice;
    m_hostContext.device = m_device;
    m_hostContext.queueFamily = m_queueFamily;
    m_hostContext.getInstanceProcAddr = reinterpret_cast<void *>(m_vk.vkGetInstanceProcAddr);
    m_hostContext.swapchainEnabled = swapchainEnabled;
    if (queueCount >= 2u)
    {
        ps2vk::VkQueue hostQueue = nullptr;
        m_vk.vkGetDeviceQueue(m_device, m_queueFamily, 1u, &hostQueue);
        m_hostContext.queue = hostQueue;
    }
    else
    {
        m_hostContext.queue = m_queue;
        m_hostContext.queueLock = &m_mutex;
    }

    constexpr ps2vk::VkFlags kHostVisCoh = ps2vk::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | ps2vk::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    constexpr ps2vk::VkFlags kHostCached = kHostVisCoh | ps2vk::VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    if (!createBuffer(m_vramGpu, kVramBytes,
                      ps2vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | ps2vk::VK_BUFFER_USAGE_TRANSFER_SRC_BIT | ps2vk::VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      ps2vk::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
        !createBuffer(m_clutGpu, kClutBytes,
                      ps2vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | ps2vk::VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      ps2vk::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
        !createBuffer(m_presentOut, kPresentBytes,
                      ps2vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | ps2vk::VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      kHostVisCoh, kHostCached) ||
        !createBuffer(m_vramReadback, kVramBytes,
                      ps2vk::VK_BUFFER_USAGE_TRANSFER_DST_BIT, kHostVisCoh, kHostCached))
    {
        if (outReason) *outReason = "Vulkan buffer allocation failed";
        destroyVulkan();
        return false;
    }
    FrameSlot *const slots[2] = {&m_cur, &m_spare};
    for (FrameSlot *slot : slots)
    {
        if (!createBuffer(slot->uploadStaging, kStagingBytes,
                          ps2vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | ps2vk::VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          kHostVisCoh, kHostCached) ||
            !createBuffer(slot->spriteBuf, static_cast<ps2vk::VkDeviceSize>(kMaxSprites) * sizeof(GpuSpriteDraw),
                          ps2vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisCoh, kHostCached) ||
            !createBuffer(slot->triangleBuf, static_cast<ps2vk::VkDeviceSize>(kMaxTriangles) * sizeof(GpuTriangleDraw),
                          ps2vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisCoh, kHostCached) ||
            !createBuffer(slot->tileHeaderBuf, static_cast<ps2vk::VkDeviceSize>(kMaxTileHeaders) * sizeof(GpuTileHeader),
                          ps2vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisCoh, kHostCached) ||
            !createBuffer(slot->tileIndexBuf, static_cast<ps2vk::VkDeviceSize>(kMaxTileIndices) * sizeof(uint32_t),
                          ps2vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisCoh, kHostCached))
        {
            if (outReason) *outReason = "Vulkan buffer allocation failed";
            destroyVulkan();
            return false;
        }
    }

    ps2vk::VkDescriptorSetLayoutBinding bindings7[7]{};
    for (uint32_t i = 0u; i < 7u; ++i)
        bindings7[i] = {i, ps2vk::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, ps2vk::VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    ps2vk::VkDescriptorSetLayoutCreateInfo dsl3{};
    dsl3.sType = ps2vk::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl3.bindingCount = 3u;
    dsl3.pBindings = bindings7;
    ps2vk::VkDescriptorSetLayoutCreateInfo dsl7{};
    dsl7.sType = ps2vk::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl7.bindingCount = 7u;
    dsl7.pBindings = bindings7;

    if (m_vk.vkCreateDescriptorSetLayout(m_device, &dsl3, nullptr, &m_descLayout3) != ps2vk::VK_SUCCESS ||
        m_vk.vkCreateDescriptorSetLayout(m_device, &dsl7, nullptr, &m_descLayout7) != ps2vk::VK_SUCCESS)
    {
        if (outReason) *outReason = "vkCreateDescriptorSetLayout failed";
        destroyVulkan();
        return false;
    }

    if (!createComputePipeline(ps2vk_shaders::kGsUploadSpv, std::size(ps2vk_shaders::kGsUploadSpv), sizeof(TransferPush), m_descLayout3, m_uploadLayout, m_uploadPipeline) ||
        !createComputePipeline(ps2vk_shaders::kGsTileRasterSpv, std::size(ps2vk_shaders::kGsTileRasterSpv), sizeof(uint32_t), m_descLayout7, m_tileLayout, m_tilePipeline) ||
        !createComputePipeline(ps2vk_shaders::kGsPresentSpv, std::size(ps2vk_shaders::kGsPresentSpv), sizeof(PresentPush), m_descLayout3, m_presentLayout, m_presentPipeline))
    {
        if (outReason) *outReason = "Vulkan compute pipeline creation failed";
        destroyVulkan();
        return false;
    }

    // Per slot: upload set (3) + tile set (7); plus one shared present set (3).
    const ps2vk::VkDescriptorPoolSize poolSize{ps2vk::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2u * (3u + 7u) + 3u};
    ps2vk::VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = ps2vk::VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 5u;
    poolInfo.poolSizeCount = 1u;
    poolInfo.pPoolSizes = &poolSize;
    if (m_vk.vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descPool) != ps2vk::VK_SUCCESS)
    {
        if (outReason) *outReason = "vkCreateDescriptorPool failed";
        destroyVulkan();
        return false;
    }

    const ps2vk::VkDescriptorSetLayout layouts[5] = {m_descLayout3, m_descLayout7, m_descLayout3, m_descLayout7, m_descLayout3};
    ps2vk::VkDescriptorSet sets[5]{};
    ps2vk::VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = ps2vk::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descPool;
    allocInfo.descriptorSetCount = 5u;
    allocInfo.pSetLayouts = layouts;
    if (m_vk.vkAllocateDescriptorSets(m_device, &allocInfo, sets) != ps2vk::VK_SUCCESS)
    {
        if (outReason) *outReason = "vkAllocateDescriptorSets failed";
        destroyVulkan();
        return false;
    }
    m_cur.uploadDescSet = sets[0];
    m_cur.tileDescSet = sets[1];
    m_spare.uploadDescSet = sets[2];
    m_spare.tileDescSet = sets[3];
    m_presentDescSet = sets[4];

    auto bindSet = [&](ps2vk::VkDescriptorSet ds, const ps2vk::VkDescriptorBufferInfo *infos, uint32_t count)
    {
        ps2vk::VkWriteDescriptorSet writes[7]{};
        for (uint32_t i = 0u; i < count; ++i)
        {
            writes[i].sType = ps2vk::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = ds;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1u;
            writes[i].descriptorType = ps2vk::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        m_vk.vkUpdateDescriptorSets(m_device, count, writes, 0u, nullptr);
    };

    for (const FrameSlot *slot : slots)
    {
        const ps2vk::VkDescriptorBufferInfo uploadInfos[3] = {
            {m_vramGpu.buffer, 0u, m_vramGpu.size},
            {m_clutGpu.buffer, 0u, m_clutGpu.size},
            {slot->uploadStaging.buffer, 0u, slot->uploadStaging.size},
        };
        const ps2vk::VkDescriptorBufferInfo tileInfos[7] = {
            {m_vramGpu.buffer, 0u, m_vramGpu.size},
            {m_clutGpu.buffer, 0u, m_clutGpu.size},
            {slot->uploadStaging.buffer, 0u, slot->uploadStaging.size},
            {slot->spriteBuf.buffer, 0u, slot->spriteBuf.size},
            {slot->triangleBuf.buffer, 0u, slot->triangleBuf.size},
            {slot->tileHeaderBuf.buffer, 0u, slot->tileHeaderBuf.size},
            {slot->tileIndexBuf.buffer, 0u, slot->tileIndexBuf.size},
        };
        bindSet(slot->uploadDescSet, uploadInfos, 3u);
        bindSet(slot->tileDescSet, tileInfos, 7u);
    }
    const ps2vk::VkDescriptorBufferInfo presentInfos[3] = {
        {m_vramGpu.buffer, 0u, m_vramGpu.size},
        {m_clutGpu.buffer, 0u, m_clutGpu.size},
        {m_presentOut.buffer, 0u, m_presentOut.size},
    };
    bindSet(m_presentDescSet, presentInfos, 3u);

    // RESET_COMMAND_BUFFER_BIT: vkBeginCommandBuffer resets each buffer on its own, so one slot can be
    // re-recorded while the other is still executing (vkResetCommandPool would reset both).
    ps2vk::VkCommandPoolCreateInfo cpInfo{};
    cpInfo.sType = ps2vk::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpInfo.flags = ps2vk::VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpInfo.queueFamilyIndex = m_queueFamily;
    if (m_vk.vkCreateCommandPool(m_device, &cpInfo, nullptr, &m_cmdPool) != ps2vk::VK_SUCCESS)
    {
        if (outReason) *outReason = "vkCreateCommandPool failed";
        destroyVulkan();
        return false;
    }

    ps2vk::VkCommandBuffer cmdBufs[3]{};
    ps2vk::VkCommandBufferAllocateInfo cbInfo{};
    cbInfo.sType = ps2vk::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbInfo.commandPool = m_cmdPool;
    cbInfo.level = ps2vk::VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = 3u;
    if (m_vk.vkAllocateCommandBuffers(m_device, &cbInfo, cmdBufs) != ps2vk::VK_SUCCESS)
    {
        if (outReason) *outReason = "vkAllocateCommandBuffers failed";
        destroyVulkan();
        return false;
    }
    m_cur.cmdBuf = cmdBufs[0];
    m_spare.cmdBuf = cmdBufs[1];
    m_presentCmdBuf = cmdBufs[2];

    ps2vk::VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = ps2vk::VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (m_vk.vkCreateFence(m_device, &fenceInfo, nullptr, &m_cur.fence) != ps2vk::VK_SUCCESS ||
        m_vk.vkCreateFence(m_device, &fenceInfo, nullptr, &m_spare.fence) != ps2vk::VK_SUCCESS ||
        m_vk.vkCreateFence(m_device, &fenceInfo, nullptr, &m_presentFence) != ps2vk::VK_SUCCESS)
    {
        if (outReason) *outReason = "vkCreateFence failed";
        destroyVulkan();
        return false;
    }

    return true;
}

void GSVulkanBackend::destroyVulkan()
{
    FrameSlot *const slots[2] = {&m_cur, &m_spare};
    if (m_device)
    {
        if (m_vk.vkDeviceWaitIdle)
            m_vk.vkDeviceWaitIdle(m_device);
        for (FrameSlot *slot : slots)
            if (slot->fence) m_vk.vkDestroyFence(m_device, slot->fence, nullptr);
        if (m_presentFence) m_vk.vkDestroyFence(m_device, m_presentFence, nullptr);
        if (m_cmdPool) m_vk.vkDestroyCommandPool(m_device, m_cmdPool, nullptr);
        if (m_descPool) m_vk.vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
        if (m_presentPipeline) m_vk.vkDestroyPipeline(m_device, m_presentPipeline, nullptr);
        if (m_presentLayout) m_vk.vkDestroyPipelineLayout(m_device, m_presentLayout, nullptr);
        if (m_tilePipeline) m_vk.vkDestroyPipeline(m_device, m_tilePipeline, nullptr);
        if (m_tileLayout) m_vk.vkDestroyPipelineLayout(m_device, m_tileLayout, nullptr);
        if (m_uploadPipeline) m_vk.vkDestroyPipeline(m_device, m_uploadPipeline, nullptr);
        if (m_uploadLayout) m_vk.vkDestroyPipelineLayout(m_device, m_uploadLayout, nullptr);
        if (m_descLayout7) m_vk.vkDestroyDescriptorSetLayout(m_device, m_descLayout7, nullptr);
        if (m_descLayout3) m_vk.vkDestroyDescriptorSetLayout(m_device, m_descLayout3, nullptr);
        destroyBuffer(m_vramReadback);
        destroyBuffer(m_presentOut);
        for (FrameSlot *slot : slots)
        {
            destroyBuffer(slot->tileIndexBuf);
            destroyBuffer(slot->tileHeaderBuf);
            destroyBuffer(slot->triangleBuf);
            destroyBuffer(slot->spriteBuf);
            destroyBuffer(slot->uploadStaging);
        }
        destroyBuffer(m_clutGpu);
        destroyBuffer(m_vramGpu);
        m_vk.vkDestroyDevice(m_device, nullptr);
    }
    if (m_instance && m_vk.vkDestroyInstance)
        m_vk.vkDestroyInstance(m_instance, nullptr);

    m_cur = {};
    m_spare = {};
    m_presentFence = nullptr;
    m_presentCmdBuf = nullptr;
    m_cmdPool = nullptr;
    m_descPool = nullptr;
    m_presentPipeline = nullptr;
    m_presentLayout = nullptr;
    m_tilePipeline = nullptr;
    m_tileLayout = nullptr;
    m_uploadPipeline = nullptr;
    m_uploadLayout = nullptr;
    m_descLayout7 = nullptr;
    m_descLayout3 = nullptr;
    m_device = nullptr;
    m_instance = nullptr;
    m_physicalDevice = nullptr;
    m_queue = nullptr;
    m_gpuInFlight = false;
    unloadVulkanLibrary(m_vk);
}

bool GSVulkanBackend::createBuffer(GpuBuffer &buf, ps2vk::VkDeviceSize size, ps2vk::VkFlags usage,
                                   ps2vk::VkFlags requiredProps, ps2vk::VkFlags preferredProps)
{
    ps2vk::VkBufferCreateInfo info{};
    info.sType = ps2vk::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = ps2vk::VK_SHARING_MODE_EXCLUSIVE;
    if (m_vk.vkCreateBuffer(m_device, &info, nullptr, &buf.buffer) != ps2vk::VK_SUCCESS)
        return false;

    ps2vk::VkMemoryRequirements req{};
    m_vk.vkGetBufferMemoryRequirements(m_device, buf.buffer, &req);

    auto findType = [&](ps2vk::VkFlags props) -> uint32_t
    {
        for (uint32_t i = 0u; i < m_memProps.memoryTypeCount; ++i)
        {
            if ((req.memoryTypeBits & (1u << i)) != 0u &&
                (m_memProps.memoryTypes[i].propertyFlags & props) == props)
                return i;
        }
        return UINT32_MAX;
    };

    uint32_t memType = preferredProps ? findType(preferredProps) : UINT32_MAX;
    if (memType == UINT32_MAX)
        memType = findType(requiredProps);
    if (memType == UINT32_MAX)
    {
        m_vk.vkDestroyBuffer(m_device, buf.buffer, nullptr);
        buf.buffer = nullptr;
        return false;
    }

    ps2vk::VkMemoryAllocateInfo alloc{};
    alloc.sType = ps2vk::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = memType;
    if (m_vk.vkAllocateMemory(m_device, &alloc, nullptr, &buf.memory) != ps2vk::VK_SUCCESS)
    {
        m_vk.vkDestroyBuffer(m_device, buf.buffer, nullptr);
        buf.buffer = nullptr;
        return false;
    }
    if (m_vk.vkBindBufferMemory(m_device, buf.buffer, buf.memory, 0u) != ps2vk::VK_SUCCESS)
    {
        destroyBuffer(buf);
        return false;
    }
    if ((requiredProps & ps2vk::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0u)
    {
        if (m_vk.vkMapMemory(m_device, buf.memory, 0u, size, 0u, &buf.mapped) != ps2vk::VK_SUCCESS)
        {
            destroyBuffer(buf);
            return false;
        }
    }
    buf.size = size;
    return true;
}

void GSVulkanBackend::destroyBuffer(GpuBuffer &buf)
{
    if (!m_device)
        return;
    if (buf.mapped)
    {
        m_vk.vkUnmapMemory(m_device, buf.memory);
        buf.mapped = nullptr;
    }
    if (buf.buffer)
    {
        m_vk.vkDestroyBuffer(m_device, buf.buffer, nullptr);
        buf.buffer = nullptr;
    }
    if (buf.memory)
    {
        m_vk.vkFreeMemory(m_device, buf.memory, nullptr);
        buf.memory = nullptr;
    }
    buf.size = 0u;
}

bool GSVulkanBackend::createComputePipeline(const uint32_t *spv, size_t wordCount, uint32_t pushBytes,
                                            ps2vk::VkDescriptorSetLayout descLayout,
                                            ps2vk::VkPipelineLayout &outLayout, ps2vk::VkPipeline &outPipeline)
{
    const ps2vk::VkPushConstantRange range{ps2vk::VK_SHADER_STAGE_COMPUTE_BIT, 0u, pushBytes};
    ps2vk::VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = ps2vk::VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1u;
    layoutInfo.pSetLayouts = &descLayout;
    layoutInfo.pushConstantRangeCount = 1u;
    layoutInfo.pPushConstantRanges = &range;
    if (m_vk.vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &outLayout) != ps2vk::VK_SUCCESS)
        return false;

    ps2vk::VkShaderModuleCreateInfo modInfo{};
    modInfo.sType = ps2vk::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    modInfo.codeSize = wordCount * sizeof(uint32_t);
    modInfo.pCode = spv;
    ps2vk::VkShaderModule module = nullptr;
    if (m_vk.vkCreateShaderModule(m_device, &modInfo, nullptr, &module) != ps2vk::VK_SUCCESS)
        return false;

    ps2vk::VkComputePipelineCreateInfo pipeInfo{};
    pipeInfo.sType = ps2vk::VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeInfo.stage.sType = ps2vk::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeInfo.stage.stage = ps2vk::VK_SHADER_STAGE_COMPUTE_BIT;
    pipeInfo.stage.module = module;
    pipeInfo.stage.pName = "main";
    pipeInfo.layout = outLayout;

    const ps2vk::VkResult res = m_vk.vkCreateComputePipelines(m_device, nullptr, 1u, &pipeInfo, nullptr, &outPipeline);
    m_vk.vkDestroyShaderModule(m_device, module, nullptr);
    return res == ps2vk::VK_SUCCESS;
}

// Waits for the previous submission (m_spare). m_cur is never in flight, so this drains the GPU slots.
void GSVulkanBackend::waitForInFlightUnlocked() const
{
    if (!m_gpuInFlight)
        return;
    if (m_vk.vkWaitForFences(m_device, 1u, &m_spare.fence, ps2vk::VK_TRUE, UINT64_MAX) == ps2vk::VK_SUCCESS)
        m_vk.vkResetFences(m_device, 1u, &m_spare.fence);
    m_gpuInFlight = false;
}

// Begins a one-shot command buffer whose first command orders it after everything submitted before it
// (queue submission order + this barrier), since submissions no longer wait for each other on the host.
bool GSVulkanBackend::beginCmdUnlocked(ps2vk::VkCommandBuffer cmd) const
{
    ps2vk::VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = ps2vk::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = ps2vk::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (m_vk.vkBeginCommandBuffer(cmd, &beginInfo) != ps2vk::VK_SUCCESS)
        return false;
    ps2vk::VkMemoryBarrier barrier{};
    barrier.sType = ps2vk::VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = ps2vk::VK_ACCESS_SHADER_WRITE_BIT | ps2vk::VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = ps2vk::VK_ACCESS_SHADER_READ_BIT | ps2vk::VK_ACCESS_SHADER_WRITE_BIT |
                            ps2vk::VK_ACCESS_TRANSFER_READ_BIT | ps2vk::VK_ACCESS_TRANSFER_WRITE_BIT;
    const ps2vk::VkFlags stages = ps2vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | ps2vk::VK_PIPELINE_STAGE_TRANSFER_BIT;
    m_vk.vkCmdPipelineBarrier(cmd, stages, stages, 0u, 1u, &barrier, 0u, nullptr, 0u, nullptr);
    return true;
}

void GSVulkanBackend::submitBufferCopyUnlocked(ps2vk::VkBuffer src, ps2vk::VkBuffer dst, ps2vk::VkDeviceSize size) const
{
    if (!beginCmdUnlocked(m_cur.cmdBuf))
        return;
    const ps2vk::VkBufferCopy copy{0u, 0u, size};
    m_vk.vkCmdCopyBuffer(m_cur.cmdBuf, src, dst, 1u, &copy);
    if (m_vk.vkEndCommandBuffer(m_cur.cmdBuf) != ps2vk::VK_SUCCESS)
        return;
    ps2vk::VkSubmitInfo submitInfo{};
    submitInfo.sType = ps2vk::VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1u;
    submitInfo.pCommandBuffers = &m_cur.cmdBuf;
    if (m_vk.vkQueueSubmit(m_queue, 1u, &submitInfo, m_cur.fence) == ps2vk::VK_SUCCESS)
    {
        // Synchronous: m_cur must be free again before the caller records into it.
        if (m_vk.vkWaitForFences(m_device, 1u, &m_cur.fence, ps2vk::VK_TRUE, UINT64_MAX) == ps2vk::VK_SUCCESS)
            m_vk.vkResetFences(m_device, 1u, &m_cur.fence);
    }
    m_stagingOffset = 0u;
    m_spriteCount = 0u;
    m_triangleCount = 0u;
    m_tileHeaderCount = 0u;
    m_tileIndexCount = 0u;
}

void GSVulkanBackend::emitUploadCmdUnlocked(const TransferPush &tr, uint32_t groupsX, uint32_t firstPage, uint32_t lastPage) const
{
    // Any upload that touches pages written or read by the open tile batch must close the batch first.
    if (!m_batchDraws.empty() &&
        (maskOverlapsRange(m_batchWritePages, firstPage, lastPage) ||
         maskOverlapsRange(m_batchReadPages, firstPage, lastPage)))
    {
        closeTileBatchUnlocked();
    }

    const bool needBarrier = maskOverlapsRange(m_gpuWritePages, firstPage, lastPage) ||
                             maskOverlapsRange(m_gpuReadPages, firstPage, lastPage);
    if (needBarrier)
    {
        m_gpuWritePages.fill(0ull);
        m_gpuReadPages.fill(0ull);
        m_gpuClutDirty = false;
    }

    PendingCmd cmd{};
    cmd.kind = CmdKind::Upload;
    cmd.barrierBefore = needBarrier;
    cmd.groupsX = groupsX;
    cmd.transfer = tr;
    m_cmds.push_back(cmd);
    setPageRange(m_gpuWritePages, firstPage, lastPage);
}

void GSVulkanBackend::materializeDeferredUploadsUnlocked(uint32_t firstPage, uint32_t lastPage, bool all) const
{
    if (m_deferredUploads.empty())
        return;
    size_t writePos = 0u;
    for (size_t i = 0u; i < m_deferredUploads.size(); ++i)
    {
        const DeferredUpload u = m_deferredUploads[i];
        const bool overlaps = all || ((u.firstPage <= lastPage) && (firstPage <= u.lastPage));
        if (overlaps)
        {
            TransferPush tr{};
            tr.mode = 0u;
            tr.dbp = u.dbp;
            tr.dbw = std::max<uint32_t>(u.dbw, 1u);
            tr.dpsm = u.dpsm;
            tr.dsax = u.dsax;
            tr.dsay = u.dsay;
            tr.rrw = u.rrw;
            tr.rrh = u.rrh;
            tr.startPixel = 0u;
            tr.pixelCount = u.totalPixels;
            tr.stagingByteOffset = u.stagingByteOffset;
            emitUploadCmdUnlocked(tr, (u.totalPixels + 63u) / 64u, u.firstPage, u.lastPage);
        }
        else
        {
            m_deferredUploads[writePos++] = u;
        }
    }
    m_deferredUploads.resize(writePos);
}

void GSVulkanBackend::flushPartialContiguousTransferUnlocked() const
{
    if (!m_activeTransferContiguous)
        return;
    m_activeTransferContiguous = false;
    if (m_activeTransferStagingBytes == 0u || m_transferState.copiedPixels == 0u)
        return;
    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    if (rrw == 0u || rrh == 0u)
        return;
    const uint32_t dsax = m_transfer.trxpos.dsax;
    const uint32_t dsay = m_transfer.trxpos.dsay;
    const uint32_t y1 = dsay + ((m_transferState.copiedPixels - 1u) / rrw);
    uint32_t firstPage = 0u, lastPage = 0u;
    computePsmPageRange(m_transfer.bitbltbuf.dpsm, m_transfer.bitbltbuf.dbp, m_transfer.bitbltbuf.dbw,
                        dsax, dsay, dsax + rrw - 1u, y1, firstPage, lastPage);
    materializeDeferredUploadsUnlocked(firstPage, lastPage, false);
    TransferPush tr{};
    tr.mode = 0u;
    tr.dbp = m_transfer.bitbltbuf.dbp;
    tr.dbw = std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u);
    tr.dpsm = m_transfer.bitbltbuf.dpsm;
    tr.dsax = dsax;
    tr.dsay = dsay;
    tr.rrw = rrw;
    tr.rrh = rrh;
    tr.startPixel = 0u;
    tr.pixelCount = m_transferState.copiedPixels;
    tr.stagingByteOffset = m_activeTransferStagingOff;
    emitUploadCmdUnlocked(tr, (m_transferState.copiedPixels + 63u) / 64u, firstPage, lastPage);
}

void GSVulkanBackend::closeTileBatchUnlocked() const
{
    if (m_batchDraws.empty())
        return;

    // 2-pass counting sort of m_batchDraws into 8x8 screen tiles over the batch's tight tile bbox.
    uint32_t activeTileCount = 0u;
    uint32_t minTx = kTileGridW, minTy = kTileGridH, maxTx = 0u, maxTy = 0u;
    for (const BinnedDraw &d : m_batchDraws)
    {
        if (d.tx0 < minTx) minTx = d.tx0;
        if (d.ty0 < minTy) minTy = d.ty0;
        if (d.tx1 > maxTx) maxTx = d.tx1;
        if (d.ty1 > maxTy) maxTy = d.ty1;
        for (uint32_t ty = d.ty0; ty <= d.ty1; ++ty)
        {
            const uint32_t rowBase = ty * kTileGridW;
            for (uint32_t tx = d.tx0; tx <= d.tx1; ++tx)
            {
                if (m_tileCounts[rowBase + tx]++ == 0u)
                    ++activeTileCount;
            }
        }
    }

    if (activeTileCount == 0u)
    {
        m_batchDraws.clear();
        m_batchWritePages.fill(0ull);
        m_batchReadPages.fill(0ull);
        m_batchFrameKey = 0xFFFFFFFFu;
        m_batchReadsClut = false;
        m_batchEstimatedIndices = 0u;
        return;
    }

    const uint32_t firstTile = m_tileHeaderCount;
    GpuTileHeader *headers = static_cast<GpuTileHeader *>(m_cur.tileHeaderBuf.mapped);
    uint32_t *indices = static_cast<uint32_t *>(m_cur.tileIndexBuf.mapped);
    uint32_t curIndexOff = m_tileIndexCount;

    for (uint32_t ty = minTy; ty <= maxTy; ++ty)
    {
        const uint32_t rowBase = ty * kTileGridW;
        for (uint32_t tx = minTx; tx <= maxTx; ++tx)
        {
            const uint32_t c = m_tileCounts[rowBase + tx];
            if (c == 0u)
                continue;
            m_tileCounts[rowBase + tx] = 0u;
            headers[m_tileHeaderCount++] = {tx, ty, curIndexOff, c};
            m_tileCursor[rowBase + tx] = curIndexOff;
            curIndexOff += c;
        }
    }
    m_tileIndexCount = curIndexOff;

    for (const BinnedDraw &d : m_batchDraws)
    {
        for (uint32_t ty = d.ty0; ty <= d.ty1; ++ty)
        {
            const uint32_t rowBase = ty * kTileGridW;
            for (uint32_t tx = d.tx0; tx <= d.tx1; ++tx)
                indices[m_tileCursor[rowBase + tx]++] = d.encodedIdx;
        }
    }

    const bool needBarrier =
        (m_batchReadsClut && m_gpuClutDirty) ||
        masksOverlap(m_batchReadPages, m_gpuWritePages) ||
        masksOverlap(m_batchWritePages, m_gpuWritePages) ||
        masksOverlap(m_batchWritePages, m_gpuReadPages);
    if (needBarrier)
    {
        m_gpuWritePages.fill(0ull);
        m_gpuReadPages.fill(0ull);
        m_gpuClutDirty = false;
    }

    PendingCmd cmd{};
    cmd.kind = CmdKind::TileBatch;
    cmd.barrierBefore = needBarrier;
    cmd.firstTile = firstTile;
    cmd.tileCount = activeTileCount;
    m_cmds.push_back(cmd);

    mergeMasks(m_gpuWritePages, m_batchWritePages);
    mergeMasks(m_gpuReadPages, m_batchReadPages);

    m_batchDraws.clear();
    m_batchWritePages.fill(0ull);
    m_batchReadPages.fill(0ull);
    m_batchFrameKey = 0xFFFFFFFFu;
    m_batchReadsClut = false;
    m_batchEstimatedIndices = 0u;
}

// Records and submits the present dispatch on its own command buffer and fence. Does not wait: the
// caller waits on m_presentFence after releasing m_mutex. Caller holds m_presentMutex and m_mutex.
bool GSVulkanBackend::submitPresentUnlocked(const PresentPush &presentPush, uint32_t presentHeight) const
{
    const uint32_t clampedH = std::clamp(presentHeight, 1u, kHostFrameHeight);
    if (!beginCmdUnlocked(m_presentCmdBuf))
        return false;

    m_vk.vkCmdBindPipeline(m_presentCmdBuf, ps2vk::VK_PIPELINE_BIND_POINT_COMPUTE, m_presentPipeline);
    m_vk.vkCmdBindDescriptorSets(m_presentCmdBuf, ps2vk::VK_PIPELINE_BIND_POINT_COMPUTE, m_presentLayout, 0u, 1u, &m_presentDescSet, 0u, nullptr);
    m_vk.vkCmdPushConstants(m_presentCmdBuf, m_presentLayout, ps2vk::VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(PresentPush), &presentPush);
    m_vk.vkCmdDispatch(m_presentCmdBuf, (kHostFrameWidth + 15u) / 16u, (clampedH + 7u) / 8u, 1u);
    // Extension point: post-processing passes and an output image sized apart from VRAM/kHostFrame* go here, on this present-only command buffer.

    ps2vk::VkBufferMemoryBarrier hostBarrier{};
    hostBarrier.sType = ps2vk::VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    hostBarrier.srcAccessMask = ps2vk::VK_ACCESS_SHADER_WRITE_BIT;
    hostBarrier.dstAccessMask = ps2vk::VK_ACCESS_HOST_READ_BIT;
    hostBarrier.srcQueueFamilyIndex = ps2vk::VK_QUEUE_FAMILY_IGNORED;
    hostBarrier.dstQueueFamilyIndex = ps2vk::VK_QUEUE_FAMILY_IGNORED;
    hostBarrier.buffer = m_presentOut.buffer;
    hostBarrier.offset = 0u;
    hostBarrier.size = static_cast<ps2vk::VkDeviceSize>(clampedH) * kHostFrameWidth * 4u;
    m_vk.vkCmdPipelineBarrier(m_presentCmdBuf,
                              ps2vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              ps2vk::VK_PIPELINE_STAGE_HOST_BIT,
                              0u, 0u, nullptr, 1u, &hostBarrier, 0u, nullptr);

    if (m_vk.vkEndCommandBuffer(m_presentCmdBuf) != ps2vk::VK_SUCCESS)
        return false;

    ps2vk::VkSubmitInfo submitInfo{};
    submitInfo.sType = ps2vk::VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1u;
    submitInfo.pCommandBuffers = &m_presentCmdBuf;
    return m_vk.vkQueueSubmit(m_queue, 1u, &submitInfo, m_presentFence) == ps2vk::VK_SUCCESS;
}

void GSVulkanBackend::submitPendingUnlocked(bool copyVramToReadback, bool waitForCompletion) const
{
    // Flush any partial contiguous transfer, deferred scratch uploads, and open tile batch before submitting.
    flushPartialContiguousTransferUnlocked();
    materializeDeferredUploadsUnlocked(0u, 511u, true);
    closeTileBatchUnlocked();

    if (m_cmds.empty() && !copyVramToReadback && !m_cpuVramDirtyForGpu)
    {
        if (waitForCompletion)
            waitForInFlightUnlocked();
        return;
    }

    const ps2vk::VkCommandBuffer cb = m_cur.cmdBuf;
    if (!beginCmdUnlocked(cb))
        return;

    auto emitComputeBarrier = [&]()
    {
        ps2vk::VkMemoryBarrier barrier{};
        barrier.sType = ps2vk::VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = ps2vk::VK_ACCESS_SHADER_WRITE_BIT | ps2vk::VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = ps2vk::VK_ACCESS_SHADER_READ_BIT | ps2vk::VK_ACCESS_SHADER_WRITE_BIT;
        m_vk.vkCmdPipelineBarrier(cb,
                                  ps2vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | ps2vk::VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  ps2vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0u, 1u, &barrier, 0u, nullptr, 0u, nullptr);
    };

    if (m_cpuVramDirtyForGpu && m_vram)
    {
        const uint32_t copyOffset = (m_stagingOffset + kVramBytes <= kStagingBytes) ? m_stagingOffset : 0u;
        std::memcpy(static_cast<uint8_t *>(m_cur.uploadStaging.mapped) + copyOffset, m_vram, kVramBytes);
        const ps2vk::VkBufferCopy copy{copyOffset, 0u, kVramBytes};
        m_vk.vkCmdCopyBuffer(cb, m_cur.uploadStaging.buffer, m_vramGpu.buffer, 1u, &copy);
        emitComputeBarrier();
        m_cpuVramDirtyForGpu = false;
    }

    ps2vk::VkPipeline boundPipeline = nullptr;
    for (const PendingCmd &cmd : m_cmds)
    {
        if (cmd.barrierBefore)
            emitComputeBarrier();

        if (cmd.kind == CmdKind::Upload)
        {
            if (boundPipeline != m_uploadPipeline)
            {
                m_vk.vkCmdBindPipeline(cb, ps2vk::VK_PIPELINE_BIND_POINT_COMPUTE, m_uploadPipeline);
                m_vk.vkCmdBindDescriptorSets(cb, ps2vk::VK_PIPELINE_BIND_POINT_COMPUTE, m_uploadLayout, 0u, 1u, &m_cur.uploadDescSet, 0u, nullptr);
                boundPipeline = m_uploadPipeline;
            }
            m_vk.vkCmdPushConstants(cb, m_uploadLayout, ps2vk::VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(TransferPush), &cmd.transfer);
            m_vk.vkCmdDispatch(cb, cmd.groupsX, 1u, 1u);
        }
        else if (cmd.kind == CmdKind::TileBatch)
        {
            if (boundPipeline != m_tilePipeline)
            {
                m_vk.vkCmdBindPipeline(cb, ps2vk::VK_PIPELINE_BIND_POINT_COMPUTE, m_tilePipeline);
                m_vk.vkCmdBindDescriptorSets(cb, ps2vk::VK_PIPELINE_BIND_POINT_COMPUTE, m_tileLayout, 0u, 1u, &m_cur.tileDescSet, 0u, nullptr);
                boundPipeline = m_tilePipeline;
            }
            m_vk.vkCmdPushConstants(cb, m_tileLayout, ps2vk::VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(uint32_t), &cmd.firstTile);
            m_vk.vkCmdDispatch(cb, cmd.tileCount, 1u, 1u);
        }
    }

    if (!m_cmds.empty())
        m_gpuVramDirtyForCpu = true;

    if (copyVramToReadback)
    {
        ps2vk::VkBufferMemoryBarrier vramToTransfer{};
        vramToTransfer.sType = ps2vk::VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        vramToTransfer.srcAccessMask = ps2vk::VK_ACCESS_SHADER_WRITE_BIT | ps2vk::VK_ACCESS_TRANSFER_WRITE_BIT;
        vramToTransfer.dstAccessMask = ps2vk::VK_ACCESS_TRANSFER_READ_BIT;
        vramToTransfer.srcQueueFamilyIndex = ps2vk::VK_QUEUE_FAMILY_IGNORED;
        vramToTransfer.dstQueueFamilyIndex = ps2vk::VK_QUEUE_FAMILY_IGNORED;
        vramToTransfer.buffer = m_vramGpu.buffer;
        vramToTransfer.offset = 0u;
        vramToTransfer.size = kVramBytes;
        m_vk.vkCmdPipelineBarrier(cb,
                                  ps2vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | ps2vk::VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  ps2vk::VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0u, 0u, nullptr, 1u, &vramToTransfer, 0u, nullptr);

        const ps2vk::VkBufferCopy copy{0u, 0u, kVramBytes};
        m_vk.vkCmdCopyBuffer(cb, m_vramGpu.buffer, m_vramReadback.buffer, 1u, &copy);

        ps2vk::VkBufferMemoryBarrier readbackHost{};
        readbackHost.sType = ps2vk::VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        readbackHost.srcAccessMask = ps2vk::VK_ACCESS_TRANSFER_WRITE_BIT;
        readbackHost.dstAccessMask = ps2vk::VK_ACCESS_HOST_READ_BIT;
        readbackHost.srcQueueFamilyIndex = ps2vk::VK_QUEUE_FAMILY_IGNORED;
        readbackHost.dstQueueFamilyIndex = ps2vk::VK_QUEUE_FAMILY_IGNORED;
        readbackHost.buffer = m_vramReadback.buffer;
        readbackHost.offset = 0u;
        readbackHost.size = kVramBytes;
        m_vk.vkCmdPipelineBarrier(cb,
                                  ps2vk::VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  ps2vk::VK_PIPELINE_STAGE_HOST_BIT,
                                  0u, 0u, nullptr, 1u, &readbackHost, 0u, nullptr);
    }

    if (m_vk.vkEndCommandBuffer(cb) != ps2vk::VK_SUCCESS)
        return;

    ps2vk::VkSubmitInfo submitInfo{};
    submitInfo.sType = ps2vk::VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1u;
    submitInfo.pCommandBuffers = &m_cur.cmdBuf;
    const ps2vk::VkResult submitRes = m_vk.vkQueueSubmit(m_queue, 1u, &submitInfo, m_cur.fence);
    m_cmds.clear();
    m_gpuWritePages.fill(0ull);
    m_gpuReadPages.fill(0ull);
    m_gpuClutDirty = false;
    if (submitRes == ps2vk::VK_SUCCESS)
    {
        // Double buffering: retire the submission before this one, then record into its slot
        // while the GPU runs the one just submitted.
        waitForInFlightUnlocked();
        std::swap(m_cur, m_spare);
        m_gpuInFlight = true;
        if (waitForCompletion)
            waitForInFlightUnlocked();
    }
    m_stagingOffset = 0u;
    m_spriteCount = 0u;
    m_triangleCount = 0u;
    m_tileHeaderCount = 0u;
    m_tileIndexCount = 0u;
}

void GSVulkanBackend::syncGpuToCpuUnlocked() const
{
    if (!m_vram)
        return;
    if (!m_gpuVramDirtyForCpu && m_cmds.empty() && m_batchDraws.empty() && m_deferredUploads.empty() &&
        !m_activeTransferContiguous && !m_gpuInFlight)
        return;
    submitPendingUnlocked(true, true);
    std::memcpy(m_vram, m_vramReadback.mapped, kVramBytes);
    m_gpuVramDirtyForCpu = false;
    m_cpuFallback.TextureFlush();
}

void GSVulkanBackend::syncCpuToGpuUnlocked()
{
    if (!m_vram || !m_cpuVramDirtyForGpu)
        return;
    if (!m_cmds.empty() || !m_batchDraws.empty() || !m_deferredUploads.empty() || m_activeTransferContiguous)
        submitPendingUnlocked(false, true);
    std::memcpy(m_cur.uploadStaging.mapped, m_vram, kVramBytes);
    submitBufferCopyUnlocked(m_cur.uploadStaging.buffer, m_vramGpu.buffer, kVramBytes);
    m_cpuVramDirtyForGpu = false;
    m_gpuWritePages.fill(0ull);
    m_gpuReadPages.fill(0ull);
    m_gpuClutDirty = false;
}

void GSVulkanBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    if (vram && vramSize < kVramBytes)
        throw std::invalid_argument("GS Vulkan backend requires at least 4 MiB of VRAM");

    std::lock_guard<std::mutex> lock(m_mutex);
    waitForInFlightUnlocked();
    m_vram = vram;
    m_vramSize = vramSize;
    m_cpuFallback.Initialize(vram, vramSize);
    m_cmds.clear();
    m_batchDraws.clear();
    m_deferredUploads.clear();
    m_stagingOffset = 0u;
    m_spriteCount = 0u;
    m_triangleCount = 0u;
    m_tileHeaderCount = 0u;
    m_tileIndexCount = 0u;
    m_clutCbp.fill(0u);
    m_transfer = {};
    m_transfer.direction = 3u;
    m_transferState = {};
    m_transferState.direction = 3u;
    m_activeTransferContiguous = false;
    m_gpuVramDirtyForCpu = false;
    m_cpuVramDirtyForGpu = (vram != nullptr);
    m_gpuWritePages.fill(0ull);
    m_gpuReadPages.fill(0ull);
    m_gpuClutDirty = false;
    m_batchWritePages.fill(0ull);
    m_batchReadPages.fill(0ull);
    m_batchFrameKey = 0xFFFFFFFFu;
    m_batchReadsClut = false;
    m_batchEstimatedIndices = 0u;

    if (m_vram)
    {
        syncCpuToGpuUnlocked();
        std::memset(m_cur.uploadStaging.mapped, 0, kClutBytes);
        submitBufferCopyUnlocked(m_cur.uploadStaging.buffer, m_clutGpu.buffer, kClutBytes);
    }
}

void GSVulkanBackend::Reset()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_cmds.empty() || !m_batchDraws.empty() || !m_deferredUploads.empty() || m_activeTransferContiguous || m_gpuInFlight)
        submitPendingUnlocked(false, true);
    m_cpuFallback.Reset();
    m_clutCbp.fill(0u);
    m_transfer = {};
    m_transfer.direction = 3u;
    m_transferState = {};
    m_transferState.direction = 3u;
    m_activeTransferContiguous = false;
    if (m_vram)
    {
        std::memset(m_cur.uploadStaging.mapped, 0, kClutBytes);
        submitBufferCopyUnlocked(m_cur.uploadStaging.buffer, m_clutGpu.buffer, kClutBytes);
    }
}

void GSVulkanBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || (!isFourBitIndexedPsm(tex0.psm) && !isEightBitIndexedPsm(tex0.psm)))
        return;

    switch (tex0.cld)
    {
    case 0u:
    case 6u:
    case 7u:
        return;
    case 1u:
        break;
    case 2u:
        m_clutCbp[0] = tex0.cbp;
        break;
    case 3u:
        m_clutCbp[1] = tex0.cbp;
        break;
    case 4u:
        if (m_clutCbp[0] == tex0.cbp)
            return;
        m_clutCbp[0] = tex0.cbp;
        break;
    case 5u:
        if (m_clutCbp[1] == tex0.cbp)
            return;
        m_clutCbp[1] = tex0.cbp;
        break;
    default:
        return;
    }

    const bool fourBit = isFourBitIndexedPsm(tex0.psm);
    const bool sixteenBit = tex0.cpsm == GS_PSM_CT16 || tex0.cpsm == GS_PSM_CT16S;
    const bool thirtyTwoBit = tex0.cpsm == GS_PSM_CT32 || tex0.cpsm == GS_PSM_CT24;
    if (!sixteenBit && !thirtyTwoBit)
        return;

    if (m_cpuVramDirtyForGpu)
        syncCpuToGpuUnlocked();
    if (m_activeTransferContiguous)
        flushPartialContiguousTransferUnlocked();

    const uint32_t entryCount = fourBit ? 16u : 256u;
    const uint32_t csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    const uint32_t destinationBase = (static_cast<uint32_t>(tex0.csa) & csaMask) << 4u;
    const bool loadCsm1Suffix = (tex0.csm == 0u && thirtyTwoBit && !fourBit);
    const uint32_t firstEntry = loadCsm1Suffix ? destinationBase : 0u;

    uint32_t clutFirst = tex0.cbp >> 5u;
    uint32_t clutLast = (tex0.cbp + 4u) >> 5u;
    if (tex0.csm != 0u)
    {
        const uint32_t cbw = std::max<uint32_t>(texclut.cbw, 1u);
        computePsmPageRange(tex0.cpsm, tex0.cbp, cbw,
                            static_cast<uint32_t>(texclut.cou) << 4u, texclut.cov,
                            (static_cast<uint32_t>(texclut.cou) << 4u) + entryCount - 1u, texclut.cov,
                            clutFirst, clutLast);
    }

    materializeDeferredUploadsUnlocked(clutFirst, clutLast, false);
    if (!m_batchDraws.empty() &&
        (m_batchReadsClut || maskOverlapsRange(m_batchWritePages, clutFirst, clutLast)))
    {
        closeTileBatchUnlocked();
    }

    const bool needBarrier = !m_cmds.empty();
    if (needBarrier)
    {
        m_gpuWritePages.fill(0ull);
        m_gpuReadPages.fill(0ull);
    }

    PendingCmd cmd{};
    cmd.kind = CmdKind::Upload;
    cmd.barrierBefore = needBarrier;
    cmd.groupsX = (entryCount + 63u) / 64u;
    cmd.transfer.mode = 2u;
    cmd.transfer.dbp = tex0.cbp;
    cmd.transfer.dpsm = tex0.cpsm;
    cmd.transfer.startPixel = firstEntry;
    cmd.transfer.pixelCount = entryCount;
    cmd.transfer.csm = tex0.csm;
    cmd.transfer.csa = tex0.csa;
    cmd.transfer.texclutPacked = (static_cast<uint32_t>(texclut.cbw) & 0x3Fu) |
                                 ((static_cast<uint32_t>(texclut.cou) & 0x3Fu) << 6u) |
                                 ((static_cast<uint32_t>(texclut.cov) & 0x3FFu) << 12u);
    m_cmds.push_back(cmd);
    setPageRange(m_gpuReadPages, clutFirst, clutLast);
    m_gpuClutDirty = true;
}

void GSVulkanBackend::BeginTransfer(const GSTransferCommand &command)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    flushPartialContiguousTransferUnlocked();
    m_transfer = command;
    m_transferState.x = command.trxpos.dsax;
    m_transferState.y = command.trxpos.dsay;
    m_transferState.totalPixels = static_cast<uint32_t>(command.trxreg.rrw) * static_cast<uint32_t>(command.trxreg.rrh);
    m_transferState.copiedPixels = 0u;
    m_transferState.direction = command.direction;
    m_transferState.localToHostPendingBytes = 0u;
    m_activeTransferContiguous = false;

    if (!m_vram)
        return;

    if (command.direction == 0u)
    {
        // Reserve contiguous staging space for PSMCT32 transfers so multi-chunk uploads
        // can be coalesced and sampled directly by sprites without VRAM barriers.
        const uint32_t totalBytes = bytesForPixels(command.bitbltbuf.dpsm, m_transferState.totalPixels);
        const uint32_t alignedBytes = (totalBytes + 3u) & ~3u;
        if (command.bitbltbuf.dpsm == GS_PSM_CT32 &&
            alignedBytes > 0u && alignedBytes <= kStagingBytes / 4u &&
            command.trxreg.rrw > 0u && command.trxreg.rrw <= 1023u &&
            command.trxreg.rrh > 0u && command.trxreg.rrh <= 1023u &&
            command.trxpos.dsax < 2048u && command.trxpos.dsay < 2048u)
        {
            if (m_cpuVramDirtyForGpu)
                syncCpuToGpuUnlocked();
            if (m_stagingOffset + alignedBytes > kStagingBytes)
                submitPendingUnlocked(false, false);
            m_activeTransferStagingOff = m_stagingOffset;
            m_activeTransferStagingBytes = 0u;
            m_stagingOffset += alignedBytes;
            m_activeTransferContiguous = true;
        }
        return;
    }

    if (command.direction == 2u)
    {
        const uint32_t total = m_transferState.totalPixels;
        if (total == 0u)
        {
            m_transferState.direction = 3u;
            return;
        }
        uint32_t srcFirst = 0u, srcLast = 0u, dstFirst = 0u, dstLast = 0u;
        computePsmPageRange(command.bitbltbuf.spsm, command.bitbltbuf.sbp, command.bitbltbuf.sbw,
                            command.trxpos.ssax, command.trxpos.ssay,
                            command.trxpos.ssax + command.trxreg.rrw - 1u, command.trxpos.ssay + command.trxreg.rrh - 1u,
                            srcFirst, srcLast);
        computePsmPageRange(command.bitbltbuf.dpsm, command.bitbltbuf.dbp, command.bitbltbuf.dbw,
                            command.trxpos.dsax, command.trxpos.dsay,
                            command.trxpos.dsax + command.trxreg.rrw - 1u, command.trxpos.dsay + command.trxreg.rrh - 1u,
                            dstFirst, dstLast);
        const bool overlaps = (srcFirst <= dstLast) && (dstFirst <= srcLast);
        if (overlaps)
        {
            syncGpuToCpuUnlocked();
            m_cpuFallback.BeginTransfer(command);
            m_transferState = m_cpuFallback.GetTransferSnapshot();
            m_cpuVramDirtyForGpu = true;
            return;
        }
        if (m_cpuVramDirtyForGpu)
            syncCpuToGpuUnlocked();

        materializeDeferredUploadsUnlocked(srcFirst, srcLast, false);
        materializeDeferredUploadsUnlocked(dstFirst, dstLast, false);
        if (!m_batchDraws.empty() &&
            (maskOverlapsRange(m_batchWritePages, srcFirst, srcLast) ||
             maskOverlapsRange(m_batchWritePages, dstFirst, dstLast) ||
             maskOverlapsRange(m_batchReadPages, dstFirst, dstLast)))
        {
            closeTileBatchUnlocked();
        }

        const bool needBarrier = maskOverlapsRange(m_gpuWritePages, srcFirst, srcLast) ||
                                 maskOverlapsRange(m_gpuWritePages, dstFirst, dstLast) ||
                                 maskOverlapsRange(m_gpuReadPages, dstFirst, dstLast);
        if (needBarrier)
        {
            m_gpuWritePages.fill(0ull);
            m_gpuReadPages.fill(0ull);
            m_gpuClutDirty = false;
        }

        PendingCmd cmd{};
        cmd.kind = CmdKind::Upload;
        cmd.barrierBefore = needBarrier;
        cmd.groupsX = (total + 63u) / 64u;
        cmd.transfer.mode = 1u;
        cmd.transfer.dbp = command.bitbltbuf.dbp;
        cmd.transfer.dbw = std::max<uint32_t>(command.bitbltbuf.dbw, 1u);
        cmd.transfer.dpsm = command.bitbltbuf.dpsm;
        cmd.transfer.dsax = command.trxpos.dsax;
        cmd.transfer.dsay = command.trxpos.dsay;
        cmd.transfer.rrw = command.trxreg.rrw;
        cmd.transfer.rrh = command.trxreg.rrh;
        cmd.transfer.pixelCount = total;
        cmd.transfer.dir = command.trxpos.dir;
        cmd.transfer.sbp = command.bitbltbuf.sbp;
        cmd.transfer.sbw = std::max<uint32_t>(command.bitbltbuf.sbw, 1u);
        cmd.transfer.spsm = command.bitbltbuf.spsm;
        cmd.transfer.ssax = command.trxpos.ssax;
        cmd.transfer.ssay = command.trxpos.ssay;
        m_cmds.push_back(cmd);
        setPageRange(m_gpuReadPages, srcFirst, srcLast);
        setPageRange(m_gpuWritePages, dstFirst, dstLast);

        m_transferState.copiedPixels = total;
        m_transferState.direction = 3u;
    }
    else if (command.direction == 1u)
    {
        syncGpuToCpuUnlocked();
        m_cpuFallback.BeginTransfer(command);
        m_transferState = m_cpuFallback.GetTransferSnapshot();
    }
}

void GSVulkanBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!data || sizeBytes == 0u || !m_vram || m_transferState.direction != 0u)
        return;
    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t totalPixels = m_transferState.totalPixels;
    if (rrw == 0u || rrh == 0u || totalPixels == 0u)
        return;

    const uint8_t dpsm = m_transfer.bitbltbuf.dpsm;
    const uint32_t remainingPixels = (m_transferState.copiedPixels < totalPixels)
                                         ? (totalPixels - m_transferState.copiedPixels)
                                         : 0u;
    if (remainingPixels == 0u)
        return;

    uint32_t pixelsInChunk = 0u;
    uint32_t bytesConsumed = 0u;
    switch (dpsm)
    {
    case GS_PSM_CT32:
    case GS_PSM_Z32:
        pixelsInChunk = std::min(remainingPixels, sizeBytes / 4u);
        bytesConsumed = pixelsInChunk * 4u;
        break;
    case GS_PSM_CT24:
    case GS_PSM_Z24:
        pixelsInChunk = std::min(remainingPixels, sizeBytes / 3u);
        bytesConsumed = pixelsInChunk * 3u;
        break;
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
    case GS_PSM_Z16:
    case GS_PSM_Z16S:
        pixelsInChunk = std::min(remainingPixels, sizeBytes / 2u);
        bytesConsumed = pixelsInChunk * 2u;
        break;
    case GS_PSM_T8:
    case GS_PSM_T8H:
        pixelsInChunk = std::min(remainingPixels, sizeBytes);
        bytesConsumed = pixelsInChunk;
        break;
    case GS_PSM_T4:
    case GS_PSM_T4HL:
    case GS_PSM_T4HH:
        pixelsInChunk = std::min(remainingPixels, sizeBytes * 2u);
        bytesConsumed = (pixelsInChunk + 1u) / 2u;
        break;
    default:
        return;
    }

    if (pixelsInChunk == 0u)
        return;

    if (m_cpuVramDirtyForGpu)
        syncCpuToGpuUnlocked();

    const uint32_t startPixel = m_transferState.copiedPixels;
    const uint32_t copied = startPixel + pixelsInChunk;
    const uint32_t dsax = m_transfer.trxpos.dsax;
    const uint32_t dsay = m_transfer.trxpos.dsay;

    if (m_activeTransferContiguous)
    {
        std::memcpy(static_cast<uint8_t *>(m_cur.uploadStaging.mapped) + m_activeTransferStagingOff + m_activeTransferStagingBytes,
                    data, bytesConsumed);
        m_activeTransferStagingBytes += bytesConsumed;

        if (copied >= totalPixels)
        {
            m_activeTransferContiguous = false;
            const uint32_t dbp = m_transfer.bitbltbuf.dbp;
            const uint32_t dbw = std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u);
            uint32_t firstPage = 0u, lastPage = 0u;
            computePsmPageRange(dpsm, dbp, dbw, dsax, dsay, dsax + rrw - 1u, dsay + rrh - 1u, firstPage, lastPage);

            // Defer small PSMCT32 scratch uploads (e.g., 16x8 / 16x16 foreground tiles) so sprites
            // sample directly from staging[] without VRAM WAR/RAW pipeline barriers.
            // Large uploads (e.g., FMV frames) go straight to VRAM.
            if (totalPixels <= 4096u)
            {
                bool replaced = false;
                for (size_t i = 0u; i < m_deferredUploads.size(); )
                {
                    DeferredUpload &u = m_deferredUploads[i];
                    if (u.dbp == dbp && u.dbw == dbw && u.dpsm == dpsm &&
                        u.dsax == dsax && u.dsay == dsay && u.rrw == rrw && u.rrh == rrh)
                    {
                        u.stagingByteOffset = m_activeTransferStagingOff;
                        replaced = true;
                        ++i;
                    }
                    else if (u.dbp == dbp && u.dbw == dbw && u.dpsm == dpsm &&
                             (dsax + rrw <= u.dsax || u.dsax + u.rrw <= dsax ||
                              dsay + rrh <= u.dsay || u.dsay + u.rrh <= dsay))
                    {
                        // Disjoint rectangle in the same texture buffer: keep both deferred!
                        ++i;
                    }
                    else if (u.firstPage <= lastPage && firstPage <= u.lastPage)
                    {
                        // Partially overlapping rect/page: materialize older upload first.
                        TransferPush tr{};
                        tr.mode = 0u;
                        tr.dbp = u.dbp; tr.dbw = u.dbw; tr.dpsm = u.dpsm;
                        tr.dsax = u.dsax; tr.dsay = u.dsay; tr.rrw = u.rrw; tr.rrh = u.rrh;
                        tr.startPixel = 0u; tr.pixelCount = u.totalPixels;
                        tr.stagingByteOffset = u.stagingByteOffset;
                        emitUploadCmdUnlocked(tr, (u.totalPixels + 63u) / 64u, u.firstPage, u.lastPage);
                        m_deferredUploads.erase(m_deferredUploads.begin() + static_cast<ptrdiff_t>(i));
                    }
                    else
                    {
                        ++i;
                    }
                }
                if (!replaced)
                {
                    if (m_deferredUploads.size() >= 16u)
                        materializeDeferredUploadsUnlocked(0u, 511u, true);
                    m_deferredUploads.push_back({dbp, dbw, dpsm, dsax, dsay, rrw, rrh,
                                                 m_activeTransferStagingOff, totalPixels, firstPage, lastPage});
                }
            }
            else
            {
                materializeDeferredUploadsUnlocked(firstPage, lastPage, false);
                TransferPush tr{};
                tr.mode = 0u;
                tr.dbp = dbp; tr.dbw = dbw; tr.dpsm = dpsm;
                tr.dsax = dsax; tr.dsay = dsay; tr.rrw = rrw; tr.rrh = rrh;
                tr.startPixel = 0u; tr.pixelCount = totalPixels;
                tr.stagingByteOffset = m_activeTransferStagingOff;
                emitUploadCmdUnlocked(tr, (totalPixels + 63u) / 64u, firstPage, lastPage);
            }
        }
    }
    else
    {
        const uint32_t alignedBytes = (bytesConsumed + 3u) & ~3u;
        if (m_stagingOffset + alignedBytes > kStagingBytes)
        {
            submitPendingUnlocked(false, false);
            if (alignedBytes > kStagingBytes)
                return;
        }
        const uint32_t stagingOff = m_stagingOffset;
        std::memcpy(static_cast<uint8_t *>(m_cur.uploadStaging.mapped) + stagingOff, data, bytesConsumed);
        if (alignedBytes > bytesConsumed)
            std::memset(static_cast<uint8_t *>(m_cur.uploadStaging.mapped) + stagingOff + bytesConsumed, 0, alignedBytes - bytesConsumed);
        m_stagingOffset += alignedBytes;

        const uint32_t endPixel = copied - 1u;
        const uint32_t y0 = dsay + (startPixel / rrw);
        const uint32_t y1 = dsay + (endPixel / rrw);
        uint32_t firstPage = 0u, lastPage = 0u;
        computePsmPageRange(dpsm, m_transfer.bitbltbuf.dbp, m_transfer.bitbltbuf.dbw,
                            dsax, y0, dsax + rrw - 1u, y1, firstPage, lastPage);
        materializeDeferredUploadsUnlocked(firstPage, lastPage, false);

        TransferPush tr{};
        tr.mode = 0u;
        tr.dbp = m_transfer.bitbltbuf.dbp;
        tr.dbw = std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u);
        tr.dpsm = dpsm;
        tr.dsax = dsax;
        tr.dsay = dsay;
        tr.rrw = rrw;
        tr.rrh = rrh;
        tr.startPixel = startPixel;
        tr.pixelCount = pixelsInChunk;
        tr.stagingByteOffset = stagingOff;
        emitUploadCmdUnlocked(tr, (pixelsInChunk + 63u) / 64u, firstPage, lastPage);
    }

    m_transferState.copiedPixels = copied;
    if (copied >= totalPixels)
    {
        m_transferState.direction = 3u;
        m_transferState.totalPixels = 0u;
    }
    else
    {
        m_transferState.x = dsax + (copied % rrw);
        m_transferState.y = dsay + (copied / rrw);
    }
}

void GSVulkanBackend::Submit(const GSPrimitiveBatch &batch)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || batch.vertexCount == 0u)
        return;

    switch (batch.state.prim.type)
    {
    case GS_PRIM_SPRITE:
        recordSpriteUnlocked(batch);
        break;
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_TRISTRIP:
    case GS_PRIM_TRIFAN:
        recordTriangleUnlocked(batch);
        break;
    default:
        syncGpuToCpuUnlocked();
        m_cpuFallback.Submit(batch);
        m_cpuVramDirtyForGpu = true;
        break;
    }
}

void GSVulkanBackend::recordSpriteUnlocked(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const auto &ctx = state.context;

    const int ofx = ctx.xyoffset.ofx >> 4;
    const int ofy = ctx.xyoffset.ofy >> 4;
    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    int x1 = static_cast<int>(v1.x) - ofx;
    int y1 = static_cast<int>(v1.y) - ofy;
    if (x0 > x1) std::swap(x0, x1);
    if (y0 > y1) std::swap(y0, y1);

    const int unclippedX0 = x0;
    const int unclippedY0 = y0;
    const int spanX = std::max(1, x1 - x0);
    const int spanY = std::max(1, y1 - y0);
    const int unclippedX1 = unclippedX0 + spanX - 1;
    const int unclippedY1 = unclippedY0 + spanY - 1;
    if (unclippedX1 < ctx.scissor.x0 || unclippedX0 > ctx.scissor.x1 ||
        unclippedY1 < ctx.scissor.y0 || unclippedY0 > ctx.scissor.y1)
        return;

    const int drawX0 = GSInternal::clampInt(unclippedX0, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY0 = GSInternal::clampInt(unclippedY0, ctx.scissor.y0, ctx.scissor.y1);
    const int drawX1 = GSInternal::clampInt(unclippedX1, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY1 = GSInternal::clampInt(unclippedY1, ctx.scissor.y0, ctx.scissor.y1);
    const int drawW = drawX1 - drawX0 + 1;
    const int drawH = drawY1 - drawY0 + 1;
    if (drawW <= 0 || drawH <= 0)
        return;

    // Fallback to CPU if coordinates exceed the 1024x1024 tile grid.
    if (drawX0 < 0 || drawY0 < 0 || drawX1 >= static_cast<int>(kTileGridW * kTileSize) ||
        drawY1 >= static_cast<int>(kTileGridH * kTileSize))
    {
        syncGpuToCpuUnlocked();
        m_cpuFallback.Submit(batch);
        m_cpuVramDirtyForGpu = true;
        return;
    }

    if (m_cpuVramDirtyForGpu)
        syncCpuToGpuUnlocked();
    if (m_activeTransferContiguous)
        flushPartialContiguousTransferUnlocked();

    const uint16_t tx0 = static_cast<uint16_t>(drawX0 >> kTileShift);
    const uint16_t ty0 = static_cast<uint16_t>(drawY0 >> kTileShift);
    const uint16_t tx1 = static_cast<uint16_t>(drawX1 >> kTileShift);
    const uint16_t ty1 = static_cast<uint16_t>(drawY1 >> kTileShift);
    const uint32_t tileSpan = static_cast<uint32_t>(tx1 - tx0 + 1u) * static_cast<uint32_t>(ty1 - ty0 + 1u);

    if (m_spriteCount >= kMaxSprites ||
        m_tileHeaderCount + kTileGridSize > kMaxTileHeaders ||
        m_tileIndexCount + m_batchEstimatedIndices + tileSpan > kMaxTileIndices)
    {
        submitPendingUnlocked(false, false);
    }

    const uint32_t fbpBlock = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
    const uint32_t fbw = std::max<uint32_t>(ctx.frame.fbw, 1u);
    const uint32_t zbpBlock = GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);
    const bool linear = state.linearFilter;

    float u0f = 0.0f, v0f = 0.0f, u1f = 0.0f, v1f = 0.0f;
    const int texW = state.textureWidth;
    const int texH = state.textureHeight;
    if (state.prim.tme)
    {
        if (state.prim.fst)
        {
            u0f = static_cast<float>(v0.u) / 16.0f;
            v0f = static_cast<float>(v0.v) / 16.0f;
            u1f = static_cast<float>(v1.u) / 16.0f;
            v1f = static_cast<float>(v1.v) / 16.0f;
        }
        else
        {
            const float q0 = fabsQ(v0.q);
            const float q1 = fabsQ(v1.q);
            u0f = (v0.s / q0) * static_cast<float>(texW);
            v0f = (v0.t / q0) * static_cast<float>(texH);
            u1f = (v1.s / q1) * static_cast<float>(texW);
            v1f = (v1.t / q1) * static_cast<float>(texH);
        }
    }

    const float invW = 1.0f / static_cast<float>(spanX);
    const float invH = 1.0f / static_cast<float>(spanY);
    const bool fieldBlend = state.spriteFieldBlend && state.prim.tme && !linear &&
                            std::fabs(v1f - v0f) > static_cast<float>(spanY);
    const float duPerPx = (u1f - u0f) * invW;
    const float uStart = u0f + (static_cast<float>(drawX0 - unclippedX0) + 0.5f) * duPerPx;
    const int32_t uBaseFixed = static_cast<int32_t>((uStart - (linear ? 0.5f : 0.0f)) * 65536.0f);
    const int32_t uStepFixed = static_cast<int32_t>(duPerPx * 65536.0f);

    uint32_t fbFirst = 0u, fbLast = 0u, zFirst = 0u, zLast = 0u;
    computePsmPageRange(ctx.frame.psm, fbpBlock, fbw, static_cast<uint32_t>(drawX0), static_cast<uint32_t>(drawY0),
                        static_cast<uint32_t>(drawX1), static_cast<uint32_t>(drawY1), fbFirst, fbLast);
    computePsmPageRange(ctx.zbuf.psm, zbpBlock, fbw, static_cast<uint32_t>(drawX0), static_cast<uint32_t>(drawY0),
                        static_cast<uint32_t>(drawX1), static_cast<uint32_t>(drawY1), zFirst, zLast);

    // If a framebuffer/Z write touches any deferred upload page, materialize it first.
    materializeDeferredUploadsUnlocked(fbFirst, fbLast, false);
    if (!ctx.zbuf.zmask)
        materializeDeferredUploadsUnlocked(zFirst, zLast, false);

    uint32_t stagingTexOff = 0u;
    uint32_t stagingTexRect = 0u;
    bool usesVramTex = false;
    uint32_t texFirst = 0u, texLast = 0u;

    if (state.prim.tme)
    {
        // Check if this sprite can read directly from a deferred linear PSMCT32 upload in staging[].
        if (ctx.tex0.psm == GS_PSM_CT32 && !linear && !fieldBlend && !m_deferredUploads.empty())
        {
            const uint32_t tbw = std::max<uint32_t>(ctx.tex0.tbw, 1u);
            const uint8_t wrapU = static_cast<uint8_t>(ctx.clamp & 3u);
            const uint8_t wrapV = static_cast<uint8_t>((ctx.clamp >> 2u) & 3u);
            if ((wrapU == 0u || wrapU == 1u) && (wrapV == 0u || wrapV == 1u))
            {
                const int uA = uBaseFixed >> 16;
                const int uB = (uBaseFixed + (drawW - 1) * uStepFixed) >> 16;
                const int vA = static_cast<int>((v0f + (v1f - v0f) * ((static_cast<float>(drawY0 - unclippedY0) + 0.5f) * invH)) * 65536.0f) >> 16;
                const int vB = static_cast<int>((v0f + (v1f - v0f) * ((static_cast<float>(drawY1 - unclippedY0) + 0.5f) * invH)) * 65536.0f) >> 16;
                const int minU = std::clamp(std::min(uA, uB), 0, std::max(texW - 1, 0));
                const int maxU = std::clamp(std::max(uA, uB), 0, std::max(texW - 1, 0));
                const int minV = std::clamp(std::min(vA, vB), 0, std::max(texH - 1, 0));
                const int maxV = std::clamp(std::max(vA, vB), 0, std::max(texH - 1, 0));
                const bool inBounds = (uA >= 0 && uA < texW && uB >= 0 && uB < texW &&
                                       vA >= 0 && vA < texH && vB >= 0 && vB < texH);
                if (inBounds)
                {
                    for (const DeferredUpload &u : m_deferredUploads)
                    {
                        if (u.dbp == ctx.tex0.tbp0 && u.dbw == tbw && u.dpsm == GS_PSM_CT32 &&
                            minU >= static_cast<int>(u.dsax) && maxU < static_cast<int>(u.dsax + u.rrw) &&
                            minV >= static_cast<int>(u.dsay) && maxV < static_cast<int>(u.dsay + u.rrh))
                        {
                            stagingTexOff = u.stagingByteOffset | 0x80000000u;
                            stagingTexRect = (u.dsax & 0x7FFu) | ((u.dsay & 0x7FFu) << 11u) | ((u.rrw & 0x3FFu) << 22u);
                            break;
                        }
                    }
                }
            }
        }

        if ((stagingTexOff & 0x80000000u) == 0u)
        {
            usesVramTex = true;
            computePsmPageRange(ctx.tex0.psm, ctx.tex0.tbp0, ctx.tex0.tbw, 0u, 0u,
                                static_cast<uint32_t>(std::max(texW - 1, 0)),
                                static_cast<uint32_t>(std::max(texH - 1, 0)), texFirst, texLast);
            materializeDeferredUploadsUnlocked(texFirst, texLast, false);
        }
    }

    const uint32_t frameKey = (ctx.frame.fbp & 0x1FFu) | ((fbw & 0x3Fu) << 9u) | ((ctx.zbuf.zbp & 0x1FFu) << 15u);
    const bool batchHazard =
        !m_batchDraws.empty() &&
        ((usesVramTex && maskOverlapsRange(m_batchWritePages, texFirst, texLast)) ||
         (m_batchFrameKey != frameKey &&
          (maskOverlapsRange(m_batchWritePages, fbFirst, fbLast) ||
           maskOverlapsRange(m_batchWritePages, zFirst, zLast) ||
           maskOverlapsRange(m_batchReadPages, fbFirst, fbLast) ||
           (!ctx.zbuf.zmask && maskOverlapsRange(m_batchReadPages, zFirst, zLast)))));
    if (batchHazard)
        closeTileBatchUnlocked();

    GpuSpriteDraw d{};
    d.drawX0 = drawX0;
    d.drawY0 = drawY0;
    d.drawW = drawW;
    d.drawH = drawH;
    d.unclippedX0 = unclippedX0;
    d.unclippedY0 = unclippedY0;
    d.uBaseFixed = uBaseFixed;
    d.uStepFixed = uStepFixed;
    d.v0f = v0f;
    d.v1f = v1f;
    d.invH = invH;
    d.zVal = static_cast<uint32_t>(v1.z);
    d.rgba = static_cast<uint32_t>(v1.r) |
             (static_cast<uint32_t>(v1.g) << 8u) |
             (static_cast<uint32_t>(v1.b) << 16u) |
             (static_cast<uint32_t>(v1.a) << 24u);
    d.fogAndColor = static_cast<uint32_t>(v1.fog) |
                    (static_cast<uint32_t>(state.fogR) << 8u) |
                    (static_cast<uint32_t>(state.fogG) << 16u) |
                    (static_cast<uint32_t>(state.fogB) << 24u);
    d.frameInfo = (fbpBlock & 0x3FFFu) |
                  ((fbw & 0x3Fu) << 14u) |
                  ((static_cast<uint32_t>(ctx.frame.psm) & 0x3Fu) << 20u) |
                  (state.prim.tme ? (1u << 26u) : 0u) |
                  (state.prim.abe ? (1u << 27u) : 0u) |
                  (state.pabe ? (1u << 28u) : 0u) |
                  ((ctx.fba & 1ull) ? (1u << 29u) : 0u) |
                  (state.prim.fge ? (1u << 30u) : 0u) |
                  (linear ? (1u << 31u) : 0u);
    d.fbmsk = ctx.frame.fbmsk;
    d.depthInfo = (zbpBlock & 0x3FFFu) |
                  ((static_cast<uint32_t>(ctx.zbuf.psm) & 0x3Fu) << 14u) |
                  (ctx.zbuf.zmask ? (1u << 20u) : 0u);
    d.testLo = static_cast<uint32_t>(ctx.test & 0xFFFFFFFFull);
    d.alphaLo = static_cast<uint32_t>(ctx.alpha & 0xFFu);
    d.alphaFix = static_cast<uint32_t>((ctx.alpha >> 32u) & 0xFFu);
    d.tbp0 = ctx.tex0.tbp0;
    d.texMeta = packTexMeta(ctx.tex0) | (fieldBlend ? (1u << 27u) : 0u); // bit 27: shader field blend
    d.texDim = (static_cast<uint32_t>(texW) & 0xFFFFu) | (static_cast<uint32_t>(texH) << 16u);
    d.clampLo = static_cast<uint32_t>(ctx.clamp & 0xFFFFFFFFull);
    d.clampHi = static_cast<uint32_t>(ctx.clamp >> 32u);
    d.texaPacked = packTexa(state.texa);
    d.stagingTexOff = stagingTexOff;
    d.stagingTexRect = stagingTexRect;

    const uint32_t idx = m_spriteCount++;
    static_cast<GpuSpriteDraw *>(m_cur.spriteBuf.mapped)[idx] = d;

    m_batchDraws.push_back({idx, tx0, ty0, tx1, ty1});
    m_batchEstimatedIndices += tileSpan;
    m_batchFrameKey = frameKey;
    setPageRange(m_batchWritePages, fbFirst, fbLast);
    if (!ctx.zbuf.zmask)
        setPageRange(m_batchWritePages, zFirst, zLast);
    else if (((ctx.test >> 17u) & 3u) >= 2u)
        setPageRange(m_batchReadPages, zFirst, zLast);
    if (usesVramTex)
    {
        setPageRange(m_batchReadPages, texFirst, texLast);
        if (isIndexedPsm(ctx.tex0.psm))
            m_batchReadsClut = true;
    }
}

void GSVulkanBackend::recordTriangleUnlocked(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const GSVertex &v2 = batch.vertices[2];
    const auto &ctx = state.context;

    const int32_t x0 = static_cast<int32_t>(v0.x * 16.0f) - ctx.xyoffset.ofx;
    const int32_t y0 = static_cast<int32_t>(v0.y * 16.0f) - ctx.xyoffset.ofy;
    const int32_t x1 = static_cast<int32_t>(v1.x * 16.0f) - ctx.xyoffset.ofx;
    const int32_t y1 = static_cast<int32_t>(v1.y * 16.0f) - ctx.xyoffset.ofy;
    const int32_t x2 = static_cast<int32_t>(v2.x * 16.0f) - ctx.xyoffset.ofx;
    const int32_t y2 = static_cast<int32_t>(v2.y * 16.0f) - ctx.xyoffset.ofy;
    const auto ceilPixel = [](int32_t fixed) { return (fixed + 15) >> 4; };
    const int minX = std::max<int>(ceilPixel(std::min({x0, x1, x2})), ctx.scissor.x0);
    const int maxX = std::min<int>(ceilPixel(std::max({x0, x1, x2})) - 1, ctx.scissor.x1);
    const int minY = std::max<int>(ceilPixel(std::min({y0, y1, y2})), ctx.scissor.y0);
    const int maxY = std::min<int>(ceilPixel(std::max({y0, y1, y2})) - 1, ctx.scissor.y1);
    if (minX > maxX || minY > maxY)
        return;

    const int64_t area = static_cast<int64_t>(x1 - x0) * (y2 - y0) - static_cast<int64_t>(y1 - y0) * (x2 - x0);
    if (area == 0)
        return;
    const int32_t winding = area > 0 ? 1 : -1;
    const double invArea = 1.0 / static_cast<double>(area * winding);

    if (minX < 0 || minY < 0 || maxX >= static_cast<int>(kTileGridW * kTileSize) ||
        maxY >= static_cast<int>(kTileGridH * kTileSize))
    {
        syncGpuToCpuUnlocked();
        m_cpuFallback.Submit(batch);
        m_cpuVramDirtyForGpu = true;
        return;
    }

    if (m_cpuVramDirtyForGpu)
        syncCpuToGpuUnlocked();
    if (m_activeTransferContiguous)
        flushPartialContiguousTransferUnlocked();

    const uint16_t tx0 = static_cast<uint16_t>(minX >> kTileShift);
    const uint16_t ty0 = static_cast<uint16_t>(minY >> kTileShift);
    const uint16_t tx1 = static_cast<uint16_t>(maxX >> kTileShift);
    const uint16_t ty1 = static_cast<uint16_t>(maxY >> kTileShift);
    const uint32_t tileSpan = static_cast<uint32_t>(tx1 - tx0 + 1u) * static_cast<uint32_t>(ty1 - ty0 + 1u);

    if (m_triangleCount >= kMaxTriangles ||
        m_tileHeaderCount + kTileGridSize > kMaxTileHeaders ||
        m_tileIndexCount + m_batchEstimatedIndices + tileSpan > kMaxTileIndices)
    {
        submitPendingUnlocked(false, false);
    }

    const uint32_t fbpBlock = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
    const uint32_t fbw = std::max<uint32_t>(ctx.frame.fbw, 1u);
    const uint32_t zbpBlock = GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);
    const int drawW = maxX - minX + 1;
    const int drawH = maxY - minY + 1;

    uint32_t fbFirst = 0u, fbLast = 0u, zFirst = 0u, zLast = 0u;
    computePsmPageRange(ctx.frame.psm, fbpBlock, fbw, static_cast<uint32_t>(minX), static_cast<uint32_t>(minY),
                        static_cast<uint32_t>(maxX), static_cast<uint32_t>(maxY), fbFirst, fbLast);
    computePsmPageRange(ctx.zbuf.psm, zbpBlock, fbw, static_cast<uint32_t>(minX), static_cast<uint32_t>(minY),
                        static_cast<uint32_t>(maxX), static_cast<uint32_t>(maxY), zFirst, zLast);
    materializeDeferredUploadsUnlocked(fbFirst, fbLast, false);
    if (!ctx.zbuf.zmask)
        materializeDeferredUploadsUnlocked(zFirst, zLast, false);

    uint32_t texFirst = 0u, texLast = 0u;
    if (state.prim.tme)
    {
        computePsmPageRange(ctx.tex0.psm, ctx.tex0.tbp0, ctx.tex0.tbw, 0u, 0u,
                            static_cast<uint32_t>(std::max<int>(state.textureWidth - 1, 0)),
                            static_cast<uint32_t>(std::max<int>(state.textureHeight - 1, 0)), texFirst, texLast);
        materializeDeferredUploadsUnlocked(texFirst, texLast, false);
    }

    const uint32_t frameKey = (ctx.frame.fbp & 0x1FFu) | ((fbw & 0x3Fu) << 9u) | ((ctx.zbuf.zbp & 0x1FFu) << 15u);
    const bool batchHazard =
        !m_batchDraws.empty() &&
        ((state.prim.tme && maskOverlapsRange(m_batchWritePages, texFirst, texLast)) ||
         (m_batchFrameKey != frameKey &&
          (maskOverlapsRange(m_batchWritePages, fbFirst, fbLast) ||
           maskOverlapsRange(m_batchWritePages, zFirst, zLast) ||
           maskOverlapsRange(m_batchReadPages, fbFirst, fbLast) ||
           (!ctx.zbuf.zmask && maskOverlapsRange(m_batchReadPages, zFirst, zLast)))));
    if (batchHazard)
        closeTileBatchUnlocked();

    auto packVtxColor = [](const GSVertex &v) -> uint32_t
    {
        return static_cast<uint32_t>(v.r) |
               (static_cast<uint32_t>(v.g) << 8u) |
               (static_cast<uint32_t>(v.b) << 16u) |
               (static_cast<uint32_t>(v.a) << 24u);
    };

    GpuTriangleDraw d{};
    d.bounds[0] = minX;
    d.bounds[1] = minY;
    d.bounds[2] = drawW;
    d.bounds[3] = drawH;
    d.vx[0] = x0; d.vx[1] = x1; d.vx[2] = x2; d.vx[3] = winding;
    d.vy[0] = y0; d.vy[1] = y1; d.vy[2] = y2; d.vy[3] = 0;
    d.s[0] = v0.s; d.s[1] = v1.s; d.s[2] = v2.s;
    d.t[0] = v0.t; d.t[1] = v1.t; d.t[2] = v2.t;
    d.q[0] = v0.q; d.q[1] = v1.q; d.q[2] = v2.q;
    d.uv01[0] = static_cast<float>(v0.u);
    d.uv01[1] = static_cast<float>(v0.v);
    d.uv01[2] = static_cast<float>(v1.u);
    d.uv01[3] = static_cast<float>(v1.v);
    d.uv2_pad[0] = static_cast<float>(v2.u);
    d.uv2_pad[1] = static_cast<float>(v2.v);
    d.z0 = v0.z;
    d.z1 = v1.z;
    d.z2 = v2.z;
    d.invArea = invArea;
    d.rgba[0] = packVtxColor(v0);
    d.rgba[1] = packVtxColor(v1);
    d.rgba[2] = packVtxColor(v2);
    d.rgba[3] = static_cast<uint32_t>(v0.fog) |
                (static_cast<uint32_t>(v1.fog) << 8u) |
                (static_cast<uint32_t>(v2.fog) << 16u);
    d.fogColor = static_cast<uint32_t>(state.fogR) |
                 (static_cast<uint32_t>(state.fogG) << 8u) |
                 (static_cast<uint32_t>(state.fogB) << 16u);
    d.frameInfo = (fbpBlock & 0x3FFFu) |
                  ((fbw & 0x3Fu) << 14u) |
                  ((static_cast<uint32_t>(ctx.frame.psm) & 0x3Fu) << 20u) |
                  (state.prim.tme ? (1u << 26u) : 0u) |
                  (state.prim.abe ? (1u << 27u) : 0u) |
                  (state.pabe ? (1u << 28u) : 0u) |
                  ((ctx.fba & 1ull) ? (1u << 29u) : 0u) |
                  (state.prim.fge ? (1u << 30u) : 0u) |
                  (state.linearFilter ? (1u << 31u) : 0u);
    d.fbmsk = ctx.frame.fbmsk;
    d.depthInfo = (zbpBlock & 0x3FFFu) |
                  ((static_cast<uint32_t>(ctx.zbuf.psm) & 0x3Fu) << 14u) |
                  (ctx.zbuf.zmask ? (1u << 20u) : 0u) |
                  (state.prim.iip ? (1u << 21u) : 0u) |
                  (state.prim.fst ? (1u << 22u) : 0u);
    d.testLo = static_cast<uint32_t>(ctx.test & 0xFFFFFFFFull);
    d.alphaLo = static_cast<uint32_t>(ctx.alpha & 0xFFu);
    d.alphaFix = static_cast<uint32_t>((ctx.alpha >> 32u) & 0xFFu);
    d.tbp0 = ctx.tex0.tbp0;
    d.texMeta = packTexMeta(ctx.tex0);
    d.texDim = (static_cast<uint32_t>(state.textureWidth) & 0xFFFFu) |
               (static_cast<uint32_t>(state.textureHeight) << 16u);
    d.clampLo = static_cast<uint32_t>(ctx.clamp & 0xFFFFFFFFull);
    d.clampHi = static_cast<uint32_t>(ctx.clamp >> 32u);
    d.texaPacked = packTexa(state.texa);

    const uint32_t idx = m_triangleCount++;
    static_cast<GpuTriangleDraw *>(m_cur.triangleBuf.mapped)[idx] = d;

    m_batchDraws.push_back({idx | 0x80000000u, tx0, ty0, tx1, ty1});
    m_batchEstimatedIndices += tileSpan;
    m_batchFrameKey = frameKey;
    setPageRange(m_batchWritePages, fbFirst, fbLast);
    if (!ctx.zbuf.zmask)
        setPageRange(m_batchWritePages, zFirst, zLast);
    else if (((ctx.test >> 17u) & 3u) >= 2u)
        setPageRange(m_batchReadPages, zFirst, zLast);
    if (state.prim.tme)
    {
        setPageRange(m_batchReadPages, texFirst, texLast);
        if (isIndexedPsm(ctx.tex0.psm))
            m_batchReadsClut = true;
    }
}

bool GSVulkanBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || context.frame.fbw == 0u || !isPresentablePsm(context.frame.psm))
        return false;

    GSPrimitiveBatch batch{};
    batch.vertexCount = 2u;
    batch.state.context = context;
    batch.state.context.xyoffset = {0u, 0u};
    batch.state.context.test = (1ull << 17u);
    batch.state.context.zbuf.zmask = true;
    batch.state.prim.type = GS_PRIM_SPRITE;
    const uint32_t x0 = context.scissor.x0;
    const uint32_t x1 = std::max<uint32_t>(x0, context.scissor.x1);
    const uint32_t y0 = context.scissor.y0;
    const uint32_t y1 = std::max<uint32_t>(y0, context.scissor.y1);
    batch.vertices[0].x = static_cast<float>(x0);
    batch.vertices[0].y = static_cast<float>(y0);
    batch.vertices[1].x = static_cast<float>(x1 + 1u);
    batch.vertices[1].y = static_cast<float>(y1 + 1u);
    batch.vertices[1].r = static_cast<uint8_t>(rgba);
    batch.vertices[1].g = static_cast<uint8_t>(rgba >> 8u);
    batch.vertices[1].b = static_cast<uint8_t>(rgba >> 16u);
    batch.vertices[1].a = static_cast<uint8_t>(rgba >> 24u);
    recordSpriteUnlocked(batch);
    return true;
}

void GSVulkanBackend::Flush()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_cmds.empty() || !m_batchDraws.empty() || !m_deferredUploads.empty() || m_activeTransferContiguous || m_cpuVramDirtyForGpu)
        submitPendingUnlocked(false, false);
}

void GSVulkanBackend::TextureFlush()
{
    // Authoritative VRAM + exact page barriers keep textures coherent without extra flushes.
}

void GSVulkanBackend::Sync(GSSyncReason reason)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (reason == GSSyncReason::DebugReadback || reason == GSSyncReason::Reset)
        submitPendingUnlocked(false, true);
    else if (!m_cmds.empty() || !m_batchDraws.empty() || !m_deferredUploads.empty() || m_activeTransferContiguous || m_cpuVramDirtyForGpu)
        submitPendingUnlocked(false, false);
}

PresentationFrame GSVulkanBackend::Present(const GSPresentationRequest &request)
{
    return presentImpl(request, nullptr);
}

bool GSVulkanBackend::PresentGpu(const GSPresentationRequest &request, GSGpuFrame &out)
{
    presentImpl(request, &out);
    return true;
}

const GSHostGpuContext *GSVulkanBackend::HostGpuContext() const
{
    return m_device ? &m_hostContext : nullptr;
}

PresentationFrame GSVulkanBackend::presentCpuFallbackUnlocked(const GSPresentationRequest &request, GSGpuFrame *gpuOut)
{
    syncGpuToCpuUnlocked();
    PresentationFrame frame = m_cpuFallback.Present(request);
    if (gpuOut && frame)
    {
        // Stage the CPU-composed field where the GPU present leaves its output so the host draws both alike.
        std::memcpy(m_presentOut.mapped, frame.pixels.data(), std::min<size_t>(frame.pixels.size(), kPresentBytes));
        gpuOut->buffer = handleBits(m_presentOut.buffer);
        gpuOut->mapped = static_cast<const uint8_t *>(m_presentOut.mapped);
        gpuOut->rowPixels = kHostFrameWidth;
        gpuOut->width = std::min(frame.width, kHostFrameWidth);
        gpuOut->height = std::min(frame.height, kHostFrameHeight);
        gpuOut->displayFbp = frame.displayFbp;
        gpuOut->sourceFbp = frame.sourceFbp;
        gpuOut->usedPreferred = frame.usedPreferred;
    }
    return frame;
}

PresentationFrame GSVulkanBackend::presentImpl(const GSPresentationRequest &request, GSGpuFrame *gpuOut)
{
    if (gpuOut)
        *gpuOut = GSGpuFrame{};
    PresentationFrame result{};
    const bool enableCrt1 = (request.pmode & 0x1ull) != 0ull;
    const bool enableCrt2 = (request.pmode & 0x2ull) != 0ull;
    const GSFrameReg displayFrame1 = decodeDisplayFrame(request.dispfb1);
    const GSFrameReg displayFrame2 = decodeDisplayFrame(request.dispfb2);
    uint32_t width1 = 0u, height1 = 0u, width2 = 0u, height2 = 0u;
    decodeDisplaySize(request.display1, width1, height1);
    decodeDisplaySize(request.display2, width2, height2);
    const bool valid1 = enableCrt1 && hasDisplaySetup(request.display1, displayFrame1);
    const bool valid2 = enableCrt2 && hasDisplaySetup(request.display2, displayFrame2);
    if (!valid1 && !valid2)
        return result;

    const bool interlaced = (request.smode2 & 0x1ull) != 0ull;
    const bool frameMode = ((request.smode2 >> 1u) & 0x1ull) != 0ull;
    const bool fieldMode = interlaced && !frameMode;
    const bool oddField = (request.vsyncTick & 1ull) != 0ull;

    const auto decodeOrigin = [](uint64_t dispfb64, uint32_t &ox, uint32_t &oy)
    {
        ox = static_cast<uint32_t>((dispfb64 >> 32u) & 0x7FFu);
        oy = static_cast<uint32_t>((dispfb64 >> 43u) & 0x7FFu);
    };

    PresentPush push{};
    push.pmode = static_cast<uint32_t>(request.pmode & 0xFFFFu);
    push.background = static_cast<uint32_t>(request.bgcolor & 0x00FFFFFFu) |
                      (static_cast<uint32_t>((request.pmode >> 8u) & 0xFFu) << 24u);
    push.field = fieldMode ? 1u : 0u;
    push.parity = oddField ? 1u : 0u;

    bool fallbackToCpu = false;
    bool usedPreferred = false;
    uint32_t displayFbp = 0u;
    uint32_t sourceFbp = 0u;

    if (valid1 && valid2)
    {
        if (!isPresentablePsm(displayFrame1.psm) || !isPresentablePsm(displayFrame2.psm) ||
            displayFrame1.fbp == 0u || displayFrame2.fbp == 0u)
        {
            fallbackToCpu = true;
        }
        else
        {
            uint32_t ox1 = 0u, oy1 = 0u, ox2 = 0u, oy2 = 0u;
            decodeOrigin(request.dispfb1, ox1, oy1);
            decodeOrigin(request.dispfb2, ox2, oy2);
            push.crt1 = {width1, height1, GSInternal::framePageBaseToBlock(displayFrame1.fbp),
                         displayFrame1.fbw, ox1, oy1, displayFrame1.psm};
            push.crt2 = {width2, height2, GSInternal::framePageBaseToBlock(displayFrame2.fbp),
                         displayFrame2.fbw, ox2, oy2, displayFrame2.psm};
            result.width = std::max(width1, width2);
            result.height = std::max(height1, height2);
            displayFbp = displayFrame1.fbp;
            sourceFbp = displayFrame1.fbp;
        }
    }
    else
    {
        const GSFrameReg &df = valid1 ? displayFrame1 : displayFrame2;
        const uint64_t dispfb64 = valid1 ? request.dispfb1 : request.dispfb2;
        const uint32_t w = valid1 ? width1 : width2;
        const uint32_t h = valid1 ? height1 : height2;
        GSFrameReg selected = df;
        uint32_t ox = 0u, oy = 0u;
        decodeOrigin(dispfb64, ox, oy);
        uint32_t baseBlock = GSInternal::framePageBaseToBlock(df.fbp);

        if (request.hasPreferredSource && request.preferredDestFbp == df.fbp &&
            (request.preferredSource.fbw != 0u || request.preferredSource.fbp != df.fbp) &&
            isPresentablePsm(request.preferredSource.psm))
        {
            selected = request.preferredSource;
            baseBlock = selected.fbp;
            ox = 0u;
            oy = 0u;
            usedPreferred = true;
        }
        else if (!isPresentablePsm(df.psm))
        {
            fallbackToCpu = true;
        }

        if (!fallbackToCpu)
        {
            push.crt1 = {w, h, baseBlock, selected.fbw, ox, oy, selected.psm};
            result.width = w;
            result.height = h;
            displayFbp = df.fbp;
            sourceFbp = selected.fbp;
        }
    }

    // Allocate/zero output buffer before locking m_mutex only when taking the GPU path.
    if (!fallbackToCpu && !gpuOut)
        result.pixels.resize(kPresentBytes);

    // The UI thread holds only m_presentMutex while the GPU runs the present dispatch, so the EE can
    // keep recording into m_cur meanwhile.
    std::lock_guard<std::mutex> presentLock(m_presentMutex);
    std::unique_lock<std::mutex> lock(m_mutex);
    if (!m_vram)
    {
        result.pixels.clear();
        return result;
    }

    if (fallbackToCpu)
        return presentCpuFallbackUnlocked(request, gpuOut);

    const uint32_t presentHeight = std::clamp(result.height, 1u, kHostFrameHeight);
    PageMask displayPages{};
    auto markCircuitPages = [&](const PresentCircuitPush &c)
    {
        if (c.width == 0u || c.height == 0u)
            return;
        uint32_t p0 = 0u, p1 = 0u;
        const uint32_t bw = c.pitch != 0u ? c.pitch : 10u;
        computePsmPageRange(c.psm, c.baseBlock, bw, c.originX, c.originY,
                            c.originX + c.width - 1u, c.originY + c.height - 1u, p0, p1);
        setPageRange(displayPages, p0, p1);
    };
    markCircuitPages(push.crt1);
    markCircuitPages(push.crt2);

    bool displayDirtyInPending = m_cpuVramDirtyForGpu ||
                                 m_activeTransferContiguous ||
                                 masksOverlap(displayPages, m_batchWritePages) ||
                                 masksOverlap(displayPages, m_gpuWritePages);
    if (!displayDirtyInPending)
    {
        for (const DeferredUpload &u : m_deferredUploads)
        {
            if (maskOverlapsRange(displayPages, u.firstPage, u.lastPage))
            {
                displayDirtyInPending = true;
                break;
            }
        }
    }

    // Pending work that touches the display goes first; queue order puts the present after it.
    if (displayDirtyInPending)
        submitPendingUnlocked(false, false);
    const bool submitted = submitPresentUnlocked(push, presentHeight);
    // The black-output CPU fallback below must read this field's VRAM, so keep the EE out until it is decided.
    const bool mayFallback = !usedPreferred && displayFbp == 0u && !(valid1 && valid2);
    if (!mayFallback)
        lock.unlock();

    if (submitted &&
        m_vk.vkWaitForFences(m_device, 1u, &m_presentFence, ps2vk::VK_TRUE, UINT64_MAX) == ps2vk::VK_SUCCESS)
        m_vk.vkResetFences(m_device, 1u, &m_presentFence);

    if (!gpuOut)
    {
        const size_t copyBytes = static_cast<size_t>(presentHeight) * kHostFrameWidth * 4u;
        std::memcpy(result.pixels.data(), m_presentOut.mapped, copyBytes);
    }
    result.displayFbp = displayFbp;
    result.sourceFbp = sourceFbp;
    result.usedPreferred = usedPreferred;

    if (mayFallback)
    {
        const uint32_t *px = reinterpret_cast<const uint32_t *>(
            gpuOut ? static_cast<const uint8_t *>(m_presentOut.mapped) : result.pixels.data());
        bool anyNonBlack = false;
        for (uint32_t y = 0u; y < result.height && !anyNonBlack; ++y)
        {
            const uint32_t *row = px + y * kHostFrameWidth;
            for (uint32_t x = 0u; x < result.width; ++x)
            {
                if ((row[x] & 0x00FFFFFFu) != 0u)
                {
                    anyNonBlack = true;
                    break;
                }
            }
        }
        if (!anyNonBlack)
            return presentCpuFallbackUnlocked(request, gpuOut);
    }

    if (gpuOut && submitted)
    {
        gpuOut->buffer = handleBits(m_presentOut.buffer);
        gpuOut->mapped = static_cast<const uint8_t *>(m_presentOut.mapped);
        gpuOut->rowPixels = kHostFrameWidth;
        gpuOut->width = std::min(result.width, kHostFrameWidth);
        gpuOut->height = presentHeight;
        gpuOut->displayFbp = displayFbp;
        gpuOut->sourceFbp = sourceFbp;
        gpuOut->usedPreferred = usedPreferred;
    }
    return result;
}

uint32_t GSVulkanBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    const uint32_t n = m_cpuFallback.ConsumeLocalToHostBytes(dst, maxBytes);
    m_transferState.localToHostPendingBytes = m_cpuFallback.GetTransferSnapshot().localToHostPendingBytes;
    return n;
}

uint32_t GSVulkanBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    syncGpuToCpuUnlocked();
    return m_cpuFallback.ReadVram(psm, base, bw, x, y);
}

void GSVulkanBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    syncGpuToCpuUnlocked();
    m_cpuFallback.WriteVram(psm, base, bw, x, y, value);
    m_cpuVramDirtyForGpu = true;
}

void GSVulkanBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    syncGpuToCpuUnlocked();
    if (!m_vram || m_vramSize == 0u)
    {
        out.clear();
        return;
    }
    out.assign(m_vram, m_vram + m_vramSize);
}

GSTransferSnapshot GSVulkanBackend::GetTransferSnapshot() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_transferState;
}
