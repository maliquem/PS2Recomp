#include "ps2_vk_presenter.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include "imgui.h"
#include "imgui_impl_vulkan.h"
#include "ps2_fmv.h"
#include "runtime/gs/gs_vulkan_shaders.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace
{
#define PS2X_VK_INSTANCE_FNS(X)                    \
    X(vkGetDeviceProcAddr)                         \
    X(vkGetPhysicalDeviceSurfaceSupportKHR)        \
    X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR)   \
    X(vkGetPhysicalDeviceSurfaceFormatsKHR)        \
    X(vkGetPhysicalDeviceSurfacePresentModesKHR)   \
    X(vkGetPhysicalDeviceMemoryProperties)         \
    X(vkGetPhysicalDeviceQueueFamilyProperties)    \
    X(vkDestroySurfaceKHR)

#define PS2X_VK_DEVICE_FNS(X)     \
    X(vkCreateSwapchainKHR)       \
    X(vkDestroySwapchainKHR)      \
    X(vkGetSwapchainImagesKHR)    \
    X(vkAcquireNextImageKHR)      \
    X(vkQueuePresentKHR)          \
    X(vkQueueSubmit)              \
    X(vkQueueWaitIdle)            \
    X(vkCreateImageView)          \
    X(vkDestroyImageView)         \
    X(vkCreateRenderPass)         \
    X(vkDestroyRenderPass)        \
    X(vkCreateFramebuffer)        \
    X(vkDestroyFramebuffer)       \
    X(vkCreateCommandPool)        \
    X(vkDestroyCommandPool)       \
    X(vkAllocateCommandBuffers)   \
    X(vkBeginCommandBuffer)       \
    X(vkEndCommandBuffer)         \
    X(vkCmdPipelineBarrier)       \
    X(vkCmdCopyBufferToImage)     \
    X(vkCmdBlitImage)             \
    X(vkCmdClearColorImage)       \
    X(vkCmdBeginRenderPass)       \
    X(vkCmdEndRenderPass)         \
    X(vkCreateSemaphore)          \
    X(vkDestroySemaphore)         \
    X(vkCreateFence)              \
    X(vkDestroyFence)             \
    X(vkWaitForFences)            \
    X(vkResetFences)              \
    X(vkCreateImage)              \
    X(vkDestroyImage)             \
    X(vkGetImageMemoryRequirements) \
    X(vkAllocateMemory)           \
    X(vkFreeMemory)               \
    X(vkBindImageMemory)          \
    X(vkCreateBuffer)             \
    X(vkDestroyBuffer)            \
    X(vkGetBufferMemoryRequirements) \
    X(vkBindBufferMemory)         \
    X(vkMapMemory)                \
    X(vkCmdCopyImageToBuffer)     \
    X(vkCreateShaderModule)       \
    X(vkDestroyShaderModule)      \
    X(vkCreateDescriptorSetLayout) \
    X(vkDestroyDescriptorSetLayout) \
    X(vkCreatePipelineLayout)     \
    X(vkDestroyPipelineLayout)    \
    X(vkCreateComputePipelines)   \
    X(vkDestroyPipeline)          \
    X(vkCreateDescriptorPool)     \
    X(vkDestroyDescriptorPool)    \
    X(vkAllocateDescriptorSets)   \
    X(vkUpdateDescriptorSets)     \
    X(vkCmdBindPipeline)          \
    X(vkCmdBindDescriptorSets)    \
    X(vkCmdPushConstants)         \
    X(vkCmdDispatch)

    struct Fns
    {
        PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
#define PS2X_DECL(fn) PFN_##fn fn = nullptr;
        PS2X_VK_INSTANCE_FNS(PS2X_DECL)
        PS2X_VK_DEVICE_FNS(PS2X_DECL)
#undef PS2X_DECL
    };


    // Push constants of fmv_scale.comp.
    struct FmvParams
    {
        int32_t srcW, srcH, dstW, dstH;
        uint32_t filter, deband;
    };

    // §4 extension point: post-process passes run here on the scene image, between the GS copy and the
    // letterbox blit. None exist yet, so the pass is named but disabled.
    constexpr bool kPostProcessEnabled = false;
}

struct VkPresenter::Impl
{
    Fns vk{};
    SDL_Window *window = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    std::mutex *queueLock = nullptr;
    bool vsync = false;

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSurfaceFormatKHR format{};
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    VkPresentModeKHR unsyncedMode = VK_PRESENT_MODE_FIFO_KHR; // best non-vsync mode the surface offers
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkExtent2D extent{};
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    std::vector<VkFramebuffer> framebuffers;
    std::vector<VkSemaphore> renderDone; // per swapchain image: present may still wait on it after the fence
    VkRenderPass renderPass = VK_NULL_HANDLE;

    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool inFlight = false;

    VkImage scene = VK_NULL_HANDLE;
    VkDeviceMemory sceneMemory = VK_NULL_HANDLE;
    VkImageView sceneView = VK_NULL_HANDLE;
    uint32_t sceneCapW = 0, sceneCapH = 0;
    uint32_t sceneW = 0, sceneH = 0;
    bool sceneValid = false;
    bool sceneInitialized = false;
    bool sceneMovie = false; // the scene holds an MPEG movie frame (kept for redraws of the same scene)

    // FMV steps 2/5 (fmv_scale.comp): a movie frame is scaled into fmvOut, sized to the letterbox rect, which is
    // then copied 1:1. Built on the first movie frame; if anything fails, movies keep the nearest blit.
    VkDescriptorSetLayout fmvSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout fmvLayout = VK_NULL_HANDLE;
    VkPipeline fmvPipeline = VK_NULL_HANDLE;
    VkDescriptorPool fmvPool = VK_NULL_HANDLE;
    VkDescriptorSet fmvSet = VK_NULL_HANDLE;
    bool fmvSetStale = true; // scene or fmvOut was recreated since the set was written
    bool fmvFailed = false;
    bool fmvLogged = false; // last pass state logged: each movie start and end shows once
    VkImage fmvOut = VK_NULL_HANDLE;
    VkDeviceMemory fmvOutMemory = VK_NULL_HANDLE;
    VkImageView fmvOutView = VK_NULL_HANDLE;
    uint32_t fmvOutW = 0, fmvOutH = 0;

    bool imgui = false;

    // PS2X_PRESENT_DUMP=<prefix>: at present number PS2X_PRESENT_DUMP_AT (default 600) write the swapchain
    // image as presented and the scene it was blitted from, to check colors on the presented pixels.
    std::string dumpPrefix;
    uint64_t dumpAt = 0, presentCount = 0;
    std::string dumpReq; // PS2X_PRESENT_DUMP_REQ: while this file exists, dump to the prefix it holds, then delete it
    VkBuffer dumpBuffer = VK_NULL_HANDLE;
    VkDeviceMemory dumpMemory = VK_NULL_HANDLE;

    bool load(void *gipa)
    {
        vk.vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(gipa);
        if (!vk.vkGetInstanceProcAddr)
            return false;
#define PS2X_LOAD_I(fn)                                                                \
    vk.fn = reinterpret_cast<PFN_##fn>(vk.vkGetInstanceProcAddr(instance, #fn));       \
    if (!vk.fn)                                                                        \
        return false;
        PS2X_VK_INSTANCE_FNS(PS2X_LOAD_I)
#undef PS2X_LOAD_I
#define PS2X_LOAD_D(fn)                                                                \
    vk.fn = reinterpret_cast<PFN_##fn>(vk.vkGetDeviceProcAddr(device, #fn));           \
    if (!vk.fn)                                                                        \
        return false;
        PS2X_VK_DEVICE_FNS(PS2X_LOAD_D)
#undef PS2X_LOAD_D
        return true;
    }

    std::unique_lock<std::mutex> lockQueue()
    {
        return queueLock ? std::unique_lock<std::mutex>(*queueLock) : std::unique_lock<std::mutex>();
    }

    void waitFence()
    {
        if (!inFlight)
            return;
        vk.vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        inFlight = false;
    }

    void destroySwapchainViews()
    {
        for (VkFramebuffer fb : framebuffers)
            vk.vkDestroyFramebuffer(device, fb, nullptr);
        for (VkImageView v : views)
            vk.vkDestroyImageView(device, v, nullptr);
        for (VkSemaphore s : renderDone)
            vk.vkDestroySemaphore(device, s, nullptr);
        framebuffers.clear();
        views.clear();
        renderDone.clear();
        images.clear();
    }

    // False when the window has no area (minimized): nothing to present until it comes back.
    bool createSwapchain()
    {
        VkSurfaceCapabilitiesKHR caps{};
        if (vk.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &caps) != VK_SUCCESS)
            return false;
        int pw = 0, ph = 0;
        SDL_GetWindowSizeInPixels(window, &pw, &ph);
        VkExtent2D want = caps.currentExtent;
        if (want.width == UINT32_MAX)
        {
            want.width = std::clamp<uint32_t>(static_cast<uint32_t>(std::max(pw, 0)), caps.minImageExtent.width, caps.maxImageExtent.width);
            want.height = std::clamp<uint32_t>(static_cast<uint32_t>(std::max(ph, 0)), caps.minImageExtent.height, caps.maxImageExtent.height);
        }
        if (want.width == 0 || want.height == 0)
            return false;

        {
            auto lock = lockQueue();
            vk.vkQueueWaitIdle(queue); // old renderDone semaphores may still be waited on by a present
        }
        destroySwapchainViews();

        uint32_t imageCount = std::max(caps.minImageCount, presentMode == VK_PRESENT_MODE_MAILBOX_KHR ? 3u : 2u);
        if (caps.maxImageCount != 0)
            imageCount = std::min(imageCount, caps.maxImageCount);

        VkCompositeAlphaFlagBitsKHR alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        if (!(caps.supportedCompositeAlpha & alpha))
            alpha = static_cast<VkCompositeAlphaFlagBitsKHR>(caps.supportedCompositeAlpha & (~caps.supportedCompositeAlpha + 1u));

        if (dumpAt && !(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
        {
            std::fprintf(stderr, "[host] present dump disabled: swapchain cannot be a transfer source\n");
            dumpAt = 0;
        }
        VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        ci.surface = surface;
        ci.minImageCount = imageCount;
        ci.imageFormat = format.format;
        ci.imageColorSpace = format.colorSpace;
        ci.imageExtent = want;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                        (dumpAt ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0u);
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform = caps.currentTransform;
        ci.compositeAlpha = alpha;
        ci.presentMode = presentMode;
        ci.clipped = VK_TRUE;
        ci.oldSwapchain = swapchain;
        VkSwapchainKHR created = VK_NULL_HANDLE;
        VkResult r = vk.vkCreateSwapchainKHR(device, &ci, nullptr, &created);
        if (swapchain)
            vk.vkDestroySwapchainKHR(device, swapchain, nullptr);
        swapchain = created;
        if (r != VK_SUCCESS)
        {
            std::fprintf(stderr, "[host] vkCreateSwapchainKHR failed (%d)\n", static_cast<int>(r));
            swapchain = VK_NULL_HANDLE;
            return false;
        }
        extent = want;

        uint32_t count = 0;
        vk.vkGetSwapchainImagesKHR(device, swapchain, &count, nullptr);
        images.resize(count);
        vk.vkGetSwapchainImagesKHR(device, swapchain, &count, images.data());
        views.resize(count);
        framebuffers.resize(count);
        renderDone.resize(count);
        for (uint32_t i = 0; i < count; ++i)
        {
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = images[i];
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = format.format;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vk.vkCreateImageView(device, &vi, nullptr, &views[i]);
            VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            fi.renderPass = renderPass;
            fi.attachmentCount = 1;
            fi.pAttachments = &views[i];
            fi.width = extent.width;
            fi.height = extent.height;
            fi.layers = 1;
            vk.vkCreateFramebuffer(device, &fi, nullptr, &framebuffers[i]);
            VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            vk.vkCreateSemaphore(device, &si, nullptr, &renderDone[i]);
        }
        return true;
    }

    // RGBA8 device-local image and its view (storage: the FMV pass reads the scene and writes fmvOut).
    bool createImage(uint32_t w, uint32_t h, VkImageUsageFlags usage, VkImage &image, VkDeviceMemory &memory,
                     VkImageView &view)
    {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.format = VK_FORMAT_R8G8B8A8_UNORM;
        ii.extent = {w, h, 1};
        ii.mipLevels = 1;
        ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = usage;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vk.vkCreateImage(device, &ii, nullptr, &image) != VK_SUCCESS)
            return false;
        VkMemoryRequirements req{};
        vk.vkGetImageMemoryRequirements(device, image, &req);
        VkPhysicalDeviceMemoryProperties props{};
        vk.vkGetPhysicalDeviceMemoryProperties(physicalDevice, &props);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < props.memoryTypeCount && type == UINT32_MAX; ++i)
            if ((req.memoryTypeBits & (1u << i)) && (props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
                type = i;
        for (uint32_t i = 0; i < props.memoryTypeCount && type == UINT32_MAX; ++i)
            if (req.memoryTypeBits & (1u << i))
                type = i;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        if (type == UINT32_MAX || vk.vkAllocateMemory(device, &ai, nullptr, &memory) != VK_SUCCESS)
            return false;
        vk.vkBindImageMemory(device, image, memory, 0);
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = VK_FORMAT_R8G8B8A8_UNORM;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        return vk.vkCreateImageView(device, &vi, nullptr, &view) == VK_SUCCESS;
    }

    void destroyImage(VkImage &image, VkDeviceMemory &memory, VkImageView &view)
    {
        if (view)
            vk.vkDestroyImageView(device, view, nullptr);
        if (image)
            vk.vkDestroyImage(device, image, nullptr);
        if (memory)
            vk.vkFreeMemory(device, memory, nullptr);
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        view = VK_NULL_HANDLE;
    }

    bool ensureScene(uint32_t w, uint32_t h)
    {
        if (scene && w <= sceneCapW && h <= sceneCapH)
            return true;
        destroyImage(scene, sceneMemory, sceneView);
        fmvSetStale = true;
        sceneCapW = std::max(w, 640u);
        sceneCapH = std::max(h, 512u);
        sceneInitialized = false;
        if (createImage(sceneCapW, sceneCapH,
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                        scene, sceneMemory, sceneView))
            return true;
        destroyImage(scene, sceneMemory, sceneView); // half-built: next call starts over
        return false;
    }

    // Pipeline, set layout, pool and set of fmv_scale.comp, built once; on error never retried (nearest blit).
    bool ensureFmvPipeline()
    {
        if (fmvSet)
            return true;
        if (fmvFailed)
            return false;
        fmvFailed = true;
        VkDescriptorSetLayoutBinding b[2]{};
        for (uint32_t i = 0; i < 2; ++i)
        {
            b[i].binding = i;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = 2;
        li.pBindings = b;
        if (vk.vkCreateDescriptorSetLayout(device, &li, nullptr, &fmvSetLayout) != VK_SUCCESS)
            return false;
        const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FmvParams)};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &fmvSetLayout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &range;
        if (vk.vkCreatePipelineLayout(device, &pli, nullptr, &fmvLayout) != VK_SUCCESS)
            return false;
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize = sizeof(ps2vk_shaders::kFmvScaleSpv);
        mi.pCode = ps2vk_shaders::kFmvScaleSpv;
        VkShaderModule module = VK_NULL_HANDLE;
        if (vk.vkCreateShaderModule(device, &mi, nullptr, &module) != VK_SUCCESS)
            return false;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = module;
        ci.stage.pName = "main";
        ci.layout = fmvLayout;
        const VkResult res = vk.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &fmvPipeline);
        vk.vkDestroyShaderModule(device, module, nullptr);
        if (res != VK_SUCCESS)
        {
            fmvPipeline = VK_NULL_HANDLE;
            return false;
        }
        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &size;
        if (vk.vkCreateDescriptorPool(device, &dpi, nullptr, &fmvPool) != VK_SUCCESS)
            return false;
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = fmvPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &fmvSetLayout;
        if (vk.vkAllocateDescriptorSets(device, &ai, &fmvSet) != VK_SUCCESS)
        {
            fmvSet = VK_NULL_HANDLE;
            return false;
        }
        fmvFailed = false;
        return true;
    }

    // fmvOut sized to the letterbox rect, and the set pointing at the current scene and fmvOut.
    // Called before the command buffer uses either, and after waitFence: nothing in flight reads the old ones.
    bool ensureFmvTarget(uint32_t w, uint32_t h)
    {
        if (fmvOut && (w != fmvOutW || h != fmvOutH))
            destroyImage(fmvOut, fmvOutMemory, fmvOutView);
        if (!fmvOut)
        {
            fmvSetStale = true;
            if (!createImage(w, h, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, fmvOut, fmvOutMemory,
                             fmvOutView))
            {
                destroyImage(fmvOut, fmvOutMemory, fmvOutView);
                return false;
            }
            fmvOutW = w;
            fmvOutH = h;
        }
        if (fmvSetStale)
        {
            const VkDescriptorImageInfo img[2] = {{VK_NULL_HANDLE, sceneView, VK_IMAGE_LAYOUT_GENERAL},
                                                  {VK_NULL_HANDLE, fmvOutView, VK_IMAGE_LAYOUT_GENERAL}};
            VkWriteDescriptorSet wr[2]{};
            for (uint32_t i = 0; i < 2; ++i)
            {
                wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wr[i].dstSet = fmvSet;
                wr[i].dstBinding = i;
                wr[i].descriptorCount = 1;
                wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                wr[i].pImageInfo = &img[i];
            }
            vk.vkUpdateDescriptorSets(device, 2, wr, 0, nullptr);
            fmvSetStale = false;
        }
        return true;
    }

    void destroyFmv()
    {
        destroyImage(fmvOut, fmvOutMemory, fmvOutView);
        if (fmvPool)
            vk.vkDestroyDescriptorPool(device, fmvPool, nullptr); // frees the set
        if (fmvPipeline)
            vk.vkDestroyPipeline(device, fmvPipeline, nullptr);
        if (fmvLayout)
            vk.vkDestroyPipelineLayout(device, fmvLayout, nullptr);
        if (fmvSetLayout)
            vk.vkDestroyDescriptorSetLayout(device, fmvSetLayout, nullptr);
    }

    static void imageBarrier(Fns &vk, VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
                             VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                             VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
    {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = srcAccess;
        b.dstAccessMask = dstAccess;
        b.oldLayout = from;
        b.newLayout = to;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vk.vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
    }

    void recordSceneCopy(const GSGpuFrame &frame)
    {
        sceneMovie = frame.movie;
        const VkImageLayout from = sceneInitialized ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
        imageBarrier(vk, cmd, scene, from, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy region{};
        region.bufferRowLength = frame.rowPixels;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {frame.width, frame.height, 1};
        vk.vkCmdCopyBufferToImage(cmd, (VkBuffer)frame.buffer, scene,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        imageBarrier(vk, cmd, scene, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        sceneInitialized = true;
        sceneW = frame.width;
        sceneH = frame.height;
        sceneValid = true;
    }

    // Records the readback copies after the ImGui pass (image in PRESENT_SRC); false if no host buffer.
    bool recordDump(VkImage target)
    {
        const VkDeviceSize presentBytes = VkDeviceSize(extent.width) * extent.height * 4u;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = presentBytes + VkDeviceSize(sceneW) * sceneH * 4u;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vk.vkCreateBuffer(device, &bi, nullptr, &dumpBuffer) != VK_SUCCESS)
            return false;
        VkMemoryRequirements req{};
        vk.vkGetBufferMemoryRequirements(device, dumpBuffer, &req);
        VkPhysicalDeviceMemoryProperties props{};
        vk.vkGetPhysicalDeviceMemoryProperties(physicalDevice, &props);
        const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = UINT32_MAX;
        for (uint32_t i = 0; i < props.memoryTypeCount && ai.memoryTypeIndex == UINT32_MAX; ++i)
            if ((req.memoryTypeBits & (1u << i)) && (props.memoryTypes[i].propertyFlags & want) == want)
                ai.memoryTypeIndex = i;
        if (ai.memoryTypeIndex == UINT32_MAX || vk.vkAllocateMemory(device, &ai, nullptr, &dumpMemory) != VK_SUCCESS)
        {
            vk.vkDestroyBuffer(device, dumpBuffer, nullptr);
            dumpBuffer = VK_NULL_HANDLE;
            return false;
        }
        vk.vkBindBufferMemory(device, dumpBuffer, dumpMemory, 0);

        imageBarrier(vk, cmd, target, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                     // ALL_COMMANDS: the render pass's final layout change runs at BOTTOM_OF_PIPE.
                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent.width, extent.height, 1};
        vk.vkCmdCopyImageToBuffer(cmd, target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dumpBuffer, 1, &region);
        region.bufferOffset = presentBytes;
        region.imageExtent = {sceneW, sceneH, 1};
        vk.vkCmdCopyImageToBuffer(cmd, scene, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dumpBuffer, 1, &region);
        imageBarrier(vk, cmd, target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     VK_ACCESS_TRANSFER_READ_BIT, 0,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vk.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0, nullptr);
        return true;
    }

    // Binary PPMs (RGB); the swapchain may be BGRA.
    void writeDump(const std::string &dumpPrefix)
    {
        waitFence();
        void *mapped = nullptr;
        if (vk.vkMapMemory(device, dumpMemory, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS)
        {
            const bool bgra = format.format == VK_FORMAT_B8G8R8A8_UNORM || format.format == VK_FORMAT_B8G8R8A8_SRGB;
            auto write = [](const std::string &path, const uint8_t *px, uint32_t w, uint32_t h, bool swap) {
                std::FILE *f = std::fopen(path.c_str(), "wb");
                if (!f)
                    return;
                std::fprintf(f, "P6\n%u %u\n255\n", w, h);
                std::vector<uint8_t> row(size_t(w) * 3u);
                for (uint32_t y = 0; y < h; ++y)
                {
                    for (uint32_t x = 0; x < w; ++x)
                    {
                        const uint8_t *s = px + (size_t(y) * w + x) * 4u;
                        row[x * 3 + 0] = swap ? s[2] : s[0];
                        row[x * 3 + 1] = s[1];
                        row[x * 3 + 2] = swap ? s[0] : s[2];
                    }
                    std::fwrite(row.data(), 1, row.size(), f);
                }
                std::fclose(f);
            };
            const auto *bytes = static_cast<const uint8_t *>(mapped);
            write(dumpPrefix + ".present.ppm", bytes, extent.width, extent.height, bgra);
            write(dumpPrefix + ".scene.ppm", bytes + size_t(extent.width) * extent.height * 4u, sceneW, sceneH, false);
            std::fprintf(stderr, "[host] present dump %s (format=%d colorspace=%d)\n", dumpPrefix.c_str(),
                         static_cast<int>(format.format), static_cast<int>(format.colorSpace));
        }
        freeDump();
    }

    void freeDump()
    {
        vk.vkDestroyBuffer(device, dumpBuffer, nullptr);
        vk.vkFreeMemory(device, dumpMemory, nullptr); // frees the mapping too
        dumpBuffer = VK_NULL_HANDLE;
        dumpMemory = VK_NULL_HANDLE;
    }

    void recordPostProcess()
    {
        if (!kPostProcessEnabled)
            return;
    }

    // FMV steps 2/5: scales the movie scene into fmvOut (rect size); true when the letterbox should copy fmvOut.
    bool recordFmvScale(const OutputRect &r)
    {
        const int filter = ps2fmv::scaleFilter();
        const bool on = sceneValid && sceneMovie && filter != ps2fmv::Pixel;
        const uint32_t w = static_cast<uint32_t>(r.w), h = static_cast<uint32_t>(r.h);
        const bool run = on && w && h && ensureFmvPipeline() && ensureFmvTarget(w, h);
        if (run != fmvLogged)
        {
            std::fprintf(stderr, "[host] FMV scale pass %s (filter=%s %ux%u -> %ux%u deband=%d)\n", run ? "on" : "off",
                         ps2fmv::kScaleNames[filter], sceneW, sceneH, w, h, ps2fmv::deband.load() ? 1 : 0);
            fmvLogged = run;
        }
        if (!run)
            return false;
        imageBarrier(vk, cmd, scene, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        imageBarrier(vk, cmd, fmvOut, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                     VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        vk.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fmvPipeline);
        vk.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fmvLayout, 0, 1, &fmvSet, 0, nullptr);
        const FmvParams params{static_cast<int32_t>(sceneW), static_cast<int32_t>(sceneH), static_cast<int32_t>(w),
                               static_cast<int32_t>(h), static_cast<uint32_t>(filter),
                               ps2fmv::deband.load(std::memory_order_relaxed) ? 1u : 0u};
        vk.vkCmdPushConstants(cmd, fmvLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
        vk.vkCmdDispatch(cmd, (w + 15) / 16, (h + 7) / 8, 1);
        imageBarrier(vk, cmd, fmvOut, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        // The next frame's copy and the dump expect the scene back in TRANSFER_SRC.
        imageBarrier(vk, cmd, scene, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        return true;
    }

    void recordLetterbox(VkImage target)
    {
        const OutputRect r = letterbox43(static_cast<float>(extent.width), static_cast<float>(extent.height));
        const bool scaled = recordFmvScale(r);
        imageBarrier(vk, cmd, target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     0, VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkClearColorValue black{};
        black.float32[3] = 1.0f;
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vk.vkCmdClearColorImage(cmd, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
        if (sceneValid)
        {
            const int32_t x0 = static_cast<int32_t>(r.x);
            const int32_t y0 = static_cast<int32_t>(r.y);
            const float dw = r.w, dh = r.h;
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.srcOffsets[1] = {static_cast<int32_t>(scaled ? fmvOutW : sceneW),
                                  static_cast<int32_t>(scaled ? fmvOutH : sceneH), 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.dstOffsets[0] = {x0, y0, 0};
            blit.dstOffsets[1] = {x0 + static_cast<int32_t>(dw), y0 + static_cast<int32_t>(dh), 1};
            // Point filtering, as the raylib path drew it; a filtered movie frame is already rect-sized (1:1).
            vk.vkCmdBlitImage(cmd, scaled ? fmvOut : scene, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
        }
        imageBarrier(vk, cmd, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    }
};

std::string readDumpRequest(const char *path)
{
    std::FILE *f = std::fopen(path, "rb");
    if (!f)
        return {};
    char prefix[512]{};
    const bool ok = std::fgets(prefix, sizeof(prefix), f) != nullptr;
    std::fclose(f);
    prefix[std::strcspn(prefix, "\r\n")] = 0;
    if (!ok || !prefix[0])
        std::remove(path);
    return prefix;
}

VkPresenter::VkPresenter() : m(std::make_unique<Impl>()) {}
VkPresenter::~VkPresenter() { shutdown(); }

bool VkPresenter::init(SDL_Window *window, const GSHostGpuContext &ctx, bool vsync, std::string &error)
{
    Impl &p = *m;
    p.window = window;
    p.instance = static_cast<VkInstance>(ctx.instance);
    p.physicalDevice = static_cast<VkPhysicalDevice>(ctx.physicalDevice);
    p.device = static_cast<VkDevice>(ctx.device);
    p.queue = static_cast<VkQueue>(ctx.queue);
    p.queueFamily = ctx.queueFamily;
    p.queueLock = ctx.queueLock;
    p.vsync = vsync;
    if (const char *dump = std::getenv("PS2X_PRESENT_DUMP"); dump && *dump)
    {
        const char *at = std::getenv("PS2X_PRESENT_DUMP_AT");
        p.dumpPrefix = dump;
        p.dumpAt = (at && *at) ? std::strtoull(at, nullptr, 10) : 600u;
    }
    if (const char *req = std::getenv("PS2X_PRESENT_DUMP_REQ"); req && *req)
    {
        p.dumpReq = req;
        if (!p.dumpAt)
            p.dumpAt = UINT64_MAX; // never reached: only enables the transfer-source swapchain the dump needs
    }
    if (!ctx.swapchainEnabled || !p.instance || !p.device || !p.queue)
    {
        error = "GS Vulkan device has no swapchain support";
        return false;
    }
    if (!p.load(ctx.getInstanceProcAddr))
    {
        error = "missing Vulkan surface/swapchain entry points";
        return false;
    }
    if (!SDL_Vulkan_CreateSurface(window, p.instance, nullptr, &p.surface))
    {
        error = std::string("SDL_Vulkan_CreateSurface: ") + SDL_GetError();
        return false;
    }
    // The GS picks its queue family; the blits and the ImGui render pass also need graphics.
    uint32_t familyCount = 0;
    p.vk.vkGetPhysicalDeviceQueueFamilyProperties(p.physicalDevice, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    p.vk.vkGetPhysicalDeviceQueueFamilyProperties(p.physicalDevice, &familyCount, families.data());
    if (p.queueFamily >= familyCount || !(families[p.queueFamily].queueFlags & VK_QUEUE_GRAPHICS_BIT))
    {
        error = "GS queue family has no graphics support";
        return false;
    }
    VkBool32 supported = VK_FALSE;
    p.vk.vkGetPhysicalDeviceSurfaceSupportKHR(p.physicalDevice, p.queueFamily, p.surface, &supported);
    if (!supported)
    {
        error = "GS queue family cannot present to the window";
        return false;
    }

    uint32_t count = 0;
    p.vk.vkGetPhysicalDeviceSurfaceFormatsKHR(p.physicalDevice, p.surface, &count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(count);
    p.vk.vkGetPhysicalDeviceSurfaceFormatsKHR(p.physicalDevice, p.surface, &count, formats.data());
    if (formats.empty())
    {
        error = "surface reports no formats";
        return false;
    }
    // UNORM on purpose: the GS output is already display-encoded, as raylib showed it.
    p.format = formats[0];
    for (const VkSurfaceFormatKHR &f : formats)
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            p.format = f;
            break;
        }
    // The dump reads 4 bytes per pixel: other formats (e.g. 16-bit float) would overrun its buffer.
    if (p.dumpAt && p.format.format != VK_FORMAT_B8G8R8A8_UNORM && p.format.format != VK_FORMAT_R8G8B8A8_UNORM)
    {
        std::fprintf(stderr, "[host] present dump disabled: swapchain format %d is not 8-bit RGBA\n",
                     static_cast<int>(p.format.format));
        p.dumpAt = 0;
    }

    p.vk.vkGetPhysicalDeviceSurfacePresentModesKHR(p.physicalDevice, p.surface, &count, nullptr);
    std::vector<VkPresentModeKHR> modes(count);
    p.vk.vkGetPhysicalDeviceSurfacePresentModesKHR(p.physicalDevice, p.surface, &count, modes.data());
    auto has = [&](VkPresentModeKHR mode) { return std::find(modes.begin(), modes.end(), mode) != modes.end(); };
    if (has(VK_PRESENT_MODE_MAILBOX_KHR))
        p.unsyncedMode = VK_PRESENT_MODE_MAILBOX_KHR;
    else if (has(VK_PRESENT_MODE_IMMEDIATE_KHR))
        p.unsyncedMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    p.presentMode = vsync ? VK_PRESENT_MODE_FIFO_KHR : p.unsyncedMode;

    VkAttachmentDescription att{};
    att.format = p.format.format;
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &ref;
    VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rp.attachmentCount = 1;
    rp.pAttachments = &att;
    rp.subpassCount = 1;
    rp.pSubpasses = &sub;
    if (p.vk.vkCreateRenderPass(p.device, &rp, nullptr, &p.renderPass) != VK_SUCCESS)
    {
        error = "vkCreateRenderPass failed";
        return false;
    }

    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pi.queueFamilyIndex = p.queueFamily;
    VkCommandBufferAllocateInfo ci{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (p.vk.vkCreateCommandPool(p.device, &pi, nullptr, &p.pool) != VK_SUCCESS ||
        (ci.commandPool = p.pool, ci.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, ci.commandBufferCount = 1,
         p.vk.vkAllocateCommandBuffers(p.device, &ci, &p.cmd) != VK_SUCCESS) ||
        p.vk.vkCreateSemaphore(p.device, &si, nullptr, &p.acquired) != VK_SUCCESS ||
        p.vk.vkCreateFence(p.device, &fi, nullptr, &p.fence) != VK_SUCCESS || !p.ensureScene(640, 512))
    {
        error = "presenter command/sync objects failed";
        return false;
    }
    m_resizePending = true;
    std::fprintf(stderr, "[host] Vulkan swapchain format=%d colorspace=%d present_mode=%d vsync=%d\n",
                 static_cast<int>(p.format.format), static_cast<int>(p.format.colorSpace),
                 static_cast<int>(p.presentMode), vsync ? 1 : 0);
    return true;
}

void VkPresenter::shutdown()
{
    if (!m || !m->device)
        return;
    Impl &p = *m;
    p.waitFence();
    {
        auto lock = p.lockQueue();
        p.vk.vkQueueWaitIdle(p.queue);
    }
    imguiShutdown();
    p.destroySwapchainViews();
    if (p.swapchain)
        p.vk.vkDestroySwapchainKHR(p.device, p.swapchain, nullptr);
    if (p.surface)
        p.vk.vkDestroySurfaceKHR(p.instance, p.surface, nullptr);
    p.destroyFmv();
    p.destroyImage(p.scene, p.sceneMemory, p.sceneView);
    if (p.renderPass)
        p.vk.vkDestroyRenderPass(p.device, p.renderPass, nullptr);
    if (p.fence)
        p.vk.vkDestroyFence(p.device, p.fence, nullptr);
    if (p.acquired)
        p.vk.vkDestroySemaphore(p.device, p.acquired, nullptr);
    if (p.pool)
        p.vk.vkDestroyCommandPool(p.device, p.pool, nullptr);
    m = std::make_unique<Impl>();
}

void VkPresenter::waitFrame()
{
    if (m && m->device)
        m->waitFence();
}

void VkPresenter::present(const GSGpuFrame *frame, ImDrawData *imgui)
{
    Impl &p = *m;
    if (!p.device)
        return;
    p.waitFence();
    if (m_resizePending || !p.swapchain)
    {
        if (!p.createSwapchain())
            return; // minimized; the scene copy waits for the next fresh field
        m_resizePending = false;
    }

    uint32_t index = 0;
    VkResult r = p.vk.vkAcquireNextImageKHR(p.device, p.swapchain, UINT64_MAX, p.acquired, VK_NULL_HANDLE, &index);
    if (r == VK_ERROR_OUT_OF_DATE_KHR)
    {
        m_resizePending = true;
        return;
    }
    if (r == VK_SUBOPTIMAL_KHR)
        m_resizePending = true;
    else if (r != VK_SUCCESS)
        return;

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    p.vk.vkBeginCommandBuffer(p.cmd, &bi);
    if (frame && *frame && p.ensureScene(frame->width, frame->height))
        p.recordSceneCopy(*frame);
    else if (frame)
        p.sceneValid = false; // a latched tick without a field shows black, as the raylib path did
    p.recordPostProcess();
    p.recordLetterbox(p.images[index]);

    // Shared queue: ImGui's texture uploads submit on it too, so hold the lock from here.
    auto lock = p.lockQueue();
    VkRenderPassBeginInfo rb{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rb.renderPass = p.renderPass;
    rb.framebuffer = p.framebuffers[index];
    rb.renderArea.extent = p.extent;
    p.vk.vkCmdBeginRenderPass(p.cmd, &rb, VK_SUBPASS_CONTENTS_INLINE);
    if (imgui && p.imgui)
        ImGui_ImplVulkan_RenderDrawData(imgui, p.cmd);
    p.vk.vkCmdEndRenderPass(p.cmd);
    const bool dumpNow = p.dumpAt && ++p.presentCount == p.dumpAt;
    // dumpAt is zero when the swapchain cannot be read back. A request that cannot be served this present
    // (no scene, no host memory) stays in its file for the next one.
    const std::string request =
        (!dumpNow && p.dumpAt && !p.dumpReq.empty()) ? readDumpRequest(p.dumpReq.c_str()) : std::string();
    const bool dump = (dumpNow || !request.empty()) && p.sceneValid && p.recordDump(p.images[index]);
    if (dumpNow && !dump)
        std::fprintf(stderr, "[host] present dump skipped at present %llu (no scene or no host memory)\n",
                     static_cast<unsigned long long>(p.presentCount));
    p.vk.vkEndCommandBuffer(p.cmd);

    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &p.acquired;
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &p.cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &p.renderDone[index];
    p.vk.vkResetFences(p.device, 1, &p.fence);
    if (p.vk.vkQueueSubmit(p.queue, 1, &si, p.fence) != VK_SUCCESS)
    {
        std::fprintf(stderr, "[host] present submit failed\n");
        if (dump)
            p.freeDump();
        return;
    }
    p.inFlight = true;

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &p.renderDone[index];
    pi.swapchainCount = 1;
    pi.pSwapchains = &p.swapchain;
    pi.pImageIndices = &index;
    r = p.vk.vkQueuePresentKHR(p.queue, &pi);
    lock = {}; // writeDump waits on the fence; the queue is free again
    if (dump)
    {
        p.writeDump(request.empty() ? p.dumpPrefix : request);
        if (!request.empty())
            std::remove(p.dumpReq.c_str());
    }
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR)
        m_resizePending = true;
}

bool VkPresenter::imguiInit()
{
    Impl &p = *m;
    if (!p.device || p.imgui)
        return p.imgui;
    struct LoaderCtx
    {
        PFN_vkGetInstanceProcAddr gipa;
        VkInstance instance;
    } ctx{p.vk.vkGetInstanceProcAddr, p.instance};
    if (!ImGui_ImplVulkan_LoadFunctions(VK_API_VERSION_1_0, [](const char *name, void *user) {
            auto *c = static_cast<LoaderCtx *>(user);
            return c->gipa(c->instance, name);
        }, &ctx))
        return false;
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = VK_API_VERSION_1_0;
    info.Instance = p.instance;
    info.PhysicalDevice = p.physicalDevice;
    info.Device = p.device;
    info.QueueFamily = p.queueFamily;
    info.Queue = p.queue;
    info.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE;
    info.MinImageCount = 2;
    info.ImageCount = 3; // only paces ImGui's buffer/texture reuse; we keep one frame in flight
    info.PipelineInfoMain.RenderPass = p.renderPass;
    info.PipelineInfoMain.Subpass = 0;
    info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    p.imgui = ImGui_ImplVulkan_Init(&info);
    return p.imgui;
}

void VkPresenter::setVsync(bool vsync)
{
    const VkPresentModeKHR mode = vsync ? VK_PRESENT_MODE_FIFO_KHR : m->unsyncedMode;
    if (mode == m->presentMode)
        return;
    m->presentMode = mode;
    m->vsync = vsync;
    m_resizePending = true; // the next present recreates the swapchain with the new mode
}

void VkPresenter::imguiNewFrame()
{
    if (m->imgui)
        ImGui_ImplVulkan_NewFrame();
}

void VkPresenter::imguiShutdown()
{
    if (!m->imgui)
        return;
    m->waitFence();
    ImGui_ImplVulkan_Shutdown();
    m->imgui = false;
}
