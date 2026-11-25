#pragma once

#include <cstddef>
#include <cstdint>

// Minimal self-contained Vulkan 1.0 compute-only declarations + dynamic loader
// so PS2Runtime compiles cleanly on Windows and Linux without requiring the Vulkan SDK.

#if defined(_WIN32)
#define PS2_VKAPI_PTR __stdcall
#else
#define PS2_VKAPI_PTR
#endif

namespace ps2vk
{

using VkFlags = uint32_t;
using VkBool32 = uint32_t;
using VkDeviceSize = uint64_t;

using VkInstance = struct VkInstance_T *;
using VkPhysicalDevice = struct VkPhysicalDevice_T *;
using VkDevice = struct VkDevice_T *;
using VkQueue = struct VkQueue_T *;
using VkCommandBuffer = struct VkCommandBuffer_T *;

#if defined(__LP64__) || defined(_WIN64) || (defined(__x86_64__) && !defined(__ILP32__)) || defined(_M_X64) || defined(__aarch64__)
using VkBuffer = struct VkBuffer_T *;
using VkDeviceMemory = struct VkDeviceMemory_T *;
using VkShaderModule = struct VkShaderModule_T *;
using VkPipelineLayout = struct VkPipelineLayout_T *;
using VkPipeline = struct VkPipeline_T *;
using VkPipelineCache = struct VkPipelineCache_T *;
using VkDescriptorSetLayout = struct VkDescriptorSetLayout_T *;
using VkDescriptorPool = struct VkDescriptorPool_T *;
using VkDescriptorSet = struct VkDescriptorSet_T *;
using VkCommandPool = struct VkCommandPool_T *;
using VkFence = struct VkFence_T *;
using VkSampler = struct VkSampler_T *;
using VkBufferView = struct VkBufferView_T *;
#else
using VkBuffer = uint64_t;
using VkDeviceMemory = uint64_t;
using VkShaderModule = uint64_t;
using VkPipelineLayout = uint64_t;
using VkPipeline = uint64_t;
using VkPipelineCache = uint64_t;
using VkDescriptorSetLayout = uint64_t;
using VkDescriptorPool = uint64_t;
using VkDescriptorSet = uint64_t;
using VkCommandPool = uint64_t;
using VkFence = uint64_t;
using VkSampler = uint64_t;
using VkBufferView = uint64_t;
#endif

constexpr uint32_t VK_MAKE_VERSION(uint32_t major, uint32_t minor, uint32_t patch)
{
    return (major << 22) | (minor << 12) | patch;
}

constexpr uint32_t VK_API_VERSION_1_0 = VK_MAKE_VERSION(1, 0, 0);
constexpr VkBool32 VK_FALSE = 0;
constexpr VkBool32 VK_TRUE = 1;
constexpr uint32_t VK_QUEUE_FAMILY_IGNORED = ~0u;

enum VkResult : int32_t
{
    VK_SUCCESS = 0,
    VK_NOT_READY = 1,
    VK_TIMEOUT = 2,
    VK_ERROR_OUT_OF_HOST_MEMORY = -1,
    VK_ERROR_OUT_OF_DEVICE_MEMORY = -2,
    VK_ERROR_INITIALIZATION_FAILED = -3,
    VK_ERROR_DEVICE_LOST = -4,
};

enum VkStructureType : int32_t
{
    VK_STRUCTURE_TYPE_APPLICATION_INFO = 0,
    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1,
    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2,
    VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3,
    VK_STRUCTURE_TYPE_SUBMIT_INFO = 4,
    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5,
    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8,
    VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12,
    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16,
    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18,
    VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29,
    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32,
    VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34,
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35,
    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42,
    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER = 44,
    VK_STRUCTURE_TYPE_MEMORY_BARRIER = 46,
};

enum VkPhysicalDeviceType : int32_t
{
    VK_PHYSICAL_DEVICE_TYPE_OTHER = 0,
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2,
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3,
    VK_PHYSICAL_DEVICE_TYPE_CPU = 4,
};

enum VkSharingMode : int32_t
{
    VK_SHARING_MODE_EXCLUSIVE = 0,
};

enum VkDescriptorType : int32_t
{
    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7,
};

enum VkPipelineBindPoint : int32_t
{
    VK_PIPELINE_BIND_POINT_COMPUTE = 1,
};

enum VkCommandBufferLevel : int32_t
{
    VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0,
};

constexpr VkFlags VK_QUEUE_COMPUTE_BIT = 0x00000002u;

constexpr VkFlags VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001u;
constexpr VkFlags VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002u;
constexpr VkFlags VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004u;
constexpr VkFlags VK_MEMORY_PROPERTY_HOST_CACHED_BIT = 0x00000008u;

constexpr VkFlags VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001u;
constexpr VkFlags VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002u;
constexpr VkFlags VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020u;

constexpr VkFlags VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020u;
constexpr VkFlags VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x00000002u;
constexpr VkFlags VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001u;

constexpr VkFlags VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x00000800u;
constexpr VkFlags VK_PIPELINE_STAGE_TRANSFER_BIT = 0x00001000u;
constexpr VkFlags VK_PIPELINE_STAGE_HOST_BIT = 0x00004000u;

constexpr VkFlags VK_ACCESS_SHADER_READ_BIT = 0x00000020u;
constexpr VkFlags VK_ACCESS_SHADER_WRITE_BIT = 0x00000040u;
constexpr VkFlags VK_ACCESS_TRANSFER_READ_BIT = 0x00000800u;
constexpr VkFlags VK_ACCESS_TRANSFER_WRITE_BIT = 0x00001000u;
constexpr VkFlags VK_ACCESS_HOST_READ_BIT = 0x00002000u;

struct VkApplicationInfo
{
    VkStructureType sType;
    const void *pNext;
    const char *pApplicationName;
    uint32_t applicationVersion;
    const char *pEngineName;
    uint32_t engineVersion;
    uint32_t apiVersion;
};

struct VkInstanceCreateInfo
{
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    const VkApplicationInfo *pApplicationInfo;
    uint32_t enabledLayerCount;
    const char *const *ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char *const *ppEnabledExtensionNames;
};

struct VkPhysicalDeviceFeatures
{
    VkBool32 robustBufferAccess;
    VkBool32 fullDrawIndexUint32;
    VkBool32 imageCubeArray;
    VkBool32 independentBlend;
    VkBool32 geometryShader;
    VkBool32 tessellationShader;
    VkBool32 sampleRateShading;
    VkBool32 dualSrcBlend;
    VkBool32 logicOp;
    VkBool32 multiDrawIndirect;
    VkBool32 drawIndirectFirstInstance;
    VkBool32 depthClamp;
    VkBool32 depthBiasClamp;
    VkBool32 fillModeNonSolid;
    VkBool32 depthBounds;
    VkBool32 wideLines;
    VkBool32 largePoints;
    VkBool32 alphaToOne;
    VkBool32 multiViewport;
    VkBool32 samplerAnisotropy;
    VkBool32 textureCompressionETC2;
    VkBool32 textureCompressionASTC_LDR;
    VkBool32 textureCompressionBC;
    VkBool32 occlusionQueryPrecise;
    VkBool32 pipelineStatisticsQuery;
    VkBool32 vertexPipelineStoresAndAtomics;
    VkBool32 fragmentStoresAndAtomics;
    VkBool32 shaderTessellationAndGeometryPointSize;
    VkBool32 shaderImageGatherExtended;
    VkBool32 shaderStorageImageExtendedFormats;
    VkBool32 shaderStorageImageMultisample;
    VkBool32 shaderStorageImageReadWithoutFormat;
    VkBool32 shaderStorageImageWriteWithoutFormat;
    VkBool32 shaderUniformBufferArrayDynamicIndexing;
    VkBool32 shaderSampledImageArrayDynamicIndexing;
    VkBool32 shaderStorageBufferArrayDynamicIndexing;
    VkBool32 shaderStorageImageArrayDynamicIndexing;
    VkBool32 shaderClipDistance;
    VkBool32 shaderCullDistance;
    VkBool32 shaderFloat64;
    VkBool32 shaderInt64;
    VkBool32 shaderInt16;
    VkBool32 shaderResourceResidency;
    VkBool32 shaderResourceMinLod;
    VkBool32 sparseBinding;
    VkBool32 sparseResidencyBuffer;
    VkBool32 sparseResidencyImage2D;
    VkBool32 sparseResidencyImage3D;
    VkBool32 sparseResidency2Samples;
    VkBool32 sparseResidency4Samples;
    VkBool32 sparseResidency8Samples;
    VkBool32 sparseResidency16Samples;
    VkBool32 sparseResidencyAliased;
    VkBool32 variableMultisampleRate;
    VkBool32 inheritedQueries;
};

struct VkPhysicalDeviceLimits
{
    uint8_t raw[504];
};

struct VkPhysicalDeviceSparseProperties
{
    VkBool32 residencyStandard2DBlockShape;
    VkBool32 residencyStandard2DMultisampleBlockShape;
    VkBool32 residencyStandard3DBlockShape;
    VkBool32 residencyAlignedMipSize;
    VkBool32 residencyNonResidentStrict;
};

struct VkPhysicalDeviceProperties
{
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    VkPhysicalDeviceType deviceType;
    char deviceName[256];
    uint8_t pipelineCacheUUID[16];
    VkPhysicalDeviceLimits limits;
    VkPhysicalDeviceSparseProperties sparseProperties;
};

struct VkExtent3D
{
    uint32_t width;
    uint32_t height;
    uint32_t depth;
};

struct VkQueueFamilyProperties
{
    VkFlags queueFlags;
    uint32_t queueCount;
    uint32_t timestampValidBits;
    VkExtent3D minImageTransferGranularity;
};

struct VkMemoryType
{
    VkFlags propertyFlags;
    uint32_t heapIndex;
};

struct VkMemoryHeap
{
    VkDeviceSize size;
    VkFlags flags;
};

struct VkPhysicalDeviceMemoryProperties
{
    uint32_t memoryTypeCount;
    VkMemoryType memoryTypes[32];
    uint32_t memoryHeapCount;
    VkMemoryHeap memoryHeaps[16];
};

struct VkDeviceQueueCreateInfo
{
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
    uint32_t queueCount;
    const float *pQueuePriorities;
};

struct VkDeviceCreateInfo
{
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    uint32_t queueCreateInfoCount;
    const VkDeviceQueueCreateInfo *pQueueCreateInfos;
    uint32_t enabledLayerCount;
    const char *const *ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char *const *ppEnabledExtensionNames;
    const VkPhysicalDeviceFeatures *pEnabledFeatures;
};

struct VkBufferCreateInfo
{
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    VkDeviceSize size;
    VkFlags usage;
    VkSharingMode sharingMode;
    uint32_t queueFamilyIndexCount;
    const uint32_t *pQueueFamilyIndices;
};

struct VkMemoryRequirements
{
    VkDeviceSize size;
    VkDeviceSize alignment;
    uint32_t memoryTypeBits;
};

struct VkMemoryAllocateInfo
{
    VkStructureType sType;
    const void *pNext;
    VkDeviceSize allocationSize;
    uint32_t memoryTypeIndex;
};

struct VkDescriptorSetLayoutBinding
{
    uint32_t binding;
    VkDescriptorType descriptorType;
    uint32_t descriptorCount;
    VkFlags stageFlags;
    const VkSampler *pImmutableSamplers;
};

struct VkDescriptorSetLayoutCreateInfo
{
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    uint32_t bindingCount;
    const VkDescriptorSetLayoutBinding *pBindings;
};

struct VkPushConstantRange
{
    VkFlags stageFlags;
    uint32_t offset;
    uint32_t size;
};

struct VkPipelineLayoutCreateInfo
{
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    uint32_t setLayoutCount;
    const VkDescriptorSetLayout *pSetLayouts;
    uint32_t pushConstantRangeCount;
    const VkPushConstantRange *pPushConstantRanges;
};

struct VkShaderModuleCreateInfo
{
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    size_t codeSize;
    const uint32_t *pCode;
};

struct VkPipelineShaderStageCreateInfo
{
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    VkFlags stage;
    VkShaderModule module;
    const char *pName;
    const void *pSpecializationInfo;
};

struct VkComputePipelineCreateInfo
{
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    VkPipelineShaderStageCreateInfo stage;
    VkPipelineLayout layout;
    VkPipeline basePipelineHandle;
    int32_t basePipelineIndex;
};

struct VkDescriptorPoolSize
{
    VkDescriptorType type;
    uint32_t descriptorCount;
};

struct VkDescriptorPoolCreateInfo
{
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    uint32_t maxSets;
    uint32_t poolSizeCount;
    const VkDescriptorPoolSize *pPoolSizes;
};

struct VkDescriptorSetAllocateInfo
{
    VkStructureType sType;
    const void *pNext;
    VkDescriptorPool descriptorPool;
    uint32_t descriptorSetCount;
    const VkDescriptorSetLayout *pSetLayouts;
};

struct VkDescriptorBufferInfo
{
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize range;
};

struct VkWriteDescriptorSet
{
    VkStructureType sType;
    const void *pNext;
    VkDescriptorSet dstSet;
    uint32_t dstBinding;
    uint32_t dstArrayElement;
    uint32_t descriptorCount;
    VkDescriptorType descriptorType;
    const void *pImageInfo;
    const VkDescriptorBufferInfo *pBufferInfo;
    const VkBufferView *pTexelBufferView;
};

struct VkCommandPoolCreateInfo
{
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
};

struct VkCommandBufferAllocateInfo
{
    VkStructureType sType;
    const void *pNext;
    VkCommandPool commandPool;
    VkCommandBufferLevel level;
    uint32_t commandBufferCount;
};

struct VkCommandBufferBeginInfo
{
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    const void *pInheritanceInfo;
};

struct VkMemoryBarrier
{
    VkStructureType sType;
    const void *pNext;
    VkFlags srcAccessMask;
    VkFlags dstAccessMask;
};

struct VkBufferMemoryBarrier
{
    VkStructureType sType;
    const void *pNext;
    VkFlags srcAccessMask;
    VkFlags dstAccessMask;
    uint32_t srcQueueFamilyIndex;
    uint32_t dstQueueFamilyIndex;
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize size;
};

struct VkBufferCopy
{
    VkDeviceSize srcOffset;
    VkDeviceSize dstOffset;
    VkDeviceSize size;
};

struct VkSubmitInfo
{
    VkStructureType sType;
    const void *pNext;
    uint32_t waitSemaphoreCount;
    const void *pWaitSemaphores;
    const VkFlags *pWaitDstStageMask;
    uint32_t commandBufferCount;
    const VkCommandBuffer *pCommandBuffers;
    uint32_t signalSemaphoreCount;
    const void *pSignalSemaphores;
};

struct VkFenceCreateInfo
{
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
};

using PFN_vkVoidFunction = void(PS2_VKAPI_PTR *)();
using PFN_vkGetInstanceProcAddr = PFN_vkVoidFunction(PS2_VKAPI_PTR *)(VkInstance, const char *);
using PFN_vkGetDeviceProcAddr = PFN_vkVoidFunction(PS2_VKAPI_PTR *)(VkDevice, const char *);
using PFN_vkCreateInstance = VkResult(PS2_VKAPI_PTR *)(const VkInstanceCreateInfo *, const void *, VkInstance *);
using PFN_vkDestroyInstance = void(PS2_VKAPI_PTR *)(VkInstance, const void *);
using PFN_vkEnumeratePhysicalDevices = VkResult(PS2_VKAPI_PTR *)(VkInstance, uint32_t *, VkPhysicalDevice *);
using PFN_vkGetPhysicalDeviceProperties = void(PS2_VKAPI_PTR *)(VkPhysicalDevice, VkPhysicalDeviceProperties *);
using PFN_vkGetPhysicalDeviceFeatures = void(PS2_VKAPI_PTR *)(VkPhysicalDevice, VkPhysicalDeviceFeatures *);
using PFN_vkGetPhysicalDeviceQueueFamilyProperties = void(PS2_VKAPI_PTR *)(VkPhysicalDevice, uint32_t *, VkQueueFamilyProperties *);
using PFN_vkGetPhysicalDeviceMemoryProperties = void(PS2_VKAPI_PTR *)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties *);
using PFN_vkCreateDevice = VkResult(PS2_VKAPI_PTR *)(VkPhysicalDevice, const VkDeviceCreateInfo *, const void *, VkDevice *);
using PFN_vkDestroyDevice = void(PS2_VKAPI_PTR *)(VkDevice, const void *);
using PFN_vkGetDeviceQueue = void(PS2_VKAPI_PTR *)(VkDevice, uint32_t, uint32_t, VkQueue *);
using PFN_vkDeviceWaitIdle = VkResult(PS2_VKAPI_PTR *)(VkDevice);
using PFN_vkQueueSubmit = VkResult(PS2_VKAPI_PTR *)(VkQueue, uint32_t, const VkSubmitInfo *, VkFence);
using PFN_vkCreateBuffer = VkResult(PS2_VKAPI_PTR *)(VkDevice, const VkBufferCreateInfo *, const void *, VkBuffer *);
using PFN_vkDestroyBuffer = void(PS2_VKAPI_PTR *)(VkDevice, VkBuffer, const void *);
using PFN_vkGetBufferMemoryRequirements = void(PS2_VKAPI_PTR *)(VkDevice, VkBuffer, VkMemoryRequirements *);
using PFN_vkAllocateMemory = VkResult(PS2_VKAPI_PTR *)(VkDevice, const VkMemoryAllocateInfo *, const void *, VkDeviceMemory *);
using PFN_vkFreeMemory = void(PS2_VKAPI_PTR *)(VkDevice, VkDeviceMemory, const void *);
using PFN_vkBindBufferMemory = VkResult(PS2_VKAPI_PTR *)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
using PFN_vkMapMemory = VkResult(PS2_VKAPI_PTR *)(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, void **);
using PFN_vkUnmapMemory = void(PS2_VKAPI_PTR *)(VkDevice, VkDeviceMemory);
using PFN_vkCreateDescriptorSetLayout = VkResult(PS2_VKAPI_PTR *)(VkDevice, const VkDescriptorSetLayoutCreateInfo *, const void *, VkDescriptorSetLayout *);
using PFN_vkDestroyDescriptorSetLayout = void(PS2_VKAPI_PTR *)(VkDevice, VkDescriptorSetLayout, const void *);
using PFN_vkCreatePipelineLayout = VkResult(PS2_VKAPI_PTR *)(VkDevice, const VkPipelineLayoutCreateInfo *, const void *, VkPipelineLayout *);
using PFN_vkDestroyPipelineLayout = void(PS2_VKAPI_PTR *)(VkDevice, VkPipelineLayout, const void *);
using PFN_vkCreateShaderModule = VkResult(PS2_VKAPI_PTR *)(VkDevice, const VkShaderModuleCreateInfo *, const void *, VkShaderModule *);
using PFN_vkDestroyShaderModule = void(PS2_VKAPI_PTR *)(VkDevice, VkShaderModule, const void *);
using PFN_vkCreateComputePipelines = VkResult(PS2_VKAPI_PTR *)(VkDevice, VkPipelineCache, uint32_t, const VkComputePipelineCreateInfo *, const void *, VkPipeline *);
using PFN_vkDestroyPipeline = void(PS2_VKAPI_PTR *)(VkDevice, VkPipeline, const void *);
using PFN_vkCreateDescriptorPool = VkResult(PS2_VKAPI_PTR *)(VkDevice, const VkDescriptorPoolCreateInfo *, const void *, VkDescriptorPool *);
using PFN_vkDestroyDescriptorPool = void(PS2_VKAPI_PTR *)(VkDevice, VkDescriptorPool, const void *);
using PFN_vkAllocateDescriptorSets = VkResult(PS2_VKAPI_PTR *)(VkDevice, const VkDescriptorSetAllocateInfo *, VkDescriptorSet *);
using PFN_vkUpdateDescriptorSets = void(PS2_VKAPI_PTR *)(VkDevice, uint32_t, const VkWriteDescriptorSet *, uint32_t, const void *);
using PFN_vkCreateCommandPool = VkResult(PS2_VKAPI_PTR *)(VkDevice, const VkCommandPoolCreateInfo *, const void *, VkCommandPool *);
using PFN_vkDestroyCommandPool = void(PS2_VKAPI_PTR *)(VkDevice, VkCommandPool, const void *);
using PFN_vkResetCommandPool = VkResult(PS2_VKAPI_PTR *)(VkDevice, VkCommandPool, VkFlags);
using PFN_vkAllocateCommandBuffers = VkResult(PS2_VKAPI_PTR *)(VkDevice, const VkCommandBufferAllocateInfo *, VkCommandBuffer *);
using PFN_vkBeginCommandBuffer = VkResult(PS2_VKAPI_PTR *)(VkCommandBuffer, const VkCommandBufferBeginInfo *);
using PFN_vkEndCommandBuffer = VkResult(PS2_VKAPI_PTR *)(VkCommandBuffer);
using PFN_vkCmdBindPipeline = void(PS2_VKAPI_PTR *)(VkCommandBuffer, VkPipelineBindPoint, VkPipeline);
using PFN_vkCmdBindDescriptorSets = void(PS2_VKAPI_PTR *)(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet *, uint32_t, const uint32_t *);
using PFN_vkCmdPushConstants = void(PS2_VKAPI_PTR *)(VkCommandBuffer, VkPipelineLayout, VkFlags, uint32_t, uint32_t, const void *);
using PFN_vkCmdDispatch = void(PS2_VKAPI_PTR *)(VkCommandBuffer, uint32_t, uint32_t, uint32_t);
using PFN_vkCmdPipelineBarrier = void(PS2_VKAPI_PTR *)(VkCommandBuffer, VkFlags, VkFlags, VkFlags, uint32_t, const VkMemoryBarrier *, uint32_t, const VkBufferMemoryBarrier *, uint32_t, const void *);
using PFN_vkCmdCopyBuffer = void(PS2_VKAPI_PTR *)(VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const VkBufferCopy *);
using PFN_vkCreateFence = VkResult(PS2_VKAPI_PTR *)(VkDevice, const VkFenceCreateInfo *, const void *, VkFence *);
using PFN_vkDestroyFence = void(PS2_VKAPI_PTR *)(VkDevice, VkFence, const void *);
using PFN_vkWaitForFences = VkResult(PS2_VKAPI_PTR *)(VkDevice, uint32_t, const VkFence *, VkBool32, uint64_t);
using PFN_vkResetFences = VkResult(PS2_VKAPI_PTR *)(VkDevice, uint32_t, const VkFence *);

struct Dispatch
{
    void *library = nullptr;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr = nullptr;
    PFN_vkCreateInstance vkCreateInstance = nullptr;
    PFN_vkDestroyInstance vkDestroyInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = nullptr;
    PFN_vkGetPhysicalDeviceFeatures vkGetPhysicalDeviceFeatures = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkCreateDevice vkCreateDevice = nullptr;
    PFN_vkDestroyDevice vkDestroyDevice = nullptr;
    PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;
    PFN_vkDeviceWaitIdle vkDeviceWaitIdle = nullptr;
    PFN_vkQueueSubmit vkQueueSubmit = nullptr;
    PFN_vkCreateBuffer vkCreateBuffer = nullptr;
    PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = nullptr;
    PFN_vkAllocateMemory vkAllocateMemory = nullptr;
    PFN_vkFreeMemory vkFreeMemory = nullptr;
    PFN_vkBindBufferMemory vkBindBufferMemory = nullptr;
    PFN_vkMapMemory vkMapMemory = nullptr;
    PFN_vkUnmapMemory vkUnmapMemory = nullptr;
    PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout = nullptr;
    PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout = nullptr;
    PFN_vkCreatePipelineLayout vkCreatePipelineLayout = nullptr;
    PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout = nullptr;
    PFN_vkCreateShaderModule vkCreateShaderModule = nullptr;
    PFN_vkDestroyShaderModule vkDestroyShaderModule = nullptr;
    PFN_vkCreateComputePipelines vkCreateComputePipelines = nullptr;
    PFN_vkDestroyPipeline vkDestroyPipeline = nullptr;
    PFN_vkCreateDescriptorPool vkCreateDescriptorPool = nullptr;
    PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool = nullptr;
    PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets = nullptr;
    PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets = nullptr;
    PFN_vkCreateCommandPool vkCreateCommandPool = nullptr;
    PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;
    PFN_vkResetCommandPool vkResetCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr;
    PFN_vkCmdBindPipeline vkCmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets = nullptr;
    PFN_vkCmdPushConstants vkCmdPushConstants = nullptr;
    PFN_vkCmdDispatch vkCmdDispatch = nullptr;
    PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier = nullptr;
    PFN_vkCmdCopyBuffer vkCmdCopyBuffer = nullptr;
    PFN_vkCreateFence vkCreateFence = nullptr;
    PFN_vkDestroyFence vkDestroyFence = nullptr;
    PFN_vkWaitForFences vkWaitForFences = nullptr;
    PFN_vkResetFences vkResetFences = nullptr;
};

} // namespace ps2vk
