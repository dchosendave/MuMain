/*
  SDL GPU driver for WebGPU (browser, Emscripten + emdawnwebgpu).

  Compiled into SDL's own target as its "private" GPU driver (SDL_GPU_PRIVATE, see
  SDL_gpu.c's backend list), so SDL itself is unchanged. The driver is called "webgpu".

  Shaders are WGSL text passed with SDL_GPU_SHADERFORMAT_PRIVATE. They follow SDL's resource
  model with this fixed layout:
    @group(0)  vertex resources:   sampler slot i -> texture @binding(2i), sampler @binding(2i+1)
                                   storage texture k -> @binding(32+k), storage buffer j -> @binding(48+j)
    @group(1)  vertex uniforms:    uniform slot u -> @binding(u)
    @group(2)  fragment resources  (as group 0)
    @group(3)  fragment uniforms   (as group 1)

  Design notes:
  - Uniform data is appended to per command buffer blocks and bound with dynamic offsets; the
    blocks are written with queue.writeBuffer right before the command buffer is submitted.
  - Uploads and copies are recorded into the command encoder (staging buffers mapped at
    creation), so they keep SDL's ordering relative to the passes of the same command buffer.
  - Waiting (device creation, fences, readback) needs Asyncify: the browser has to run its
    event loop for WebGPU promises to resolve.
  - Not implemented: compute passes, storage textures (the client does not use them).
*/
#include "SDL_internal.h"

#include "gpu/SDL_sysgpu.h"

#include <emscripten.h>
#include <webgpu/webgpu.h>

#define WEBGPU_DRIVER_NAME "webgpu"
#define WEBGPU_MAX_SAMPLERS 16
#define WEBGPU_MAX_STORAGE_BUFFERS 8
#define WEBGPU_MAX_UNIFORMS MAX_UNIFORM_BUFFERS_PER_STAGE
#define WEBGPU_BINDING_STORAGE_TEXTURE 32
#define WEBGPU_BINDING_STORAGE_BUFFER 48
#define WEBGPU_UNIFORM_ALIGNMENT 256
#define WEBGPU_UNIFORM_BINDING_SIZE 16384
#define WEBGPU_UNIFORM_BLOCK_SIZE (1024 * 1024)
#define WEBGPU_STAGING_CHUNK_SIZE (4 * 1024 * 1024)
#define WEBGPU_STAGING_POOL_LIMIT 8
// A fence of a command buffer without downloads only has to wait for the GPU once this much upload
// data went out without a wait, so the queue cannot run arbitrarily far ahead (see SubmitInternal).
#define WEBGPU_UNFENCED_UPLOAD_LIMIT (64ull * 1024 * 1024)
#define WEBGPU_COPY_ROW_ALIGNMENT 256
#define WEBGPU_MAX_LOGGED_ERRORS 64

static WGPUStringView WebGPU_String(const char *text)
{
    WGPUStringView view;
    view.data = text;
    view.length = text ? SDL_strlen(text) : 0;
    return view;
}

static Uint32 WebGPU_Align(Uint32 value, Uint32 alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

// --- Objects ------------------------------------------------------------------------------

typedef struct WebGPURenderer WebGPURenderer;
typedef struct WebGPUWindowData WebGPUWindowData;

typedef struct WebGPUTexture
{
    TextureCommonHeader common; // must be first (SDL_gpu.c reads it)
    WGPUTexture texture;
    WGPUTextureFormat format;
    WGPUTextureView sampleView;
    WGPUTextureView *targetViews; // one per (layer, mip level), created on first use as a target
    Uint32 targetViewCount;
    Uint32 layers; // array layers (6 per cube), or depth for 3D
    bool isSwapchain;
} WebGPUTexture;

typedef struct WebGPUBuffer
{
    WGPUBuffer buffer;
    Uint32 size;
} WebGPUBuffer;

typedef struct WebGPUDownload
{
    WGPUBuffer readback;
    Uint32 transferOffset;       // where the first row goes in the transfer buffer
    Uint32 readbackPitch;        // bytes per row in the readback buffer (256 aligned for textures)
    Uint32 rowBytes;             // bytes copied per row
    Uint32 transferPitch;        // bytes per row in the transfer buffer
    Uint32 rowsPerLayer;         // rows per layer in the readback buffer
    Uint32 transferRowsPerLayer; // rows per layer in the transfer buffer
    Uint32 rows;                 // rows to copy per layer
    Uint32 layers;
} WebGPUDownload;

typedef struct WebGPUTransferBuffer
{
    Uint8 *data;
    Uint32 size;
    SDL_GPUTransferBufferUsage usage;
    WebGPUDownload *downloads;
    Uint32 downloadCount;
    Uint32 downloadCapacity;
} WebGPUTransferBuffer;

typedef struct WebGPUSampler
{
    WGPUSampler sampler;
    bool comparison;
} WebGPUSampler;

typedef enum WebGPUBindingKind
{
    WEBGPU_BINDING_UNIFORM,
    WEBGPU_BINDING_STORAGE_BUFFER_KIND,
    WEBGPU_BINDING_TEXTURE,
    WEBGPU_BINDING_SAMPLER,
} WebGPUBindingKind;

typedef struct WebGPUShaderBinding
{
    Uint32 group;
    Uint32 binding;
    WebGPUBindingKind kind;
    WGPUTextureSampleType sampleType;
    WGPUTextureViewDimension viewDimension;
    bool multisampled;
    bool comparisonSampler;
} WebGPUShaderBinding;

#define WEBGPU_MAX_SHADER_BINDINGS 64

typedef struct WebGPUShader
{
    WGPUShaderModule module;
    char *entrypoint;
    SDL_GPUShaderStage stage;
    Uint32 numSamplers;
    Uint32 numStorageBuffers;
    Uint32 numUniformBuffers;
    WebGPUShaderBinding bindings[WEBGPU_MAX_SHADER_BINDINGS];
    Uint32 bindingCount;
} WebGPUShader;

typedef struct WebGPUGraphicsPipeline
{
    GraphicsPipelineCommonHeader common; // must be first
    WGPURenderPipeline pipelines[2];      // [0] uint16 strip index format (or list), [1] uint32
    bool isStrip;
    WGPUBindGroupLayout resourceLayouts[2]; // vertex (group 0), fragment (group 2)
    WGPUPipelineLayout layout;
    WebGPUShaderBinding vertexBindings[WEBGPU_MAX_SHADER_BINDINGS];
    Uint32 vertexBindingCount;
    WebGPUShaderBinding fragmentBindings[WEBGPU_MAX_SHADER_BINDINGS];
    Uint32 fragmentBindingCount;
    Uint32 vertexUniforms;
    Uint32 fragmentUniforms;
} WebGPUGraphicsPipeline;

typedef struct WebGPUUniformBlock
{
    WGPUBuffer buffer;
    Uint8 *cpu;
    Uint32 used;
    WGPUBindGroup bindGroups[2][WEBGPU_MAX_UNIFORMS + 1]; // [stage][uniform count]
} WebGPUUniformBlock;

// Upload source data: filled on the CPU while recording, written to the GPU buffer with one queue
// write at submit (like the uniform blocks), then reused. Queue operations run in order, so the next
// command buffer's write cannot reach the buffer before this one's copies have read it.
typedef struct WebGPUStagingChunk
{
    WGPUBuffer buffer;
    Uint8 *cpu;
    Uint32 size;
    Uint32 used;
} WebGPUStagingChunk;

typedef struct WebGPUFence
{
    SDL_AtomicInt signaled;
    SDL_AtomicInt references; // the user and the pending work-done callback
} WebGPUFence;

typedef struct WebGPUStageUniforms
{
    Uint32 offset[WEBGPU_MAX_UNIFORMS];
    Uint32 length[WEBGPU_MAX_UNIFORMS];
    bool set[WEBGPU_MAX_UNIFORMS];
} WebGPUStageUniforms;

typedef struct WebGPUCommandBuffer
{
    CommandBufferCommonHeader common; // must be first
    WebGPURenderer *renderer;
    WGPUCommandEncoder encoder;
    WGPURenderPassEncoder renderPass;
    Uint32 targetWidth;
    Uint32 targetHeight;

    WebGPUGraphicsPipeline *pipeline;
    WGPUIndexFormat indexFormat;

    WebGPUTexture *textures[2][WEBGPU_MAX_SAMPLERS];
    WebGPUSampler *samplers[2][WEBGPU_MAX_SAMPLERS];
    WebGPUBuffer *storageBuffers[2][WEBGPU_MAX_STORAGE_BUFFERS];
    bool resourcesDirty[2];
    WGPUBindGroup resourceGroups[2];

    WebGPUStageUniforms uniforms[2];
    WebGPUUniformBlock *uniformBlock;
    WebGPUUniformBlock **usedBlocks;
    Uint32 usedBlockCount;
    Uint32 usedBlockCapacity;

    WebGPUStagingChunk *staging;
    Uint32 stagingCount;
    Uint32 stagingCapacity;

    bool hasDownloads; // its fence has to wait for the GPU: the downloads are read after it

    WebGPUWindowData *presentWindows[4]; // windows whose back buffer is copied to the canvas at submit
    Uint32 presentWindowCount;

    WGPUBindGroup *transientGroups; // released after submit
    Uint32 transientGroupCount;
    Uint32 transientGroupCapacity;
} WebGPUCommandBuffer;

struct WebGPUWindowData
{
    SDL_Window *window;
    WGPUSurface surface;
    WGPUTextureFormat format;
    WGPUTextureUsage usage;
    Uint32 width;
    Uint32 height;
    // What the application renders into. The canvas texture expires whenever the page yields to the
    // browser, which the client can do in the middle of a frame (fence waits, data loading), so the
    // frame is drawn here and copied to the canvas texture right before the submit.
    WebGPUTexture swapchainTexture;
};

struct WebGPURenderer
{
    SDL_GPUDevice *sdlDevice;
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;
    SDL_PropertiesID properties;

    bool hasBC;
    bool hasDepth32Stencil8;
    bool hasClipDistances;
    bool hasFloat32Filterable;

    WGPUBindGroupLayout uniformLayouts[2][WEBGPU_MAX_UNIFORMS + 1];
    WGPUBindGroupLayout emptyLayout;
    WGPUBindGroup emptyGroup;

    WebGPUUniformBlock **freeBlocks;
    Uint32 freeBlockCount;
    Uint32 freeBlockCapacity;

    WebGPUStagingChunk *freeStaging; // standard size chunks, ready for the next command buffer
    Uint32 freeStagingCount;
    Uint32 freeStagingCapacity;
    // Upload bytes submitted since the last fence that really waited for the GPU.
    Uint64 unfencedUploadBytes;

    WebGPUWindowData **windows;
    Uint32 windowCount;
    Uint32 windowCapacity;

    // Fallbacks for resource slots the application left unbound.
    WebGPUTexture *dummyTexture;
    WebGPUSampler *dummySampler;
    WebGPUBuffer *dummyStorage;

    // SDL_GPU_BlitCommon state.
    SDL_GPUShader *blitVertexShader;
    SDL_GPUShader *blitFrom2DShader;
    SDL_GPUShader *blitFrom2DArrayShader;
    SDL_GPUShader *blitFrom3DShader;
    SDL_GPUShader *blitFromCubeShader;
    SDL_GPUShader *blitFromCubeArrayShader;
    SDL_GPUSampler *blitNearestSampler;
    SDL_GPUSampler *blitLinearSampler;
    BlitPipelineCacheEntry *blitPipelines;
    Uint32 blitPipelineCount;
    Uint32 blitPipelineCapacity;

    int loggedErrors;
};

// --- Waiting ------------------------------------------------------------------------------

typedef struct WebGPURequestResult
{
    bool done;
    bool ok;
    void *handle;
} WebGPURequestResult;

static void WebGPU_WaitFor(WebGPURenderer *renderer, WGPUFuture future)
{
    WGPUFutureWaitInfo info = WGPU_FUTURE_WAIT_INFO_INIT;
    info.future = future;
    wgpuInstanceWaitAny(renderer->instance, 1, &info, UINT64_MAX);
}

static void WebGPU_OnAdapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void *userdata1, void *userdata2)
{
    WebGPURequestResult *result = (WebGPURequestResult *)userdata1;
    (void)userdata2;
    result->done = true;
    result->ok = status == WGPURequestAdapterStatus_Success;
    result->handle = adapter;
    if (!result->ok) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "WebGPU: no adapter: %.*s", (int)message.length, message.data ? message.data : "");
    }
}

static void WebGPU_OnDevice(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void *userdata1, void *userdata2)
{
    WebGPURequestResult *result = (WebGPURequestResult *)userdata1;
    (void)userdata2;
    result->done = true;
    result->ok = status == WGPURequestDeviceStatus_Success;
    result->handle = device;
    if (!result->ok) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "WebGPU: no device: %.*s", (int)message.length, message.data ? message.data : "");
    }
}

static void WebGPU_OnUncapturedError(WGPUDevice const *device, WGPUErrorType type, WGPUStringView message, void *userdata1, void *userdata2)
{
    WebGPURenderer *renderer = (WebGPURenderer *)userdata1;
    (void)device;
    (void)userdata2;
    if (renderer && renderer->loggedErrors++ < WEBGPU_MAX_LOGGED_ERRORS) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "WebGPU error %d: %.*s", (int)type, (int)message.length, message.data ? message.data : "");
        if (renderer->loggedErrors == WEBGPU_MAX_LOGGED_ERRORS) {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "WebGPU: further errors are not logged");
        }
    }
}

static void WebGPU_OnDeviceLost(WGPUDevice const *device, WGPUDeviceLostReason reason, WGPUStringView message, void *userdata1, void *userdata2)
{
    (void)device;
    (void)userdata1;
    (void)userdata2;
    if (reason != WGPUDeviceLostReason_Destroyed && reason != WGPUDeviceLostReason_CallbackCancelled) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "WebGPU device lost (%d): %.*s", (int)reason, (int)message.length, message.data ? message.data : "");
    }
}

static void WebGPU_OnWorkDone(WGPUQueueWorkDoneStatus status, WGPUStringView message, void *userdata1, void *userdata2)
{
    WebGPUFence *fence = (WebGPUFence *)userdata1;
    (void)status;
    (void)message;
    (void)userdata2;
    SDL_SetAtomicInt(&fence->signaled, 1);
    if (SDL_AddAtomicInt(&fence->references, -1) == 1) {
        SDL_free(fence);
    }
}

static WebGPUFence *WebGPU_SignalWhenIdle(WebGPURenderer *renderer)
{
    WebGPUFence *fence = (WebGPUFence *)SDL_calloc(1, sizeof(WebGPUFence));
    WGPUQueueWorkDoneCallbackInfo info = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    SDL_SetAtomicInt(&fence->references, 2);
    info.mode = WGPUCallbackMode_AllowSpontaneous;
    info.callback = WebGPU_OnWorkDone;
    info.userdata1 = fence;
    wgpuQueueOnSubmittedWorkDone(renderer->queue, info);
    return fence;
}

// A fence that is already signaled: nothing the application can observe depends on the GPU having
// finished (upload data is copied out of transfer buffers while recording, and WebGPU keeps released
// objects alive for work in flight). Waiting for real costs a trip through the browser's event loop.
static WebGPUFence *WebGPU_SignaledFence(void)
{
    WebGPUFence *fence = (WebGPUFence *)SDL_calloc(1, sizeof(WebGPUFence));
    SDL_SetAtomicInt(&fence->signaled, 1);
    SDL_SetAtomicInt(&fence->references, 1);
    return fence;
}

static void WebGPU_ReleaseFenceReference(WebGPUFence *fence)
{
    if (SDL_AddAtomicInt(&fence->references, -1) == 1) {
        SDL_free(fence);
    }
}

static void WebGPU_WaitForFence(WebGPUFence *fence)
{
    // The work-done promise resolves on the browser's event loop.
    while (!SDL_GetAtomicInt(&fence->signaled)) {
        emscripten_sleep(0);
    }
}

// --- Format and enum mapping --------------------------------------------------------------

static WGPUTextureFormat WebGPU_TextureFormat(WebGPURenderer *renderer, SDL_GPUTextureFormat format)
{
    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_R8_UNORM: return WGPUTextureFormat_R8Unorm;
    case SDL_GPU_TEXTUREFORMAT_R8G8_UNORM: return WGPUTextureFormat_RG8Unorm;
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM: return WGPUTextureFormat_RGBA8Unorm;
    case SDL_GPU_TEXTUREFORMAT_R10G10B10A2_UNORM: return WGPUTextureFormat_RGB10A2Unorm;
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM: return WGPUTextureFormat_BGRA8Unorm;
    case SDL_GPU_TEXTUREFORMAT_R8_SNORM: return WGPUTextureFormat_R8Snorm;
    case SDL_GPU_TEXTUREFORMAT_R8G8_SNORM: return WGPUTextureFormat_RG8Snorm;
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_SNORM: return WGPUTextureFormat_RGBA8Snorm;
    case SDL_GPU_TEXTUREFORMAT_R16_FLOAT: return WGPUTextureFormat_R16Float;
    case SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT: return WGPUTextureFormat_RG16Float;
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT: return WGPUTextureFormat_RGBA16Float;
    case SDL_GPU_TEXTUREFORMAT_R32_FLOAT: return WGPUTextureFormat_R32Float;
    case SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT: return WGPUTextureFormat_RG32Float;
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT: return WGPUTextureFormat_RGBA32Float;
    case SDL_GPU_TEXTUREFORMAT_R11G11B10_UFLOAT: return WGPUTextureFormat_RG11B10Ufloat;
    case SDL_GPU_TEXTUREFORMAT_R8_UINT: return WGPUTextureFormat_R8Uint;
    case SDL_GPU_TEXTUREFORMAT_R8G8_UINT: return WGPUTextureFormat_RG8Uint;
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UINT: return WGPUTextureFormat_RGBA8Uint;
    case SDL_GPU_TEXTUREFORMAT_R16_UINT: return WGPUTextureFormat_R16Uint;
    case SDL_GPU_TEXTUREFORMAT_R16G16_UINT: return WGPUTextureFormat_RG16Uint;
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UINT: return WGPUTextureFormat_RGBA16Uint;
    case SDL_GPU_TEXTUREFORMAT_R32_UINT: return WGPUTextureFormat_R32Uint;
    case SDL_GPU_TEXTUREFORMAT_R32G32_UINT: return WGPUTextureFormat_RG32Uint;
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_UINT: return WGPUTextureFormat_RGBA32Uint;
    case SDL_GPU_TEXTUREFORMAT_R8_INT: return WGPUTextureFormat_R8Sint;
    case SDL_GPU_TEXTUREFORMAT_R8G8_INT: return WGPUTextureFormat_RG8Sint;
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_INT: return WGPUTextureFormat_RGBA8Sint;
    case SDL_GPU_TEXTUREFORMAT_R16_INT: return WGPUTextureFormat_R16Sint;
    case SDL_GPU_TEXTUREFORMAT_R16G16_INT: return WGPUTextureFormat_RG16Sint;
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_INT: return WGPUTextureFormat_RGBA16Sint;
    case SDL_GPU_TEXTUREFORMAT_R32_INT: return WGPUTextureFormat_R32Sint;
    case SDL_GPU_TEXTUREFORMAT_R32G32_INT: return WGPUTextureFormat_RG32Sint;
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_INT: return WGPUTextureFormat_RGBA32Sint;
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB: return WGPUTextureFormat_RGBA8UnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB: return WGPUTextureFormat_BGRA8UnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_D16_UNORM: return WGPUTextureFormat_Depth16Unorm;
    case SDL_GPU_TEXTUREFORMAT_D24_UNORM: return WGPUTextureFormat_Depth24Plus;
    case SDL_GPU_TEXTUREFORMAT_D32_FLOAT: return WGPUTextureFormat_Depth32Float;
    case SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT: return WGPUTextureFormat_Depth24PlusStencil8;
    case SDL_GPU_TEXTUREFORMAT_D32_FLOAT_S8_UINT:
        return renderer->hasDepth32Stencil8 ? WGPUTextureFormat_Depth32FloatStencil8 : WGPUTextureFormat_Undefined;
    default:
        break;
    }

    if (!renderer->hasBC) {
        return WGPUTextureFormat_Undefined;
    }

    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM: return WGPUTextureFormat_BC1RGBAUnorm;
    case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM: return WGPUTextureFormat_BC2RGBAUnorm;
    case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM: return WGPUTextureFormat_BC3RGBAUnorm;
    case SDL_GPU_TEXTUREFORMAT_BC4_R_UNORM: return WGPUTextureFormat_BC4RUnorm;
    case SDL_GPU_TEXTUREFORMAT_BC5_RG_UNORM: return WGPUTextureFormat_BC5RGUnorm;
    case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM: return WGPUTextureFormat_BC7RGBAUnorm;
    case SDL_GPU_TEXTUREFORMAT_BC6H_RGB_FLOAT: return WGPUTextureFormat_BC6HRGBFloat;
    case SDL_GPU_TEXTUREFORMAT_BC6H_RGB_UFLOAT: return WGPUTextureFormat_BC6HRGBUfloat;
    case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM_SRGB: return WGPUTextureFormat_BC1RGBAUnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM_SRGB: return WGPUTextureFormat_BC2RGBAUnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM_SRGB: return WGPUTextureFormat_BC3RGBAUnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM_SRGB: return WGPUTextureFormat_BC7RGBAUnormSrgb;
    default:
        return WGPUTextureFormat_Undefined;
    }
}

static bool WebGPU_IsDepthFormat(SDL_GPUTextureFormat format)
{
    return IsDepthFormat(format);
}

static bool WebGPU_HasStencil(SDL_GPUTextureFormat format)
{
    return IsStencilFormat(format);
}

static SDL_GPUTextureFormat WebGPU_SDLFormat(WGPUTextureFormat format)
{
    switch (format) {
    case WGPUTextureFormat_BGRA8Unorm: return SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM;
    case WGPUTextureFormat_RGBA8Unorm: return SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    case WGPUTextureFormat_RGBA16Float: return SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;
    default: return SDL_GPU_TEXTUREFORMAT_INVALID;
    }
}

static WGPUCompareFunction WebGPU_Compare(SDL_GPUCompareOp op)
{
    switch (op) {
    case SDL_GPU_COMPAREOP_NEVER: return WGPUCompareFunction_Never;
    case SDL_GPU_COMPAREOP_LESS: return WGPUCompareFunction_Less;
    case SDL_GPU_COMPAREOP_EQUAL: return WGPUCompareFunction_Equal;
    case SDL_GPU_COMPAREOP_LESS_OR_EQUAL: return WGPUCompareFunction_LessEqual;
    case SDL_GPU_COMPAREOP_GREATER: return WGPUCompareFunction_Greater;
    case SDL_GPU_COMPAREOP_NOT_EQUAL: return WGPUCompareFunction_NotEqual;
    case SDL_GPU_COMPAREOP_GREATER_OR_EQUAL: return WGPUCompareFunction_GreaterEqual;
    case SDL_GPU_COMPAREOP_ALWAYS:
    default: return WGPUCompareFunction_Always;
    }
}

static WGPUStencilOperation WebGPU_StencilOp(SDL_GPUStencilOp op)
{
    switch (op) {
    case SDL_GPU_STENCILOP_ZERO: return WGPUStencilOperation_Zero;
    case SDL_GPU_STENCILOP_REPLACE: return WGPUStencilOperation_Replace;
    case SDL_GPU_STENCILOP_INCREMENT_AND_CLAMP: return WGPUStencilOperation_IncrementClamp;
    case SDL_GPU_STENCILOP_DECREMENT_AND_CLAMP: return WGPUStencilOperation_DecrementClamp;
    case SDL_GPU_STENCILOP_INVERT: return WGPUStencilOperation_Invert;
    case SDL_GPU_STENCILOP_INCREMENT_AND_WRAP: return WGPUStencilOperation_IncrementWrap;
    case SDL_GPU_STENCILOP_DECREMENT_AND_WRAP: return WGPUStencilOperation_DecrementWrap;
    case SDL_GPU_STENCILOP_KEEP:
    default: return WGPUStencilOperation_Keep;
    }
}

static WGPUBlendFactor WebGPU_BlendFactor(SDL_GPUBlendFactor factor)
{
    switch (factor) {
    case SDL_GPU_BLENDFACTOR_ZERO: return WGPUBlendFactor_Zero;
    case SDL_GPU_BLENDFACTOR_ONE: return WGPUBlendFactor_One;
    case SDL_GPU_BLENDFACTOR_SRC_COLOR: return WGPUBlendFactor_Src;
    case SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_COLOR: return WGPUBlendFactor_OneMinusSrc;
    case SDL_GPU_BLENDFACTOR_DST_COLOR: return WGPUBlendFactor_Dst;
    case SDL_GPU_BLENDFACTOR_ONE_MINUS_DST_COLOR: return WGPUBlendFactor_OneMinusDst;
    case SDL_GPU_BLENDFACTOR_SRC_ALPHA: return WGPUBlendFactor_SrcAlpha;
    case SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA: return WGPUBlendFactor_OneMinusSrcAlpha;
    case SDL_GPU_BLENDFACTOR_DST_ALPHA: return WGPUBlendFactor_DstAlpha;
    case SDL_GPU_BLENDFACTOR_ONE_MINUS_DST_ALPHA: return WGPUBlendFactor_OneMinusDstAlpha;
    case SDL_GPU_BLENDFACTOR_CONSTANT_COLOR: return WGPUBlendFactor_Constant;
    case SDL_GPU_BLENDFACTOR_ONE_MINUS_CONSTANT_COLOR: return WGPUBlendFactor_OneMinusConstant;
    case SDL_GPU_BLENDFACTOR_SRC_ALPHA_SATURATE: return WGPUBlendFactor_SrcAlphaSaturated;
    default: return WGPUBlendFactor_One;
    }
}

static WGPUBlendOperation WebGPU_BlendOp(SDL_GPUBlendOp op)
{
    switch (op) {
    case SDL_GPU_BLENDOP_SUBTRACT: return WGPUBlendOperation_Subtract;
    case SDL_GPU_BLENDOP_REVERSE_SUBTRACT: return WGPUBlendOperation_ReverseSubtract;
    case SDL_GPU_BLENDOP_MIN: return WGPUBlendOperation_Min;
    case SDL_GPU_BLENDOP_MAX: return WGPUBlendOperation_Max;
    case SDL_GPU_BLENDOP_ADD:
    default: return WGPUBlendOperation_Add;
    }
}

static WGPUVertexFormat WebGPU_VertexFormat(SDL_GPUVertexElementFormat format)
{
    switch (format) {
    case SDL_GPU_VERTEXELEMENTFORMAT_INT: return WGPUVertexFormat_Sint32;
    case SDL_GPU_VERTEXELEMENTFORMAT_INT2: return WGPUVertexFormat_Sint32x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_INT3: return WGPUVertexFormat_Sint32x3;
    case SDL_GPU_VERTEXELEMENTFORMAT_INT4: return WGPUVertexFormat_Sint32x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_UINT: return WGPUVertexFormat_Uint32;
    case SDL_GPU_VERTEXELEMENTFORMAT_UINT2: return WGPUVertexFormat_Uint32x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_UINT3: return WGPUVertexFormat_Uint32x3;
    case SDL_GPU_VERTEXELEMENTFORMAT_UINT4: return WGPUVertexFormat_Uint32x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_FLOAT: return WGPUVertexFormat_Float32;
    case SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2: return WGPUVertexFormat_Float32x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3: return WGPUVertexFormat_Float32x3;
    case SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4: return WGPUVertexFormat_Float32x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_BYTE2: return WGPUVertexFormat_Sint8x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_BYTE4: return WGPUVertexFormat_Sint8x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_UBYTE2: return WGPUVertexFormat_Uint8x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4: return WGPUVertexFormat_Uint8x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_BYTE2_NORM: return WGPUVertexFormat_Snorm8x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_BYTE4_NORM: return WGPUVertexFormat_Snorm8x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_UBYTE2_NORM: return WGPUVertexFormat_Unorm8x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4_NORM: return WGPUVertexFormat_Unorm8x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_SHORT2: return WGPUVertexFormat_Sint16x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_SHORT4: return WGPUVertexFormat_Sint16x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_USHORT2: return WGPUVertexFormat_Uint16x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_USHORT4: return WGPUVertexFormat_Uint16x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_SHORT2_NORM: return WGPUVertexFormat_Snorm16x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_SHORT4_NORM: return WGPUVertexFormat_Snorm16x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_USHORT2_NORM: return WGPUVertexFormat_Unorm16x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_USHORT4_NORM: return WGPUVertexFormat_Unorm16x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_HALF2: return WGPUVertexFormat_Float16x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_HALF4: return WGPUVertexFormat_Float16x4;
    default: return WGPUVertexFormat_Float32;
    }
}

static WGPUAddressMode WebGPU_AddressMode(SDL_GPUSamplerAddressMode mode)
{
    switch (mode) {
    case SDL_GPU_SAMPLERADDRESSMODE_MIRRORED_REPEAT: return WGPUAddressMode_MirrorRepeat;
    case SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE: return WGPUAddressMode_ClampToEdge;
    case SDL_GPU_SAMPLERADDRESSMODE_REPEAT:
    default: return WGPUAddressMode_Repeat;
    }
}

static WGPUTextureViewDimension WebGPU_ViewDimension(SDL_GPUTextureType type)
{
    switch (type) {
    case SDL_GPU_TEXTURETYPE_2D_ARRAY: return WGPUTextureViewDimension_2DArray;
    case SDL_GPU_TEXTURETYPE_3D: return WGPUTextureViewDimension_3D;
    case SDL_GPU_TEXTURETYPE_CUBE: return WGPUTextureViewDimension_Cube;
    case SDL_GPU_TEXTURETYPE_CUBE_ARRAY: return WGPUTextureViewDimension_CubeArray;
    case SDL_GPU_TEXTURETYPE_2D:
    default: return WGPUTextureViewDimension_2D;
    }
}

static WGPULoadOp WebGPU_LoadOp(SDL_GPULoadOp op)
{
    // WebGPU has no "don't care": clearing is the cheap equivalent.
    return op == SDL_GPU_LOADOP_LOAD ? WGPULoadOp_Load : WGPULoadOp_Clear;
}

static WGPUStoreOp WebGPU_StoreOp(SDL_GPUStoreOp op)
{
    return (op == SDL_GPU_STOREOP_STORE || op == SDL_GPU_STOREOP_RESOLVE_AND_STORE) ? WGPUStoreOp_Store : WGPUStoreOp_Discard;
}

// --- Device queries -----------------------------------------------------------------------

static SDL_PropertiesID WEBGPU_GetDeviceProperties(SDL_GPUDevice *device)
{
    return ((WebGPURenderer *)device->driverData)->properties;
}

static bool WEBGPU_SupportsTextureFormat(SDL_GPURenderer *driverData, SDL_GPUTextureFormat format, SDL_GPUTextureType type, SDL_GPUTextureUsageFlags usage)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    (void)type;
    if (WebGPU_TextureFormat(renderer, format) == WGPUTextureFormat_Undefined) {
        return false;
    }
    if (usage & (SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ |
                 SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE)) {
        return false;
    }
    if ((usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) && IsCompressedFormat(format)) {
        return false;
    }
    return true;
}

static bool WEBGPU_SupportsSampleCount(SDL_GPURenderer *driverData, SDL_GPUTextureFormat format, SDL_GPUSampleCount sampleCount)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    if (WebGPU_TextureFormat(renderer, format) == WGPUTextureFormat_Undefined) {
        return false;
    }
    return sampleCount == SDL_GPU_SAMPLECOUNT_1 || sampleCount == SDL_GPU_SAMPLECOUNT_4;
}

// --- Textures -----------------------------------------------------------------------------

static WebGPUTexture *WebGPU_CreateTextureInternal(WebGPURenderer *renderer, const SDL_GPUTextureCreateInfo *createinfo, const char *label)
{
    WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    WGPUTextureViewDescriptor viewDesc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    WebGPUTexture *texture;
    const WGPUTextureFormat format = WebGPU_TextureFormat(renderer, createinfo->format);
    Uint32 layers = createinfo->layer_count_or_depth;

    if (format == WGPUTextureFormat_Undefined) {
        SDL_SetError("WebGPU: unsupported texture format %d", (int)createinfo->format);
        return NULL;
    }

    if (createinfo->type == SDL_GPU_TEXTURETYPE_CUBE) {
        layers = 6;
    } else if (createinfo->type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY) {
        layers = createinfo->layer_count_or_depth * 6;
    }
    if (layers == 0) {
        layers = 1;
    }

    desc.label = WebGPU_String(label);
    desc.dimension = createinfo->type == SDL_GPU_TEXTURETYPE_3D ? WGPUTextureDimension_3D : WGPUTextureDimension_2D;
    desc.size.width = createinfo->width;
    desc.size.height = createinfo->height;
    desc.size.depthOrArrayLayers = layers;
    desc.format = format;
    desc.mipLevelCount = createinfo->num_levels ? createinfo->num_levels : 1;
    desc.sampleCount = createinfo->sample_count == SDL_GPU_SAMPLECOUNT_4 ? 4 : 1;
    desc.usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_CopySrc;
    if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_SAMPLER) {
        desc.usage |= WGPUTextureUsage_TextureBinding;
    }
    if (createinfo->usage & (SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET)) {
        desc.usage |= WGPUTextureUsage_RenderAttachment;
    }
    if (desc.sampleCount > 1) {
        desc.usage &= ~(WGPUTextureUsage_CopyDst | WGPUTextureUsage_CopySrc);
    }

    texture = (WebGPUTexture *)SDL_calloc(1, sizeof(WebGPUTexture));
    texture->common.info = *createinfo;
    texture->format = format;
    texture->layers = layers;
    texture->texture = wgpuDeviceCreateTexture(renderer->device, &desc);

    if (desc.usage & WGPUTextureUsage_TextureBinding) {
        viewDesc.dimension = WebGPU_ViewDimension(createinfo->type);
        viewDesc.mipLevelCount = desc.mipLevelCount;
        viewDesc.arrayLayerCount = createinfo->type == SDL_GPU_TEXTURETYPE_3D ? 1 : layers;
        if (WebGPU_IsDepthFormat(createinfo->format)) {
            viewDesc.aspect = WGPUTextureAspect_DepthOnly;
        }
        texture->sampleView = wgpuTextureCreateView(texture->texture, &viewDesc);
    }

    texture->targetViewCount = (createinfo->type == SDL_GPU_TEXTURETYPE_3D ? 1 : layers) * desc.mipLevelCount;
    texture->targetViews = (WGPUTextureView *)SDL_calloc(texture->targetViewCount, sizeof(WGPUTextureView));
    return texture;
}

static SDL_GPUTexture *WEBGPU_CreateTexture(SDL_GPURenderer *driverData, const SDL_GPUTextureCreateInfo *createinfo)
{
    return (SDL_GPUTexture *)WebGPU_CreateTextureInternal((WebGPURenderer *)driverData, createinfo, NULL);
}

// The view a render pass draws into: one mip level of one layer (all depth slices for 3D).
static WGPUTextureView WebGPU_TargetView(WebGPUTexture *texture, Uint32 mipLevel, Uint32 layer)
{
    const SDL_GPUTextureCreateInfo *info = &texture->common.info;
    const Uint32 levels = info->num_levels ? info->num_levels : 1;
    const bool is3D = info->type == SDL_GPU_TEXTURETYPE_3D;
    const Uint32 index = (is3D ? 0 : layer) * levels + mipLevel;
    WGPUTextureViewDescriptor desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;

    if (texture->isSwapchain) {
        return texture->sampleView;
    }
    if (index >= texture->targetViewCount) {
        return NULL;
    }
    if (!texture->targetViews[index]) {
        desc.dimension = is3D ? WGPUTextureViewDimension_3D : WGPUTextureViewDimension_2D;
        desc.baseMipLevel = mipLevel;
        desc.mipLevelCount = 1;
        desc.baseArrayLayer = is3D ? 0 : layer;
        desc.arrayLayerCount = 1;
        texture->targetViews[index] = wgpuTextureCreateView(texture->texture, &desc);
    }
    return texture->targetViews[index];
}

static void WebGPU_DestroyTextureObjects(WebGPUTexture *texture)
{
    Uint32 i;
    for (i = 0; i < texture->targetViewCount; i++) {
        if (texture->targetViews[i]) {
            wgpuTextureViewRelease(texture->targetViews[i]);
        }
    }
    SDL_free(texture->targetViews);
    texture->targetViews = NULL;
    texture->targetViewCount = 0;
    if (texture->sampleView) {
        wgpuTextureViewRelease(texture->sampleView);
        texture->sampleView = NULL;
    }
    if (texture->texture) {
        if (!texture->isSwapchain) {
            wgpuTextureDestroy(texture->texture);
        }
        wgpuTextureRelease(texture->texture);
        texture->texture = NULL;
    }
}

static void WEBGPU_ReleaseTexture(SDL_GPURenderer *driverData, SDL_GPUTexture *texture)
{
    (void)driverData;
    WebGPU_DestroyTextureObjects((WebGPUTexture *)texture);
    SDL_free(texture);
}

static void WEBGPU_SetTextureName(SDL_GPURenderer *driverData, SDL_GPUTexture *texture, const char *text)
{
    (void)driverData;
    if (((WebGPUTexture *)texture)->texture) {
        wgpuTextureSetLabel(((WebGPUTexture *)texture)->texture, WebGPU_String(text));
    }
}

// --- Buffers ------------------------------------------------------------------------------

static SDL_GPUBuffer *WEBGPU_CreateBuffer(SDL_GPURenderer *driverData, SDL_GPUBufferUsageFlags usageFlags, Uint32 size, const char *debugName)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    WebGPUBuffer *buffer = (WebGPUBuffer *)SDL_calloc(1, sizeof(WebGPUBuffer));

    // Rounded up to 4 bytes so copies of any size can be padded to WebGPU's copy alignment.
    buffer->size = WebGPU_Align(size ? size : 4, 4);
    desc.label = WebGPU_String(debugName);
    desc.size = buffer->size;
    desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
    if (usageFlags & SDL_GPU_BUFFERUSAGE_VERTEX) {
        desc.usage |= WGPUBufferUsage_Vertex;
    }
    if (usageFlags & SDL_GPU_BUFFERUSAGE_INDEX) {
        desc.usage |= WGPUBufferUsage_Index;
    }
    if (usageFlags & SDL_GPU_BUFFERUSAGE_INDIRECT) {
        desc.usage |= WGPUBufferUsage_Indirect;
    }
    if (usageFlags & (SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE)) {
        desc.usage |= WGPUBufferUsage_Storage;
    }
    buffer->buffer = wgpuDeviceCreateBuffer(renderer->device, &desc);
    return (SDL_GPUBuffer *)buffer;
}

static void WEBGPU_ReleaseBuffer(SDL_GPURenderer *driverData, SDL_GPUBuffer *buffer)
{
    WebGPUBuffer *webgpuBuffer = (WebGPUBuffer *)buffer;
    (void)driverData;
    wgpuBufferDestroy(webgpuBuffer->buffer);
    wgpuBufferRelease(webgpuBuffer->buffer);
    SDL_free(webgpuBuffer);
}

static void WEBGPU_SetBufferName(SDL_GPURenderer *driverData, SDL_GPUBuffer *buffer, const char *text)
{
    (void)driverData;
    wgpuBufferSetLabel(((WebGPUBuffer *)buffer)->buffer, WebGPU_String(text));
}

static SDL_GPUTransferBuffer *WEBGPU_CreateTransferBuffer(SDL_GPURenderer *driverData, SDL_GPUTransferBufferUsage usage, Uint32 size, const char *debugName)
{
    WebGPUTransferBuffer *buffer = (WebGPUTransferBuffer *)SDL_calloc(1, sizeof(WebGPUTransferBuffer));
    (void)driverData;
    (void)debugName;
    buffer->data = (Uint8 *)SDL_calloc(1, size ? size : 1);
    buffer->size = size;
    buffer->usage = usage;
    return (SDL_GPUTransferBuffer *)buffer;
}

static void WebGPU_DiscardDownloads(WebGPUTransferBuffer *buffer)
{
    Uint32 i;
    for (i = 0; i < buffer->downloadCount; i++) {
        // Destroyed, not only released: the readback memory goes now instead of on garbage collection.
        wgpuBufferDestroy(buffer->downloads[i].readback);
        wgpuBufferRelease(buffer->downloads[i].readback);
    }
    buffer->downloadCount = 0;
}

static void WEBGPU_ReleaseTransferBuffer(SDL_GPURenderer *driverData, SDL_GPUTransferBuffer *transferBuffer)
{
    WebGPUTransferBuffer *buffer = (WebGPUTransferBuffer *)transferBuffer;
    (void)driverData;
    WebGPU_DiscardDownloads(buffer);
    SDL_free(buffer->downloads);
    SDL_free(buffer->data);
    SDL_free(buffer);
}

typedef struct WebGPUMapResult
{
    bool done;
    bool ok;
} WebGPUMapResult;

static void WebGPU_OnMapped(WGPUMapAsyncStatus status, WGPUStringView message, void *userdata1, void *userdata2)
{
    WebGPUMapResult *result = (WebGPUMapResult *)userdata1;
    (void)message;
    (void)userdata2;
    result->done = true;
    result->ok = status == WGPUMapAsyncStatus_Success;
}

// Copies finished downloads into the transfer buffer's memory (waits for the GPU).
static void WebGPU_ResolveDownloads(WebGPURenderer *renderer, WebGPUTransferBuffer *buffer)
{
    Uint32 i, layer, row;
    for (i = 0; i < buffer->downloadCount; i++) {
        WebGPUDownload *download = &buffer->downloads[i];
        const Uint64 size = (Uint64)download->readbackPitch * download->rowsPerLayer * download->layers;
        WGPUBufferMapCallbackInfo info = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        WebGPUMapResult result = { false, false };
        const Uint8 *mapped;

        info.mode = WGPUCallbackMode_WaitAnyOnly;
        info.callback = WebGPU_OnMapped;
        info.userdata1 = &result;
        WebGPU_WaitFor(renderer, wgpuBufferMapAsync(download->readback, WGPUMapMode_Read, 0, (size_t)size, info));
        if (!result.ok) {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "WebGPU: could not map a download");
            continue;
        }

        mapped = (const Uint8 *)wgpuBufferGetConstMappedRange(download->readback, 0, (size_t)size);
        if (mapped) {
            for (layer = 0; layer < download->layers; layer++) {
                for (row = 0; row < download->rows; row++) {
                    const Uint64 source = ((Uint64)layer * download->rowsPerLayer + row) * download->readbackPitch;
                    const Uint64 target = download->transferOffset + ((Uint64)layer * download->transferRowsPerLayer + row) * download->transferPitch;
                    if (target + download->rowBytes <= buffer->size) {
                        SDL_memcpy(buffer->data + target, mapped + source, download->rowBytes);
                    }
                }
            }
        }
        wgpuBufferUnmap(download->readback);
    }
    WebGPU_DiscardDownloads(buffer);
}

static void *WEBGPU_MapTransferBuffer(SDL_GPURenderer *driverData, SDL_GPUTransferBuffer *transferBuffer, bool cycle)
{
    WebGPUTransferBuffer *buffer = (WebGPUTransferBuffer *)transferBuffer;
    (void)cycle;
    if (buffer->downloadCount > 0) {
        WebGPU_ResolveDownloads((WebGPURenderer *)driverData, buffer);
    }
    return buffer->data;
}

static void WEBGPU_UnmapTransferBuffer(SDL_GPURenderer *driverData, SDL_GPUTransferBuffer *transferBuffer)
{
    (void)driverData;
    (void)transferBuffer;
}

// --- Samplers -----------------------------------------------------------------------------

static SDL_GPUSampler *WEBGPU_CreateSampler(SDL_GPURenderer *driverData, const SDL_GPUSamplerCreateInfo *createinfo)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WGPUSamplerDescriptor desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    WebGPUSampler *sampler = (WebGPUSampler *)SDL_calloc(1, sizeof(WebGPUSampler));

    desc.addressModeU = WebGPU_AddressMode(createinfo->address_mode_u);
    desc.addressModeV = WebGPU_AddressMode(createinfo->address_mode_v);
    desc.addressModeW = WebGPU_AddressMode(createinfo->address_mode_w);
    desc.magFilter = createinfo->mag_filter == SDL_GPU_FILTER_LINEAR ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
    desc.minFilter = createinfo->min_filter == SDL_GPU_FILTER_LINEAR ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
    desc.mipmapFilter = createinfo->mipmap_mode == SDL_GPU_SAMPLERMIPMAPMODE_LINEAR ? WGPUMipmapFilterMode_Linear : WGPUMipmapFilterMode_Nearest;
    desc.lodMinClamp = createinfo->min_lod;
    desc.lodMaxClamp = createinfo->max_lod < createinfo->min_lod ? createinfo->min_lod : createinfo->max_lod;
    if (desc.lodMaxClamp > 32.0f) {
        desc.lodMaxClamp = 32.0f;
    }
    if (createinfo->enable_compare) {
        desc.compare = WebGPU_Compare(createinfo->compare_op);
        sampler->comparison = true;
    }
    // WebGPU allows anisotropy only when every filter is linear.
    if (createinfo->enable_anisotropy && createinfo->max_anisotropy > 1.0f &&
        desc.magFilter == WGPUFilterMode_Linear && desc.minFilter == WGPUFilterMode_Linear &&
        desc.mipmapFilter == WGPUMipmapFilterMode_Linear) {
        desc.maxAnisotropy = (Uint16)SDL_min(16.0f, createinfo->max_anisotropy);
    }
    sampler->sampler = wgpuDeviceCreateSampler(renderer->device, &desc);
    return (SDL_GPUSampler *)sampler;
}

static void WEBGPU_ReleaseSampler(SDL_GPURenderer *driverData, SDL_GPUSampler *sampler)
{
    (void)driverData;
    wgpuSamplerRelease(((WebGPUSampler *)sampler)->sampler);
    SDL_free(sampler);
}

// --- Shaders ------------------------------------------------------------------------------

static const char *WebGPU_SkipSpace(const char *p, const char *end)
{
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) {
        p++;
    }
    return p;
}

static bool WebGPU_ParseUint(const char **p, const char *end, Uint32 *value)
{
    const char *q = WebGPU_SkipSpace(*p, end);
    Uint32 result = 0;
    bool any = false;
    while (q < end && *q >= '0' && *q <= '9') {
        result = result * 10 + (Uint32)(*q - '0');
        q++;
        any = true;
    }
    *p = q;
    *value = result;
    return any;
}

static bool WebGPU_StartsWith(const char *p, const char *end, const char *prefix)
{
    const size_t length = SDL_strlen(prefix);
    return (size_t)(end - p) >= length && SDL_memcmp(p, prefix, length) == 0;
}

// Reads the resource declarations of a WGSL module: "@group(G) @binding(B) var<...> name : type;".
static void WebGPU_ReflectShader(WebGPUShader *shader, const char *code, size_t length)
{
    const char *p = code;
    const char *end = code + length;

    while ((p = SDL_strstr(p, "@group(")) != NULL && p < end) {
        WebGPUShaderBinding binding;
        Uint32 group, index;
        const char *type;

        SDL_zero(binding);
        p += 7;
        if (!WebGPU_ParseUint(&p, end, &group)) {
            continue;
        }
        p = SDL_strstr(p, "@binding(");
        if (!p) {
            break;
        }
        p += 9;
        if (!WebGPU_ParseUint(&p, end, &index)) {
            continue;
        }
        p = SDL_strstr(p, "var");
        if (!p) {
            break;
        }
        p += 3;
        binding.group = group;
        binding.binding = index;
        if (WebGPU_StartsWith(p, end, "<uniform")) {
            binding.kind = WEBGPU_BINDING_UNIFORM;
        } else if (WebGPU_StartsWith(p, end, "<storage")) {
            binding.kind = WEBGPU_BINDING_STORAGE_BUFFER_KIND;
        } else {
            type = SDL_strchr(p, ':');
            if (!type) {
                break;
            }
            type = WebGPU_SkipSpace(type + 1, end);
            if (WebGPU_StartsWith(type, end, "sampler_comparison")) {
                binding.kind = WEBGPU_BINDING_SAMPLER;
                binding.comparisonSampler = true;
            } else if (WebGPU_StartsWith(type, end, "sampler")) {
                binding.kind = WEBGPU_BINDING_SAMPLER;
            } else if (WebGPU_StartsWith(type, end, "texture_")) {
                binding.kind = WEBGPU_BINDING_TEXTURE;
                binding.sampleType = WGPUTextureSampleType_Float;
                binding.viewDimension = WGPUTextureViewDimension_2D;
                if (WebGPU_StartsWith(type, end, "texture_depth")) {
                    binding.sampleType = WGPUTextureSampleType_Depth;
                }
                if (WebGPU_StartsWith(type, end, "texture_multisampled") || WebGPU_StartsWith(type, end, "texture_depth_multisampled")) {
                    binding.multisampled = true;
                }
                if (SDL_strstr(type, "_2d_array") && SDL_strstr(type, "_2d_array") < SDL_strchr(type, ';')) {
                    binding.viewDimension = WGPUTextureViewDimension_2DArray;
                } else if (SDL_strstr(type, "_cube_array") && SDL_strstr(type, "_cube_array") < SDL_strchr(type, ';')) {
                    binding.viewDimension = WGPUTextureViewDimension_CubeArray;
                } else if (SDL_strstr(type, "_cube") && SDL_strstr(type, "_cube") < SDL_strchr(type, ';')) {
                    binding.viewDimension = WGPUTextureViewDimension_Cube;
                } else if (SDL_strstr(type, "_3d") && SDL_strstr(type, "_3d") < SDL_strchr(type, ';')) {
                    binding.viewDimension = WGPUTextureViewDimension_3D;
                }
                if (SDL_strstr(type, "<u32>") && SDL_strstr(type, "<u32>") < SDL_strchr(type, ';')) {
                    binding.sampleType = WGPUTextureSampleType_Uint;
                } else if (SDL_strstr(type, "<i32>") && SDL_strstr(type, "<i32>") < SDL_strchr(type, ';')) {
                    binding.sampleType = WGPUTextureSampleType_Sint;
                }
            } else {
                continue;
            }
        }
        if (shader->bindingCount < WEBGPU_MAX_SHADER_BINDINGS) {
            shader->bindings[shader->bindingCount++] = binding;
        }
    }
}

// Removes the lines marked "// webgpu:clip-distances" when the browser lacks that feature.
static char *WebGPU_PreprocessShader(WebGPURenderer *renderer, const char *code, size_t length)
{
    char *result = (char *)SDL_malloc(length + 1);
    const char *line = code;
    const char *end = code + length;
    size_t written = 0;

    while (line < end) {
        const char *lineEnd = line;
        while (lineEnd < end && *lineEnd != '\n') {
            lineEnd++;
        }
        if (lineEnd < end) {
            lineEnd++; // keep the newline
        }
        bool keep = true;
        if (!renderer->hasClipDistances) {
            const char *marker = SDL_strstr(line, "webgpu:clip-distances");
            keep = !(marker && marker < lineEnd);
        }
        if (keep) {
            SDL_memcpy(result + written, line, (size_t)(lineEnd - line));
            written += (size_t)(lineEnd - line);
        }
        line = lineEnd;
    }
    result[written] = '\0';
    return result;
}

static SDL_GPUShader *WEBGPU_CreateShader(SDL_GPURenderer *driverData, const SDL_GPUShaderCreateInfo *createinfo)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
    WGPUShaderModuleDescriptor desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    WebGPUShader *shader;
    char *code;

    if (createinfo->format != SDL_GPU_SHADERFORMAT_PRIVATE) {
        SDL_SetError("WebGPU: shaders must be WGSL (SDL_GPU_SHADERFORMAT_PRIVATE)");
        return NULL;
    }

    code = WebGPU_PreprocessShader(renderer, (const char *)createinfo->code, createinfo->code_size);
    shader = (WebGPUShader *)SDL_calloc(1, sizeof(WebGPUShader));
    shader->stage = createinfo->stage;
    shader->numSamplers = createinfo->num_samplers;
    shader->numStorageBuffers = createinfo->num_storage_buffers;
    shader->numUniformBuffers = createinfo->num_uniform_buffers;
    shader->entrypoint = SDL_strdup(createinfo->entrypoint ? createinfo->entrypoint : "main");
    WebGPU_ReflectShader(shader, code, SDL_strlen(code));

    source.code = WebGPU_String(code);
    desc.nextInChain = &source.chain;
    shader->module = wgpuDeviceCreateShaderModule(renderer->device, &desc);
    SDL_free(code);
    return (SDL_GPUShader *)shader;
}

static void WEBGPU_ReleaseShader(SDL_GPURenderer *driverData, SDL_GPUShader *shader)
{
    WebGPUShader *webgpuShader = (WebGPUShader *)shader;
    (void)driverData;
    wgpuShaderModuleRelease(webgpuShader->module);
    SDL_free(webgpuShader->entrypoint);
    SDL_free(webgpuShader);
}

// --- Pipelines ----------------------------------------------------------------------------

static WGPUBindGroupLayout WebGPU_CreateResourceLayout(WebGPURenderer *renderer, const WebGPUShader *shader, WGPUShaderStage visibility, Uint32 group)
{
    WGPUBindGroupLayoutEntry entries[WEBGPU_MAX_SHADER_BINDINGS];
    WGPUBindGroupLayoutDescriptor desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    Uint32 i, count = 0;

    for (i = 0; i < shader->bindingCount; i++) {
        const WebGPUShaderBinding *binding = &shader->bindings[i];
        WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        if (binding->group != group) {
            continue;
        }
        entry.binding = binding->binding;
        entry.visibility = visibility;
        switch (binding->kind) {
        case WEBGPU_BINDING_TEXTURE:
            entry.texture.sampleType = binding->sampleType;
            entry.texture.viewDimension = binding->viewDimension;
            entry.texture.multisampled = binding->multisampled;
            break;
        case WEBGPU_BINDING_SAMPLER:
            entry.sampler.type = binding->comparisonSampler ? WGPUSamplerBindingType_Comparison : WGPUSamplerBindingType_Filtering;
            break;
        case WEBGPU_BINDING_STORAGE_BUFFER_KIND:
            entry.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
            break;
        default:
            continue;
        }
        entries[count++] = entry;
    }

    if (count == 0) {
        wgpuBindGroupLayoutAddRef(renderer->emptyLayout);
        return renderer->emptyLayout;
    }
    desc.entryCount = count;
    desc.entries = entries;
    return wgpuDeviceCreateBindGroupLayout(renderer->device, &desc);
}

static Uint32 WebGPU_UniformCount(const WebGPUShader *shader)
{
    Uint32 i, count = shader->numUniformBuffers;
    for (i = 0; i < shader->bindingCount; i++) {
        if (shader->bindings[i].kind == WEBGPU_BINDING_UNIFORM && shader->bindings[i].binding + 1 > count) {
            count = shader->bindings[i].binding + 1;
        }
    }
    return SDL_min(count, (Uint32)WEBGPU_MAX_UNIFORMS);
}

static SDL_GPUGraphicsPipeline *WEBGPU_CreateGraphicsPipeline(SDL_GPURenderer *driverData, const SDL_GPUGraphicsPipelineCreateInfo *createinfo)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WebGPUShader *vertexShader = (WebGPUShader *)createinfo->vertex_shader;
    WebGPUShader *fragmentShader = (WebGPUShader *)createinfo->fragment_shader;
    WebGPUGraphicsPipeline *pipeline;
    WGPUBindGroupLayout layouts[4];
    WGPUPipelineLayoutDescriptor layoutDesc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    WGPURenderPipelineDescriptor desc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    WGPUVertexBufferLayout buffers[MAX_VERTEX_BUFFERS];
    WGPUVertexAttribute attributes[MAX_VERTEX_ATTRIBUTES];
    WGPUColorTargetState targets[MAX_COLOR_TARGET_BINDINGS];
    WGPUBlendState blends[MAX_COLOR_TARGET_BINDINGS];
    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    WGPUDepthStencilState depthStencil = WGPU_DEPTH_STENCIL_STATE_INIT;
    Uint32 i, b, attributeCount = 0;
    int variant;

    if (!vertexShader || !fragmentShader) {
        SDL_SetError("WebGPU: a pipeline needs vertex and fragment shaders");
        return NULL;
    }

    pipeline = (WebGPUGraphicsPipeline *)SDL_calloc(1, sizeof(WebGPUGraphicsPipeline));
    pipeline->common.num_vertex_samplers = vertexShader->numSamplers;
    pipeline->common.num_vertex_storage_buffers = vertexShader->numStorageBuffers;
    pipeline->common.num_vertex_uniform_buffers = vertexShader->numUniformBuffers;
    pipeline->common.num_fragment_samplers = fragmentShader->numSamplers;
    pipeline->common.num_fragment_storage_buffers = fragmentShader->numStorageBuffers;
    pipeline->common.num_fragment_uniform_buffers = fragmentShader->numUniformBuffers;

    SDL_memcpy(pipeline->vertexBindings, vertexShader->bindings, sizeof(vertexShader->bindings));
    pipeline->vertexBindingCount = vertexShader->bindingCount;
    SDL_memcpy(pipeline->fragmentBindings, fragmentShader->bindings, sizeof(fragmentShader->bindings));
    pipeline->fragmentBindingCount = fragmentShader->bindingCount;
    pipeline->vertexUniforms = WebGPU_UniformCount(vertexShader);
    pipeline->fragmentUniforms = WebGPU_UniformCount(fragmentShader);

    pipeline->resourceLayouts[0] = WebGPU_CreateResourceLayout(renderer, vertexShader, WGPUShaderStage_Vertex, 0);
    pipeline->resourceLayouts[1] = WebGPU_CreateResourceLayout(renderer, fragmentShader, WGPUShaderStage_Fragment, 2);
    layouts[0] = pipeline->resourceLayouts[0];
    layouts[1] = renderer->uniformLayouts[0][pipeline->vertexUniforms];
    layouts[2] = pipeline->resourceLayouts[1];
    layouts[3] = renderer->uniformLayouts[1][pipeline->fragmentUniforms];
    layoutDesc.bindGroupLayoutCount = 4;
    layoutDesc.bindGroupLayouts = layouts;
    pipeline->layout = wgpuDeviceCreatePipelineLayout(renderer->device, &layoutDesc);

    // Vertex input. SDL addresses vertex buffers by slot, WebGPU by index in this array, so the
    // layouts are placed at their slot; unused slots in between stay empty (no attributes).
    {
        Uint32 bufferCount = 0;
        for (b = 0; b < MAX_VERTEX_BUFFERS; b++) {
            buffers[b] = (WGPUVertexBufferLayout)WGPU_VERTEX_BUFFER_LAYOUT_INIT;
            buffers[b].stepMode = WGPUVertexStepMode_Undefined;
        }
        for (b = 0; b < createinfo->vertex_input_state.num_vertex_buffers; b++) {
            const SDL_GPUVertexBufferDescription *description = &createinfo->vertex_input_state.vertex_buffer_descriptions[b];
            WGPUVertexBufferLayout *layout;
            const Uint32 first = attributeCount;
            if (description->slot >= MAX_VERTEX_BUFFERS) {
                continue;
            }
            for (i = 0; i < createinfo->vertex_input_state.num_vertex_attributes; i++) {
                const SDL_GPUVertexAttribute *attribute = &createinfo->vertex_input_state.vertex_attributes[i];
                if (attribute->buffer_slot != description->slot || attributeCount >= MAX_VERTEX_ATTRIBUTES) {
                    continue;
                }
                attributes[attributeCount].nextInChain = NULL;
                attributes[attributeCount].format = WebGPU_VertexFormat(attribute->format);
                attributes[attributeCount].offset = attribute->offset;
                attributes[attributeCount].shaderLocation = attribute->location;
                attributeCount++;
            }
            layout = &buffers[description->slot];
            layout->arrayStride = description->pitch;
            layout->stepMode = description->input_rate == SDL_GPU_VERTEXINPUTRATE_INSTANCE ? WGPUVertexStepMode_Instance : WGPUVertexStepMode_Vertex;
            layout->attributeCount = attributeCount - first;
            layout->attributes = &attributes[first];
            bufferCount = SDL_max(bufferCount, description->slot + 1);
        }
        desc.vertex.bufferCount = bufferCount;
    }

    desc.layout = pipeline->layout;
    desc.vertex.module = vertexShader->module;
    desc.vertex.entryPoint = WebGPU_String(vertexShader->entrypoint);
    desc.vertex.buffers = buffers;

    switch (createinfo->primitive_type) {
    case SDL_GPU_PRIMITIVETYPE_TRIANGLESTRIP: desc.primitive.topology = WGPUPrimitiveTopology_TriangleStrip; pipeline->isStrip = true; break;
    case SDL_GPU_PRIMITIVETYPE_LINELIST: desc.primitive.topology = WGPUPrimitiveTopology_LineList; break;
    case SDL_GPU_PRIMITIVETYPE_LINESTRIP: desc.primitive.topology = WGPUPrimitiveTopology_LineStrip; pipeline->isStrip = true; break;
    case SDL_GPU_PRIMITIVETYPE_POINTLIST: desc.primitive.topology = WGPUPrimitiveTopology_PointList; break;
    case SDL_GPU_PRIMITIVETYPE_TRIANGLELIST:
    default: desc.primitive.topology = WGPUPrimitiveTopology_TriangleList; break;
    }
    desc.primitive.frontFace = createinfo->rasterizer_state.front_face == SDL_GPU_FRONTFACE_CLOCKWISE ? WGPUFrontFace_CW : WGPUFrontFace_CCW;
    switch (createinfo->rasterizer_state.cull_mode) {
    case SDL_GPU_CULLMODE_FRONT: desc.primitive.cullMode = WGPUCullMode_Front; break;
    case SDL_GPU_CULLMODE_BACK: desc.primitive.cullMode = WGPUCullMode_Back; break;
    default: desc.primitive.cullMode = WGPUCullMode_None; break;
    }

    desc.multisample.count = createinfo->multisample_state.sample_count == SDL_GPU_SAMPLECOUNT_4 ? 4 : 1;
    desc.multisample.mask = createinfo->multisample_state.enable_mask ? createinfo->multisample_state.sample_mask : 0xFFFFFFFF;
    desc.multisample.alphaToCoverageEnabled = createinfo->multisample_state.enable_alpha_to_coverage && desc.multisample.count > 1;

    if (createinfo->target_info.has_depth_stencil_target) {
        const SDL_GPUDepthStencilState *state = &createinfo->depth_stencil_state;
        const bool triangles = desc.primitive.topology == WGPUPrimitiveTopology_TriangleList || desc.primitive.topology == WGPUPrimitiveTopology_TriangleStrip;
        depthStencil.format = WebGPU_TextureFormat(renderer, createinfo->target_info.depth_stencil_format);
        depthStencil.depthWriteEnabled = (state->enable_depth_test && state->enable_depth_write) ? WGPUOptionalBool_True : WGPUOptionalBool_False;
        depthStencil.depthCompare = state->enable_depth_test ? WebGPU_Compare(state->compare_op) : WGPUCompareFunction_Always;
        if (state->enable_stencil_test && WebGPU_HasStencil(createinfo->target_info.depth_stencil_format)) {
            depthStencil.stencilFront.compare = WebGPU_Compare(state->front_stencil_state.compare_op);
            depthStencil.stencilFront.failOp = WebGPU_StencilOp(state->front_stencil_state.fail_op);
            depthStencil.stencilFront.depthFailOp = WebGPU_StencilOp(state->front_stencil_state.depth_fail_op);
            depthStencil.stencilFront.passOp = WebGPU_StencilOp(state->front_stencil_state.pass_op);
            depthStencil.stencilBack.compare = WebGPU_Compare(state->back_stencil_state.compare_op);
            depthStencil.stencilBack.failOp = WebGPU_StencilOp(state->back_stencil_state.fail_op);
            depthStencil.stencilBack.depthFailOp = WebGPU_StencilOp(state->back_stencil_state.depth_fail_op);
            depthStencil.stencilBack.passOp = WebGPU_StencilOp(state->back_stencil_state.pass_op);
            depthStencil.stencilReadMask = state->compare_mask;
            depthStencil.stencilWriteMask = state->write_mask;
        } else {
            depthStencil.stencilFront.compare = WGPUCompareFunction_Always;
            depthStencil.stencilBack.compare = WGPUCompareFunction_Always;
            depthStencil.stencilReadMask = 0;
            depthStencil.stencilWriteMask = 0;
        }
        if (createinfo->rasterizer_state.enable_depth_bias && triangles) {
            depthStencil.depthBias = (int32_t)createinfo->rasterizer_state.depth_bias_constant_factor;
            depthStencil.depthBiasSlopeScale = createinfo->rasterizer_state.depth_bias_slope_factor;
            depthStencil.depthBiasClamp = createinfo->rasterizer_state.depth_bias_clamp;
        }
        desc.depthStencil = &depthStencil;
    }

    for (i = 0; i < createinfo->target_info.num_color_targets; i++) {
        const SDL_GPUColorTargetDescription *target = &createinfo->target_info.color_target_descriptions[i];
        const SDL_GPUColorTargetBlendState *blend = &target->blend_state;
        targets[i] = (WGPUColorTargetState)WGPU_COLOR_TARGET_STATE_INIT;
        targets[i].format = WebGPU_TextureFormat(renderer, target->format);
        targets[i].writeMask = blend->enable_color_write_mask ? (WGPUColorWriteMask)blend->color_write_mask : WGPUColorWriteMask_All;
        if (blend->enable_blend) {
            blends[i].color.operation = WebGPU_BlendOp(blend->color_blend_op);
            blends[i].color.srcFactor = WebGPU_BlendFactor(blend->src_color_blendfactor);
            blends[i].color.dstFactor = WebGPU_BlendFactor(blend->dst_color_blendfactor);
            blends[i].alpha.operation = WebGPU_BlendOp(blend->alpha_blend_op);
            blends[i].alpha.srcFactor = WebGPU_BlendFactor(blend->src_alpha_blendfactor);
            blends[i].alpha.dstFactor = WebGPU_BlendFactor(blend->dst_alpha_blendfactor);
            // Min/max ignore the factors in SDL; WebGPU requires them to be one.
            if (blends[i].color.operation == WGPUBlendOperation_Min || blends[i].color.operation == WGPUBlendOperation_Max) {
                blends[i].color.srcFactor = blends[i].color.dstFactor = WGPUBlendFactor_One;
            }
            if (blends[i].alpha.operation == WGPUBlendOperation_Min || blends[i].alpha.operation == WGPUBlendOperation_Max) {
                blends[i].alpha.srcFactor = blends[i].alpha.dstFactor = WGPUBlendFactor_One;
            }
            targets[i].blend = &blends[i];
        }
    }
    fragment.module = fragmentShader->module;
    fragment.entryPoint = WebGPU_String(fragmentShader->entrypoint);
    fragment.targetCount = createinfo->target_info.num_color_targets;
    fragment.targets = targets;
    desc.fragment = &fragment;

    for (variant = 0; variant < (pipeline->isStrip ? 2 : 1); variant++) {
        if (pipeline->isStrip) {
            desc.primitive.stripIndexFormat = variant == 0 ? WGPUIndexFormat_Uint16 : WGPUIndexFormat_Uint32;
        }
        pipeline->pipelines[variant] = wgpuDeviceCreateRenderPipeline(renderer->device, &desc);
    }
    if (!pipeline->isStrip) {
        pipeline->pipelines[1] = pipeline->pipelines[0];
        wgpuRenderPipelineAddRef(pipeline->pipelines[1]);
    }
    return (SDL_GPUGraphicsPipeline *)pipeline;
}

static void WEBGPU_ReleaseGraphicsPipeline(SDL_GPURenderer *driverData, SDL_GPUGraphicsPipeline *graphicsPipeline)
{
    WebGPUGraphicsPipeline *pipeline = (WebGPUGraphicsPipeline *)graphicsPipeline;
    (void)driverData;
    wgpuRenderPipelineRelease(pipeline->pipelines[0]);
    wgpuRenderPipelineRelease(pipeline->pipelines[1]);
    wgpuBindGroupLayoutRelease(pipeline->resourceLayouts[0]);
    wgpuBindGroupLayoutRelease(pipeline->resourceLayouts[1]);
    wgpuPipelineLayoutRelease(pipeline->layout);
    SDL_free(pipeline);
}

static SDL_GPUComputePipeline *WEBGPU_CreateComputePipeline(SDL_GPURenderer *driverData, const SDL_GPUComputePipelineCreateInfo *createinfo)
{
    (void)driverData;
    (void)createinfo;
    SDL_SetError("WebGPU: compute pipelines are not implemented");
    return NULL;
}

static void WEBGPU_ReleaseComputePipeline(SDL_GPURenderer *driverData, SDL_GPUComputePipeline *computePipeline)
{
    (void)driverData;
    (void)computePipeline;
}

// --- Command buffers ----------------------------------------------------------------------

static void WebGPU_AddTransientGroup(WebGPUCommandBuffer *cb, WGPUBindGroup group)
{
    EXPAND_ARRAY_IF_NEEDED(cb->transientGroups, WGPUBindGroup, cb->transientGroupCount + 1, cb->transientGroupCapacity, cb->transientGroupCapacity * 2 + 16);
    cb->transientGroups[cb->transientGroupCount++] = group;
}

static WebGPUUniformBlock *WebGPU_AcquireUniformBlock(WebGPURenderer *renderer)
{
    WebGPUUniformBlock *block;
    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    if (renderer->freeBlockCount > 0) {
        block = renderer->freeBlocks[--renderer->freeBlockCount];
        block->used = 0;
        return block;
    }
    block = (WebGPUUniformBlock *)SDL_calloc(1, sizeof(WebGPUUniformBlock));
    // Room past the last offset for a whole binding window.
    desc.size = WEBGPU_UNIFORM_BLOCK_SIZE + WEBGPU_UNIFORM_BINDING_SIZE;
    desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    block->buffer = wgpuDeviceCreateBuffer(renderer->device, &desc);
    block->cpu = (Uint8 *)SDL_calloc(1, WEBGPU_UNIFORM_BLOCK_SIZE);
    return block;
}

static WGPUBindGroup WebGPU_UniformBindGroup(WebGPURenderer *renderer, WebGPUUniformBlock *block, int stage, Uint32 count)
{
    if (count == 0) {
        return renderer->emptyGroup;
    }
    if (!block->bindGroups[stage][count]) {
        WGPUBindGroupEntry entries[WEBGPU_MAX_UNIFORMS];
        WGPUBindGroupDescriptor desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        Uint32 i;
        for (i = 0; i < count; i++) {
            entries[i] = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
            entries[i].binding = i;
            entries[i].buffer = block->buffer;
            entries[i].offset = 0;
            entries[i].size = WEBGPU_UNIFORM_BINDING_SIZE;
        }
        desc.layout = renderer->uniformLayouts[stage][count];
        desc.entryCount = count;
        desc.entries = entries;
        block->bindGroups[stage][count] = wgpuDeviceCreateBindGroup(renderer->device, &desc);
    }
    return block->bindGroups[stage][count];
}

static Uint32 WebGPU_AllocateUniform(WebGPUCommandBuffer *cb, Uint32 length)
{
    WebGPURenderer *renderer = cb->renderer;
    const Uint32 size = WebGPU_Align(length ? length : 16, WEBGPU_UNIFORM_ALIGNMENT);
    int stage;
    Uint32 slot;

    if (!cb->uniformBlock || cb->uniformBlock->used + size > WEBGPU_UNIFORM_BLOCK_SIZE) {
        WebGPUUniformBlock *previous = cb->uniformBlock;
        WebGPUUniformBlock *block = WebGPU_AcquireUniformBlock(renderer);
        EXPAND_ARRAY_IF_NEEDED(cb->usedBlocks, WebGPUUniformBlock *, cb->usedBlockCount + 1, cb->usedBlockCapacity, cb->usedBlockCapacity * 2 + 4);
        cb->usedBlocks[cb->usedBlockCount++] = block;
        cb->uniformBlock = block;
        // Offsets are relative to the bound block: carry the current values of every slot over.
        if (previous) {
            for (stage = 0; stage < 2; stage++) {
                for (slot = 0; slot < WEBGPU_MAX_UNIFORMS; slot++) {
                    if (cb->uniforms[stage].set[slot]) {
                        const Uint32 copySize = WebGPU_Align(cb->uniforms[stage].length[slot], WEBGPU_UNIFORM_ALIGNMENT);
                        SDL_memcpy(block->cpu + block->used, previous->cpu + cb->uniforms[stage].offset[slot], cb->uniforms[stage].length[slot]);
                        cb->uniforms[stage].offset[slot] = block->used;
                        block->used += copySize;
                    }
                }
            }
        }
    }

    {
        const Uint32 offset = cb->uniformBlock->used;
        cb->uniformBlock->used += size;
        return offset;
    }
}

static void WebGPU_PushUniform(WebGPUCommandBuffer *cb, int stage, Uint32 slot, const void *data, Uint32 length)
{
    Uint32 offset;
    if (slot >= WEBGPU_MAX_UNIFORMS) {
        return;
    }
    if (length > WEBGPU_UNIFORM_BINDING_SIZE) {
        length = WEBGPU_UNIFORM_BINDING_SIZE;
    }
    offset = WebGPU_AllocateUniform(cb, length);
    SDL_memcpy(cb->uniformBlock->cpu + offset, data, length);
    cb->uniforms[stage].offset[slot] = offset;
    cb->uniforms[stage].length[slot] = length;
    cb->uniforms[stage].set[slot] = true;
}

static SDL_GPUCommandBuffer *WEBGPU_AcquireCommandBuffer(SDL_GPURenderer *driverData)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)SDL_calloc(1, sizeof(WebGPUCommandBuffer));
    WGPUCommandEncoderDescriptor desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    cb->renderer = renderer;
    cb->encoder = wgpuDeviceCreateCommandEncoder(renderer->device, &desc);
    cb->indexFormat = WGPUIndexFormat_Uint16;
    return (SDL_GPUCommandBuffer *)cb;
}

static void WebGPU_RecycleStagingChunk(WebGPURenderer *renderer, WebGPUStagingChunk *chunk);

static void WebGPU_FreeCommandBuffer(WebGPUCommandBuffer *cb, bool submitted)
{
    WebGPURenderer *renderer = cb->renderer;
    Uint32 i;

    // A cancelled command buffer never wrote its chunks, so they are as reusable as submitted ones.
    (void)submitted;
    for (i = 0; i < cb->stagingCount; i++) {
        WebGPU_RecycleStagingChunk(renderer, &cb->staging[i]);
    }
    SDL_free(cb->staging);

    // Queue operations run in order, so a block can be refilled for the next command buffer.
    for (i = 0; i < cb->usedBlockCount; i++) {
        EXPAND_ARRAY_IF_NEEDED(renderer->freeBlocks, WebGPUUniformBlock *, renderer->freeBlockCount + 1, renderer->freeBlockCapacity, renderer->freeBlockCapacity * 2 + 4);
        renderer->freeBlocks[renderer->freeBlockCount++] = cb->usedBlocks[i];
    }
    SDL_free(cb->usedBlocks);

    for (i = 0; i < cb->transientGroupCount; i++) {
        wgpuBindGroupRelease(cb->transientGroups[i]);
    }
    SDL_free(cb->transientGroups);

    if (cb->renderPass) {
        wgpuRenderPassEncoderRelease(cb->renderPass);
    }
    if (cb->encoder) {
        wgpuCommandEncoderRelease(cb->encoder);
    }
    SDL_free(cb);
}

static void WebGPU_CopyToCanvas(WebGPUCommandBuffer *cb, WebGPUWindowData *data, WGPUTexture *canvasOut);

static bool WebGPU_SubmitInternal(WebGPUCommandBuffer *cb, WebGPUFence **fenceOut)
{
    WebGPURenderer *renderer = cb->renderer;
    WGPUCommandBufferDescriptor desc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    WGPUCommandBuffer commands;
    Uint32 i;

    for (i = 0; i < cb->usedBlockCount; i++) {
        WebGPUUniformBlock *block = cb->usedBlocks[i];
        if (block->used > 0) {
            wgpuQueueWriteBuffer(renderer->queue, block->buffer, 0, block->cpu, block->used);
        }
    }
    for (i = 0; i < cb->stagingCount; i++) {
        WebGPUStagingChunk *chunk = &cb->staging[i];
        if (chunk->used > 0) {
            wgpuQueueWriteBuffer(renderer->queue, chunk->buffer, 0, chunk->cpu, chunk->used);
            renderer->unfencedUploadBytes += chunk->used;
        }
    }

    WGPUTexture canvases[4] = { NULL, NULL, NULL, NULL };
    for (i = 0; i < cb->presentWindowCount; i++) {
        WebGPU_CopyToCanvas(cb, cb->presentWindows[i], &canvases[i]);
    }

    commands = wgpuCommandEncoderFinish(cb->encoder, &desc);
    wgpuQueueSubmit(renderer->queue, 1, &commands);
    wgpuCommandBufferRelease(commands);
    for (i = 0; i < cb->presentWindowCount; i++) {
        if (canvases[i]) {
            wgpuTextureRelease(canvases[i]);
        }
    }

    // Only downloads need the GPU to finish (see WebGPU_SignaledFence). A real wait now and then
    // still keeps the client, which uploads every texture with its own fence while loading, from
    // queueing unbounded upload data.
    if (fenceOut) {
        if (cb->hasDownloads || renderer->unfencedUploadBytes >= WEBGPU_UNFENCED_UPLOAD_LIMIT) {
            *fenceOut = WebGPU_SignalWhenIdle(renderer);
            renderer->unfencedUploadBytes = 0;
        } else {
            *fenceOut = WebGPU_SignaledFence();
        }
    }
    WebGPU_FreeCommandBuffer(cb, true);
    return true;
}

static bool WEBGPU_Submit(SDL_GPUCommandBuffer *commandBuffer)
{
    return WebGPU_SubmitInternal((WebGPUCommandBuffer *)commandBuffer, NULL);
}

static SDL_GPUFence *WEBGPU_SubmitAndAcquireFence(SDL_GPUCommandBuffer *commandBuffer)
{
    WebGPUFence *fence = NULL;
    WebGPU_SubmitInternal((WebGPUCommandBuffer *)commandBuffer, &fence);
    return (SDL_GPUFence *)fence;
}

static bool WEBGPU_Cancel(SDL_GPUCommandBuffer *commandBuffer)
{
    WebGPU_FreeCommandBuffer((WebGPUCommandBuffer *)commandBuffer, false);
    return true;
}

static bool WEBGPU_Wait(SDL_GPURenderer *driverData)
{
    WebGPUFence *fence = WebGPU_SignalWhenIdle((WebGPURenderer *)driverData);
    WebGPU_WaitForFence(fence);
    WebGPU_ReleaseFenceReference(fence);
    return true;
}

static bool WEBGPU_WaitForFences(SDL_GPURenderer *driverData, bool waitAll, SDL_GPUFence *const *fences, Uint32 numFences)
{
    Uint32 i;
    (void)driverData;
    if (waitAll) {
        for (i = 0; i < numFences; i++) {
            WebGPU_WaitForFence((WebGPUFence *)fences[i]);
        }
        return true;
    }
    for (;;) {
        for (i = 0; i < numFences; i++) {
            if (SDL_GetAtomicInt(&((WebGPUFence *)fences[i])->signaled)) {
                return true;
            }
        }
        emscripten_sleep(0);
    }
}

static bool WEBGPU_QueryFence(SDL_GPURenderer *driverData, SDL_GPUFence *fence)
{
    (void)driverData;
    return SDL_GetAtomicInt(&((WebGPUFence *)fence)->signaled) != 0;
}

static void WEBGPU_ReleaseFence(SDL_GPURenderer *driverData, SDL_GPUFence *fence)
{
    (void)driverData;
    if (fence) {
        WebGPU_ReleaseFenceReference((WebGPUFence *)fence);
    }
}

// --- Debug labels -------------------------------------------------------------------------

static void WEBGPU_InsertDebugLabel(SDL_GPUCommandBuffer *commandBuffer, const char *text)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    if (cb->renderPass) {
        wgpuRenderPassEncoderInsertDebugMarker(cb->renderPass, WebGPU_String(text));
    } else {
        wgpuCommandEncoderInsertDebugMarker(cb->encoder, WebGPU_String(text));
    }
}

static void WEBGPU_PushDebugGroup(SDL_GPUCommandBuffer *commandBuffer, const char *name)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    if (cb->renderPass) {
        wgpuRenderPassEncoderPushDebugGroup(cb->renderPass, WebGPU_String(name));
    } else {
        wgpuCommandEncoderPushDebugGroup(cb->encoder, WebGPU_String(name));
    }
}

static void WEBGPU_PopDebugGroup(SDL_GPUCommandBuffer *commandBuffer)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    if (cb->renderPass) {
        wgpuRenderPassEncoderPopDebugGroup(cb->renderPass);
    } else {
        wgpuCommandEncoderPopDebugGroup(cb->encoder);
    }
}

// --- Render passes ------------------------------------------------------------------------

static void WEBGPU_BeginRenderPass(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUColorTargetInfo *colorTargetInfos, Uint32 numColorTargets, const SDL_GPUDepthStencilTargetInfo *depthStencilTargetInfo)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    WGPURenderPassColorAttachment colors[MAX_COLOR_TARGET_BINDINGS];
    WGPURenderPassDepthStencilAttachment depth = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
    WGPURenderPassDescriptor desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    Uint32 i;

    cb->targetWidth = 0;
    cb->targetHeight = 0;
    for (i = 0; i < numColorTargets; i++) {
        const SDL_GPUColorTargetInfo *info = &colorTargetInfos[i];
        WebGPUTexture *texture = (WebGPUTexture *)info->texture;
        const bool is3D = texture->common.info.type == SDL_GPU_TEXTURETYPE_3D;
        colors[i] = (WGPURenderPassColorAttachment)WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
        colors[i].view = WebGPU_TargetView(texture, info->mip_level, is3D ? 0 : info->layer_or_depth_plane);
        colors[i].depthSlice = is3D ? info->layer_or_depth_plane : WGPU_DEPTH_SLICE_UNDEFINED;
        colors[i].loadOp = WebGPU_LoadOp(info->load_op);
        colors[i].storeOp = WebGPU_StoreOp(info->store_op);
        colors[i].clearValue.r = info->clear_color.r;
        colors[i].clearValue.g = info->clear_color.g;
        colors[i].clearValue.b = info->clear_color.b;
        colors[i].clearValue.a = info->clear_color.a;
        if ((info->store_op == SDL_GPU_STOREOP_RESOLVE || info->store_op == SDL_GPU_STOREOP_RESOLVE_AND_STORE) && info->resolve_texture) {
            colors[i].resolveTarget = WebGPU_TargetView((WebGPUTexture *)info->resolve_texture, info->resolve_mip_level, info->resolve_layer);
        }
        if (cb->targetWidth == 0) {
            cb->targetWidth = SDL_max(1u, texture->common.info.width >> info->mip_level);
            cb->targetHeight = SDL_max(1u, texture->common.info.height >> info->mip_level);
        }
    }

    desc.colorAttachmentCount = numColorTargets;
    desc.colorAttachments = colors;

    if (depthStencilTargetInfo && depthStencilTargetInfo->texture) {
        WebGPUTexture *texture = (WebGPUTexture *)depthStencilTargetInfo->texture;
        const SDL_GPUTextureFormat format = texture->common.info.format;
        depth.view = WebGPU_TargetView(texture, depthStencilTargetInfo->mip_level, depthStencilTargetInfo->layer);
        depth.depthLoadOp = WebGPU_LoadOp(depthStencilTargetInfo->load_op);
        depth.depthStoreOp = WebGPU_StoreOp(depthStencilTargetInfo->store_op);
        depth.depthClearValue = depthStencilTargetInfo->clear_depth;
        if (WebGPU_HasStencil(format)) {
            depth.stencilLoadOp = WebGPU_LoadOp(depthStencilTargetInfo->stencil_load_op);
            depth.stencilStoreOp = WebGPU_StoreOp(depthStencilTargetInfo->stencil_store_op);
            depth.stencilClearValue = depthStencilTargetInfo->clear_stencil;
        } else {
            depth.stencilLoadOp = WGPULoadOp_Undefined;
            depth.stencilStoreOp = WGPUStoreOp_Undefined;
        }
        desc.depthStencilAttachment = &depth;
        if (cb->targetWidth == 0) {
            cb->targetWidth = SDL_max(1u, texture->common.info.width >> depthStencilTargetInfo->mip_level);
            cb->targetHeight = SDL_max(1u, texture->common.info.height >> depthStencilTargetInfo->mip_level);
        }
    }

    cb->renderPass = wgpuCommandEncoderBeginRenderPass(cb->encoder, &desc);
    cb->pipeline = NULL;
    cb->resourcesDirty[0] = cb->resourcesDirty[1] = true;
}

static void WEBGPU_EndRenderPass(SDL_GPUCommandBuffer *commandBuffer)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    if (cb->renderPass) {
        wgpuRenderPassEncoderEnd(cb->renderPass);
        wgpuRenderPassEncoderRelease(cb->renderPass);
        cb->renderPass = NULL;
    }
    cb->pipeline = NULL;
    SDL_zeroa(cb->textures);
    SDL_zeroa(cb->samplers);
    SDL_zeroa(cb->storageBuffers);
}

static void WEBGPU_BindGraphicsPipeline(SDL_GPUCommandBuffer *commandBuffer, SDL_GPUGraphicsPipeline *graphicsPipeline)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUGraphicsPipeline *pipeline = (WebGPUGraphicsPipeline *)graphicsPipeline;
    cb->pipeline = pipeline;
    wgpuRenderPassEncoderSetPipeline(cb->renderPass, pipeline->pipelines[cb->indexFormat == WGPUIndexFormat_Uint32 ? 1 : 0]);
    cb->resourcesDirty[0] = cb->resourcesDirty[1] = true;
}

static void WEBGPU_SetViewport(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUViewport *viewport)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    float x = viewport->x, y = viewport->y, w = viewport->w, h = viewport->h;
    // WebGPU rejects viewports outside the target instead of clipping them.
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if (x + w > (float)cb->targetWidth) { w = (float)cb->targetWidth - x; }
    if (y + h > (float)cb->targetHeight) { h = (float)cb->targetHeight - y; }
    if (w <= 0 || h <= 0) {
        w = h = 0;
        x = y = 0;
    }
    wgpuRenderPassEncoderSetViewport(cb->renderPass, x, y, w, h, SDL_clamp(viewport->min_depth, 0.0f, 1.0f), SDL_clamp(viewport->max_depth, 0.0f, 1.0f));
}

static void WEBGPU_SetScissor(SDL_GPUCommandBuffer *commandBuffer, const SDL_Rect *scissor)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    Sint32 x = SDL_max(0, scissor->x);
    Sint32 y = SDL_max(0, scissor->y);
    Sint32 right = SDL_min((Sint32)cb->targetWidth, scissor->x + scissor->w);
    Sint32 bottom = SDL_min((Sint32)cb->targetHeight, scissor->y + scissor->h);
    if (right < x) {
        right = x;
    }
    if (bottom < y) {
        bottom = y;
    }
    if (x > (Sint32)cb->targetWidth) {
        x = right = (Sint32)cb->targetWidth;
    }
    if (y > (Sint32)cb->targetHeight) {
        y = bottom = (Sint32)cb->targetHeight;
    }
    wgpuRenderPassEncoderSetScissorRect(cb->renderPass, (Uint32)x, (Uint32)y, (Uint32)(right - x), (Uint32)(bottom - y));
}

static void WEBGPU_SetBlendConstants(SDL_GPUCommandBuffer *commandBuffer, SDL_FColor blendConstants)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    WGPUColor color;
    color.r = blendConstants.r;
    color.g = blendConstants.g;
    color.b = blendConstants.b;
    color.a = blendConstants.a;
    wgpuRenderPassEncoderSetBlendConstant(cb->renderPass, &color);
}

static void WEBGPU_SetStencilReference(SDL_GPUCommandBuffer *commandBuffer, Uint8 reference)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    wgpuRenderPassEncoderSetStencilReference(cb->renderPass, reference);
}

static void WEBGPU_BindVertexBuffers(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, const SDL_GPUBufferBinding *bindings, Uint32 numBindings)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    Uint32 i;
    for (i = 0; i < numBindings; i++) {
        WebGPUBuffer *buffer = (WebGPUBuffer *)bindings[i].buffer;
        wgpuRenderPassEncoderSetVertexBuffer(cb->renderPass, firstSlot + i, buffer->buffer, bindings[i].offset, WGPU_WHOLE_SIZE);
    }
}

static void WEBGPU_BindIndexBuffer(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUBufferBinding *binding, SDL_GPUIndexElementSize indexElementSize)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    const WGPUIndexFormat format = indexElementSize == SDL_GPU_INDEXELEMENTSIZE_32BIT ? WGPUIndexFormat_Uint32 : WGPUIndexFormat_Uint16;
    wgpuRenderPassEncoderSetIndexBuffer(cb->renderPass, ((WebGPUBuffer *)binding->buffer)->buffer, format, binding->offset, WGPU_WHOLE_SIZE);
    if (format != cb->indexFormat) {
        cb->indexFormat = format;
        if (cb->pipeline && cb->pipeline->isStrip) {
            wgpuRenderPassEncoderSetPipeline(cb->renderPass, cb->pipeline->pipelines[format == WGPUIndexFormat_Uint32 ? 1 : 0]);
        }
    }
}

static void WebGPU_BindSamplers(WebGPUCommandBuffer *cb, int stage, Uint32 firstSlot, const SDL_GPUTextureSamplerBinding *bindings, Uint32 numBindings)
{
    Uint32 i;
    for (i = 0; i < numBindings && firstSlot + i < WEBGPU_MAX_SAMPLERS; i++) {
        cb->textures[stage][firstSlot + i] = (WebGPUTexture *)bindings[i].texture;
        cb->samplers[stage][firstSlot + i] = (WebGPUSampler *)bindings[i].sampler;
    }
    cb->resourcesDirty[stage] = true;
}

static void WEBGPU_BindVertexSamplers(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, const SDL_GPUTextureSamplerBinding *textureSamplerBindings, Uint32 numBindings)
{
    WebGPU_BindSamplers((WebGPUCommandBuffer *)commandBuffer, 0, firstSlot, textureSamplerBindings, numBindings);
}

static void WEBGPU_BindFragmentSamplers(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, const SDL_GPUTextureSamplerBinding *textureSamplerBindings, Uint32 numBindings)
{
    WebGPU_BindSamplers((WebGPUCommandBuffer *)commandBuffer, 1, firstSlot, textureSamplerBindings, numBindings);
}

static void WebGPU_BindStorageBuffers(WebGPUCommandBuffer *cb, int stage, Uint32 firstSlot, SDL_GPUBuffer *const *buffers, Uint32 numBindings)
{
    Uint32 i;
    for (i = 0; i < numBindings && firstSlot + i < WEBGPU_MAX_STORAGE_BUFFERS; i++) {
        cb->storageBuffers[stage][firstSlot + i] = (WebGPUBuffer *)buffers[i];
    }
    cb->resourcesDirty[stage] = true;
}

static void WEBGPU_BindVertexStorageBuffers(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, SDL_GPUBuffer *const *storageBuffers, Uint32 numBindings)
{
    WebGPU_BindStorageBuffers((WebGPUCommandBuffer *)commandBuffer, 0, firstSlot, storageBuffers, numBindings);
}

static void WEBGPU_BindFragmentStorageBuffers(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, SDL_GPUBuffer *const *storageBuffers, Uint32 numBindings)
{
    WebGPU_BindStorageBuffers((WebGPUCommandBuffer *)commandBuffer, 1, firstSlot, storageBuffers, numBindings);
}

static void WEBGPU_BindVertexStorageTextures(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, SDL_GPUTexture *const *storageTextures, Uint32 numBindings)
{
    (void)commandBuffer;
    (void)firstSlot;
    (void)storageTextures;
    (void)numBindings;
    SDL_LogError(SDL_LOG_CATEGORY_GPU, "WebGPU: storage textures are not implemented");
}

static void WEBGPU_BindFragmentStorageTextures(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, SDL_GPUTexture *const *storageTextures, Uint32 numBindings)
{
    WEBGPU_BindVertexStorageTextures(commandBuffer, firstSlot, storageTextures, numBindings);
}

static void WEBGPU_PushVertexUniformData(SDL_GPUCommandBuffer *commandBuffer, Uint32 slotIndex, const void *data, Uint32 length)
{
    WebGPU_PushUniform((WebGPUCommandBuffer *)commandBuffer, 0, slotIndex, data, length);
}

static void WEBGPU_PushFragmentUniformData(SDL_GPUCommandBuffer *commandBuffer, Uint32 slotIndex, const void *data, Uint32 length)
{
    WebGPU_PushUniform((WebGPUCommandBuffer *)commandBuffer, 1, slotIndex, data, length);
}

// Creates the resource bind group (group 0 or 2) for the current pipeline and bindings.
static WGPUBindGroup WebGPU_CreateResourceGroup(WebGPUCommandBuffer *cb, int stage)
{
    WebGPURenderer *renderer = cb->renderer;
    const WebGPUGraphicsPipeline *pipeline = cb->pipeline;
    const WebGPUShaderBinding *bindings = stage == 0 ? pipeline->vertexBindings : pipeline->fragmentBindings;
    const Uint32 bindingCount = stage == 0 ? pipeline->vertexBindingCount : pipeline->fragmentBindingCount;
    const Uint32 group = stage == 0 ? 0 : 2;
    WGPUBindGroupEntry entries[WEBGPU_MAX_SHADER_BINDINGS];
    WGPUBindGroupDescriptor desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    Uint32 i, count = 0;

    for (i = 0; i < bindingCount; i++) {
        const WebGPUShaderBinding *binding = &bindings[i];
        WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
        if (binding->group != group) {
            continue;
        }
        entry.binding = binding->binding;
        if (binding->kind == WEBGPU_BINDING_TEXTURE) {
            const Uint32 slot = binding->binding / 2;
            WebGPUTexture *texture = slot < WEBGPU_MAX_SAMPLERS ? cb->textures[stage][slot] : NULL;
            if (!texture || !texture->sampleView) {
                texture = renderer->dummyTexture;
            }
            entry.textureView = texture->sampleView;
        } else if (binding->kind == WEBGPU_BINDING_SAMPLER) {
            const Uint32 slot = binding->binding / 2;
            WebGPUSampler *sampler = slot < WEBGPU_MAX_SAMPLERS ? cb->samplers[stage][slot] : NULL;
            entry.sampler = (sampler ? sampler : renderer->dummySampler)->sampler;
        } else if (binding->kind == WEBGPU_BINDING_STORAGE_BUFFER_KIND) {
            const Uint32 slot = binding->binding - WEBGPU_BINDING_STORAGE_BUFFER;
            WebGPUBuffer *buffer = slot < WEBGPU_MAX_STORAGE_BUFFERS ? cb->storageBuffers[stage][slot] : NULL;
            if (!buffer) {
                buffer = renderer->dummyStorage;
            }
            entry.buffer = buffer->buffer;
            entry.offset = 0;
            entry.size = buffer->size;
        } else {
            continue;
        }
        entries[count++] = entry;
    }

    if (count == 0) {
        return renderer->emptyGroup;
    }
    desc.layout = pipeline->resourceLayouts[stage];
    desc.entryCount = count;
    desc.entries = entries;
    {
        WGPUBindGroup group_ = wgpuDeviceCreateBindGroup(renderer->device, &desc);
        WebGPU_AddTransientGroup(cb, group_);
        return group_;
    }
}

static void WebGPU_ApplyUniforms(WebGPUCommandBuffer *cb, int stage, Uint32 count)
{
    Uint32 offsets[WEBGPU_MAX_UNIFORMS];
    Uint32 slot;
    if (count == 0) {
        wgpuRenderPassEncoderSetBindGroup(cb->renderPass, stage == 0 ? 1 : 3, cb->renderer->emptyGroup, 0, NULL);
        return;
    }
    for (slot = 0; slot < count; slot++) {
        if (!cb->uniforms[stage].set[slot]) {
            static const Uint8 zeros[WEBGPU_UNIFORM_ALIGNMENT] = { 0 };
            WebGPU_PushUniform(cb, stage, slot, zeros, sizeof(zeros));
        }
    }
    for (slot = 0; slot < count; slot++) {
        offsets[slot] = cb->uniforms[stage].offset[slot];
    }
    wgpuRenderPassEncoderSetBindGroup(cb->renderPass, stage == 0 ? 1 : 3,
                                      WebGPU_UniformBindGroup(cb->renderer, cb->uniformBlock, stage, count), count, offsets);
}

static bool WebGPU_PrepareDraw(WebGPUCommandBuffer *cb)
{
    int stage;
    if (!cb->pipeline || !cb->renderPass) {
        return false;
    }
    for (stage = 0; stage < 2; stage++) {
        if (cb->resourcesDirty[stage]) {
            cb->resourceGroups[stage] = WebGPU_CreateResourceGroup(cb, stage);
            wgpuRenderPassEncoderSetBindGroup(cb->renderPass, stage == 0 ? 0 : 2, cb->resourceGroups[stage], 0, NULL);
            cb->resourcesDirty[stage] = false;
        }
    }
    // Each stage binds the block its offsets refer to. A block switch while filling the fragment
    // slots leaves the vertex binding valid: the old block keeps its data until the submit.
    WebGPU_ApplyUniforms(cb, 0, cb->pipeline->vertexUniforms);
    WebGPU_ApplyUniforms(cb, 1, cb->pipeline->fragmentUniforms);
    return true;
}

static void WEBGPU_DrawPrimitives(SDL_GPUCommandBuffer *commandBuffer, Uint32 numVertices, Uint32 numInstances, Uint32 firstVertex, Uint32 firstInstance)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    if (WebGPU_PrepareDraw(cb)) {
        wgpuRenderPassEncoderDraw(cb->renderPass, numVertices, numInstances, firstVertex, firstInstance);
    }
}

static void WEBGPU_DrawIndexedPrimitives(SDL_GPUCommandBuffer *commandBuffer, Uint32 numIndices, Uint32 numInstances, Uint32 firstIndex, Sint32 vertexOffset, Uint32 firstInstance)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    if (WebGPU_PrepareDraw(cb)) {
        wgpuRenderPassEncoderDrawIndexed(cb->renderPass, numIndices, numInstances, firstIndex, vertexOffset, firstInstance);
    }
}

static void WEBGPU_DrawPrimitivesIndirect(SDL_GPUCommandBuffer *commandBuffer, SDL_GPUBuffer *buffer, Uint32 offset, Uint32 drawCount)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    Uint32 i;
    if (WebGPU_PrepareDraw(cb)) {
        for (i = 0; i < drawCount; i++) {
            wgpuRenderPassEncoderDrawIndirect(cb->renderPass, ((WebGPUBuffer *)buffer)->buffer, offset + i * sizeof(SDL_GPUIndirectDrawCommand));
        }
    }
}

static void WEBGPU_DrawIndexedPrimitivesIndirect(SDL_GPUCommandBuffer *commandBuffer, SDL_GPUBuffer *buffer, Uint32 offset, Uint32 drawCount)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    Uint32 i;
    if (WebGPU_PrepareDraw(cb)) {
        for (i = 0; i < drawCount; i++) {
            wgpuRenderPassEncoderDrawIndexedIndirect(cb->renderPass, ((WebGPUBuffer *)buffer)->buffer, offset + i * sizeof(SDL_GPUIndexedIndirectDrawCommand));
        }
    }
}

// --- Compute (not implemented) ------------------------------------------------------------

static void WEBGPU_BeginComputePass(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUStorageTextureReadWriteBinding *storageTextureBindings, Uint32 numStorageTextureBindings, const SDL_GPUStorageBufferReadWriteBinding *storageBufferBindings, Uint32 numStorageBufferBindings)
{
    (void)commandBuffer;
    (void)storageTextureBindings;
    (void)numStorageTextureBindings;
    (void)storageBufferBindings;
    (void)numStorageBufferBindings;
    SDL_LogError(SDL_LOG_CATEGORY_GPU, "WebGPU: compute passes are not implemented");
}

static void WEBGPU_BindComputePipeline(SDL_GPUCommandBuffer *commandBuffer, SDL_GPUComputePipeline *computePipeline) { (void)commandBuffer; (void)computePipeline; }
static void WEBGPU_BindComputeSamplers(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, const SDL_GPUTextureSamplerBinding *textureSamplerBindings, Uint32 numBindings) { (void)commandBuffer; (void)firstSlot; (void)textureSamplerBindings; (void)numBindings; }
static void WEBGPU_BindComputeStorageTextures(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, SDL_GPUTexture *const *storageTextures, Uint32 numBindings) { (void)commandBuffer; (void)firstSlot; (void)storageTextures; (void)numBindings; }
static void WEBGPU_BindComputeStorageBuffers(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, SDL_GPUBuffer *const *storageBuffers, Uint32 numBindings) { (void)commandBuffer; (void)firstSlot; (void)storageBuffers; (void)numBindings; }
static void WEBGPU_PushComputeUniformData(SDL_GPUCommandBuffer *commandBuffer, Uint32 slotIndex, const void *data, Uint32 length) { (void)commandBuffer; (void)slotIndex; (void)data; (void)length; }
static void WEBGPU_DispatchCompute(SDL_GPUCommandBuffer *commandBuffer, Uint32 groupcountX, Uint32 groupcountY, Uint32 groupcountZ) { (void)commandBuffer; (void)groupcountX; (void)groupcountY; (void)groupcountZ; }
static void WEBGPU_DispatchComputeIndirect(SDL_GPUCommandBuffer *commandBuffer, SDL_GPUBuffer *buffer, Uint32 offset) { (void)commandBuffer; (void)buffer; (void)offset; }
static void WEBGPU_EndComputePass(SDL_GPUCommandBuffer *commandBuffer) { (void)commandBuffer; }

// --- Copy passes --------------------------------------------------------------------------

static void WEBGPU_BeginCopyPass(SDL_GPUCommandBuffer *commandBuffer) { (void)commandBuffer; }
static void WEBGPU_EndCopyPass(SDL_GPUCommandBuffer *commandBuffer) { (void)commandBuffer; }

static void WebGPU_DestroyStagingChunk(WebGPUStagingChunk *chunk)
{
    wgpuBufferDestroy(chunk->buffer);
    wgpuBufferRelease(chunk->buffer);
    SDL_free(chunk->cpu);
}

// A standard chunk from the pool, or a new one (one of exactly `size` bytes when more is needed).
static bool WebGPU_AcquireStagingChunk(WebGPURenderer *renderer, Uint32 size, WebGPUStagingChunk *chunk)
{
    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    if (size <= WEBGPU_STAGING_CHUNK_SIZE && renderer->freeStagingCount > 0) {
        *chunk = renderer->freeStaging[--renderer->freeStagingCount];
        chunk->used = 0;
        return true;
    }
    chunk->size = SDL_max(size, (Uint32)WEBGPU_STAGING_CHUNK_SIZE);
    chunk->used = 0;
    chunk->cpu = (Uint8 *)SDL_malloc(chunk->size);
    if (!chunk->cpu) {
        return false;
    }
    desc.size = chunk->size;
    desc.usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    chunk->buffer = wgpuDeviceCreateBuffer(renderer->device, &desc);
    return true;
}

// Done with a chunk: its queue write and the copies from it are already submitted.
static void WebGPU_RecycleStagingChunk(WebGPURenderer *renderer, WebGPUStagingChunk *chunk)
{
    if (chunk->size != WEBGPU_STAGING_CHUNK_SIZE || renderer->freeStagingCount >= WEBGPU_STAGING_POOL_LIMIT) {
        WebGPU_DestroyStagingChunk(chunk);
        return;
    }
    EXPAND_ARRAY_IF_NEEDED(renderer->freeStaging, WebGPUStagingChunk, renderer->freeStagingCount + 1, renderer->freeStagingCapacity, renderer->freeStagingCapacity * 2 + 4);
    renderer->freeStaging[renderer->freeStagingCount++] = *chunk;
}

// Space in a staging buffer that is copied from in this command buffer; NULL when out of memory.
static Uint8 *WebGPU_AllocateStaging(WebGPUCommandBuffer *cb, Uint32 size, WGPUBuffer *bufferOut, Uint32 *offsetOut)
{
    WebGPUStagingChunk *chunk = cb->stagingCount > 0 ? &cb->staging[cb->stagingCount - 1] : NULL;
    const Uint32 alignedSize = WebGPU_Align(size, 16);

    if (!chunk || chunk->used + alignedSize > chunk->size) {
        WebGPUStagingChunk fresh;
        if (!WebGPU_AcquireStagingChunk(cb->renderer, alignedSize, &fresh)) {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "WebGPU: out of memory for %u bytes of upload data", (unsigned)size);
            return NULL;
        }
        EXPAND_ARRAY_IF_NEEDED(cb->staging, WebGPUStagingChunk, cb->stagingCount + 1, cb->stagingCapacity, cb->stagingCapacity * 2 + 4);
        chunk = &cb->staging[cb->stagingCount++];
        *chunk = fresh;
    }

    *bufferOut = chunk->buffer;
    *offsetOut = chunk->used;
    chunk->used += alignedSize;
    return chunk->cpu + *offsetOut;
}

static void WebGPU_TextureCopyGeometry(const WebGPUTexture *texture, Uint32 w, Uint32 h, Uint32 *blockRows, Uint32 *rowBytesForWidth)
{
    const SDL_GPUTextureFormat format = texture->common.info.format;
    const Uint32 blockHeight = (Uint32)Texture_GetBlockHeight(format);
    *blockRows = (h + blockHeight - 1) / blockHeight;
    *rowBytesForWidth = BytesPerRow((Sint32)w, format);
}

static void WEBGPU_UploadToTexture(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUTextureTransferInfo *source, const SDL_GPUTextureRegion *destination, bool cycle)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUTransferBuffer *transfer = (WebGPUTransferBuffer *)source->transfer_buffer;
    WebGPUTexture *texture = (WebGPUTexture *)destination->texture;
    const SDL_GPUTextureFormat format = texture->common.info.format;
    const bool is3D = texture->common.info.type == SDL_GPU_TEXTURETYPE_3D;
    const Uint32 depth = SDL_max(1u, destination->d);
    Uint32 sourcePitch, tightPitch, blockRows, sourceRows, stagingPitch, layer, row, stagingOffset;
    Uint32 unused;
    WGPUBuffer stagingBuffer;
    Uint8 *staging;
    WGPUTexelCopyBufferInfo copySource;
    WGPUTexelCopyTextureInfo copyDestination;
    WGPUExtent3D extent;
    (void)cycle;

    WebGPU_TextureCopyGeometry(texture, destination->w, destination->h, &blockRows, &tightPitch);
    sourcePitch = BytesPerRow((Sint32)(source->pixels_per_row ? source->pixels_per_row : destination->w), format);
    WebGPU_TextureCopyGeometry(texture, destination->w, source->rows_per_layer ? source->rows_per_layer : destination->h, &sourceRows, &unused);
    stagingPitch = WebGPU_Align(tightPitch, WEBGPU_COPY_ROW_ALIGNMENT);

    staging = WebGPU_AllocateStaging(cb, stagingPitch * blockRows * depth, &stagingBuffer, &stagingOffset);
    if (!staging) {
        return;
    }
    for (layer = 0; layer < depth; layer++) {
        for (row = 0; row < blockRows; row++) {
            const Uint64 from = source->offset + ((Uint64)layer * sourceRows + row) * sourcePitch;
            if (from + tightPitch <= transfer->size) {
                SDL_memcpy(staging + ((Uint64)layer * blockRows + row) * stagingPitch, transfer->data + from, tightPitch);
            }
        }
    }

    copySource.buffer = stagingBuffer;
    copySource.layout.offset = stagingOffset;
    copySource.layout.bytesPerRow = stagingPitch;
    copySource.layout.rowsPerImage = blockRows;
    copyDestination.texture = texture->texture;
    copyDestination.mipLevel = destination->mip_level;
    copyDestination.origin.x = destination->x;
    copyDestination.origin.y = destination->y;
    copyDestination.origin.z = is3D ? destination->z : destination->layer;
    copyDestination.aspect = WGPUTextureAspect_All;
    extent.width = destination->w;
    extent.height = destination->h;
    extent.depthOrArrayLayers = depth;
    wgpuCommandEncoderCopyBufferToTexture(cb->encoder, &copySource, &copyDestination, &extent);
}

static void WEBGPU_UploadToBuffer(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUTransferBufferLocation *source, const SDL_GPUBufferRegion *destination, bool cycle)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUTransferBuffer *transfer = (WebGPUTransferBuffer *)source->transfer_buffer;
    WebGPUBuffer *buffer = (WebGPUBuffer *)destination->buffer;
    const Uint32 size = WebGPU_Align(destination->size, 4);
    WGPUBuffer stagingBuffer;
    Uint32 stagingOffset;
    Uint8 *staging;
    (void)cycle;

    if ((destination->offset & 3) != 0 || destination->offset + size > buffer->size) {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "WebGPU: buffer uploads need a 4 byte aligned destination");
        return;
    }
    staging = WebGPU_AllocateStaging(cb, size, &stagingBuffer, &stagingOffset);
    if (!staging) {
        return;
    }
    SDL_memcpy(staging, transfer->data + source->offset, SDL_min(destination->size, transfer->size - source->offset));
    wgpuCommandEncoderCopyBufferToBuffer(cb->encoder, stagingBuffer, stagingOffset, buffer->buffer, destination->offset, size);
}

static void WEBGPU_CopyTextureToTexture(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUTextureLocation *source, const SDL_GPUTextureLocation *destination, Uint32 w, Uint32 h, Uint32 d, bool cycle)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUTexture *from = (WebGPUTexture *)source->texture;
    WebGPUTexture *to = (WebGPUTexture *)destination->texture;
    WGPUTexelCopyTextureInfo src, dst;
    WGPUExtent3D extent;
    (void)cycle;
    src.texture = from->texture;
    src.mipLevel = source->mip_level;
    src.origin.x = source->x;
    src.origin.y = source->y;
    src.origin.z = from->common.info.type == SDL_GPU_TEXTURETYPE_3D ? source->z : source->layer;
    src.aspect = WGPUTextureAspect_All;
    dst.texture = to->texture;
    dst.mipLevel = destination->mip_level;
    dst.origin.x = destination->x;
    dst.origin.y = destination->y;
    dst.origin.z = to->common.info.type == SDL_GPU_TEXTURETYPE_3D ? destination->z : destination->layer;
    dst.aspect = WGPUTextureAspect_All;
    extent.width = w;
    extent.height = h;
    extent.depthOrArrayLayers = SDL_max(1u, d);
    wgpuCommandEncoderCopyTextureToTexture(cb->encoder, &src, &dst, &extent);
}

static void WEBGPU_CopyBufferToBuffer(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUBufferLocation *source, const SDL_GPUBufferLocation *destination, Uint32 size, bool cycle)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    (void)cycle;
    wgpuCommandEncoderCopyBufferToBuffer(cb->encoder, ((WebGPUBuffer *)source->buffer)->buffer, source->offset,
                                         ((WebGPUBuffer *)destination->buffer)->buffer, destination->offset, WebGPU_Align(size, 4));
}

static void WebGPU_AddDownload(WebGPUTransferBuffer *transfer, const WebGPUDownload *download)
{
    EXPAND_ARRAY_IF_NEEDED(transfer->downloads, WebGPUDownload, transfer->downloadCount + 1, transfer->downloadCapacity, transfer->downloadCapacity * 2 + 2);
    transfer->downloads[transfer->downloadCount++] = *download;
}

static void WEBGPU_DownloadFromTexture(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUTextureRegion *source, const SDL_GPUTextureTransferInfo *destination)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUTexture *texture = (WebGPUTexture *)source->texture;
    WebGPUTransferBuffer *transfer = (WebGPUTransferBuffer *)destination->transfer_buffer;
    const bool is3D = texture->common.info.type == SDL_GPU_TEXTURETYPE_3D;
    WebGPUDownload download;
    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    WGPUTexelCopyTextureInfo src;
    WGPUTexelCopyBufferInfo dst;
    WGPUExtent3D extent;
    Uint32 tightPitch, blockRows, transferRows, unused;

    SDL_zero(download);
    WebGPU_TextureCopyGeometry(texture, source->w, source->h, &blockRows, &tightPitch);
    WebGPU_TextureCopyGeometry(texture, source->w, destination->rows_per_layer ? destination->rows_per_layer : source->h, &transferRows, &unused);
    download.rowBytes = tightPitch;
    download.transferPitch = destination->pixels_per_row ? BytesPerRow((Sint32)destination->pixels_per_row, texture->common.info.format) : tightPitch;
    download.readbackPitch = WebGPU_Align(tightPitch, WEBGPU_COPY_ROW_ALIGNMENT);
    download.rowsPerLayer = blockRows;
    download.transferRowsPerLayer = transferRows;
    download.rows = blockRows;
    download.layers = SDL_max(1u, source->d);
    download.transferOffset = destination->offset;

    desc.size = (Uint64)download.readbackPitch * download.rowsPerLayer * download.layers;
    desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    download.readback = wgpuDeviceCreateBuffer(cb->renderer->device, &desc);

    src.texture = texture->texture;
    src.mipLevel = source->mip_level;
    src.origin.x = source->x;
    src.origin.y = source->y;
    src.origin.z = is3D ? source->z : source->layer;
    src.aspect = WGPUTextureAspect_All;
    dst.buffer = download.readback;
    dst.layout.offset = 0;
    dst.layout.bytesPerRow = download.readbackPitch;
    dst.layout.rowsPerImage = download.rowsPerLayer;
    extent.width = source->w;
    extent.height = source->h;
    extent.depthOrArrayLayers = download.layers;
    wgpuCommandEncoderCopyTextureToBuffer(cb->encoder, &src, &dst, &extent);
    WebGPU_AddDownload(transfer, &download);
    cb->hasDownloads = true;
}

static void WEBGPU_DownloadFromBuffer(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUBufferRegion *source, const SDL_GPUTransferBufferLocation *destination)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUTransferBuffer *transfer = (WebGPUTransferBuffer *)destination->transfer_buffer;
    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    WebGPUDownload download;
    const Uint32 size = WebGPU_Align(source->size, 4);

    SDL_zero(download);
    desc.size = size;
    desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    download.readback = wgpuDeviceCreateBuffer(cb->renderer->device, &desc);
    download.transferOffset = destination->offset;
    download.readbackPitch = size;
    download.rowBytes = source->size;
    download.transferPitch = source->size;
    download.rowsPerLayer = 1;
    download.transferRowsPerLayer = 1;
    download.rows = 1;
    download.layers = 1;
    wgpuCommandEncoderCopyBufferToBuffer(cb->encoder, ((WebGPUBuffer *)source->buffer)->buffer, source->offset, download.readback, 0, size);
    WebGPU_AddDownload(transfer, &download);
    cb->hasDownloads = true;
}

static void WEBGPU_Blit(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUBlitInfo *info)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    WebGPURenderer *renderer = cb->renderer;
    SDL_GPU_BlitCommon(commandBuffer, info, renderer->blitLinearSampler, renderer->blitNearestSampler,
                       renderer->blitVertexShader, renderer->blitFrom2DShader, renderer->blitFrom2DArrayShader,
                       renderer->blitFrom3DShader, renderer->blitFromCubeShader, renderer->blitFromCubeArrayShader,
                       &renderer->blitPipelines, &renderer->blitPipelineCount, &renderer->blitPipelineCapacity);
}

static void WEBGPU_GenerateMipmaps(SDL_GPUCommandBuffer *commandBuffer, SDL_GPUTexture *texture)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUTexture *webgpuTexture = (WebGPUTexture *)texture;
    const SDL_GPUTextureCreateInfo *info = &webgpuTexture->common.info;
    const Uint32 layers = info->type == SDL_GPU_TEXTURETYPE_3D ? info->layer_count_or_depth : webgpuTexture->layers;
    Uint32 level, layer;

    cb->common.ignore_render_pass_texture_validation = true;
    for (layer = 0; layer < layers; layer++) {
        for (level = 1; level < info->num_levels; level++) {
            SDL_GPUBlitInfo blit;
            SDL_zero(blit);
            blit.source.texture = texture;
            blit.source.mip_level = level - 1;
            blit.source.layer_or_depth_plane = layer;
            blit.source.w = SDL_max(1u, info->width >> (level - 1));
            blit.source.h = SDL_max(1u, info->height >> (level - 1));
            blit.destination.texture = texture;
            blit.destination.mip_level = level;
            blit.destination.layer_or_depth_plane = layer;
            blit.destination.w = SDL_max(1u, info->width >> level);
            blit.destination.h = SDL_max(1u, info->height >> level);
            blit.load_op = SDL_GPU_LOADOP_DONT_CARE;
            blit.filter = SDL_GPU_FILTER_LINEAR;
            WEBGPU_Blit(commandBuffer, &blit);
        }
    }
    cb->common.ignore_render_pass_texture_validation = false;
}

// --- Swapchain ----------------------------------------------------------------------------

static WebGPUWindowData *WebGPU_FindWindow(WebGPURenderer *renderer, SDL_Window *window)
{
    Uint32 i;
    for (i = 0; i < renderer->windowCount; i++) {
        if (renderer->windows[i]->window == window) {
            return renderer->windows[i];
        }
    }
    return NULL;
}

static void WebGPU_ConfigureSurface(WebGPURenderer *renderer, WebGPUWindowData *data, Uint32 width, Uint32 height)
{
    WGPUSurfaceConfiguration config = WGPU_SURFACE_CONFIGURATION_INIT;
    config.device = renderer->device;
    config.format = data->format;
    config.usage = data->usage;
    config.width = SDL_max(1u, width);
    config.height = SDL_max(1u, height);
    config.alphaMode = WGPUCompositeAlphaMode_Opaque;
    config.presentMode = WGPUPresentMode_Fifo;
    wgpuSurfaceConfigure(data->surface, &config);
    data->width = config.width;
    data->height = config.height;

    {
        WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
        WebGPUTexture *back = &data->swapchainTexture;
        if (back->sampleView) {
            wgpuTextureViewRelease(back->sampleView);
        }
        if (back->texture) {
            wgpuTextureDestroy(back->texture);
            wgpuTextureRelease(back->texture);
        }
        desc.label = WebGPU_String("SDL swapchain back buffer");
        desc.size.width = data->width;
        desc.size.height = data->height;
        desc.size.depthOrArrayLayers = 1;
        desc.format = data->format;
        desc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding;
        back->texture = wgpuDeviceCreateTexture(renderer->device, &desc);
        back->sampleView = wgpuTextureCreateView(back->texture, NULL);
        back->common.info.width = data->width;
        back->common.info.height = data->height;
    }
}

static void WebGPU_CopyToCanvas(WebGPUCommandBuffer *cb, WebGPUWindowData *data, WGPUTexture *canvasOut)
{
    WGPUSurfaceTexture surfaceTexture = WGPU_SURFACE_TEXTURE_INIT;
    WGPUTexelCopyTextureInfo src, dst;
    WGPUExtent3D extent;
    (void)cb;

    *canvasOut = NULL;
    wgpuSurfaceGetCurrentTexture(data->surface, &surfaceTexture);
    if (!surfaceTexture.texture ||
        (surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
         surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal)) {
        return;
    }
    SDL_zero(src);
    SDL_zero(dst);
    src.texture = data->swapchainTexture.texture;
    src.aspect = WGPUTextureAspect_All;
    dst.texture = surfaceTexture.texture;
    dst.aspect = WGPUTextureAspect_All;
    extent.width = SDL_min(data->width, wgpuTextureGetWidth(surfaceTexture.texture));
    extent.height = SDL_min(data->height, wgpuTextureGetHeight(surfaceTexture.texture));
    extent.depthOrArrayLayers = 1;
    wgpuCommandEncoderCopyTextureToTexture(cb->encoder, &src, &dst, &extent);
    *canvasOut = surfaceTexture.texture;
}

static bool WEBGPU_SupportsSwapchainComposition(SDL_GPURenderer *driverData, SDL_Window *window, SDL_GPUSwapchainComposition swapchainComposition)
{
    (void)driverData;
    (void)window;
    return swapchainComposition == SDL_GPU_SWAPCHAINCOMPOSITION_SDR;
}

static bool WEBGPU_SupportsPresentMode(SDL_GPURenderer *driverData, SDL_Window *window, SDL_GPUPresentMode presentMode)
{
    (void)driverData;
    (void)window;
    // The browser presents on its own refresh.
    return presentMode == SDL_GPU_PRESENTMODE_VSYNC;
}

static bool WEBGPU_ClaimWindow(SDL_GPURenderer *driverData, SDL_Window *window)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WGPUEmscriptenSurfaceSourceCanvasHTMLSelector canvas = WGPU_EMSCRIPTEN_SURFACE_SOURCE_CANVAS_HTML_SELECTOR_INIT;
    WGPUSurfaceDescriptor desc = WGPU_SURFACE_DESCRIPTOR_INIT;
    WGPUSurfaceCapabilities caps = WGPU_SURFACE_CAPABILITIES_INIT;
    WebGPUWindowData *data;
    const char *canvasId;
    char selector[256];
    int width = 0, height = 0;

    if (WebGPU_FindWindow(renderer, window)) {
        return SDL_SetError("WebGPU: window already claimed");
    }

    canvasId = SDL_GetStringProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_EMSCRIPTEN_CANVAS_ID_STRING, "#canvas");
    SDL_snprintf(selector, sizeof(selector), "%s%s", canvasId[0] == '#' ? "" : "#", canvasId);
    canvas.selector = WebGPU_String(selector);
    desc.nextInChain = &canvas.chain;

    data = (WebGPUWindowData *)SDL_calloc(1, sizeof(WebGPUWindowData));
    data->window = window;
    data->surface = wgpuInstanceCreateSurface(renderer->instance, &desc);
    if (!data->surface) {
        SDL_free(data);
        return SDL_SetError("WebGPU: could not create a surface for %s", selector);
    }

    wgpuSurfaceGetCapabilities(data->surface, renderer->adapter, &caps);
    data->format = caps.formatCount > 0 ? caps.formats[0] : WGPUTextureFormat_BGRA8Unorm;
    data->usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst;
    wgpuSurfaceCapabilitiesFreeMembers(caps);

    SDL_GetWindowSizeInPixels(window, &width, &height);
    WebGPU_ConfigureSurface(renderer, data, (Uint32)width, (Uint32)height);

    data->swapchainTexture.isSwapchain = true;
    data->swapchainTexture.format = data->format;
    data->swapchainTexture.layers = 1;
    data->swapchainTexture.common.info.type = SDL_GPU_TEXTURETYPE_2D;
    data->swapchainTexture.common.info.format = WebGPU_SDLFormat(data->format);
    data->swapchainTexture.common.info.usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET;
    data->swapchainTexture.common.info.num_levels = 1;
    data->swapchainTexture.common.info.layer_count_or_depth = 1;
    data->swapchainTexture.common.info.sample_count = SDL_GPU_SAMPLECOUNT_1;

    EXPAND_ARRAY_IF_NEEDED(renderer->windows, WebGPUWindowData *, renderer->windowCount + 1, renderer->windowCapacity, renderer->windowCapacity * 2 + 1);
    renderer->windows[renderer->windowCount++] = data;
    SDL_LogInfo(SDL_LOG_CATEGORY_GPU, "WebGPU: presenting to %s (%dx%d)", selector, width, height);
    return true;
}

static void WebGPU_ReleaseSwapchainFrame(WebGPUWindowData *data)
{
    if (data->swapchainTexture.sampleView) {
        wgpuTextureViewRelease(data->swapchainTexture.sampleView);
        data->swapchainTexture.sampleView = NULL;
    }
    if (data->swapchainTexture.texture) {
        wgpuTextureDestroy(data->swapchainTexture.texture);
        wgpuTextureRelease(data->swapchainTexture.texture);
        data->swapchainTexture.texture = NULL;
    }
}

static void WEBGPU_ReleaseWindow(SDL_GPURenderer *driverData, SDL_Window *window)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    Uint32 i;
    for (i = 0; i < renderer->windowCount; i++) {
        WebGPUWindowData *data = renderer->windows[i];
        if (data->window == window) {
            WebGPU_ReleaseSwapchainFrame(data);
            wgpuSurfaceUnconfigure(data->surface);
            wgpuSurfaceRelease(data->surface);
            SDL_free(data);
            renderer->windows[i] = renderer->windows[--renderer->windowCount];
            return;
        }
    }
}

static bool WEBGPU_SetSwapchainParameters(SDL_GPURenderer *driverData, SDL_Window *window, SDL_GPUSwapchainComposition swapchainComposition, SDL_GPUPresentMode presentMode)
{
    (void)driverData;
    (void)window;
    if (swapchainComposition != SDL_GPU_SWAPCHAINCOMPOSITION_SDR) {
        return SDL_SetError("WebGPU: only SDR swapchains are supported");
    }
    if (presentMode != SDL_GPU_PRESENTMODE_VSYNC) {
        return SDL_SetError("WebGPU: the browser always presents with vsync");
    }
    return true;
}

static bool WEBGPU_SetAllowedFramesInFlight(SDL_GPURenderer *driverData, Uint32 allowedFramesInFlight)
{
    (void)driverData;
    (void)allowedFramesInFlight;
    return true;
}

static SDL_GPUTextureFormat WEBGPU_GetSwapchainTextureFormat(SDL_GPURenderer *driverData, SDL_Window *window)
{
    WebGPUWindowData *data = WebGPU_FindWindow((WebGPURenderer *)driverData, window);
    return data ? WebGPU_SDLFormat(data->format) : SDL_GPU_TEXTUREFORMAT_INVALID;
}

static bool WEBGPU_AcquireSwapchainTexture(SDL_GPUCommandBuffer *commandBuffer, SDL_Window *window, SDL_GPUTexture **swapchainTexture, Uint32 *swapchainTextureWidth, Uint32 *swapchainTextureHeight)
{
    WebGPUCommandBuffer *cb = (WebGPUCommandBuffer *)commandBuffer;
    WebGPURenderer *renderer = cb->renderer;
    WebGPUWindowData *data = WebGPU_FindWindow(renderer, window);
    int width = 0, height = 0;
    Uint32 i;

    *swapchainTexture = NULL;
    if (!data) {
        return SDL_SetError("WebGPU: window not claimed");
    }

    SDL_GetWindowSizeInPixels(window, &width, &height);
    if (width <= 0 || height <= 0) {
        return true; // minimized: no texture this frame
    }
    if ((Uint32)width != data->width || (Uint32)height != data->height) {
        WebGPU_ConfigureSurface(renderer, data, (Uint32)width, (Uint32)height);
    }

    for (i = 0; i < cb->presentWindowCount && cb->presentWindows[i] != data; i++) {
    }
    if (i == cb->presentWindowCount && cb->presentWindowCount < SDL_arraysize(cb->presentWindows)) {
        cb->presentWindows[cb->presentWindowCount++] = data;
    }
    *swapchainTexture = (SDL_GPUTexture *)&data->swapchainTexture;
    if (swapchainTextureWidth) {
        *swapchainTextureWidth = data->width;
    }
    if (swapchainTextureHeight) {
        *swapchainTextureHeight = data->height;
    }
    return true;
}

static bool WEBGPU_WaitForSwapchain(SDL_GPURenderer *driverData, SDL_Window *window)
{
    (void)driverData;
    (void)window;
    return true;
}

static bool WEBGPU_WaitAndAcquireSwapchainTexture(SDL_GPUCommandBuffer *commandBuffer, SDL_Window *window, SDL_GPUTexture **swapchainTexture, Uint32 *swapchainTextureWidth, Uint32 *swapchainTextureHeight)
{
    return WEBGPU_AcquireSwapchainTexture(commandBuffer, window, swapchainTexture, swapchainTextureWidth, swapchainTextureHeight);
}

// --- Built-in blit shaders ----------------------------------------------------------------

static const char WebGPU_BlitVertexSource[] =
    "struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };\n"
    "@vertex fn main(@builtin(vertex_index) id: u32) -> Out {\n"
    "  let uv = vec2<f32>(f32((id << 1u) & 2u), f32(id & 2u));\n"
    "  var o: Out;\n"
    "  o.pos = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);\n"
    "  o.uv = uv;\n"
    "  return o;\n"
    "}\n";

#define WEBGPU_BLIT_UNIFORMS \
    "struct Blit { left: f32, top: f32, width: f32, height: f32, mip_level: u32, layer_or_depth: f32 };\n" \
    "@group(3) @binding(0) var<uniform> blit: Blit;\n" \
    "@group(2) @binding(1) var s: sampler;\n"

static const char WebGPU_BlitFrom2DSource[] =
    WEBGPU_BLIT_UNIFORMS
    "@group(2) @binding(0) var t: texture_2d<f32>;\n"
    "@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {\n"
    "  let c = vec2<f32>(blit.left, blit.top) + uv * vec2<f32>(blit.width, blit.height);\n"
    "  return textureSampleLevel(t, s, c, f32(blit.mip_level));\n"
    "}\n";

static const char WebGPU_BlitFrom2DArraySource[] =
    WEBGPU_BLIT_UNIFORMS
    "@group(2) @binding(0) var t: texture_2d_array<f32>;\n"
    "@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {\n"
    "  let c = vec2<f32>(blit.left, blit.top) + uv * vec2<f32>(blit.width, blit.height);\n"
    "  return textureSampleLevel(t, s, c, i32(blit.layer_or_depth), f32(blit.mip_level));\n"
    "}\n";

static const char WebGPU_BlitFrom3DSource[] =
    WEBGPU_BLIT_UNIFORMS
    "@group(2) @binding(0) var t: texture_3d<f32>;\n"
    "@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {\n"
    "  let c = vec2<f32>(blit.left, blit.top) + uv * vec2<f32>(blit.width, blit.height);\n"
    "  return textureSampleLevel(t, s, vec3<f32>(c, blit.layer_or_depth), f32(blit.mip_level));\n"
    "}\n";

#define WEBGPU_BLIT_CUBE_DIRECTION \
    "fn direction(c: vec2<f32>, face: u32) -> vec3<f32> {\n" \
    "  let p = c * 2.0 - 1.0;\n" \
    "  switch face {\n" \
    "    case 0u: { return vec3<f32>(1.0, -p.y, -p.x); }\n" \
    "    case 1u: { return vec3<f32>(-1.0, -p.y, p.x); }\n" \
    "    case 2u: { return vec3<f32>(p.x, 1.0, p.y); }\n" \
    "    case 3u: { return vec3<f32>(p.x, -1.0, -p.y); }\n" \
    "    case 4u: { return vec3<f32>(p.x, -p.y, 1.0); }\n" \
    "    default: { return vec3<f32>(-p.x, -p.y, -1.0); }\n" \
    "  }\n" \
    "}\n"

static const char WebGPU_BlitFromCubeSource[] =
    WEBGPU_BLIT_UNIFORMS
    WEBGPU_BLIT_CUBE_DIRECTION
    "@group(2) @binding(0) var t: texture_cube<f32>;\n"
    "@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {\n"
    "  let c = vec2<f32>(blit.left, blit.top) + uv * vec2<f32>(blit.width, blit.height);\n"
    "  return textureSampleLevel(t, s, direction(c, u32(blit.layer_or_depth)), f32(blit.mip_level));\n"
    "}\n";

static const char WebGPU_BlitFromCubeArraySource[] =
    WEBGPU_BLIT_UNIFORMS
    WEBGPU_BLIT_CUBE_DIRECTION
    "@group(2) @binding(0) var t: texture_cube_array<f32>;\n"
    "@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {\n"
    "  let c = vec2<f32>(blit.left, blit.top) + uv * vec2<f32>(blit.width, blit.height);\n"
    "  let layer = u32(blit.layer_or_depth);\n"
    "  return textureSampleLevel(t, s, direction(c, layer % 6u), i32(layer / 6u), f32(blit.mip_level));\n"
    "}\n";

static SDL_GPUShader *WebGPU_CreateBuiltinShader(WebGPURenderer *renderer, const char *source, SDL_GPUShaderStage stage, Uint32 samplers, Uint32 uniforms)
{
    SDL_GPUShaderCreateInfo info;
    SDL_zero(info);
    info.code = (const Uint8 *)source;
    info.code_size = SDL_strlen(source);
    info.entrypoint = "main";
    info.format = SDL_GPU_SHADERFORMAT_PRIVATE;
    info.stage = stage;
    info.num_samplers = samplers;
    info.num_uniform_buffers = uniforms;
    return WEBGPU_CreateShader((SDL_GPURenderer *)renderer, &info);
}

// --- Device -------------------------------------------------------------------------------

static void WEBGPU_DestroyDevice(SDL_GPUDevice *device)
{
    WebGPURenderer *renderer = (WebGPURenderer *)device->driverData;
    Uint32 i;
    int stage;

    while (renderer->windowCount > 0) {
        WEBGPU_ReleaseWindow((SDL_GPURenderer *)renderer, renderer->windows[0]->window);
    }
    SDL_free(renderer->windows);

    for (i = 0; i < renderer->blitPipelineCount; i++) {
        WEBGPU_ReleaseGraphicsPipeline((SDL_GPURenderer *)renderer, renderer->blitPipelines[i].pipeline);
    }
    SDL_free(renderer->blitPipelines);
    WEBGPU_ReleaseShader((SDL_GPURenderer *)renderer, renderer->blitVertexShader);
    WEBGPU_ReleaseShader((SDL_GPURenderer *)renderer, renderer->blitFrom2DShader);
    WEBGPU_ReleaseShader((SDL_GPURenderer *)renderer, renderer->blitFrom2DArrayShader);
    WEBGPU_ReleaseShader((SDL_GPURenderer *)renderer, renderer->blitFrom3DShader);
    WEBGPU_ReleaseShader((SDL_GPURenderer *)renderer, renderer->blitFromCubeShader);
    WEBGPU_ReleaseShader((SDL_GPURenderer *)renderer, renderer->blitFromCubeArrayShader);
    WEBGPU_ReleaseSampler((SDL_GPURenderer *)renderer, renderer->blitNearestSampler);
    WEBGPU_ReleaseSampler((SDL_GPURenderer *)renderer, renderer->blitLinearSampler);

    WEBGPU_ReleaseTexture((SDL_GPURenderer *)renderer, (SDL_GPUTexture *)renderer->dummyTexture);
    WEBGPU_ReleaseSampler((SDL_GPURenderer *)renderer, (SDL_GPUSampler *)renderer->dummySampler);
    WEBGPU_ReleaseBuffer((SDL_GPURenderer *)renderer, (SDL_GPUBuffer *)renderer->dummyStorage);

    for (i = 0; i < renderer->freeBlockCount; i++) {
        WebGPUUniformBlock *block = renderer->freeBlocks[i];
        for (stage = 0; stage < 2; stage++) {
            Uint32 count;
            for (count = 0; count <= WEBGPU_MAX_UNIFORMS; count++) {
                if (block->bindGroups[stage][count]) {
                    wgpuBindGroupRelease(block->bindGroups[stage][count]);
                }
            }
        }
        wgpuBufferDestroy(block->buffer);
        wgpuBufferRelease(block->buffer);
        SDL_free(block->cpu);
        SDL_free(block);
    }
    SDL_free(renderer->freeBlocks);

    for (i = 0; i < renderer->freeStagingCount; i++) {
        WebGPU_DestroyStagingChunk(&renderer->freeStaging[i]);
    }
    SDL_free(renderer->freeStaging);

    for (stage = 0; stage < 2; stage++) {
        for (i = 0; i <= WEBGPU_MAX_UNIFORMS; i++) {
            if (renderer->uniformLayouts[stage][i]) {
                wgpuBindGroupLayoutRelease(renderer->uniformLayouts[stage][i]);
            }
        }
    }
    wgpuBindGroupRelease(renderer->emptyGroup);
    wgpuBindGroupLayoutRelease(renderer->emptyLayout);

    wgpuQueueRelease(renderer->queue);
    wgpuDeviceDestroy(renderer->device);
    wgpuDeviceRelease(renderer->device);
    wgpuAdapterRelease(renderer->adapter);
    wgpuInstanceRelease(renderer->instance);
    SDL_DestroyProperties(renderer->properties);
    SDL_free(renderer);
    SDL_free(device);
}

EM_JS(int, WebGPU_BrowserSupportsWebGPU, (void), {
    return (typeof navigator !== 'undefined' && navigator.gpu) ? 1 : 0;
});

static bool WEBGPU_PrepareDriver(SDL_VideoDevice *_this, SDL_PropertiesID props)
{
    (void)_this;
    if (!SDL_GetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_PRIVATE_BOOLEAN, false)) {
        return false;
    }
    return WebGPU_BrowserSupportsWebGPU() != 0;
}

static void WebGPU_CreateSharedLayouts(WebGPURenderer *renderer)
{
    WGPUBindGroupLayoutDescriptor desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    WGPUBindGroupDescriptor groupDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    int stage;
    Uint32 count, i;

    renderer->emptyLayout = wgpuDeviceCreateBindGroupLayout(renderer->device, &desc);
    groupDesc.layout = renderer->emptyLayout;
    renderer->emptyGroup = wgpuDeviceCreateBindGroup(renderer->device, &groupDesc);

    for (stage = 0; stage < 2; stage++) {
        renderer->uniformLayouts[stage][0] = renderer->emptyLayout;
        wgpuBindGroupLayoutAddRef(renderer->emptyLayout);
        for (count = 1; count <= WEBGPU_MAX_UNIFORMS; count++) {
            WGPUBindGroupLayoutEntry entries[WEBGPU_MAX_UNIFORMS];
            for (i = 0; i < count; i++) {
                entries[i] = (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
                entries[i].binding = i;
                entries[i].visibility = stage == 0 ? WGPUShaderStage_Vertex : WGPUShaderStage_Fragment;
                entries[i].buffer.type = WGPUBufferBindingType_Uniform;
                entries[i].buffer.hasDynamicOffset = true;
            }
            desc.entryCount = count;
            desc.entries = entries;
            renderer->uniformLayouts[stage][count] = wgpuDeviceCreateBindGroupLayout(renderer->device, &desc);
        }
    }
}

static void WebGPU_CreateBuiltins(WebGPURenderer *renderer)
{
    SDL_GPUTextureCreateInfo textureInfo;
    SDL_GPUSamplerCreateInfo samplerInfo;
    WGPUTexelCopyTextureInfo destination;
    WGPUTexelCopyBufferLayout layout;
    WGPUExtent3D extent = { 1, 1, 1 };
    static const Uint8 white[4] = { 255, 255, 255, 255 };

    SDL_zero(textureInfo);
    textureInfo.type = SDL_GPU_TEXTURETYPE_2D;
    textureInfo.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    textureInfo.usage = SDL_GPU_TEXTUREUSAGE_SAMPLER;
    textureInfo.width = textureInfo.height = 1;
    textureInfo.layer_count_or_depth = 1;
    textureInfo.num_levels = 1;
    renderer->dummyTexture = WebGPU_CreateTextureInternal(renderer, &textureInfo, "SDL dummy texture");
    destination.texture = renderer->dummyTexture->texture;
    destination.mipLevel = 0;
    destination.origin.x = destination.origin.y = destination.origin.z = 0;
    destination.aspect = WGPUTextureAspect_All;
    layout.offset = 0;
    layout.bytesPerRow = 4;
    layout.rowsPerImage = 1;
    wgpuQueueWriteTexture(renderer->queue, &destination, white, sizeof(white), &layout, &extent);

    SDL_zero(samplerInfo);
    samplerInfo.min_filter = samplerInfo.mag_filter = SDL_GPU_FILTER_NEAREST;
    samplerInfo.address_mode_u = samplerInfo.address_mode_v = samplerInfo.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    samplerInfo.max_lod = 1000.0f;
    renderer->dummySampler = (WebGPUSampler *)WEBGPU_CreateSampler((SDL_GPURenderer *)renderer, &samplerInfo);
    renderer->blitNearestSampler = WEBGPU_CreateSampler((SDL_GPURenderer *)renderer, &samplerInfo);
    samplerInfo.min_filter = samplerInfo.mag_filter = SDL_GPU_FILTER_LINEAR;
    renderer->blitLinearSampler = WEBGPU_CreateSampler((SDL_GPURenderer *)renderer, &samplerInfo);

    renderer->dummyStorage = (WebGPUBuffer *)WEBGPU_CreateBuffer((SDL_GPURenderer *)renderer, SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ, 256, "SDL dummy storage");

    renderer->blitVertexShader = WebGPU_CreateBuiltinShader(renderer, WebGPU_BlitVertexSource, SDL_GPU_SHADERSTAGE_VERTEX, 0, 0);
    renderer->blitFrom2DShader = WebGPU_CreateBuiltinShader(renderer, WebGPU_BlitFrom2DSource, SDL_GPU_SHADERSTAGE_FRAGMENT, 1, 1);
    renderer->blitFrom2DArrayShader = WebGPU_CreateBuiltinShader(renderer, WebGPU_BlitFrom2DArraySource, SDL_GPU_SHADERSTAGE_FRAGMENT, 1, 1);
    renderer->blitFrom3DShader = WebGPU_CreateBuiltinShader(renderer, WebGPU_BlitFrom3DSource, SDL_GPU_SHADERSTAGE_FRAGMENT, 1, 1);
    renderer->blitFromCubeShader = WebGPU_CreateBuiltinShader(renderer, WebGPU_BlitFromCubeSource, SDL_GPU_SHADERSTAGE_FRAGMENT, 1, 1);
    renderer->blitFromCubeArrayShader = WebGPU_CreateBuiltinShader(renderer, WebGPU_BlitFromCubeArraySource, SDL_GPU_SHADERSTAGE_FRAGMENT, 1, 1);
}

static SDL_GPUDevice *WEBGPU_CreateDevice(bool debugMode, bool preferLowPower, SDL_PropertiesID props)
{
    WebGPURenderer *renderer;
    SDL_GPUDevice *result;
    WGPUInstanceDescriptor instanceDesc = WGPU_INSTANCE_DESCRIPTOR_INIT;
    WGPUInstanceFeatureName instanceFeatures[] = { WGPUInstanceFeatureName_TimedWaitAny };
    WGPURequestAdapterOptions adapterOptions = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    WGPURequestAdapterCallbackInfo adapterCallback = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    WGPURequestDeviceCallbackInfo deviceCallback = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    WGPUDeviceDescriptor deviceDesc = WGPU_DEVICE_DESCRIPTOR_INIT;
    WGPUFeatureName features[8];
    size_t featureCount = 0;
    WebGPURequestResult adapterResult = { false, false, NULL };
    WebGPURequestResult deviceResult = { false, false, NULL };
    (void)props;

    renderer = (WebGPURenderer *)SDL_calloc(1, sizeof(WebGPURenderer));
    instanceDesc.requiredFeatureCount = 1;
    instanceDesc.requiredFeatures = instanceFeatures;
    renderer->instance = wgpuCreateInstance(&instanceDesc);
    if (!renderer->instance) {
        SDL_free(renderer);
        SDL_SetError("WebGPU: could not create an instance");
        return NULL;
    }

    adapterOptions.powerPreference = preferLowPower ? WGPUPowerPreference_LowPower : WGPUPowerPreference_HighPerformance;
    adapterCallback.mode = WGPUCallbackMode_WaitAnyOnly;
    adapterCallback.callback = WebGPU_OnAdapter;
    adapterCallback.userdata1 = &adapterResult;
    WebGPU_WaitFor(renderer, wgpuInstanceRequestAdapter(renderer->instance, &adapterOptions, adapterCallback));
    if (!adapterResult.ok) {
        wgpuInstanceRelease(renderer->instance);
        SDL_free(renderer);
        SDL_SetError("WebGPU: no adapter");
        return NULL;
    }
    renderer->adapter = (WGPUAdapter)adapterResult.handle;

#define WEBGPU_OPTIONAL_FEATURE(name, flag)                         \
    if (wgpuAdapterHasFeature(renderer->adapter, name)) {           \
        features[featureCount++] = name;                            \
        renderer->flag = true;                                      \
    }
    WEBGPU_OPTIONAL_FEATURE(WGPUFeatureName_TextureCompressionBC, hasBC)
    WEBGPU_OPTIONAL_FEATURE(WGPUFeatureName_Depth32FloatStencil8, hasDepth32Stencil8)
    WEBGPU_OPTIONAL_FEATURE(WGPUFeatureName_ClipDistances, hasClipDistances)
    WEBGPU_OPTIONAL_FEATURE(WGPUFeatureName_Float32Filterable, hasFloat32Filterable)
#undef WEBGPU_OPTIONAL_FEATURE

    deviceDesc.label = WebGPU_String("SDL GPU");
    deviceDesc.requiredFeatureCount = featureCount;
    deviceDesc.requiredFeatures = features;
    deviceDesc.uncapturedErrorCallbackInfo.callback = WebGPU_OnUncapturedError;
    deviceDesc.uncapturedErrorCallbackInfo.userdata1 = renderer;
    deviceDesc.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
    deviceDesc.deviceLostCallbackInfo.callback = WebGPU_OnDeviceLost;
    deviceCallback.mode = WGPUCallbackMode_WaitAnyOnly;
    deviceCallback.callback = WebGPU_OnDevice;
    deviceCallback.userdata1 = &deviceResult;
    WebGPU_WaitFor(renderer, wgpuAdapterRequestDevice(renderer->adapter, &deviceDesc, deviceCallback));
    if (!deviceResult.ok) {
        wgpuAdapterRelease(renderer->adapter);
        wgpuInstanceRelease(renderer->instance);
        SDL_free(renderer);
        SDL_SetError("WebGPU: no device");
        return NULL;
    }
    renderer->device = (WGPUDevice)deviceResult.handle;
    renderer->queue = wgpuDeviceGetQueue(renderer->device);

    WebGPU_CreateSharedLayouts(renderer);
    WebGPU_CreateBuiltins(renderer);

    renderer->properties = SDL_CreateProperties();
    SDL_SetStringProperty(renderer->properties, SDL_PROP_GPU_DEVICE_NAME_STRING, "WebGPU");
    SDL_SetStringProperty(renderer->properties, SDL_PROP_GPU_DEVICE_DRIVER_NAME_STRING, WEBGPU_DRIVER_NAME);

    result = (SDL_GPUDevice *)SDL_calloc(1, sizeof(SDL_GPUDevice));
    ASSIGN_DRIVER(WEBGPU)
    result->driverData = (SDL_GPURenderer *)renderer;
    result->shader_formats = SDL_GPU_SHADERFORMAT_PRIVATE;
    result->debug_mode = debugMode;
    renderer->sdlDevice = result;

    SDL_LogInfo(SDL_LOG_CATEGORY_GPU, "WebGPU device created (BC textures: %s, clip distances: %s)",
                renderer->hasBC ? "yes" : "no", renderer->hasClipDistances ? "yes" : "no");
    return result;
}

SDL_GPUBootstrap PrivateGPUDriver = {
    WEBGPU_DRIVER_NAME,
    WEBGPU_PrepareDriver,
    WEBGPU_CreateDevice
};
