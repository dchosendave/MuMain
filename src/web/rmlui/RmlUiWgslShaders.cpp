// Web build: WGSL versions of RmlUi's SDL GPU shaders (ThirdParty/RmlUi/Backends/RmlUi_SDL_GPU/*.vert|frag),
// used by the WebGPU SDL GPU driver (web/sdl_gpu_webgpu). The build compiles a copy of
// RmlUi_Renderer_SDL_GPU.cpp that calls MuWebCreateRmlUiShader for SDL's "private" shader format.
//
// Bindings follow the driver's layout: vertex uniforms @group(1), fragment textures/samplers
// @group(2) (slot i -> @binding(2i), @binding(2i+1)), fragment uniforms @group(3). Vertex inputs
// are RmlUi's layout: position @location(0), colour @location(1), texture coordinate @location(2).
// HLSL mul(M, v) with column-major constant buffers is M * v in WGSL.
#include <RmlUi/Core/Log.h>
#include <SDL3/SDL_gpu.h>

namespace
{
#define RMLUI_WGSL_VARYINGS \
    "struct Varyings { @builtin(position) pos: vec4<f32>, @location(0) color: vec4<f32>, @location(1) uv: vec2<f32> };\n"
#define RMLUI_WGSL_POST_VARYINGS \
    "struct PostVaryings { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };\n"
#define RMLUI_WGSL_BLUR_VARYINGS                                                                                 \
    "struct BlurVaryings { @builtin(position) pos: vec4<f32>, @location(0) uv0: vec2<f32>, @location(1) uv1: vec2<f32>,\n" \
    "  @location(2) uv2: vec2<f32>, @location(3) uv3: vec2<f32>, @location(4) uv4: vec2<f32>, @location(5) uv5: vec2<f32>,\n" \
    "  @location(6) uv6: vec2<f32> };\n"
#define RMLUI_WGSL_TEXTURE                                     \
    "@group(2) @binding(0) var tex: texture_2d<f32>;\n"        \
    "@group(2) @binding(1) var texSampler: sampler;\n"

const char kVertMain[] =
    RMLUI_WGSL_VARYINGS
    "struct Transform { transform: mat4x4<f32> };\n"
    "struct Translate { translate: vec2<f32> };\n"
    "@group(1) @binding(0) var<uniform> t: Transform;\n"
    "@group(1) @binding(1) var<uniform> tr: Translate;\n"
    "@vertex fn main(@location(0) position: vec2<f32>, @location(1) color: vec4<f32>, @location(2) uv: vec2<f32>) -> Varyings {\n"
    "  var o: Varyings;\n"
    "  o.uv = uv;\n"
    "  o.color = color;\n"
    "  o.pos = t.transform * vec4<f32>(position + tr.translate, 0.0, 1.0);\n"
    "  return o;\n"
    "}\n";

const char kVertPassthrough[] =
    RMLUI_WGSL_POST_VARYINGS
    "struct Quad { quadPosition: vec4<f32>, quadTexCoord: vec4<f32> };\n"
    "@group(1) @binding(0) var<uniform> q: Quad;\n"
    "@vertex fn main(@location(2) uv: vec2<f32>) -> PostVaryings {\n"
    "  var o: PostVaryings;\n"
    "  o.uv = q.quadTexCoord.xy + uv * q.quadTexCoord.zw;\n"
    "  o.pos = vec4<f32>(mix(q.quadPosition.xy, q.quadPosition.zw, uv), 0.0, 1.0);\n"
    "  return o;\n"
    "}\n";

// BLUR_SIZE 7, BLUR_NUM_WEIGHTS 4: tex_coord - (i - 3) * TexelOffset for i = 0..6.
const char kVertBlur[] =
    RMLUI_WGSL_BLUR_VARYINGS
    "struct Blur { quadPosition: vec4<f32>, quadTexCoord: vec4<f32>, texelOffset: vec2<f32> };\n"
    "@group(1) @binding(0) var<uniform> b: Blur;\n"
    "@vertex fn main(@location(2) uv: vec2<f32>) -> BlurVaryings {\n"
    "  let c = b.quadTexCoord.xy + uv * b.quadTexCoord.zw;\n"
    "  var o: BlurVaryings;\n"
    "  o.uv0 = c + 3.0 * b.texelOffset;\n"
    "  o.uv1 = c + 2.0 * b.texelOffset;\n"
    "  o.uv2 = c + 1.0 * b.texelOffset;\n"
    "  o.uv3 = c;\n"
    "  o.uv4 = c - 1.0 * b.texelOffset;\n"
    "  o.uv5 = c - 2.0 * b.texelOffset;\n"
    "  o.uv6 = c - 3.0 * b.texelOffset;\n"
    "  o.pos = vec4<f32>(mix(b.quadPosition.xy, b.quadPosition.zw, uv), 0.0, 1.0);\n"
    "  return o;\n"
    "}\n";

const char kFragColor[] =
    "@fragment fn main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {\n"
    "  return color;\n"
    "}\n";

const char kFragTexture[] =
    RMLUI_WGSL_TEXTURE
    "@fragment fn main(@location(0) color: vec4<f32>, @location(1) uv: vec2<f32>) -> @location(0) vec4<f32> {\n"
    "  return color * textureSample(tex, texSampler, uv);\n"
    "}\n";

// MAX_NUM_STOPS 16, packed four stop positions per vec4.
const char kFragGradient[] =
    "struct Gradient { p: vec2<f32>, v: vec2<f32>, func: i32, numStops: i32,\n"
    "  stopColors: array<vec4<f32>, 16>, stopPositions: array<vec4<f32>, 4> };\n"
    "@group(3) @binding(0) var<uniform> g: Gradient;\n"
    "fn stopPos(i: i32) -> f32 { return g.stopPositions[i >> 2u][i & 3]; }\n"
    "fn glslMod(x: f32, y: f32) -> f32 { return x - y * floor(x / y); }\n"
    "fn lerpStopColors(t: f32) -> vec4<f32> {\n"
    "  var color = g.stopColors[0];\n"
    "  for (var i = 1; i < g.numStops; i++) {\n"
    "    color = mix(color, g.stopColors[i], smoothstep(stopPos(i - 1), stopPos(i), t));\n"
    "  }\n"
    "  return color;\n"
    "}\n"
    "@fragment fn main(@location(0) color: vec4<f32>, @location(1) uv: vec2<f32>) -> @location(0) vec4<f32> {\n"
    "  var t = 0.0;\n"
    "  if (g.func == 0 || g.func == 3) {\n"
    "    let delta = uv - g.p;\n"
    "    t = dot(g.v, delta) / dot(g.v, g.v);\n"
    "  } else if (g.func == 1 || g.func == 4) {\n"
    "    t = length(g.v * (uv - g.p));\n"
    "  } else if (g.func == 2 || g.func == 5) {\n"
    // HLSL: float2x2 R = float2x2(V.x, -V.y, V.y, V.x) (rows); mul(delta, R).
    "    let d = uv - g.p;\n"
    "    let rotated = vec2<f32>(d.x * g.v.x + d.y * g.v.y, -d.x * g.v.y + d.y * g.v.x);\n"
    "    t = 0.5 + atan2(-rotated.x, rotated.y) / (2.0 * 3.14159265);\n"
    "  }\n"
    "  if (g.func == 3 || g.func == 4 || g.func == 5) {\n"
    "    let t0 = stopPos(0);\n"
    "    let t1 = stopPos(g.numStops - 1);\n"
    "    t = t0 + glslMod(t - t0, t1 - t0);\n"
    "  }\n"
    "  return color * lerpStopColors(t);\n"
    "}\n";

const char kFragCreation[] =
    "struct Creation { dimensions: vec2<f32>, value: f32 };\n"
    "@group(3) @binding(0) var<uniform> cr: Creation;\n"
    "fn glslMod2(x: vec2<f32>, y: f32) -> vec2<f32> { return x - y * floor(x / y); }\n"
    "@fragment fn main(@location(0) color: vec4<f32>, @location(1) uvIn: vec2<f32>) -> @location(0) vec4<f32> {\n"
    "  let t = cr.value;\n"
    "  var c = vec3<f32>(0.0);\n"
    "  var l = 0.0;\n"
    "  for (var i = 0; i < 3; i++) {\n"
    "    var p = uvIn;\n"
    "    var uv = p;\n"
    "    p -= vec2<f32>(0.5);\n"
    "    p.x *= cr.dimensions.x / cr.dimensions.y;\n"
    "    let z = t + f32(i) * 0.07;\n"
    "    l = length(p);\n"
    "    uv += p / l * (sin(z) + 1.0) * abs(sin(l * 9.0 - z - z));\n"
    "    c[i] = 0.01 / length(glslMod2(uv, 1.0) - vec2<f32>(0.5));\n"
    "  }\n"
    "  return vec4<f32>(c / l, color.a);\n"
    "}\n";

const char kFragPassthrough[] =
    RMLUI_WGSL_TEXTURE
    "@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {\n"
    "  return textureSample(tex, texSampler, uv);\n"
    "}\n";

const char kFragColorMatrix[] =
    RMLUI_WGSL_TEXTURE
    "struct ColorMatrix { m: mat4x4<f32> };\n"
    "@group(3) @binding(0) var<uniform> cm: ColorMatrix;\n"
    "@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {\n"
    "  let texColor = textureSample(tex, texSampler, uv);\n"
    "  return vec4<f32>((cm.m * texColor).rgb, texColor.a);\n"
    "}\n";

const char kFragBlendMask[] =
    RMLUI_WGSL_TEXTURE
    "@group(2) @binding(2) var maskTex: texture_2d<f32>;\n"
    "@group(2) @binding(3) var maskSampler: sampler;\n"
    "@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {\n"
    "  let texColor = textureSample(tex, texSampler, uv);\n"
    "  let maskAlpha = textureSample(maskTex, maskSampler, uv).a;\n"
    "  return texColor * maskAlpha;\n"
    "}\n";

// Weights[abs(i - 3)] for i = 0..6.
const char kFragBlur[] =
    RMLUI_WGSL_TEXTURE
    RMLUI_WGSL_BLUR_VARYINGS
    "struct Blur { weights: vec4<f32>, texCoordMin: vec2<f32>, texCoordMax: vec2<f32> };\n"
    "@group(3) @binding(0) var<uniform> b: Blur;\n"
    "fn tap(uv: vec2<f32>, weight: f32) -> vec4<f32> {\n"
    "  let inRegion = step(b.texCoordMin, uv) * step(uv, b.texCoordMax);\n"
    "  return textureSample(tex, texSampler, uv) * inRegion.x * inRegion.y * weight;\n"
    "}\n"
    "@fragment fn main(input: BlurVaryings) -> @location(0) vec4<f32> {\n"
    "  var color = tap(input.uv0, b.weights[3]);\n"
    "  color += tap(input.uv1, b.weights[2]);\n"
    "  color += tap(input.uv2, b.weights[1]);\n"
    "  color += tap(input.uv3, b.weights[0]);\n"
    "  color += tap(input.uv4, b.weights[1]);\n"
    "  color += tap(input.uv5, b.weights[2]);\n"
    "  color += tap(input.uv6, b.weights[3]);\n"
    "  return color;\n"
    "}\n";

const char kFragDropShadow[] =
    RMLUI_WGSL_TEXTURE
    "struct DropShadow { texCoordMin: vec2<f32>, texCoordMax: vec2<f32>, color: vec4<f32> };\n"
    "@group(3) @binding(0) var<uniform> d: DropShadow;\n"
    "@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {\n"
    "  let inRegion = step(d.texCoordMin, uv) * step(uv, d.texCoordMax);\n"
    "  return textureSample(tex, texSampler, uv).a * inRegion.x * inRegion.y * d.color;\n"
    "}\n";

// In RenderInterface_SDL_GPU::ShaderId order.
const char* const kShaders[] = {
    kVertMain, kVertPassthrough, kVertBlur, kFragColor, kFragTexture, kFragGradient,
    kFragCreation, kFragPassthrough, kFragColorMatrix, kFragBlendMask, kFragBlur, kFragDropShadow,
};
} // namespace

SDL_GPUShader* MuWebCreateRmlUiShader(SDL_GPUDevice* device, int id, SDL_GPUShaderStage stage, int samplers, int uniforms)
{
    if (id < 0 || id >= static_cast<int>(sizeof(kShaders) / sizeof(kShaders[0])))
    {
        Rml::Log::Message(Rml::Log::LT_ERROR, "No WGSL shader for RmlUi shader %d", id);
        return nullptr;
    }

    SDL_GPUShaderCreateInfo info{};
    info.code = reinterpret_cast<const Uint8*>(kShaders[id]);
    info.code_size = SDL_strlen(kShaders[id]);
    info.entrypoint = "main";
    info.format = SDL_GPU_SHADERFORMAT_PRIVATE;
    info.stage = stage;
    info.num_samplers = static_cast<Uint32>(samplers);
    info.num_uniform_buffers = static_cast<Uint32>(uniforms);
    SDL_GPUShader* shader = SDL_CreateGPUShader(device, &info);
    if (!shader)
    {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create WGSL shader %d: %s", id, SDL_GetError());
    }
    return shader;
}
