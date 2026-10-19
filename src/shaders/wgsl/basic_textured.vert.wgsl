// WGSL port of basic_textured.vert.hlsl.
// The textured vertex shaders (basic_textured, skinned_textured, skinned_shadow) share one output
// struct, so any of them pairs with any textured fragment shader: WebGPU requires every fragment
// input to be written by the vertex stage.
enable clip_distances; // webgpu:clip-distances

struct Transform { mvp: mat4x4<f32>, fogStart: f32, fogEnd: f32, fogPad: vec2<f32> };
@group(1) @binding(0) var<uniform> u: Transform;

struct VSOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) fogFactor: f32,
    @location(3) eyeDepth: f32,
    @location(4) normal: vec3<f32>,
    @location(5) viewVector: vec3<f32>,
    @location(6) @interpolate(flat) lightDirectionAndSpecular: vec4<f32>,
    @location(7) worldPositionAndDynamicLights: vec4<f32>,
    @location(8) worldNormal: vec3<f32>,
    @builtin(clip_distances) clip: array<f32, 1>, // webgpu:clip-distances
};

@vertex fn main(@location(0) pos: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) color: vec4<f32>) -> VSOutput {
    var o: VSOutput;
    o.pos = u.mvp * vec4<f32>(pos, 1.0);
    o.uv = uv;
    o.color = color;
    // Near-plane user clip: clip-space w = -eye_z; geometry with w < 1 is discarded.
    o.clip[0] = o.pos.w - 1.0; // webgpu:clip-distances
    let dist = abs(o.pos.w);
    let range = u.fogEnd - u.fogStart;
    o.fogFactor = select(1.0, saturate((u.fogEnd - dist) / range), range > 0.001);
    o.eyeDepth = dist;
    o.normal = vec3<f32>(0.0);
    o.viewVector = vec3<f32>(0.0);
    o.lightDirectionAndSpecular = vec4<f32>(0.0);
    o.worldPositionAndDynamicLights = vec4<f32>(0.0);
    o.worldNormal = vec3<f32>(0.0);
    return o;
}
