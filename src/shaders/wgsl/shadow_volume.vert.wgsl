// WGSL port of shadow_volume.vert.hlsl.
struct MVP { mvp: mat4x4<f32> };
@group(1) @binding(0) var<uniform> u: MVP;

@vertex fn main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return u.mvp * vec4<f32>(pos, 1.0);
}
