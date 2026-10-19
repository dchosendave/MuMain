// WGSL port of basic_colored.vert.hlsl (web build; bindings: see web/sdl_gpu_webgpu).
struct ScreenSize { screenSize: vec2<f32>, pad: vec2<f32> };
@group(1) @binding(0) var<uniform> u: ScreenSize;

struct VSOutput { @builtin(position) pos: vec4<f32>, @location(0) color: vec4<f32> };

@vertex fn main(@location(0) pos: vec2<f32>, @location(1) color: vec4<f32>) -> VSOutput {
    var o: VSOutput;
    o.pos = vec4<f32>(pos / u.screenSize * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    o.color = color;
    return o;
}
