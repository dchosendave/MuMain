// WGSL port of post_fullscreen.vert.hlsl: one triangle covering the target, uv (0,0) top-left.
struct VSOutput { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex fn main(@builtin(vertex_index) vertexId: u32) -> VSOutput {
    let uv = vec2<f32>(f32((vertexId << 1u) & 2u), f32(vertexId & 2u));
    var o: VSOutput;
    o.pos = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    o.uv = uv;
    return o;
}
