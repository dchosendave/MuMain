// WGSL port of basic_textured.frag.hlsl.
@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var s: sampler;

struct FogUniforms {
    fogEnabled: u32,
    alphaDiscardEnabled: u32,
    alphaThreshold: f32,
    alphaToCoverage: u32,
    fogStart: f32,
    fogEnd: f32,
    fogColor: vec4<f32>,
};
@group(3) @binding(0) var<uniform> fog: FogUniforms;

@fragment fn main(@location(0) uv: vec2<f32>, @location(1) vertexColor: vec4<f32>, @location(2) fogFactor: f32) -> @location(0) vec4<f32> {
    var color = textureSample(tex, s, uv) * vertexColor;
    if (fog.alphaToCoverage != 0u) {
        let edge = saturate((color.a - fog.alphaThreshold) / max(fwidth(color.a), 0.0001) + 0.5);
        color.a *= edge;
        if (color.a <= 0.0) {
            discard;
        }
    } else if (fog.alphaDiscardEnabled != 0u && color.a <= fog.alphaThreshold) {
        discard;
    }
    // fogFactor 1: no fog (close), 0: full fog (far).
    if (fog.fogEnabled != 0u) {
        color = vec4<f32>(mix(fog.fogColor.rgb, color.rgb, fogFactor), color.a);
    }
    return color;
}
