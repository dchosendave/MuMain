// WGSL port of basic_textured_fog.frag.hlsl.
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

struct SceneLighting {
    sceneLightDirection: vec4<f32>,
    sceneLightColor: vec4<f32>,
    diffuseCurve: vec4<f32>,
    rimParameters: vec4<f32>,
    specularParameters: vec4<f32>,
    sceneFogColor: vec4<f32>,
    sceneFogParameters: vec4<f32>,
    cameraPosition: vec4<f32>,
};
@group(3) @binding(1) var<uniform> scene: SceneLighting;

// Same curve as skinned_lit.frag and mu::EnhancedFogAmount() (SceneLighting.cpp).
fn EnhancedFogAmount(eyeDepth: f32) -> f32 {
    let scaledDistance = max(eyeDepth - scene.sceneFogParameters.x, 0.0) * scene.sceneFogParameters.z;
    return saturate(1.0 - exp(-scaledDistance * scaledDistance));
}

@fragment fn main(@location(0) uv: vec2<f32>, @location(1) vertexColor: vec4<f32>, @location(3) eyeDepth: f32) -> @location(0) vec4<f32> {
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
    if (fog.fogEnabled != 0u && scene.sceneFogColor.w > 0.0) {
        color = vec4<f32>(mix(color.rgb, scene.sceneFogColor.rgb, EnhancedFogAmount(eyeDepth)), color.a);
    }
    return color;
}
