// WGSL port of post_composite.frag.hlsl: scene + screen-blended bloom, colour graded.
@group(2) @binding(0) var scene: texture_2d<f32>;
@group(2) @binding(1) var sceneSampler: sampler;
@group(2) @binding(2) var bloomTexture: texture_2d<f32>;
@group(2) @binding(3) var bloomSampler: sampler;
@group(2) @binding(4) var grading: texture_2d<f32>;
@group(2) @binding(5) var gradingSampler: sampler;

struct CompositeUniforms { inverseTargetSize: vec2<f32>, bloomScale: f32, gradingStrength: f32 };
@group(3) @binding(0) var<uniform> composite: CompositeUniforms;

const kGradingSize = 32.0;

// The lookup table is a 32x32x32 colour cube stored as 32 slices of 32x32 side by side (1024x32):
// red runs along x inside a slice, green along y, blue picks the slice.
fn Grade(color: vec3<f32>) -> vec3<f32> {
    let lastEntry = kGradingSize - 1.0;
    let blue = color.b * lastEntry;
    let lowerSlice = floor(blue);
    let upperSlice = min(lowerSlice + 1.0, lastEntry);
    let sliceBlend = blue - lowerSlice;
    let x = (color.r * lastEntry + 0.5) / (kGradingSize * kGradingSize);
    let y = (color.g * lastEntry + 0.5) / kGradingSize;
    let sliceWidth = 1.0 / kGradingSize;
    let lower = textureSample(grading, gradingSampler, vec2<f32>(x + lowerSlice * sliceWidth, y)).rgb;
    let upper = textureSample(grading, gradingSampler, vec2<f32>(x + upperSlice * sliceWidth, y)).rgb;
    return mix(lower, upper, sliceBlend);
}

@fragment fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = pos.xy * composite.inverseTargetSize;
    var color = textureSample(scene, sceneSampler, uv);
    // Branches on uniform values keep the control flow uniform, as textureSample requires.
    if (composite.bloomScale > 0.0) {
        let glow = saturate(textureSample(bloomTexture, bloomSampler, uv).rgb * composite.bloomScale);
        color = vec4<f32>(1.0 - (1.0 - color.rgb) * (1.0 - glow), color.a);
    }
    if (composite.gradingStrength > 0.0) {
        color = vec4<f32>(mix(color.rgb, Grade(color.rgb), composite.gradingStrength), color.a);
    }
    return color;
}
