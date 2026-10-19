// WGSL port of post_bloom_up.frag.hlsl (dual filter upsample).
@group(2) @binding(0) var source: texture_2d<f32>;
@group(2) @binding(1) var linearSampler: sampler;

struct BloomUniforms { sourceTexelSize: vec2<f32>, levelWeight: f32, padding: f32 };
@group(3) @binding(0) var<uniform> bloom: BloomUniforms;

const kDiagonalWeight = 2.0;
const kTotalWeight = 12.0;

@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let h = bloom.sourceTexelSize * 0.5;
    var sum = textureSample(source, linearSampler, uv + vec2<f32>(-h.x * 2.0, 0.0)).rgb;
    sum += textureSample(source, linearSampler, uv + vec2<f32>(h.x * 2.0, 0.0)).rgb;
    sum += textureSample(source, linearSampler, uv + vec2<f32>(0.0, -h.y * 2.0)).rgb;
    sum += textureSample(source, linearSampler, uv + vec2<f32>(0.0, h.y * 2.0)).rgb;
    sum += textureSample(source, linearSampler, uv + vec2<f32>(-h.x, -h.y)).rgb * kDiagonalWeight;
    sum += textureSample(source, linearSampler, uv + vec2<f32>(h.x, -h.y)).rgb * kDiagonalWeight;
    sum += textureSample(source, linearSampler, uv + vec2<f32>(-h.x, h.y)).rgb * kDiagonalWeight;
    sum += textureSample(source, linearSampler, uv + vec2<f32>(h.x, h.y)).rgb * kDiagonalWeight;
    return vec4<f32>(sum / kTotalWeight * bloom.levelWeight, 1.0);
}
