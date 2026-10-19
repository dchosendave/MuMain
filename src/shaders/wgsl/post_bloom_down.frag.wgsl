// WGSL port of post_bloom_down.frag.hlsl (dual filter downsample).
@group(2) @binding(0) var source: texture_2d<f32>;
@group(2) @binding(1) var linearSampler: sampler;

struct BloomUniforms { sourceTexelSize: vec2<f32>, levelWeight: f32, padding: f32 };
@group(3) @binding(0) var<uniform> bloom: BloomUniforms;

const kCentreWeight = 4.0;
const kTotalWeight = 8.0;

@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let halfTexel = bloom.sourceTexelSize * 0.5;
    var sum = textureSample(source, linearSampler, uv).rgb * kCentreWeight;
    sum += textureSample(source, linearSampler, uv - halfTexel).rgb;
    sum += textureSample(source, linearSampler, uv + halfTexel).rgb;
    sum += textureSample(source, linearSampler, uv + vec2<f32>(halfTexel.x, -halfTexel.y)).rgb;
    sum += textureSample(source, linearSampler, uv - vec2<f32>(halfTexel.x, -halfTexel.y)).rgb;
    return vec4<f32>(sum / kTotalWeight, 1.0);
}
