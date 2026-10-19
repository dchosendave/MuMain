// WGSL port of skinned_lit.frag.hlsl: Enhanced per-pixel lighting for GPU-skinned BMD meshes.
//   diffuse  = Classic curve max(dot(n, L) * scale + ambient, floor) * object light, per pixel
//   rim      = strength * (1 - |dot(n, v)|)^power * object light, added after the texture
//   dynamic  = sum of the frame's point lights: colour * linear falloff * wrapped diffuse
//   specular = Blinn-Phong, opt-in per item model, scaled by the scene and object light
//   fog      = Enhanced exponential fog (see basic_textured_fog.frag)
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
    diffuseCurve: vec4<f32>, // scale, ambient, floor, highlight knee
    rimParameters: vec4<f32>,
    specularParameters: vec4<f32>,
    sceneFogColor: vec4<f32>,
    sceneFogParameters: vec4<f32>,
    cameraPosition: vec4<f32>,
};
@group(3) @binding(1) var<uniform> scene: SceneLighting;

const MaxDynamicLights = 32u; // mu::kMaxDynamicLights
struct DynamicLights {
    dynamicLightCount: u32,
    dynamicLightWrap: f32,
    dynamicLightPadding: vec2<u32>,
    dynamicLightPositionAndRadius: array<vec4<f32>, 32>,
    dynamicLightColor: array<vec4<f32>, 32>,
};
@group(3) @binding(2) var<uniform> lights: DynamicLights;

fn RollsOffHighlight(knee: f32) -> bool {
    return knee > 0.0 && knee < 1.0;
}

fn RollOffChannel(value: f32, knee: f32) -> f32 {
    let range = 1.0 - knee;
    return select(value, knee + range * (1.0 - exp(-max(value - knee, 0.0) / range)), value > knee);
}

fn RollOffHighlight(color: vec3<f32>, knee: f32) -> vec3<f32> {
    return saturate(vec3<f32>(RollOffChannel(color.r, knee), RollOffChannel(color.g, knee), RollOffChannel(color.b, knee)));
}

fn EnhancedFogAmount(eyeDepth: f32) -> f32 {
    let scaledDistance = max(eyeDepth - scene.sceneFogParameters.x, 0.0) * scene.sceneFogParameters.z;
    return saturate(1.0 - exp(-scaledDistance * scaledDistance));
}

fn SafeNormalize(value: vec3<f32>) -> vec3<f32> {
    let lengthSquared = dot(value, value);
    return select(vec3<f32>(0.0), value * inverseSqrt(lengthSquared), lengthSquared > 1e-12);
}

fn DynamicLighting(worldPosition: vec3<f32>, worldNormal: vec3<f32>) -> vec3<f32> {
    var light = vec3<f32>(0.0);
    let count = min(lights.dynamicLightCount, MaxDynamicLights);
    for (var i = 0u; i < count; i++) {
        let positionAndRadius = lights.dynamicLightPositionAndRadius[i];
        let toLight = positionAndRadius.xyz - worldPosition;
        let radius = positionAndRadius.w;
        let horizontalDistance = length(toLight.xy);
        if (horizontalDistance >= radius) {
            continue;
        }
        let falloff = (radius - horizontalDistance) / radius;
        let wrapped = saturate(dot(worldNormal, SafeNormalize(toLight)) * (1.0 - lights.dynamicLightWrap) + lights.dynamicLightWrap);
        light += lights.dynamicLightColor[i].rgb * (falloff * wrapped);
    }
    return light;
}

struct FSInput {
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) fogFactor: f32,
    @location(3) eyeDepth: f32,
    @location(4) normal: vec3<f32>,
    @location(5) viewVector: vec3<f32>,
    @location(6) @interpolate(flat) lightDirectionAndSpecular: vec4<f32>,
    @location(7) worldPositionAndDynamicLights: vec4<f32>,
    @location(8) worldNormal: vec3<f32>,
};

@fragment fn main(input: FSInput) -> @location(0) vec4<f32> {
    // Sampled first: textureSample needs uniform control flow, which the varying branches below break.
    let texel = textureSample(tex, s, input.uv);

    let objectLight = input.color.rgb;
    let normal = SafeNormalize(input.normal);
    let view = SafeNormalize(input.viewVector);
    // The diffuse term uses the interpolated (not normalized) normal and light, as Classic does.
    let light = input.lightDirectionAndSpecular.xyz;
    let lightDot = dot(input.normal, light);

    let luminosity = max(lightDot * scene.diffuseCurve.x + scene.diffuseCurve.y, scene.diffuseCurve.z);
    var dynamicLight = vec3<f32>(0.0);
    if (input.worldPositionAndDynamicLights.w > 0.5) {
        dynamicLight = DynamicLighting(input.worldPositionAndDynamicLights.xyz, SafeNormalize(input.worldNormal));
    }
    var diffuse = saturate(objectLight * luminosity + dynamicLight);
    if (RollsOffHighlight(scene.diffuseCurve.w)) {
        diffuse = RollOffHighlight(objectLight * luminosity + dynamicLight, scene.diffuseCurve.w);
    }

    // Two-sided meshes show back faces; |dot| keeps their interiors from rim-lighting.
    let facing = saturate(abs(dot(normal, view)));
    let rim = scene.rimParameters.x * pow(1.0 - facing, scene.rimParameters.y) * objectLight;

    let specularStrength = input.lightDirectionAndSpecular.w * scene.specularParameters.x;
    var specular = vec3<f32>(0.0);
    if (specularStrength > 0.0 && lightDot > 0.0) {
        let halfVector = SafeNormalize(SafeNormalize(light) + view);
        let highlight = pow(saturate(dot(normal, halfVector)), scene.specularParameters.y);
        specular = specularStrength * highlight * scene.sceneLightColor.rgb * objectLight;
    }

    var color = vec4<f32>(texel.rgb * diffuse + rim + specular, texel.a * input.color.a);
    // fwidth is computed before any branch: derivatives need uniform control flow too.
    let alphaWidth = fwidth(color.a);
    if (fog.alphaToCoverage != 0u) {
        let edge = saturate((color.a - fog.alphaThreshold) / max(alphaWidth, 0.0001) + 0.5);
        color.a *= edge;
        if (color.a <= 0.0) {
            discard;
        }
    } else if (fog.alphaDiscardEnabled != 0u && color.a <= fog.alphaThreshold) {
        discard;
    }
    if (fog.fogEnabled != 0u && scene.sceneFogColor.w > 0.0) {
        color = vec4<f32>(mix(color.rgb, scene.sceneFogColor.rgb, EnhancedFogAmount(input.eyeDepth)), color.a);
    }
    return color;
}
