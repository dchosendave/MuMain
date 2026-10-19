// WGSL port of skinned_textured.vert.hlsl: GPU bone skinning for BMD rest-pose geometry.
// palette.w selects the lighting: 0 unlit, 1 Classic per-vertex lighting, 2 Enhanced lighting
// (skinned_lit.frag lights each pixel), 3 Enhanced plus the frame's point lights.
enable clip_distances; // webgpu:clip-distances

@group(0) @binding(48) var<storage, read> boneRows: array<vec4<f32>>;

struct Skinning {
    mvp: mat4x4<f32>,
    bodyOriginAndScale: vec4<f32>,
    boneScaleAndRestPoseScale: vec4<f32>,
    palette: vec4<u32>,
    lightDirection: vec4<f32>, // xyz: the model light vector; w: the highlight knee
    fogParameters: vec4<f32>,
    textureCoordinateMode: vec4<u32>,
    chromeParameters: vec4<f32>,
    textureCoordinateParameters: vec4<f32>,
    eyePositionAndSpecular: vec4<f32>, // xyz: camera position in vertex space; w: specular strength
    vertexToWorldRows: array<vec4<f32>, 3>,
};
@group(1) @binding(0) var<uniform> u: Skinning;

const LightingClassic = 1u;
const LightingEnhancedDynamic = 3u;

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

fn transformPosition(position: vec3<f32>, boneIndex: i32) -> vec3<f32> {
    if (boneIndex < 0 || boneIndex >= i32(u.palette.y)) {
        return position;
    }
    let row = u.palette.x + u32(boneIndex) * 3u;
    let row0 = boneRows[row];
    let row1 = boneRows[row + 1u];
    let row2 = boneRows[row + 2u];
    var transformed: vec3<f32>;
    if (u.boneScaleAndRestPoseScale.x == 1.0) {
        let restScale = select(1.0, u.boneScaleAndRestPoseScale.y, u.boneScaleAndRestPoseScale.y != 0.0);
        let restPosition = vec4<f32>(position * restScale, 1.0);
        transformed = vec3<f32>(dot(row0, restPosition), dot(row1, restPosition), dot(row2, restPosition));
    } else {
        transformed = vec3<f32>(dot(row0.xyz, position), dot(row1.xyz, position), dot(row2.xyz, position));
        transformed = transformed * u.boneScaleAndRestPoseScale.x + vec3<f32>(row0.w, row1.w, row2.w);
    }
    if (u.palette.z != 0u) {
        transformed = u.bodyOriginAndScale.xyz + transformed * u.bodyOriginAndScale.w;
    }
    return transformed;
}

fn transformNormal(normal: vec3<f32>, boneIndex: i32) -> vec3<f32> {
    if (boneIndex < 0 || boneIndex >= i32(u.palette.y)) {
        return normal;
    }
    let row = u.palette.x + u32(boneIndex) * 3u;
    return vec3<f32>(dot(boneRows[row].xyz, normal), dot(boneRows[row + 1u].xyz, normal), dot(boneRows[row + 2u].xyz, normal));
}

fn getTextureCoordinates(restCoordinates: vec2<f32>, normal: vec3<f32>) -> vec2<f32> {
    let mode = u.textureCoordinateMode.x;
    let chromeWave = u.chromeParameters.x;
    let chromeWave2 = u.chromeParameters.y;
    let chromeLight = u.chromeParameters.zw;
    let offset = u.textureCoordinateParameters.xy;

    if (mode == 1u) {
        return vec2<f32>(normal.z * 0.5 + chromeWave, normal.y * 0.5 + chromeWave * 2.0);
    }
    if (mode == 2u) {
        return vec2<f32>((normal.z + normal.x) * 0.8 + chromeWave2 * 2.0, normal.y + normal.x + chromeWave2 * 3.0);
    }
    if (mode == 3u) {
        let uCoordinate = dot(normal, vec3<f32>(0.0, -0.1, -0.8));
        return vec2<f32>(uCoordinate, 1.0 - uCoordinate);
    }
    if (mode == 4u || mode == 5u) {
        let lightDot = dot(normal, vec3<f32>(chromeLight, 1.0));
        let isFour = mode == 4u;
        let scale = select(2.5, 0.5, isFour);
        let uScale = select(3.0, 0.5, isFour);
        let lightScale = select(5.0, 3.0, isFour);
        let waveScale = select(1.0, 3.0, isFour);
        return vec2<f32>(lightDot + normal.y * uScale + chromeLight.y * lightScale,
                         1.0 - lightDot - normal.z * scale - chromeWave * waveScale) + offset;
    }
    if (mode == 6u) {
        let coordinate = (normal.z + normal.x) * 0.8 + chromeWave2 * 2.0;
        return vec2<f32>(coordinate, coordinate);
    }
    if (mode == 7u) {
        let coordinate = (normal.z + normal.x) * 0.8 + u.textureCoordinateParameters.z;
        return vec2<f32>(coordinate, coordinate);
    }
    if (mode == 8u) {
        return normal.xy * restCoordinates + offset;
    }
    if (mode == 9u) {
        return vec2<f32>(normal.z * 0.5 + 0.2, normal.y * 0.5 + 0.5);
    }
    return restCoordinates;
}

@vertex fn main(@location(0) pos: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>,
                @location(3) color: vec4<f32>, @location(4) positionBoneIndex: i32, @location(5) normalBoneIndex: i32) -> VSOutput {
    var o: VSOutput;
    let skinnedPosition = transformPosition(pos, positionBoneIndex);
    let skinnedNormal = transformNormal(normal, normalBoneIndex);
    o.pos = u.mvp * vec4<f32>(skinnedPosition, 1.0);
    o.uv = getTextureCoordinates(uv, skinnedNormal);
    o.color = color;

    if (u.palette.w == LightingClassic) {
        let luminosity = max(dot(skinnedNormal, u.lightDirection.xyz) * 0.8 + 0.4, 0.2);
        o.color = vec4<f32>(saturate(color.rgb * luminosity), color.a);
        if (RollsOffHighlight(u.lightDirection.w)) {
            o.color = vec4<f32>(RollOffHighlight(color.rgb * luminosity, u.lightDirection.w), color.a);
        }
    }

    o.clip[0] = o.pos.w - 1.0; // webgpu:clip-distances
    let range = u.fogParameters.y - u.fogParameters.x;
    o.fogFactor = select(1.0, saturate((u.fogParameters.y - abs(o.pos.w)) / range), range > 0.001);
    o.eyeDepth = abs(o.pos.w);
    o.normal = skinnedNormal;
    o.viewVector = u.eyePositionAndSpecular.xyz - skinnedPosition;
    o.lightDirectionAndSpecular = vec4<f32>(u.lightDirection.xyz, u.eyePositionAndSpecular.w);
    let position = vec4<f32>(skinnedPosition, 1.0);
    o.worldPositionAndDynamicLights = vec4<f32>(dot(u.vertexToWorldRows[0], position), dot(u.vertexToWorldRows[1], position),
                                                dot(u.vertexToWorldRows[2], position),
                                                select(0.0, 1.0, u.palette.w == LightingEnhancedDynamic));
    o.worldNormal = vec3<f32>(dot(u.vertexToWorldRows[0].xyz, skinnedNormal), dot(u.vertexToWorldRows[1].xyz, skinnedNormal),
                              dot(u.vertexToWorldRows[2].xyz, skinnedNormal));
    return o;
}
