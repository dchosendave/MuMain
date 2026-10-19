// WGSL port of skinned_shadow.vert.hlsl: BMD body shadow, skinned and projected onto the terrain
// as CalcShadowPosition() does.
enable clip_distances; // webgpu:clip-distances

@group(0) @binding(48) var<storage, read> boneRows: array<vec4<f32>>;
// Render::TerrainHeightGrid: 256 * 256 heights, then one TW_HEIGHT bit per cell.
@group(0) @binding(49) var<storage, read> terrainWords: array<u32>;

struct Skinning {
    mvp: mat4x4<f32>,
    bodyOriginAndScale: vec4<f32>,
    boneScaleAndRestPoseScale: vec4<f32>,
    palette: vec4<u32>,
    lightDirection: vec4<f32>,
    fogParameters: vec4<f32>,
    textureCoordinateMode: vec4<u32>,
    chromeParameters: vec4<f32>,
    textureCoordinateParameters: vec4<f32>,
};
@group(1) @binding(0) var<uniform> u: Skinning;

struct Shadow {
    shadowOrigin: vec4<f32>,     // xyz: BodyOrigin, w: height of TW_HEIGHT cells
    shadowProjection: vec4<f32>, // x, y: sx, sy; z: 0 where RequestTerrainHeight() returns 0
};
@group(1) @binding(1) var<uniform> shadow: Shadow;

const TerrainSize = 256u;
const TerrainCells = 65536u;

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

fn shearShadow(position: vec3<f32>) -> vec3<f32> {
    var relative = position - shadow.shadowOrigin.xyz;
    relative.x += relative.z * (relative.x + shadow.shadowProjection.x) / (relative.z - shadow.shadowProjection.y);
    return relative + shadow.shadowOrigin.xyz;
}

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

fn terrainCornerHeight(x: i32, y: i32) -> f32 {
    return bitcast<f32>(terrainWords[u32(y & 255) * TerrainSize + u32(x & 255)]);
}

// Render::SampleTerrainHeightGrid().
fn terrainHeight(xIn: f32, yIn: f32) -> f32 {
    if (shadow.shadowProjection.z == 0.0 || xIn < 0.0 || yIn < 0.0) {
        return 0.0;
    }
    let x = xIn / 100.0;
    let y = yIn / 100.0;
    let xi = i32(x);
    let yi = i32(y);

    // An x past the last column runs into the next row; only the corners wrap.
    let index = u32(yi * i32(TerrainSize) + xi);
    if (index >= TerrainCells || ((terrainWords[TerrainCells + (index >> 5u)] >> (index & 31u)) & 1u) != 0u) {
        return shadow.shadowOrigin.w;
    }

    let xd = x - f32(xi);
    let yd = y - f32(yi);
    let h1 = terrainCornerHeight(xi, yi);
    let h2 = terrainCornerHeight(xi, yi + 1);
    let h3 = terrainCornerHeight(xi + 1, yi);
    let h4 = terrainCornerHeight(xi + 1, yi + 1);
    let left = h1 + (h2 - h1) * yd;
    let right = h3 + (h4 - h3) * yd;
    return left + (right - left) * xd;
}

@vertex fn main(@location(0) pos: vec3<f32>, @location(3) color: vec4<f32>, @location(4) positionBoneIndex: i32) -> VSOutput {
    var o: VSOutput;
    var position = shearShadow(transformPosition(pos, positionBoneIndex));
    position.z = terrainHeight(position.x, position.y) + 5.0;

    o.pos = u.mvp * vec4<f32>(position, 1.0);
    o.uv = vec2<f32>(0.0, 0.0);
    o.color = color;
    o.clip[0] = o.pos.w - 1.0; // webgpu:clip-distances
    let range = u.fogParameters.y - u.fogParameters.x;
    o.fogFactor = select(1.0, saturate((u.fogParameters.y - abs(o.pos.w)) / range), range > 0.001);
    o.eyeDepth = abs(o.pos.w);
    o.normal = vec3<f32>(0.0);
    o.viewVector = vec3<f32>(0.0);
    o.lightDirectionAndSpecular = vec4<f32>(0.0);
    o.worldPositionAndDynamicLights = vec4<f32>(0.0);
    o.worldNormal = vec3<f32>(0.0);
    return o;
}
