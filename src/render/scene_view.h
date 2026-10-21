// The scene as the renderer sees it: flat arrays of plain structs behind raw pointers. The same struct
// points into host memory for the CPU renderer and into device memory for the GPU renderer, so the
// integrator code is shared verbatim.
#pragma once

#include "core/math.h"

namespace photon {

// A BVH node stores the bounds of both children, so traversal tests the two boxes from one 64-byte
// node (four 16-byte loads on the GPU). A child with count > 0 is a leaf holding the triangles
// [index, index + count) in BVH order; otherwise index is the child node.
struct alignas(16) BvhNode {
    vec3 lmin;
    int left;
    vec3 lmax;
    int lcount;
    vec3 rmin;
    int right;
    vec3 rmax;
    int rcount;
};

// Triangle geometry for intersection, in BVH order: a vertex and two edges (Moller-Trumbore).
struct alignas(16) TriGeom {
    vec3 v0;
    int material;
    vec3 e1;
    int flags;  // kTriAlphaTested
    vec3 e2;
    int pad;
};

constexpr int kTriAlphaTested = 1;

enum class AlphaMode : int { Opaque = 0, Mask = 1, Blend = 2 };

struct TextureInfo {
    int width = 0, height = 0;
    uint32_t offset = 0;  // into the texel pool (RGBA8)
    int wrap_repeat = 1;  // 1: repeat, 0: clamp to edge
    int srgb = 0;         // color data stored sRGB-encoded
};

// glTF 2.0 metallic-roughness material with the transmission, IOR and emissive-strength extensions.
struct Material {
    vec4 base_color = vec4(1, 1, 1, 1);
    vec3 emission = vec3(0);  // emissive factor * strength
    float metallic = 1.0f;
    float roughness = 1.0f;
    float ior = 1.5f;
    float transmission = 0.0f;
    float normal_scale = 1.0f;
    float alpha_cutoff = 0.5f;
    AlphaMode alpha_mode = AlphaMode::Opaque;
    int double_sided = 0;
    int base_color_tex = -1;
    int metallic_roughness_tex = -1;  // G: roughness, B: metallic
    int normal_tex = -1;
    int emissive_tex = -1;
    int occlusion_tex = -1;  // unused by the path tracer (it computes occlusion)
};

// One emissive triangle, for light sampling.
struct EmissiveTri {
    int tri;
    float area;
};

// Walker's alias method: O(1) sampling of a discrete distribution.
struct AliasEntry {
    float prob;   // probability of keeping this bin
    int alias;    // the bin to use otherwise
    float pmf;    // this bin's probability
    int pad;
};

enum class LightType : int { Point = 0, Spot = 1, Directional = 2 };

// glTF KHR_lights_punctual lights: delta lights, sampled only by next event estimation.
struct PunctualLight {
    vec3 position;
    LightType type;
    vec3 direction;  // the direction the light points to
    float cos_inner;
    vec3 intensity;  // color * intensity (W/sr for point and spot, lux for directional)
    float cos_outer;
};

// The environment: an equirectangular HDR image with a 2D importance-sampling distribution, or a
// constant color (or a procedural sky when sky_turbidity > 0).
struct EnvironmentView {
    const vec3* texels = nullptr;
    const float* marginal_cdf = nullptr;     // height + 1 entries
    const float* conditional_cdf = nullptr;  // height rows of width + 1 entries
    int width = 0, height = 0;
    float intensity = 1.0f;
    float rotation = 0.0f;  // around +y, in radians
    vec3 color = vec3(0);   // used when there is no image
    float marginal_integral = 0.0f;
    int sky = 0;            // procedural sky
    vec3 sun_direction = vec3(0.3f, 0.8f, 0.4f);
    float sun_intensity = 0.0f;
};

struct CameraView {
    vec3 position = vec3(0, 0, 5);
    float tan_half_fov = 0.41421356f;  // tan(22.5 deg): a 45 degree vertical field of view
    vec3 forward = vec3(0, 0, -1);
    float aspect = 1.0f;
    vec3 right = vec3(1, 0, 0);
    float aperture = 0.0f;  // lens radius (depth of field when > 0)
    vec3 up = vec3(0, 1, 0);
    float focus_distance = 5.0f;
};

struct SceneView {
    // Geometry, in BVH order.
    const BvhNode* nodes = nullptr;
    const TriGeom* tris = nullptr;
    const int* indices = nullptr;  // 3 vertex indices per triangle
    const vec3* normals = nullptr;
    const vec2* uvs = nullptr;
    const vec4* tangents = nullptr;  // xyz + handedness
    int num_tris = 0;
    int num_nodes = 0;

    // Appearance.
    const Material* materials = nullptr;
    const TextureInfo* textures = nullptr;
    const uint32_t* texels = nullptr;
    const float* srgb_lut = nullptr;  // 256 entries: sRGB byte -> linear

    // Lights.
    const EmissiveTri* emissive = nullptr;
    const int* tri_emissive = nullptr;  // per triangle: its index in `emissive`, or -1
    const AliasEntry* emissive_alias = nullptr;
    int num_emissive = 0;
    float emissive_power = 0.0f;
    const PunctualLight* punctual = nullptr;
    int num_punctual = 0;
    EnvironmentView env;
    // Next event estimation picks one light group with these probabilities (they sum to 1, or are all 0
    // when the scene has no lights), then a light within the group: emissive triangles by power,
    // punctual lights uniformly, the environment by its image.
    float p_env = 0.0f, p_emissive = 0.0f, p_punctual = 0.0f;

    // Scene bounds (for directional and environment light offsets).
    vec3 bounds_min, bounds_max;
};

}  // namespace photon
