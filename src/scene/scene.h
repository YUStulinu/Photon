// The scene on the host: geometry, materials, textures and lights as loaded, plus the derived data the
// renderer needs (BVH, light sampling tables, environment distribution), built by `Scene::build`.
#pragma once

#include <string>
#include <vector>

#include "bvh/bvh.h"
#include "render/scene_view.h"

namespace photon {

struct Texture {
    int width = 0, height = 0;
    std::vector<uint32_t> texels;  // RGBA8, row 0 at the top
    bool repeat = true;
    bool srgb = false;
    std::string name;
};

struct Environment {
    std::vector<vec3> texels;  // equirectangular, linear radiance
    int width = 0, height = 0;
    vec3 color = vec3(0);      // used when there is no image
    float intensity = 1.0f;
    float rotation = 0.0f;
    std::string name;
};

struct Camera {
    vec3 position = vec3(0, 1, 5);
    vec3 target = vec3(0, 1, 0);
    vec3 up = vec3(0, 1, 0);
    float vfov = 45.0f;  // degrees
    float aperture = 0.0f;
    float focus_distance = 0.0f;  // 0: the distance to the target

    CameraView view(float aspect) const;
};

class Scene {
public:
    // Geometry: shared vertex arrays and three indices per triangle.
    std::vector<vec3> positions;
    std::vector<vec3> normals;   // empty, or one per vertex
    std::vector<vec2> uvs;       // empty, or one per vertex
    std::vector<vec4> tangents;  // empty, or one per vertex
    std::vector<int> indices;
    std::vector<int> tri_material;

    std::vector<Material> materials;
    std::vector<Texture> textures;
    std::vector<PunctualLight> punctual;
    Environment env;
    Camera camera;
    std::vector<Camera> cameras;  // all cameras found in the file
    std::string name;

    int num_tris() const { return int(indices.size() / 3); }

    // Appends a triangle mesh with one material (normals and uvs may be empty).
    void add_mesh(const std::vector<vec3>& pos, const std::vector<int>& idx, int material, const std::vector<vec3>& nrm = {},
                  const std::vector<vec2>& uv = {});
    int add_material(const Material& m);

    // Computes missing tangents from UVs (for normal mapping), builds the BVH and the light tables.
    void build(const BvhBuildOptions& options = {});
    // After changing materials or the environment of a built scene: updates the light tables (the BVH
    // stays as it is).
    void rebuild_lights();

    // A view over the built data in host memory (for the CPU renderer).
    SceneView view() const;

    // Built data.
    Bvh bvh;
    std::vector<TriGeom> tris;       // BVH order
    std::vector<int> bvh_indices;    // BVH order, 3 per triangle
    std::vector<EmissiveTri> emissive;
    std::vector<AliasEntry> emissive_alias;
    std::vector<int> tri_emissive;
    float emissive_power = 0;
    std::vector<uint32_t> texel_pool;
    std::vector<TextureInfo> texture_infos;
    std::vector<float> env_marginal, env_conditional;
    float env_integral = 0;
    std::vector<float> srgb_lut;
    float p_env = 0, p_emissive = 0, p_punctual = 0;
    bool built = false;

private:
    void compute_tangents();
    void build_lights();
    void build_environment();
};

// Builds the alias table of a discrete distribution (Vose's algorithm).
std::vector<AliasEntry> build_alias_table(const std::vector<float>& weights);

}  // namespace photon
