#include <filesystem>
#include <fstream>

#include "scene/builtin.h"
#include "scene/gltf.h"
#include "scene/json.h"
#include "test.h"

using namespace photon;

namespace {

std::string temp_path(const std::string& name) { return (std::filesystem::temp_directory_path() / ("photon_test_" + name)).string(); }

std::string base64(const std::vector<uint8_t>& bin) {
    static const char* b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string enc;
    for (size_t i = 0; i < bin.size(); i += 3) {
        uint32_t v = uint32_t(bin[i]) << 16;
        if (i + 1 < bin.size()) v |= uint32_t(bin[i + 1]) << 8;
        if (i + 2 < bin.size()) v |= bin[i + 2];
        enc += b64[(v >> 18) & 63];
        enc += b64[(v >> 12) & 63];
        enc += i + 1 < bin.size() ? b64[(v >> 6) & 63] : '=';
        enc += i + 2 < bin.size() ? b64[v & 63] : '=';
    }
    return enc;
}

double total_area(const Scene& s) {
    double a = 0;
    for (int t = 0; t < s.num_tris(); t++) {
        vec3 p0 = s.positions[size_t(s.indices[size_t(t) * 3])], p1 = s.positions[size_t(s.indices[size_t(t) * 3 + 1])],
             p2 = s.positions[size_t(s.indices[size_t(t) * 3 + 2])];
        a += 0.5 * length(cross(p1 - p0, p2 - p0));
    }
    return a;
}

void write_file(const std::string& path, const std::string& text) {
    std::ofstream o(path, std::ios::binary);
    o << text;
}

}  // namespace

TEST(json_parser) {
    json::Value v = json::parse(R"({"a": [1, 2.5e1, -3], "b": {"c": "xé\n"}, "t": true, "n": null})");
    CHECK(v["a"].size() == 3);
    CHECK_NEAR(v["a"][1].number(), 25, 1e-12);
    CHECK(v["a"][2].integer() == -3);
    CHECK(v["b"]["c"].string() == "x\xc3\xa9\n");
    CHECK(v["t"].boolean());
    CHECK(!v.has("missing"));
    CHECK(v["missing"]["deeper"].integer(-7) == -7);
    bool threw = false;
    try {
        json::parse("{\"a\": [1, 2}");
    } catch (const json::ParseError&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(gltf_round_trip) {
    Scene a;
    make_builtin_scene("materials", a);
    PunctualLight l{};
    l.type = LightType::Spot;
    l.position = vec3(1, 4, 2);
    l.direction = normalize(vec3(0, -1, 0.2f));
    l.intensity = vec3(30, 20, 10);
    l.cos_inner = std::cos(0.3f);
    l.cos_outer = std::cos(0.5f);
    a.punctual.push_back(l);
    std::string path = temp_path("roundtrip.gltf"), error;
    CHECK(save_gltf(path, a, error));
    Scene b;
    CHECK_MSG(load_gltf(path, b, error), error);
    CHECK(b.num_tris() == a.num_tris());
    CHECK(b.materials.size() == a.materials.size());
    CHECK_NEAR(total_area(b), total_area(a), 1e-3 * total_area(a));
    for (size_t i = 0; i < a.materials.size() && i < b.materials.size(); i++) {
        const Material &x = a.materials[i], &y = b.materials[i];
        CHECK_NEAR(x.base_color.x, y.base_color.x, 1e-5);
        CHECK_NEAR(x.metallic, y.metallic, 1e-5);
        CHECK_NEAR(x.roughness, y.roughness, 1e-5);
        CHECK_NEAR(x.transmission, y.transmission, 1e-5);
        CHECK_NEAR(x.ior, y.ior, 1e-5);
        CHECK_NEAR(luminance(x.emission), luminance(y.emission), 1e-4);
    }
    // Every triangle keeps its material.
    std::vector<int> ca(a.materials.size(), 0), cb(b.materials.size(), 0);
    for (int m : a.tri_material) ca[size_t(m)]++;
    for (int m : b.tri_material) cb[size_t(m)]++;
    CHECK(ca == cb);
    // Camera and light.
    CHECK(b.cameras.size() == 1);
    CHECK_NEAR(length(b.camera.position - a.camera.position), 0, 1e-4);
    vec3 fa = normalize(a.camera.target - a.camera.position), fb = normalize(b.camera.target - b.camera.position);
    CHECK_NEAR(dot(fa, fb), 1, 1e-5);
    CHECK_NEAR(b.camera.vfov, a.camera.vfov, 1e-3);
    CHECK(b.punctual.size() == 1);
    if (!b.punctual.empty()) {
        const PunctualLight& m = b.punctual[0];
        CHECK(m.type == LightType::Spot);
        CHECK_NEAR(length(m.position - l.position), 0, 1e-4);
        CHECK_NEAR(dot(m.direction, l.direction), 1, 1e-5);
        CHECK_NEAR(m.intensity.x, 30, 1e-3);
        CHECK_NEAR(m.cos_outer, l.cos_outer, 1e-5);
    }
    std::filesystem::remove(path);
}

TEST(gltf_node_transforms_and_winding) {
    // One triangle, instanced by two nodes: a translated child of a scaled parent, and a mirrored copy
    // (negative determinant), whose winding the loader must flip to keep the face pointing outwards.
    std::vector<uint8_t> bin;
    float pos[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
    uint16_t idx[3] = {0, 1, 2};
    bin.insert(bin.end(), reinterpret_cast<uint8_t*>(pos), reinterpret_cast<uint8_t*>(pos) + sizeof(pos));
    bin.insert(bin.end(), reinterpret_cast<uint8_t*>(idx), reinterpret_cast<uint8_t*>(idx) + sizeof(idx));
    bin.push_back(0);
    bin.push_back(0);
    std::string doc = R"({
      "asset": {"version": "2.0"},
      "buffers": [{"byteLength": )" + std::to_string(bin.size()) + R"(, "uri": "data:application/octet-stream;base64,)" + base64(bin) + R"("}],
      "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}, {"buffer": 0, "byteOffset": 36, "byteLength": 6}],
      "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0]},
                    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}],
      "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.2, 0.4, 0.6, 1], "metallicFactor": 0.0, "roughnessFactor": 0.3},
                     "emissiveFactor": [1, 0.5, 0], "extensions": {"KHR_materials_emissive_strength": {"emissiveStrength": 4}}}],
      "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
      "nodes": [{"scale": [2, 2, 2], "children": [1]},
                {"translation": [0, 0, 3], "mesh": 0},
                {"scale": [-1, 1, 1], "mesh": 0}],
      "scenes": [{"nodes": [0, 2]}],
      "scene": 0
    })";
    std::string path = temp_path("transforms.gltf"), error;
    write_file(path, doc);
    Scene s;
    CHECK_MSG(load_gltf(path, s, error), error);
    CHECK(s.num_tris() == 2);
    if (s.num_tris() == 2) {
        auto tri = [&](int t, int k) { return s.positions[size_t(s.indices[size_t(t) * 3 + k])]; };
        auto normal = [&](int t) { return normalize(cross(tri(t, 1) - tri(t, 0), tri(t, 2) - tri(t, 0))); };
        // Node 1 under node 0: scaled by 2, then translated by (0, 0, 3) * 2.
        CHECK_NEAR(length(tri(0, 0) - vec3(0, 0, 6)), 0, 1e-5);
        CHECK_NEAR(length(tri(0, 1) - vec3(2, 0, 6)), 0, 1e-5);
        CHECK_NEAR(normal(0).z, 1, 1e-5);
        // The mirrored copy still faces +z.
        CHECK_NEAR(normal(1).z, 1, 1e-5);
        bool has_mirrored = false;
        for (int k = 0; k < 3; k++) has_mirrored = has_mirrored || length(tri(1, k) - vec3(-1, 0, 0)) < 1e-5f;
        CHECK(has_mirrored);
    }
    CHECK(s.materials.size() == 1);
    if (!s.materials.empty()) {
        CHECK_NEAR(s.materials[0].roughness, 0.3, 1e-6);
        CHECK_NEAR(s.materials[0].emission.x, 4, 1e-5);
        CHECK_NEAR(s.materials[0].emission.y, 2, 1e-5);
    }
    std::filesystem::remove(path);
}

TEST(gltf_errors_are_reported) {
    Scene s;
    std::string error;
    CHECK(!load_gltf(temp_path("does_not_exist.gltf"), s, error));
    CHECK(!error.empty());
    std::string path = temp_path("broken.gltf");
    write_file(path, "{\"asset\": {\"version\": \"2.0\"}, \"meshes\": [");
    error.clear();
    CHECK(!load_gltf(path, s, error));
    CHECK(!error.empty());
    std::filesystem::remove(path);
}

TEST(gltf_sample_model) {
    // A real-world model, if the samples were downloaded (scripts/fetch_models.py).
    std::string path = test::source_dir() + "/scenes/models/DamagedHelmet/DamagedHelmet.glb";
    if (!std::filesystem::exists(path)) throw test::Skip{"DamagedHelmet not downloaded"};
    Scene s;
    std::string error;
    CHECK_MSG(load_gltf(path, s, error), error);
    CHECK(s.num_tris() == 15452);
    CHECK(s.textures.size() >= 4);  // base color, metal-roughness, normal, emissive (occlusion unused)
    CHECK(!s.uvs.empty() && !s.normals.empty());
    s.build();
    CHECK(s.tangents.size() == s.positions.size());
    CHECK(!s.emissive.empty());  // the visor lights are emissive (textured)
}
