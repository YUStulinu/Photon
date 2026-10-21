#include "scene/gltf.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

#include "scene/image.h"
#include "scene/json.h"

namespace photon {

namespace {

// Paths in glTF files and on the command line are UTF-8.
std::filesystem::path utf8_path(const std::string& s) { return std::filesystem::path(std::u8string(s.begin(), s.end())); }

using json::Value;

std::string decode_uri(const std::string& uri) {
    // Percent-decoding (spaces in file names are encoded as %20).
    std::string out;
    for (size_t i = 0; i < uri.size(); i++) {
        if (uri[i] == '%' && i + 2 < uri.size()) {
            out += char(std::stoi(uri.substr(i + 1, 2), nullptr, 16));
            i += 2;
        } else {
            out += uri[i];
        }
    }
    return out;
}

bool base64_decode(const std::string& in, std::vector<uint8_t>& out) {
    static int table[256];
    static bool init = false;
    if (!init) {
        for (int& t : table) t = -1;
        const char* chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) table[uint8_t(chars[i])] = i;
        init = true;
    }
    unsigned acc = 0;
    int bits = 0;
    for (char c : in) {
        if (c == '=') break;
        int v = table[uint8_t(c)];
        if (v < 0) {
            if (c == '\n' || c == '\r' || c == ' ') continue;
            return false;
        }
        acc = (acc << 6) | unsigned(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(uint8_t((acc >> bits) & 0xFF));
        }
    }
    return true;
}

bool read_file(const std::filesystem::path& p, std::vector<uint8_t>& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

struct Loader {
    Value doc;
    std::filesystem::path dir;
    std::vector<std::vector<uint8_t>> buffers;
    Scene& scene;
    std::string error;
    int material_base = 0;
    std::map<std::pair<int, bool>, int> texture_map;  // (glTF texture, sRGB) -> scene texture
    std::vector<Texture> images;                       // decoded images, by glTF image index
    std::vector<bool> image_loaded;

    explicit Loader(Scene& s) : scene(s) {}

    bool fail(const std::string& msg) {
        error = msg;
        return false;
    }

    bool load_buffers(const std::vector<uint8_t>* glb_bin) {
        const Value& bufs = doc["buffers"];
        for (size_t i = 0; i < bufs.size(); i++) {
            const Value& b = bufs[i];
            std::vector<uint8_t> data;
            if (!b.has("uri")) {
                if (!glb_bin) return fail("buffer " + std::to_string(i) + " has no data");
                data = *glb_bin;
            } else {
                const std::string& uri = b["uri"].string();
                if (uri.rfind("data:", 0) == 0) {
                    size_t comma = uri.find(',');
                    if (comma == std::string::npos || !base64_decode(uri.substr(comma + 1), data)) return fail("invalid data URI");
                } else if (!read_file(dir / utf8_path(decode_uri(uri)), data)) {
                    return fail("cannot read buffer " + uri);
                }
            }
            buffers.push_back(std::move(data));
        }
        return true;
    }

    // Reads an accessor as floats (count * components), converting and normalizing integers.
    bool read_floats(int accessor, int want_components, std::vector<float>& out) {
        const Value& a = doc["accessors"][size_t(accessor)];
        if (a.is_null()) return fail("missing accessor");
        int count = a["count"].integer(0);
        static const std::map<std::string, int> comps = {{"SCALAR", 1}, {"VEC2", 2}, {"VEC3", 3}, {"VEC4", 4}, {"MAT4", 16}};
        auto it = comps.find(a["type"].string());
        if (it == comps.end()) return fail("unsupported accessor type");
        int nc = it->second;
        int ct = a["componentType"].integer();
        bool normalized = a["normalized"].boolean();
        int csize = (ct == 5126 || ct == 5125) ? 4 : (ct == 5122 || ct == 5123) ? 2 : 1;
        out.assign(size_t(count) * size_t(want_components), 0.0f);
        if (!a.has("bufferView")) return true;  // all zeros
        const Value& bv = doc["bufferViews"][size_t(a["bufferView"].integer())];
        const std::vector<uint8_t>& buf = buffers.at(size_t(bv["buffer"].integer(0)));
        size_t base = size_t(bv["byteOffset"].integer(0)) + size_t(a["byteOffset"].integer(0));
        size_t stride = bv.has("byteStride") ? size_t(bv["byteStride"].integer()) : size_t(nc * csize);
        if (count > 0 && base + stride * size_t(count - 1) + size_t(nc * csize) > buf.size()) return fail("accessor out of the buffer's range");
        for (int i = 0; i < count; i++) {
            const uint8_t* p = buf.data() + base + stride * size_t(i);
            for (int c = 0; c < std::min(nc, want_components); c++) {
                const uint8_t* q = p + size_t(c * csize);
                float v = 0;
                switch (ct) {
                    case 5126: std::memcpy(&v, q, 4); break;
                    case 5121: v = normalized ? float(q[0]) / 255.0f : float(q[0]); break;
                    case 5120: v = normalized ? std::max(float(int8_t(q[0])) / 127.0f, -1.0f) : float(int8_t(q[0])); break;
                    case 5123: {
                        uint16_t u;
                        std::memcpy(&u, q, 2);
                        v = normalized ? float(u) / 65535.0f : float(u);
                        break;
                    }
                    case 5122: {
                        int16_t s;
                        std::memcpy(&s, q, 2);
                        v = normalized ? std::max(float(s) / 32767.0f, -1.0f) : float(s);
                        break;
                    }
                    case 5125: {
                        uint32_t u;
                        std::memcpy(&u, q, 4);
                        v = float(u);
                        break;
                    }
                    default: return fail("unsupported component type");
                }
                out[size_t(i) * size_t(want_components) + size_t(c)] = v;
            }
        }
        return true;
    }

    bool read_indices(int accessor, std::vector<int>& out) {
        const Value& a = doc["accessors"][size_t(accessor)];
        int count = a["count"].integer(0);
        int ct = a["componentType"].integer();
        out.resize(size_t(count));
        if (!a.has("bufferView")) return fail("index accessor without data");
        const Value& bv = doc["bufferViews"][size_t(a["bufferView"].integer())];
        const std::vector<uint8_t>& buf = buffers.at(size_t(bv["buffer"].integer(0)));
        size_t base = size_t(bv["byteOffset"].integer(0)) + size_t(a["byteOffset"].integer(0));
        int csize = ct == 5125 ? 4 : ct == 5123 ? 2 : 1;
        size_t stride = bv.has("byteStride") ? size_t(bv["byteStride"].integer()) : size_t(csize);
        if (count > 0 && base + stride * size_t(count - 1) + size_t(csize) > buf.size()) return fail("index accessor out of range");
        for (int i = 0; i < count; i++) {
            const uint8_t* p = buf.data() + base + stride * size_t(i);
            uint32_t v = 0;
            if (csize == 4) std::memcpy(&v, p, 4);
            else if (csize == 2) {
                uint16_t u;
                std::memcpy(&u, p, 2);
                v = u;
            } else v = p[0];
            out[size_t(i)] = int(v);
        }
        return true;
    }

    bool decode_glb_image(int image, Texture& tex) {
        const Value& img = doc["images"][size_t(image)];
        std::string err;
        if (img.has("uri")) {
            const std::string& uri = img["uri"].string();
            if (uri.rfind("data:", 0) == 0) {
                std::vector<uint8_t> data;
                size_t comma = uri.find(',');
                if (comma == std::string::npos || !base64_decode(uri.substr(comma + 1), data)) return fail("invalid image data URI");
                if (!decode_image(data.data(), data.size(), tex, err)) return fail(err);
            } else {
                if (!load_image((dir / utf8_path(decode_uri(uri))).string(), tex, err)) return fail(err);
            }
        } else if (img.has("bufferView")) {
            const Value& bv = doc["bufferViews"][size_t(img["bufferView"].integer())];
            const std::vector<uint8_t>& buf = buffers.at(size_t(bv["buffer"].integer(0)));
            size_t off = size_t(bv["byteOffset"].integer(0)), len = size_t(bv["byteLength"].integer(0));
            if (off + len > buf.size()) return fail("image buffer view out of range");
            if (!decode_image(buf.data() + off, len, tex, err)) return fail(err);
        } else {
            return fail("image without data");
        }
        return true;
    }

    // The scene texture for a glTF texture used as color (sRGB) or data (linear); -1 if unusable.
    int texture(const Value& info, bool srgb) {
        if (!info.is_object() || !info.has("index")) return -1;
        int ti = info["index"].integer();
        auto key = std::make_pair(ti, srgb);
        if (auto it = texture_map.find(key); it != texture_map.end()) return it->second;
        const Value& t = doc["textures"][size_t(ti)];
        int source = t["source"].integer(-1);
        if (source < 0) {
            // Images in other formats come from extensions (KHR_texture_basisu, EXT_texture_webp...).
            const Value& ext = t["extensions"];
            if (ext.has("EXT_texture_webp")) source = ext["EXT_texture_webp"]["source"].integer(-1);
        }
        if (source < 0 || source >= int(images.size())) return -1;
        if (!image_loaded[size_t(source)]) {
            if (!decode_glb_image(source, images[size_t(source)])) {
                // A missing or undecodable image should not stop the scene from loading.
                std::fprintf(stderr, "warning: texture skipped: %s\n", error.c_str());
                error.clear();
                texture_map[key] = -1;
                return -1;
            }
            image_loaded[size_t(source)] = true;
        }
        Texture tex = images[size_t(source)];
        tex.srgb = srgb;
        const Value& sampler = doc["samplers"][size_t(t["sampler"].integer(-1))];
        tex.repeat = sampler["wrapS"].integer(10497) != 33071;
        int index = int(scene.textures.size());
        scene.textures.push_back(std::move(tex));
        texture_map[key] = index;
        return index;
    }

    void load_materials() {
        const Value& mats = doc["materials"];
        material_base = int(scene.materials.size());
        for (size_t i = 0; i < mats.size(); i++) {
            const Value& m = mats[i];
            Material mat;
            const Value& pbr = m["pbrMetallicRoughness"];
            if (pbr.has("baseColorFactor")) {
                const Value& f = pbr["baseColorFactor"];
                mat.base_color = vec4(float(f[0].number(1)), float(f[1].number(1)), float(f[2].number(1)), float(f[3].number(1)));
            }
            mat.metallic = float(pbr["metallicFactor"].number(1.0));
            mat.roughness = float(pbr["roughnessFactor"].number(1.0));
            mat.base_color_tex = texture(pbr["baseColorTexture"], true);
            mat.metallic_roughness_tex = texture(pbr["metallicRoughnessTexture"], false);
            mat.normal_tex = texture(m["normalTexture"], false);
            mat.normal_scale = float(m["normalTexture"]["scale"].number(1.0));
            mat.emissive_tex = texture(m["emissiveTexture"], true);
            vec3 e(0);
            if (m.has("emissiveFactor")) {
                const Value& f = m["emissiveFactor"];
                e = vec3(float(f[0].number()), float(f[1].number()), float(f[2].number()));
            }
            const Value& ext = m["extensions"];
            float strength = float(ext["KHR_materials_emissive_strength"]["emissiveStrength"].number(1.0));
            mat.emission = e * strength;
            mat.transmission = float(ext["KHR_materials_transmission"]["transmissionFactor"].number(0.0));
            mat.ior = float(ext["KHR_materials_ior"]["ior"].number(1.5));
            mat.double_sided = m["doubleSided"].boolean() ? 1 : 0;
            const std::string& mode = m["alphaMode"].string();
            mat.alpha_mode = mode == "MASK" ? AlphaMode::Mask : mode == "BLEND" ? AlphaMode::Blend : AlphaMode::Opaque;
            mat.alpha_cutoff = mode == "BLEND" ? 0.5f : float(m["alphaCutoff"].number(0.5));
            scene.materials.push_back(mat);
        }
    }

    int default_material_index = -1;

    // The material of primitives without one (a light gray dielectric; the glTF default would be a
    // rough white metal, which looks wrong in a path tracer).
    int default_material() {
        if (default_material_index < 0) {
            Material m;
            m.base_color = vec4(0.8f, 0.8f, 0.8f, 1);
            m.metallic = 0.0f;
            m.roughness = 0.6f;
            default_material_index = scene.add_material(m);
        }
        return default_material_index;
    }

    mat4 node_matrix(const Value& node) {
        mat4 m;
        if (node.has("matrix")) {
            const Value& a = node["matrix"];
            // glTF matrices are column-major.
            for (int c = 0; c < 4; c++)
                for (int r = 0; r < 4; r++) m.m[r][c] = float(a[size_t(c * 4 + r)].number(r == c ? 1 : 0));
            return m;
        }
        vec3 t(0), s(1);
        vec4 q(0, 0, 0, 1);
        if (node.has("translation")) t = vec3(float(node["translation"][0].number()), float(node["translation"][1].number()), float(node["translation"][2].number()));
        if (node.has("scale")) s = vec3(float(node["scale"][0].number(1)), float(node["scale"][1].number(1)), float(node["scale"][2].number(1)));
        if (node.has("rotation"))
            q = vec4(float(node["rotation"][0].number()), float(node["rotation"][1].number()), float(node["rotation"][2].number()), float(node["rotation"][3].number(1)));
        return translation(t) * rotation_quat(q) * scaling(s);
    }

    bool load_primitive(const Value& prim, const mat4& world) {
        int mode = prim["mode"].integer(4);
        if (mode != 4 && mode != 5 && mode != 6) return true;  // points and lines are not rendered
        const Value& attrs = prim["attributes"];
        if (!attrs.has("POSITION")) return true;
        std::vector<float> pos, nrm, uv, tan;
        if (!read_floats(attrs["POSITION"].integer(), 3, pos)) return false;
        if (attrs.has("NORMAL") && !read_floats(attrs["NORMAL"].integer(), 3, nrm)) return false;
        if (attrs.has("TEXCOORD_0") && !read_floats(attrs["TEXCOORD_0"].integer(), 2, uv)) return false;
        if (attrs.has("TANGENT") && !read_floats(attrs["TANGENT"].integer(), 4, tan)) return false;
        size_t nv = pos.size() / 3;
        std::vector<int> idx;
        if (prim.has("indices")) {
            if (!read_indices(prim["indices"].integer(), idx)) return false;
        } else {
            idx.resize(nv);
            for (size_t i = 0; i < nv; i++) idx[i] = int(i);
        }
        // Strips and fans to lists.
        if (mode != 4) {
            std::vector<int> list;
            for (size_t i = 2; i < idx.size(); i++) {
                if (mode == 5) {
                    if (i % 2 == 0) list.insert(list.end(), {idx[i - 2], idx[i - 1], idx[i]});
                    else list.insert(list.end(), {idx[i - 1], idx[i - 2], idx[i]});
                } else {
                    list.insert(list.end(), {idx[0], idx[i - 1], idx[i]});
                }
            }
            idx = std::move(list);
        }
        for (int i : idx)
            if (i < 0 || size_t(i) >= nv) return fail("vertex index out of range");
        mat4 normal_m = transpose(inverse(world));
        bool flip = [&] {
            const auto& m = world.m;
            float det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
            return det < 0;
        }();
        std::vector<vec3> p(nv), n;
        std::vector<vec2> t;
        for (size_t i = 0; i < nv; i++) p[i] = world.transform_point(vec3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]));
        if (!nrm.empty()) {
            n.resize(nv);
            for (size_t i = 0; i < nv; i++) n[i] = normalize(normal_m.transform_vector(vec3(nrm[3 * i], nrm[3 * i + 1], nrm[3 * i + 2])));
        }
        if (!uv.empty()) {
            t.resize(nv);
            for (size_t i = 0; i < nv; i++) t[i] = vec2(uv[2 * i], uv[2 * i + 1]);
        }
        if (flip)
            for (size_t i = 0; i + 2 < idx.size(); i += 3) std::swap(idx[i + 1], idx[i + 2]);
        int material = prim.has("material") ? material_base + prim["material"].integer() : default_material();
        size_t vbase = scene.positions.size();
        scene.add_mesh(p, idx, material, n, t);
        if (!tan.empty()) {
            scene.tangents.resize(vbase, vec4(1, 0, 0, 1));
            for (size_t i = 0; i < nv; i++) {
                vec3 tv = normalize(world.transform_vector(vec3(tan[4 * i], tan[4 * i + 1], tan[4 * i + 2])));
                scene.tangents.push_back(vec4(tv, tan[4 * i + 3] * (flip ? -1.0f : 1.0f)));
            }
        } else if (!scene.tangents.empty()) {
            scene.tangents.resize(scene.positions.size(), vec4(1, 0, 0, 1));
        }
        return true;
    }

    bool load_node(int index, const mat4& parent, int depth) {
        if (depth > 64) return fail("node hierarchy too deep (or cyclic)");
        const Value& node = doc["nodes"][size_t(index)];
        mat4 world = parent * node_matrix(node);
        if (node.has("mesh")) {
            const Value& mesh = doc["meshes"][size_t(node["mesh"].integer())];
            const Value& prims = mesh["primitives"];
            for (size_t i = 0; i < prims.size(); i++)
                if (!load_primitive(prims[i], world)) return false;
        }
        if (node.has("camera")) {
            const Value& cam = doc["cameras"][size_t(node["camera"].integer())];
            if (cam["type"].string() == "perspective") {
                Camera c;
                c.position = world.transform_point(vec3(0));
                vec3 fwd = normalize(world.transform_vector(vec3(0, 0, -1)));
                c.target = c.position + fwd;
                c.up = normalize(world.transform_vector(vec3(0, 1, 0)));
                c.vfov = float(cam["perspective"]["yfov"].number(0.8)) * 180.0f / kPi;
                scene.cameras.push_back(c);
            }
        }
        const Value& light_ref = node["extensions"]["KHR_lights_punctual"];
        if (light_ref.has("light")) {
            const Value& l = doc["extensions"]["KHR_lights_punctual"]["lights"][size_t(light_ref["light"].integer())];
            PunctualLight pl{};
            vec3 color(1);
            if (l.has("color")) color = vec3(float(l["color"][0].number(1)), float(l["color"][1].number(1)), float(l["color"][2].number(1)));
            pl.intensity = color * float(l["intensity"].number(1.0));
            pl.position = world.transform_point(vec3(0));
            pl.direction = normalize(world.transform_vector(vec3(0, 0, -1)));
            const std::string& type = l["type"].string();
            pl.type = type == "directional" ? LightType::Directional : type == "spot" ? LightType::Spot : LightType::Point;
            pl.cos_inner = std::cos(float(l["spot"]["innerConeAngle"].number(0)));
            pl.cos_outer = std::cos(float(l["spot"]["outerConeAngle"].number(kPi / 4)));
            scene.punctual.push_back(pl);
        }
        const Value& children = node["children"];
        for (size_t i = 0; i < children.size(); i++)
            if (!load_node(children[i].integer(), world, depth + 1)) return false;
        return true;
    }

    bool load(const std::string& path) {
        std::vector<uint8_t> file;
        if (!read_file(utf8_path(path), file)) return fail("cannot open " + path);
        dir = utf8_path(path).parent_path();
        std::vector<uint8_t> bin;
        bool has_bin = false;
        std::string text;
        if (file.size() >= 12 && std::memcmp(file.data(), "glTF", 4) == 0) {
            // GLB: header, a JSON chunk, an optional BIN chunk.
            size_t off = 12;
            while (off + 8 <= file.size()) {
                uint32_t len, type;
                std::memcpy(&len, &file[off], 4);
                std::memcpy(&type, &file[off + 4], 4);
                off += 8;
                if (off + len > file.size()) return fail("truncated GLB chunk");
                if (type == 0x4E4F534A) text.assign(reinterpret_cast<const char*>(&file[off]), len);
                else if (type == 0x004E4942) {
                    bin.assign(file.begin() + long(off), file.begin() + long(off + len));
                    has_bin = true;
                }
                off += (len + 3) & ~3u;
            }
        } else {
            text.assign(file.begin(), file.end());
        }
        try {
            doc = json::parse(text);
        } catch (const json::ParseError& e) {
            return fail(e.what());
        }
        if (doc["asset"]["version"].string().rfind("2", 0) != 0) return fail("only glTF 2.0 is supported");
        if (!load_buffers(has_bin ? &bin : nullptr)) return false;
        images.resize(doc["images"].size());
        image_loaded.assign(images.size(), false);
        load_materials();
        int scene_index = doc["scene"].integer(0);
        const Value& sc = doc["scenes"][size_t(scene_index)];
        mat4 identity;
        if (sc.is_null()) {
            // No scene list: load every root node.
            for (size_t i = 0; i < doc["nodes"].size(); i++)
                if (!load_node(int(i), identity, 0)) return false;
        } else {
            const Value& roots = sc["nodes"];
            for (size_t i = 0; i < roots.size(); i++)
                if (!load_node(roots[i].integer(), identity, 0)) return false;
        }
        return true;
    }
};

}  // namespace

bool load_gltf(const std::string& path, Scene& scene, std::string& error) {
    Loader loader(scene);
    bool ok = loader.load(path);
    if (!ok) error = loader.error;
    if (ok && scene.name.empty()) scene.name = utf8_path(path).stem().string();
    if (ok && !scene.cameras.empty()) scene.camera = scene.cameras.front();
    return ok;
}

// --- export ------------------------------------------------------------------------------------------

bool save_gltf(const std::string& path, const Scene& scene, std::string& error) {
    // One buffer: positions, normals (if any), indices per material group.
    std::vector<uint8_t> bin;
    auto put = [&](const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        bin.insert(bin.end(), b, b + n);
        while (bin.size() % 4) bin.push_back(0);
    };
    size_t nv = scene.positions.size();
    size_t pos_off = bin.size();
    put(scene.positions.data(), nv * sizeof(vec3));
    size_t nrm_off = bin.size();
    bool has_n = scene.normals.size() == nv;
    if (has_n) put(scene.normals.data(), nv * sizeof(vec3));
    size_t uv_off = bin.size();
    bool has_uv = scene.uvs.size() == nv;
    if (has_uv) put(scene.uvs.data(), nv * sizeof(vec2));
    vec3 lo(kInf), hi(-kInf);
    for (const vec3& p : scene.positions) {
        lo = vmin(lo, p);
        hi = vmax(hi, p);
    }
    // Group triangles by material.
    std::map<int, std::vector<uint32_t>> groups;
    for (int t = 0; t < scene.num_tris(); t++)
        for (int k = 0; k < 3; k++) groups[scene.tri_material[size_t(t)]].push_back(uint32_t(scene.indices[size_t(t) * 3 + k]));
    std::vector<std::pair<size_t, size_t>> index_views;  // offset, count
    for (auto& [m, idx] : groups) {
        index_views.push_back({bin.size(), idx.size()});
        put(idx.data(), idx.size() * 4);
    }

    std::ostringstream o;
    o.precision(9);
    o << "{\n\"asset\": {\"version\": \"2.0\", \"generator\": \"Photon\"},\n";
    o << "\"extensionsUsed\": [\"KHR_materials_emissive_strength\", \"KHR_materials_transmission\", \"KHR_materials_ior\", \"KHR_lights_punctual\"],\n";
    // Buffer as a data URI.
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
    o << "\"buffers\": [{\"byteLength\": " << bin.size() << ", \"uri\": \"data:application/octet-stream;base64," << enc << "\"}],\n";
    o << "\"bufferViews\": [";
    o << "{\"buffer\": 0, \"byteOffset\": " << pos_off << ", \"byteLength\": " << nv * 12 << "}";
    int view = 1;
    int nrm_view = -1, uv_view = -1;
    if (has_n) {
        o << ", {\"buffer\": 0, \"byteOffset\": " << nrm_off << ", \"byteLength\": " << nv * 12 << "}";
        nrm_view = view++;
    }
    if (has_uv) {
        o << ", {\"buffer\": 0, \"byteOffset\": " << uv_off << ", \"byteLength\": " << nv * 8 << "}";
        uv_view = view++;
    }
    int first_index_view = view;
    for (auto& [off, cnt] : index_views) o << ", {\"buffer\": 0, \"byteOffset\": " << off << ", \"byteLength\": " << cnt * 4 << "}";
    o << "],\n\"accessors\": [";
    o << "{\"bufferView\": 0, \"componentType\": 5126, \"count\": " << nv << ", \"type\": \"VEC3\", \"min\": [" << lo.x << ", " << lo.y << ", " << lo.z
      << "], \"max\": [" << hi.x << ", " << hi.y << ", " << hi.z << "]}";
    int acc = 1, nrm_acc = -1, uv_acc = -1;
    if (has_n) {
        o << ", {\"bufferView\": " << nrm_view << ", \"componentType\": 5126, \"count\": " << nv << ", \"type\": \"VEC3\"}";
        nrm_acc = acc++;
    }
    if (has_uv) {
        o << ", {\"bufferView\": " << uv_view << ", \"componentType\": 5126, \"count\": " << nv << ", \"type\": \"VEC2\"}";
        uv_acc = acc++;
    }
    int first_index_acc = acc;
    for (size_t i = 0; i < index_views.size(); i++)
        o << ", {\"bufferView\": " << first_index_view + int(i) << ", \"componentType\": 5125, \"count\": " << index_views[i].second << ", \"type\": \"SCALAR\"}";
    o << "],\n\"materials\": [";
    for (size_t i = 0; i < scene.materials.size(); i++) {
        const Material& m = scene.materials[i];
        float strength = std::max({m.emission.x, m.emission.y, m.emission.z, 1.0f});
        vec3 ef = m.emission / strength;
        if (i) o << ", ";
        o << "{\"pbrMetallicRoughness\": {\"baseColorFactor\": [" << m.base_color.x << ", " << m.base_color.y << ", " << m.base_color.z << ", "
          << m.base_color.w << "], \"metallicFactor\": " << m.metallic << ", \"roughnessFactor\": " << m.roughness << "}, \"emissiveFactor\": [" << ef.x
          << ", " << ef.y << ", " << ef.z << "], \"doubleSided\": " << (m.double_sided ? "true" : "false") << ", \"extensions\": {"
          << "\"KHR_materials_emissive_strength\": {\"emissiveStrength\": " << strength << "}, \"KHR_materials_transmission\": {\"transmissionFactor\": "
          << m.transmission << "}, \"KHR_materials_ior\": {\"ior\": " << m.ior << "}}}";
    }
    o << "],\n\"meshes\": [{\"primitives\": [";
    size_t gi = 0;
    for (auto& [m, idx] : groups) {
        if (gi) o << ", ";
        o << "{\"attributes\": {\"POSITION\": 0";
        if (has_n) o << ", \"NORMAL\": " << nrm_acc;
        if (has_uv) o << ", \"TEXCOORD_0\": " << uv_acc;
        o << "}, \"indices\": " << first_index_acc + int(gi) << ", \"material\": " << m << "}";
        gi++;
    }
    o << "]}],\n";
    // The camera, as a node looking along -z.
    const Camera& c = scene.camera;
    vec3 f = normalize(c.target - c.position), r = normalize(cross(f, c.up)), u = cross(r, f);
    o << "\"cameras\": [{\"type\": \"perspective\", \"perspective\": {\"yfov\": " << c.vfov * kPi / 180.0f << ", \"znear\": 0.01}}],\n";
    o << "\"extensions\": {\"KHR_lights_punctual\": {\"lights\": [";
    for (size_t i = 0; i < scene.punctual.size(); i++) {
        const PunctualLight& l = scene.punctual[i];
        float inten = std::max({l.intensity.x, l.intensity.y, l.intensity.z, 1e-6f});
        vec3 col = l.intensity / inten;
        const char* type = l.type == LightType::Directional ? "directional" : l.type == LightType::Spot ? "spot" : "point";
        if (i) o << ", ";
        o << "{\"type\": \"" << type << "\", \"color\": [" << col.x << ", " << col.y << ", " << col.z << "], \"intensity\": " << inten;
        if (l.type == LightType::Spot) o << ", \"spot\": {\"innerConeAngle\": " << std::acos(l.cos_inner) << ", \"outerConeAngle\": " << std::acos(l.cos_outer) << "}";
        o << "}";
    }
    o << "]}},\n";
    auto matrix_of = [&](vec3 pos, vec3 dir) {
        // Columns: x (right), y (up), z (backwards), translation.
        vec3 z = -normalize(dir);
        vec3 x = normalize(cross(std::fabs(z.y) > 0.99f ? vec3(1, 0, 0) : vec3(0, 1, 0), z));
        vec3 y = cross(z, x);
        std::ostringstream m;
        m.precision(9);
        m << "[" << x.x << ", " << x.y << ", " << x.z << ", 0, " << y.x << ", " << y.y << ", " << y.z << ", 0, " << z.x << ", " << z.y << ", " << z.z << ", 0, "
          << pos.x << ", " << pos.y << ", " << pos.z << ", 1]";
        return m.str();
    };
    o << "\"nodes\": [{\"mesh\": 0}, {\"camera\": 0, \"matrix\": [" << r.x << ", " << r.y << ", " << r.z << ", 0, " << u.x << ", " << u.y << ", " << u.z
      << ", 0, " << -f.x << ", " << -f.y << ", " << -f.z << ", 0, " << c.position.x << ", " << c.position.y << ", " << c.position.z << ", 1]}";
    for (size_t i = 0; i < scene.punctual.size(); i++)
        o << ", {\"matrix\": " << matrix_of(scene.punctual[i].position, scene.punctual[i].direction) << ", \"extensions\": {\"KHR_lights_punctual\": {\"light\": " << i
          << "}}}";
    o << "],\n\"scenes\": [{\"nodes\": [0, 1";
    for (size_t i = 0; i < scene.punctual.size(); i++) o << ", " << i + 2;
    o << "]}],\n\"scene\": 0\n}\n";
    std::ofstream out(utf8_path(path), std::ios::binary);
    if (!out) {
        error = "cannot write " + path;
        return false;
    }
    out << o.str();
    return true;
}

}  // namespace photon
