#include "scene/scene.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "core/color.h"
#include "core/sampling.h"
#include "render/lights.h"

namespace photon {

CameraView Camera::view(float aspect) const {
    CameraView v;
    v.position = position;
    v.forward = normalize(target - position);
    v.right = normalize(cross(v.forward, up));
    if (length_squared(v.right) < 1e-12f) v.right = vec3(1, 0, 0);
    v.up = cross(v.right, v.forward);
    v.tan_half_fov = std::tan(vfov * 0.5f * kPi / 180.0f);
    v.aspect = aspect;
    v.aperture = aperture;
    v.focus_distance = focus_distance > 0 ? focus_distance : length(target - position);
    return v;
}

void Scene::add_mesh(const std::vector<vec3>& pos, const std::vector<int>& idx, int material, const std::vector<vec3>& nrm,
                     const std::vector<vec2>& uv) {
    size_t base = positions.size();
    // Each attribute array is either empty or has one entry per vertex.
    auto append = [&](auto& arr, const auto& src, auto zero) {
        if (src.empty() && arr.empty()) return;
        arr.resize(base, zero);
        if (src.empty()) arr.resize(base + pos.size(), zero);
        else arr.insert(arr.end(), src.begin(), src.end());
    };
    append(normals, nrm, vec3(0));
    append(uvs, uv, vec2(0, 0));
    positions.insert(positions.end(), pos.begin(), pos.end());
    for (int i : idx) indices.push_back(int(base) + i);
    for (size_t t = 0; t < idx.size() / 3; t++) tri_material.push_back(material);
}

int Scene::add_material(const Material& m) {
    materials.push_back(m);
    return int(materials.size()) - 1;
}

// Per-vertex tangents from UV derivatives, accumulated over the adjacent triangles (Lengyel's method).
void Scene::compute_tangents() {
    if (uvs.empty() || !tangents.empty()) return;
    bool needed = false;
    for (const Material& m : materials) needed |= m.normal_tex >= 0;
    if (!needed) return;
    size_t nv = positions.size();
    std::vector<vec3> tan(nv, vec3(0)), bit(nv, vec3(0));
    for (size_t t = 0; t + 2 < indices.size(); t += 3) {
        int a = indices[t], b = indices[t + 1], c = indices[t + 2];
        vec3 e1 = positions[b] - positions[a], e2 = positions[c] - positions[a];
        vec2 d1 = uvs[b] - uvs[a], d2 = uvs[c] - uvs[a];
        float det = d1.x * d2.y - d2.x * d1.y;
        if (std::fabs(det) < 1e-20f) continue;
        float r = 1.0f / det;
        vec3 sdir = (e1 * d2.y - e2 * d1.y) * r;
        vec3 tdir = (e2 * d1.x - e1 * d2.x) * r;
        for (int v : {a, b, c}) {
            tan[v] += sdir;
            bit[v] += tdir;
        }
    }
    tangents.resize(nv);
    for (size_t v = 0; v < nv; v++) {
        vec3 n = normals.empty() ? vec3(0, 0, 1) : normals[v];
        vec3 t = tan[v] - n * dot(n, tan[v]);
        if (length_squared(t) < 1e-20f) t = Frame(normalize(n)).t;
        t = normalize(t);
        float w = dot(cross(n, t), bit[v]) < 0 ? -1.0f : 1.0f;
        tangents[v] = vec4(t, w);
    }
}

std::vector<AliasEntry> build_alias_table(const std::vector<float>& weights) {
    size_t n = weights.size();
    std::vector<AliasEntry> table(n);
    double total = 0;
    for (float w : weights) total += std::max(0.0f, w);
    if (n == 0) return table;
    std::vector<double> scaled(n);
    std::vector<size_t> small, large;
    for (size_t i = 0; i < n; i++) {
        double p = total > 0 ? std::max(0.0f, weights[i]) / total : 1.0 / double(n);
        table[i].pmf = float(p);
        table[i].alias = int(i);
        scaled[i] = p * double(n);
        (scaled[i] < 1.0 ? small : large).push_back(i);
    }
    while (!small.empty() && !large.empty()) {
        size_t s = small.back(), l = large.back();
        small.pop_back();
        table[s].prob = float(scaled[s]);
        table[s].alias = int(l);
        scaled[l] = scaled[l] + scaled[s] - 1.0;
        if (scaled[l] < 1.0) {
            large.pop_back();
            small.push_back(l);
        }
    }
    for (size_t i : small) table[i].prob = 1.0f;
    for (size_t i : large) table[i].prob = 1.0f;
    return table;
}

void Scene::build_lights() {
    emissive.clear();
    tri_emissive.assign(tris.size(), -1);
    std::vector<float> power;
    emissive_power = 0;
    SceneView v = view();
    for (size_t i = 0; i < tris.size(); i++) {
        const Material& m = materials[tris[i].material];
        float lum = luminance(m.emission);
        if (lum <= 0) continue;
        if (m.emissive_tex >= 0 && v.uvs) {
            // A textured emitter: estimate the triangle's mean emission from a few texture lookups, and
            // leave dark triangles to BSDF sampling (most of a typical emissive texture is black).
            static const float pts[7][2] = {{1.f / 3, 1.f / 3}, {0.1f, 0.1f}, {0.8f, 0.1f}, {0.1f, 0.8f}, {0.45f, 0.1f}, {0.45f, 0.45f}, {0.1f, 0.45f}};
            float sum = 0;
            for (const auto& b : pts) sum += luminance(emitted(v, int(i), b[0], b[1]));
            lum = sum / 7.0f;
            if (lum <= 1e-4f) continue;
        }
        float area = 0.5f * length(cross(tris[i].e1, tris[i].e2));
        if (area <= 0) continue;
        tri_emissive[i] = int(emissive.size());
        emissive.push_back({int(i), area});
        float p = area * lum * kPi * (m.double_sided ? 2.0f : 1.0f);
        power.push_back(p);
        emissive_power += p;
    }
    emissive_alias = build_alias_table(power);

    // Light group probabilities by estimated power arriving at the scene.
    Aabb b = bvh.bounds;
    float radius = b.empty() ? 1.0f : 0.5f * length(b.max - b.min);
    float env_power = 0;
    if (!env.texels.empty()) {
        double sum = 0;
        for (const vec3& t : env.texels) sum += luminance(t);
        env_power = float(sum / double(env.texels.size())) * env.intensity * kPi * kPi * radius * radius * 4.0f;
    } else {
        env_power = luminance(env.color) * env.intensity * kPi * kPi * radius * radius * 4.0f;
    }
    float punctual_power = 0;
    for (const PunctualLight& l : punctual) {
        float lum = luminance(l.intensity);
        punctual_power += l.type == LightType::Directional ? lum * kPi * radius * radius : lum * 4.0f * kPi;
    }
    float total = env_power + emissive_power + punctual_power;
    p_env = p_emissive = p_punctual = 0;
    if (total > 0) {
        p_env = env_power / total;
        p_emissive = emissive_power / total;
        p_punctual = punctual_power / total;
        // Keep every present group sampled now and then: power estimates ignore occlusion.
        auto floor_prob = [](float& p, bool present) {
            if (present) p = std::max(p, 0.05f);
        };
        floor_prob(p_env, env_power > 0);
        floor_prob(p_emissive, emissive_power > 0);
        floor_prob(p_punctual, punctual_power > 0);
        float s = p_env + p_emissive + p_punctual;
        p_env /= s;
        p_emissive /= s;
        p_punctual /= s;
    }
}

// A piecewise-constant 2D distribution over the environment image proportional to
// luminance * sin(theta), sampled by inverting a marginal CDF over rows and a conditional CDF per row.
void Scene::build_environment() {
    env_marginal.clear();
    env_conditional.clear();
    env_integral = 0;
    if (env.texels.empty()) return;
    int w = env.width, h = env.height;
    env_conditional.resize(size_t(h) * size_t(w + 1));
    env_marginal.resize(size_t(h) + 1);
    std::vector<double> row_sum(h);
    for (int y = 0; y < h; y++) {
        float sin_theta = std::sin((float(y) + 0.5f) / float(h) * kPi);
        float* cdf = &env_conditional[size_t(y) * size_t(w + 1)];
        double acc = 0;
        cdf[0] = 0;
        for (int x = 0; x < w; x++) {
            acc += double(luminance(env.texels[size_t(y) * size_t(w) + x])) * sin_theta;
            cdf[x + 1] = float(acc);
        }
        row_sum[y] = acc;
        for (int x = 1; x <= w; x++) cdf[x] = acc > 0 ? float(cdf[x] / acc) : float(x) / float(w);
    }
    double acc = 0;
    env_marginal[0] = 0;
    for (int y = 0; y < h; y++) {
        acc += row_sum[y];
        env_marginal[y + 1] = float(acc);
    }
    for (int y = 1; y <= h; y++) env_marginal[y] = acc > 0 ? float(env_marginal[y] / acc) : float(y) / float(h);
    // The average of f over the unit square: pdf(u, v) = f(u, v) / integral.
    env_integral = float(acc / (double(w) * double(h)));
}

void Scene::build(const BvhBuildOptions& options) {
    compute_tangents();
    if (materials.empty()) materials.push_back(Material{});
    int n = num_tris();
    std::vector<vec3> verts(size_t(n) * 3);
    for (int t = 0; t < n; t++)
        for (int k = 0; k < 3; k++) verts[size_t(t) * 3 + k] = positions[indices[size_t(t) * 3 + k]];
    bvh = build_bvh(verts, options);
    tris.resize(n);
    bvh_indices.resize(size_t(n) * 3);
    for (int i = 0; i < n; i++) {
        int t = bvh.order[i];
        vec3 a = verts[size_t(t) * 3], b = verts[size_t(t) * 3 + 1], c = verts[size_t(t) * 3 + 2];
        TriGeom g{};
        g.v0 = a;
        g.e1 = b - a;
        g.e2 = c - a;
        g.material = std::clamp(tri_material[t], 0, int(materials.size()) - 1);
        const Material& m = materials[g.material];
        g.flags = (m.alpha_mode != AlphaMode::Opaque) ? kTriAlphaTested : 0;
        tris[i] = g;
        for (int k = 0; k < 3; k++) bvh_indices[size_t(i) * 3 + k] = indices[size_t(t) * 3 + k];
    }

    // Textures into one pool.
    texel_pool.clear();
    texture_infos.clear();
    for (const Texture& t : textures) {
        TextureInfo info;
        info.width = t.width;
        info.height = t.height;
        info.offset = uint32_t(texel_pool.size());
        info.wrap_repeat = t.repeat ? 1 : 0;
        info.srgb = t.srgb ? 1 : 0;
        texel_pool.insert(texel_pool.end(), t.texels.begin(), t.texels.end());
        texture_infos.push_back(info);
    }
    srgb_lut.resize(256);
    for (int i = 0; i < 256; i++) srgb_lut[i] = srgb_to_linear(float(i) / 255.0f);

    build_lights();
    build_environment();
    built = true;
}

void Scene::rebuild_lights() {
    for (TriGeom& g : tris) {
        const Material& m = materials[size_t(g.material)];
        g.flags = (m.alpha_mode != AlphaMode::Opaque) ? kTriAlphaTested : 0;
    }
    build_lights();
    build_environment();
}

SceneView Scene::view() const {
    SceneView v;
    v.nodes = bvh.nodes.data();
    v.num_nodes = int(bvh.nodes.size());
    v.tris = tris.data();
    v.indices = bvh_indices.data();
    v.num_tris = int(tris.size());
    v.normals = normals.empty() ? nullptr : normals.data();
    v.uvs = uvs.empty() ? nullptr : uvs.data();
    v.tangents = tangents.empty() ? nullptr : tangents.data();
    v.materials = materials.data();
    v.textures = texture_infos.data();
    v.texels = texel_pool.data();
    v.srgb_lut = srgb_lut.data();
    v.emissive = emissive.data();
    v.emissive_alias = emissive_alias.data();
    v.tri_emissive = tri_emissive.data();
    v.num_emissive = int(emissive.size());
    v.emissive_power = emissive_power;
    v.punctual = punctual.data();
    v.num_punctual = int(punctual.size());
    v.env.texels = env.texels.empty() ? nullptr : env.texels.data();
    v.env.width = env.width;
    v.env.height = env.height;
    v.env.marginal_cdf = env_marginal.empty() ? nullptr : env_marginal.data();
    v.env.conditional_cdf = env_conditional.empty() ? nullptr : env_conditional.data();
    v.env.marginal_integral = env_integral;
    v.env.intensity = env.intensity;
    v.env.rotation = env.rotation;
    v.env.color = env.color;
    v.p_env = p_env;
    v.p_emissive = p_emissive;
    v.p_punctual = p_punctual;
    v.bounds_min = bvh.bounds.min;
    v.bounds_max = bvh.bounds.max;
    return v;
}

}  // namespace photon
