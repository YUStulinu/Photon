// Light sampling for next event estimation, and the matching densities for multiple importance
// sampling: emissive triangles (chosen by power with an alias table, then uniformly by area), glTF
// punctual lights (delta lights), and the environment (importance-sampled by its image).
#pragma once

#include "core/sampling.h"
#include "render/scene_view.h"
#include "render/texture.h"

namespace photon {

struct LightSample {
    vec3 wi;              // direction to the light
    float dist = 0;       // distance to the light point (kInf for distant lights)
    vec3 radiance;        // incident radiance (or intensity-derived for delta lights)
    float pdf = 0;        // solid angle density, including the choice of light (1 for delta lights)
    bool delta = false;   // a point or directional light: no MIS
    int tri = -1;         // the emissive triangle, if any
    vec2 bary;            // its barycentric coordinates
};

PHOTON_HD int sample_alias(const AliasEntry* table, int n, float u, float* pmf) {
    float x = u * float(n);
    int i = int(x);
    if (i >= n) i = n - 1;
    float frac = x - float(i);
    int chosen = frac < table[i].prob ? i : table[i].alias;
    *pmf = table[chosen].pmf;
    return chosen;
}

// --- the environment -------------------------------------------------------------------------------

PHOTON_HD vec3 rotate_y(vec3 d, float angle) {
    float c = cosf(angle), s = sinf(angle);
    return {c * d.x + s * d.z, d.y, -s * d.x + c * d.z};
}

PHOTON_HD vec3 env_radiance(const EnvironmentView& env, vec3 dir) {
    if (!env.texels) return env.color * env.intensity;
    vec2 uv = direction_to_equirect(rotate_y(dir, -env.rotation));
    int x = int(uv.x * float(env.width));
    int y = int(uv.y * float(env.height));
    x = x < 0 ? 0 : (x >= env.width ? env.width - 1 : x);
    y = y < 0 ? 0 : (y >= env.height ? env.height - 1 : y);
    return env.texels[y * env.width + x] * env.intensity;
}

// Index of the last CDF entry <= u (binary search over n + 1 entries).
PHOTON_HD int find_interval(const float* cdf, int n, float u) {
    int lo = 0, hi = n;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] <= u) lo = mid;
        else hi = mid;
    }
    return lo;
}

// The environment image's density over directions (excluding the light group choice).
PHOTON_HD float env_pdf_dir(const EnvironmentView& env, vec3 dir) {
    if (!env.texels) return 1.0f / (4.0f * kPi);
    vec3 d = rotate_y(dir, -env.rotation);
    vec2 uv = direction_to_equirect(d);
    float sin_theta = safe_sqrt(1 - d.y * d.y);
    if (sin_theta <= 0 || env.marginal_integral <= 0) return 0;
    int x = int(uv.x * float(env.width)), y = int(uv.y * float(env.height));
    x = x < 0 ? 0 : (x >= env.width ? env.width - 1 : x);
    y = y < 0 ? 0 : (y >= env.height ? env.height - 1 : y);
    float row_sin = sinf((float(y) + 0.5f) / float(env.height) * kPi);
    float f = luminance(env.texels[y * env.width + x]) * row_sin;
    float pdf_uv = f / env.marginal_integral;
    return pdf_uv / (2.0f * kPi * kPi * sin_theta);
}

PHOTON_HD bool sample_env(const EnvironmentView& env, vec2 u, vec3& dir, float& pdf) {
    if (!env.texels) {
        dir = uniform_sphere(u);
        pdf = 1.0f / (4.0f * kPi);
        return true;
    }
    int y = find_interval(env.marginal_cdf, env.height, u.y);
    float c0 = env.marginal_cdf[y], c1 = env.marginal_cdf[y + 1];
    float dv = c1 > c0 ? (u.y - c0) / (c1 - c0) : 0.5f;
    const float* row = env.conditional_cdf + y * (env.width + 1);
    int x = find_interval(row, env.width, u.x);
    float r0 = row[x], r1 = row[x + 1];
    float du = r1 > r0 ? (u.x - r0) / (r1 - r0) : 0.5f;
    vec2 uv((float(x) + du) / float(env.width), (float(y) + dv) / float(env.height));
    vec3 d = equirect_to_direction(uv);
    dir = rotate_y(d, env.rotation);
    float sin_theta = safe_sqrt(1 - d.y * d.y);
    if (sin_theta <= 1e-6f) return false;
    float row_sin = sinf((float(y) + 0.5f) / float(env.height) * kPi);
    float f = luminance(env.texels[y * env.width + x]) * row_sin;
    pdf = f / env.marginal_integral / (2.0f * kPi * kPi * sin_theta);
    return pdf > 0;
}

// --- emissive triangles ------------------------------------------------------------------------------

// Emitted radiance of an emissive triangle at barycentrics (u, v), towards a direction on the
// emitting side (front side, or both for double-sided materials).
PHOTON_HD vec3 emitted(const SceneView& s, int tri, float u, float v) {
    const Material& m = s.materials[s.tris[tri].material];
    vec3 e = m.emission;
    if (m.emissive_tex >= 0 && s.uvs) {
        const int* idx = s.indices + 3 * tri;
        vec2 uv = s.uvs[idx[0]] * (1 - u - v) + s.uvs[idx[1]] * u + s.uvs[idx[2]] * v;
        e *= sample_texture(s, m.emissive_tex, uv).xyz();
    }
    return e;
}

PHOTON_HD bool emits_towards(const SceneView& s, int tri, vec3 dir_from_light) {
    const TriGeom& g = s.tris[tri];
    vec3 n = cross(g.e1, g.e2);
    return s.materials[g.material].double_sided != 0 || dot(n, dir_from_light) > 0;
}

// The density with which next event estimation picks the point on emissive triangle `tri` seen at
// distance `dist` along `dir`.
PHOTON_HD float emissive_pdf(const SceneView& s, int tri, vec3 dir, float dist) {
    if (s.num_emissive == 0 || !s.tri_emissive) return 0;
    int k = s.tri_emissive[tri];
    if (k < 0) return 0;
    const EmissiveTri& e = s.emissive[k];
    const TriGeom& g = s.tris[tri];
    vec3 n = normalize(cross(g.e1, g.e2));
    float cos_l = fabsf(dot(n, dir));
    if (cos_l <= 0) return 0;
    return s.p_emissive * s.emissive_alias[k].pmf * dist * dist / (e.area * cos_l);
}

PHOTON_HD float env_pdf(const SceneView& s, vec3 dir) { return s.p_env * env_pdf_dir(s.env, dir); }

// --- picking a light ---------------------------------------------------------------------------------

PHOTON_HD LightSample sample_light(const SceneView& s, vec3 p, float u_group, vec2 u, float u_light) {
    LightSample ls;
    if (u_group < s.p_emissive && s.num_emissive > 0) {
        float pmf;
        int k = sample_alias(s.emissive_alias, s.num_emissive, u_light, &pmf);
        const EmissiveTri& e = s.emissive[k];
        const TriGeom& g = s.tris[e.tri];
        vec2 b = uniform_triangle(u);
        vec3 q = g.v0 + g.e1 * b.x + g.e2 * b.y;
        vec3 d = q - p;
        float dist2 = length_squared(d);
        if (dist2 <= 0) return ls;
        float dist = sqrtf(dist2);
        vec3 wi = d / dist;
        vec3 n = normalize(cross(g.e1, g.e2));
        float cos_l = fabsf(dot(n, wi));
        if (cos_l <= 1e-8f || !emits_towards(s, e.tri, -wi)) return ls;
        ls.wi = wi;
        ls.dist = dist;
        ls.radiance = emitted(s, e.tri, b.x, b.y);
        ls.pdf = s.p_emissive * pmf * dist2 / (e.area * cos_l);
        ls.tri = e.tri;
        ls.bary = b;
        return ls;
    }
    if (u_group < s.p_emissive + s.p_punctual && s.num_punctual > 0) {
        int k = int(u_light * float(s.num_punctual));
        if (k >= s.num_punctual) k = s.num_punctual - 1;
        const PunctualLight& l = s.punctual[k];
        float pick = s.p_punctual / float(s.num_punctual);
        ls.delta = true;
        ls.pdf = pick;
        if (l.type == LightType::Directional) {
            ls.wi = -l.direction;
            ls.dist = kInf;
            ls.radiance = l.intensity;
            return ls;
        }
        vec3 d = l.position - p;
        float dist2 = length_squared(d);
        if (dist2 <= 0) return LightSample{};
        float dist = sqrtf(dist2);
        ls.wi = d / dist;
        ls.dist = dist;
        vec3 li = l.intensity / dist2;
        if (l.type == LightType::Spot) {
            float cd = dot(-ls.wi, l.direction);
            float t = saturate((cd - l.cos_outer) / fmaxf(l.cos_inner - l.cos_outer, 1e-4f));
            li *= t * t;
        }
        ls.radiance = li;
        return ls;
    }
    if (s.p_env > 0) {
        vec3 dir;
        float pdf;
        if (!sample_env(s.env, u, dir, pdf)) return ls;
        ls.wi = dir;
        ls.dist = kInf;
        ls.radiance = env_radiance(s.env, dir);
        ls.pdf = s.p_env * pdf;
    }
    return ls;
}

}  // namespace photon
