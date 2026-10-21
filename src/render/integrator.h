// The path tracing integrator, shared by the CPU and GPU renderers.
//
// Each path vertex combines two estimators of direct light with multiple importance sampling:
//  - next event estimation: pick a light (emissive triangle, punctual light or the environment), trace
//    a shadow ray, weight by the power heuristic against the BSDF's density for that direction;
//  - BSDF sampling: continue the path; if it hits an emitter, weight that emission by the power
//    heuristic against the light-sampling density.
// Delta lights and (near-)specular surfaces skip the weighting where the other strategy cannot
// produce the sample. Paths are terminated by Russian roulette after a few bounces.
#pragma once

#include <cstring>

#include "core/color.h"
#include "render/bsdf.h"
#include "render/lights.h"
#include "render/traverse.h"

namespace photon {

struct PathOptions {
    int max_depth = 8;           // path vertices after the camera
    int rr_depth = 3;            // Russian roulette starts after this many bounces
    float clamp = 0.0f;          // clamp the luminance of indirect contributions (0: off, unbiased)
    int nee = 1;                 // next event estimation
    uint32_t* ray_count = nullptr;  // if set (host only, single thread), counts the rays traced
};

// What the first visible surface looks like, for the denoiser and debug views.
struct PrimaryInfo {
    vec3 position;
    float depth = 0;      // distance from the camera, 0 for misses
    vec3 normal;
    int material = -1;    // -1 for misses
    vec3 albedo = vec3(1);
    vec3 emission;        // radiance emitted by the first surface (or the environment on a miss)
};

PHOTON_HD uint32_t float_bits(float f) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(f);
#else
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
#endif
}

PHOTON_HD float bits_float(uint32_t u) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(u);
#else
    float f;
    std::memcpy(&f, &u, 4);
    return f;
#endif
}

// Moves a ray origin off a surface along the geometric normal by a few ULPs, scaled to the position's
// magnitude (Wachter and Binder, "A Fast and Robust Method for Avoiding Self-Intersection", 2019).
PHOTON_HD vec3 offset_ray(vec3 p, vec3 n) {
    const float origin = 1.0f / 32.0f, float_scale = 1.0f / 65536.0f, int_scale = 256.0f;
    auto off = [&](float pc, float nc) {
        int of = int(int_scale * nc);
        float pi = bits_float(float_bits(pc) + uint32_t(pc < 0 ? -of : of));
        return fabsf(pc) < origin ? pc + float_scale * nc : pi;
    };
    return {off(p.x, n.x), off(p.y, n.y), off(p.z, n.z)};
}

struct Interaction {
    vec3 p;
    vec3 ng;       // geometric normal, facing the side the ray came from for opaque surfaces
    vec3 ns;       // shading normal
    vec2 uv;
    Frame frame;   // around the shading normal
    BsdfParams params;
    vec3 emission;
    bool front;    // the ray hit the front side (by winding)
    int material;
};

PHOTON_HD Interaction surface_at(const SceneView& s, const Ray& ray, const Hit& hit) {
    Interaction it;
    const TriGeom& g = s.tris[hit.tri];
    const Material& m = s.materials[g.material];
    it.material = g.material;
    float w = 1 - hit.u - hit.v;
    it.p = ray.o + ray.d * hit.t;
    vec3 ng = normalize(cross(g.e1, g.e2));
    const int* idx = s.indices + 3 * hit.tri;
    vec3 ns = ng;
    if (s.normals) {
        vec3 n = s.normals[idx[0]] * w + s.normals[idx[1]] * hit.u + s.normals[idx[2]] * hit.v;
        if (length_squared(n) > 1e-12f) ns = normalize(n);
        if (dot(ns, ng) < 0) ns = -ns;
    }
    it.uv = s.uvs ? s.uvs[idx[0]] * w + s.uvs[idx[1]] * hit.u + s.uvs[idx[2]] * hit.v : vec2(0, 0);
    it.front = dot(ray.d, ng) < 0;

    // Normal mapping in the tangent frame.
    if (m.normal_tex >= 0 && s.tangents) {
        vec4 t4 = s.tangents[idx[0]] * w + s.tangents[idx[1]] * hit.u + s.tangents[idx[2]] * hit.v;
        vec3 t = t4.xyz();
        if (length_squared(t) > 1e-12f) {
            t = normalize(t - ns * dot(ns, t));
            vec3 b = cross(ns, t) * (s.tangents[idx[0]].w < 0 ? -1.0f : 1.0f);
            vec4 tn = sample_texture(s, m.normal_tex, it.uv);
            vec3 nt(tn.x * 2 - 1, tn.y * 2 - 1, tn.z * 2 - 1);
            nt.x *= m.normal_scale;
            nt.y *= m.normal_scale;
            vec3 n = normalize(t * nt.x + b * nt.y + ns * nt.z);
            if (dot(n, ng) > 0.01f) ns = n;
        }
    }

    // Material parameters.
    vec4 base = m.base_color;
    if (m.base_color_tex >= 0) {
        vec4 tc = sample_texture(s, m.base_color_tex, it.uv);
        base = vec4(base.x * tc.x, base.y * tc.y, base.z * tc.z, base.w * tc.w);
    }
    float metallic = m.metallic, roughness = m.roughness;
    if (m.metallic_roughness_tex >= 0) {
        vec4 mr = sample_texture(s, m.metallic_roughness_tex, it.uv);
        roughness *= mr.y;
        metallic *= mr.z;
    }
    it.params.base = base.xyz();
    it.params.metallic = metallic;
    it.params.roughness = roughness;
    it.params.transmission = m.transmission;
    it.params.ior = m.ior;
    it.emission = (it.front || m.double_sided) ? emitted(s, hit.tri, hit.u, hit.v) : vec3(0);

    // Opaque surfaces are shaded from whichever side was hit; transmissive ones keep the outward normal
    // so that the BSDF can tell entering from leaving.
    if (m.transmission <= 0 && !it.front) {
        ng = -ng;
        ns = -ns;
    }
    vec3 wo = -ray.d;
    if (m.transmission <= 0 && dot(wo, ns) <= 0) ns = ng;  // a shading normal facing away from the viewer
    it.ng = ng;
    it.ns = ns;
    it.frame = Frame(ns);
    return it;
}

// Traces one path and returns the radiance it carries to the camera.
PHOTON_HD vec3 trace_path(const SceneView& s, Ray ray, Rng& rng, const PathOptions& opt, PrimaryInfo* primary) {
    vec3 L(0), beta(1);
    float prev_pdf = 0;
    bool prev_delta = true;  // the camera ray counts as specular: emitters it sees get full weight
    bool has_lights = s.p_env + s.p_emissive + s.p_punctual > 0;
    for (int depth = 0;; depth++) {
        Hit hit;
        if (opt.ray_count) ++*opt.ray_count;
        if (!intersect(s, ray, hit)) {
            vec3 le = env_radiance(s.env, ray.d);
            float w = (prev_delta || s.p_env <= 0) ? 1.0f : power_heuristic(prev_pdf, env_pdf(s, ray.d));
            vec3 c = beta * le * w;
            if (depth > 0 && opt.clamp > 0) {
                float l = luminance(c);
                if (l > opt.clamp) c *= opt.clamp / l;
            }
            L += c;
            if (depth == 0 && primary) {
                primary->depth = 0;
                primary->material = -1;
                primary->emission = le;
                primary->albedo = vec3(1);
                primary->normal = -ray.d;
                primary->position = ray.o + ray.d * 1e4f;
            }
            break;
        }
        Interaction it = surface_at(s, ray, hit);
        vec3 wo_world = -ray.d;

        // Emission reached by BSDF sampling (or seen directly).
        if (!is_black(it.emission)) {
            float w = 1.0f;
            if (!prev_delta) w = power_heuristic(prev_pdf, emissive_pdf(s, hit.tri, ray.d, hit.t));
            vec3 c = beta * it.emission * w;
            if (depth > 0 && opt.clamp > 0) {
                float l = luminance(c);
                if (l > opt.clamp) c *= opt.clamp / l;
            }
            L += c;
        }

        Bsdf bsdf(it.params);
        if (depth == 0 && primary) {
            primary->position = it.p;
            primary->depth = hit.t;
            primary->normal = it.ns;
            primary->material = it.material;
            primary->albedo = bsdf.albedo();
            primary->emission = it.emission;
        }
        if (depth + 1 >= opt.max_depth) break;

        vec3 wo = it.frame.to_local(wo_world);
        // Next event estimation.
        bool nee = opt.nee && has_lights && bsdf.samples_lights();
        if (nee) {
            float ug = rng.next();
            vec2 ul = rng.next2();
            float uk = rng.next();
            LightSample ls = sample_light(s, it.p, ug, ul, uk);
            if (ls.pdf > 0 && !is_black(ls.radiance)) {
                vec3 wi = it.frame.to_local(ls.wi);
                float pdf_b;
                vec3 f = bsdf.eval_pdf(wo, wi, &pdf_b);
                if (!is_black(f)) {
                    // Leave the surface on the side the light is on.
                    vec3 origin = offset_ray(it.p, dot(ls.wi, it.ng) > 0 ? it.ng : -it.ng);
                    Ray shadow;
                    shadow.o = origin;
                    shadow.d = ls.wi;
                    shadow.tmax = ls.dist == kInf ? kInf : ls.dist * (1.0f - 1e-4f);
                    if (opt.ray_count) ++*opt.ray_count;
                    if (!occluded(s, shadow)) {
                        float w = ls.delta ? 1.0f : power_heuristic(ls.pdf, pdf_b);
                        vec3 c = beta * f * fabsf(wi.z) * ls.radiance * (w / ls.pdf);
                        if (depth > 0 && opt.clamp > 0) {
                            float l = luminance(c);
                            if (l > opt.clamp) c *= opt.clamp / l;
                        }
                        L += c;
                    }
                }
            }
        }

        // Continue the path by sampling the BSDF.
        float u_lobe = rng.next();
        vec2 u = rng.next2();
        float u_sub = rng.next();
        BsdfSample bs = bsdf.sample(wo, u_lobe, u, u_sub);
        if (!bs.valid) break;
        beta *= bs.f * (fabsf(bs.wi.z) / bs.pdf);
        if (!is_finite(beta) || is_black(beta)) break;
        prev_pdf = bs.pdf;
        prev_delta = !nee;  // without light sampling here, emitters found by the BSDF get full weight
        vec3 wi_world = it.frame.to_world(bs.wi);
        // With an interpolated shading normal, a direction can be on the shading side but cross the actual
        // surface: drop the path instead of letting it leak through the geometry.
        float side = dot(wi_world, it.ng) * dot(wo_world, it.ng);
        if (bs.transmitted ? side >= 0 : side <= 0) break;
        ray.o = offset_ray(it.p, dot(wi_world, it.ng) > 0 ? it.ng : -it.ng);
        ray.d = normalize(wi_world);
        ray.tmax = kInf;

        if (depth + 1 >= opt.rr_depth) {
            float q = fmaxf(0.05f, 1.0f - max_component(beta));
            if (rng.next() < q) break;
            beta *= 1.0f / (1.0f - q);
        }
    }
    return L;
}

// A camera ray through pixel (px, py) + jitter, with a thin lens when the aperture is open.
PHOTON_HD Ray camera_ray(const CameraView& c, float px, float py, int width, int height, vec2 lens) {
    float sx = (2.0f * px / float(width) - 1.0f) * c.tan_half_fov * c.aspect;
    float sy = (1.0f - 2.0f * py / float(height)) * c.tan_half_fov;
    vec3 d = normalize(c.forward + c.right * sx + c.up * sy);
    Ray r;
    r.o = c.position;
    r.d = d;
    if (c.aperture > 0) {
        vec3 focus = c.position + d * (c.focus_distance / dot(d, c.forward));
        vec2 l = concentric_disk(lens) * c.aperture;
        r.o = c.position + c.right * l.x + c.up * l.y;
        r.d = normalize(focus - r.o);
    }
    return r;
}

}  // namespace photon
