// The surface scattering model: glTF 2.0 metallic-roughness with transmission.
//
// The BSDF is a weighted sum of three components:
//
//   metal     (weight metallic)                      GGX reflection with a Schlick conductor Fresnel
//                                                    tinted by the base color;
//   opaque    (weight (1 - metallic)(1 - transmission))  F(h) * GGX reflection + a Lambertian base layer
//                                                    weighted by (1 - F(wo)) (1 - F(wi)) / (1 - F_avg):
//                                                    reciprocal and energy conserving (the glTF reference
//                                                    model's (1 - F(h)) weight gains energy at grazing
//                                                    angles: up to +37% for a glossy plastic);
//   glass     (weight (1 - metallic) transmission)    a rough dielectric (Walter et al. 2007), reflecting
//                                                    and refracting with the exact Fresnel equations.
//
// The microfacet distribution is GGX (Trowbridge-Reitz) with alpha = roughness^2, the height-correlated
// Smith masking-shadowing term, and directions are sampled from the distribution of visible normals
// (Heitz 2018), which is much less noisy than sampling the plain distribution at grazing angles.
// `pdf` returns the exact density of `sample` (the mixture over all lobes that can produce a given
// direction), which multiple importance sampling relies on.
//
// All vectors are in the local shading frame: the normal is +z, and the normal is on the outside of the
// surface (wo.z < 0 means the path is inside a transmissive object).
#pragma once

#include "core/sampling.h"

namespace photon {

struct BsdfParams {
    vec3 base = vec3(0.8f);
    float metallic = 0.0f;
    float roughness = 0.5f;
    float transmission = 0.0f;
    float ior = 1.5f;
};

struct BsdfSample {
    vec3 wi;
    vec3 f;        // BSDF value (without the cosine)
    float pdf = 0;  // solid angle density of the whole mixture
    bool transmitted = false;
    bool valid = false;
};

PHOTON_HD float ggx_d(vec3 h, float alpha) {
    float a2 = alpha * alpha;
    float c2 = h.z * h.z;
    float d = c2 * (a2 - 1.0f) + 1.0f;
    return a2 / (kPi * d * d);
}

PHOTON_HD float ggx_lambda(vec3 w, float alpha) {
    float c2 = w.z * w.z;
    if (c2 >= 1.0f) return 0.0f;
    float t2 = (1.0f - c2) / fmaxf(c2, 1e-12f);
    return 0.5f * (-1.0f + sqrtf(1.0f + alpha * alpha * t2));
}

PHOTON_HD float ggx_g1(vec3 w, float alpha) { return 1.0f / (1.0f + ggx_lambda(w, alpha)); }

PHOTON_HD float ggx_g2(vec3 wo, vec3 wi, float alpha) { return 1.0f / (1.0f + ggx_lambda(wo, alpha) + ggx_lambda(wi, alpha)); }

// A visible normal for a view direction in the upper hemisphere (Heitz 2018, "Sampling the GGX
// Distribution of Visible Normals").
PHOTON_HD vec3 sample_vndf(vec3 wo, float alpha, vec2 u) {
    vec3 vh = normalize(vec3(alpha * wo.x, alpha * wo.y, wo.z));
    float lensq = vh.x * vh.x + vh.y * vh.y;
    vec3 t1 = lensq > 0 ? vec3(-vh.y, vh.x, 0) * (1.0f / sqrtf(lensq)) : vec3(1, 0, 0);
    vec3 t2 = cross(vh, t1);
    float r = sqrtf(u.x);
    float phi = kTwoPi * u.y;
    float p1 = r * cosf(phi), p2 = r * sinf(phi);
    float s = 0.5f * (1.0f + vh.z);
    p2 = (1.0f - s) * safe_sqrt(1.0f - p1 * p1) + s * p2;
    vec3 nh = t1 * p1 + t2 * p2 + vh * safe_sqrt(1.0f - p1 * p1 - p2 * p2);
    return normalize(vec3(alpha * nh.x, alpha * nh.y, fmaxf(1e-6f, nh.z)));
}

// Density of the visible normal h for view wo (both upper hemisphere).
PHOTON_HD float vndf_pdf(vec3 wo, vec3 h, float alpha) {
    if (wo.z <= 0) return 0.0f;
    return ggx_g1(wo, alpha) * fmaxf(0.0f, dot(wo, h)) * ggx_d(h, alpha) / wo.z;
}

PHOTON_HD vec3 fresnel_schlick(vec3 f0, float cos_theta) {
    float m = saturate(1.0f - cos_theta);
    float m5 = m * m * m * m * m;
    return f0 + (vec3(1.0f) - f0) * m5;
}

// Unpolarized Fresnel reflectance of a dielectric; eta = n_t / n_i, cos_i signed against the normal.
PHOTON_HD float fresnel_dielectric(float cos_i, float eta) {
    cos_i = clampf(cos_i, -1.0f, 1.0f);
    if (cos_i < 0) {
        eta = 1.0f / eta;
        cos_i = -cos_i;
    }
    float sin2_t = (1.0f - cos_i * cos_i) / (eta * eta);
    if (sin2_t >= 1.0f) return 1.0f;  // total internal reflection
    float cos_t = safe_sqrt(1.0f - sin2_t);
    float r_parl = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    float r_perp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5f * (r_parl * r_parl + r_perp * r_perp);
}

// Hemispherical average of the dielectric Fresnel reflectance, 2 * integral of F(mu) mu dmu, for eta > 1
// (the fit of d'Eon, "A Hitchhiker's Guide to Multiple Scattering", within 0.1%).
PHOTON_HD float fresnel_dielectric_average(float eta) { return (eta - 1.0f) / (4.08567f + 1.00071f * eta); }

struct Bsdf {
    vec3 base;
    float alpha;
    float ior;
    // Component weights and the lobe selection probabilities derived from them.
    float w_metal, w_opaque, w_glass;
    float f0_dielectric;
    float diffuse_norm;  // 1 / (1 - F_avg)

    PHOTON_HD explicit Bsdf(const BsdfParams& p) {
        base = p.base;
        float r = clampf(p.roughness, 0.0f, 1.0f);
        alpha = fmaxf(r * r, 0.002f);
        ior = fmaxf(p.ior, 1.0001f);
        float m = saturate(p.metallic), t = saturate(p.transmission);
        w_metal = m;
        w_glass = (1 - m) * t;
        w_opaque = (1 - m) * (1 - t);
        f0_dielectric = sqr((ior - 1) / (ior + 1));
        diffuse_norm = 1.0f / (1.0f - fresnel_dielectric_average(ior));
    }

    // Probability of sampling the specular (rather than diffuse) lobe of the opaque component.
    PHOTON_HD float opaque_spec_prob(float cos_o) const {
        if (f0_dielectric < 1e-6f) return 0.0f;  // index matched (ior 1): no specular layer to sample
        float f = fresnel_schlick(vec3(f0_dielectric), cos_o).x;
        float d = (1 - f) * fmaxf(luminance(base), 0.01f);
        return clampf(f / (f + d), 0.1f, 0.9f);
    }

    PHOTON_HD bool is_smooth() const { return alpha <= 0.002f; }
    // Whether light sampling can find directions this BSDF scatters into: not for a mirror-like metal
    // or a smooth glass, whose lobes are too narrow (a smooth plastic still has its diffuse part).
    PHOTON_HD bool samples_lights() const { return !is_smooth() || w_opaque > 0; }

    // Albedo-like color for denoiser demodulation.
    PHOTON_HD vec3 albedo() const { return base * (w_metal + w_opaque) + vec3(w_glass); }

    // --- the glass component --------------------------------------------------------------------

    PHOTON_HD vec3 glass_eval(vec3 wo, vec3 wi, float* pdf_out) const {
        bool reflect_ = wo.z * wi.z > 0;
        float etap = wo.z > 0 ? ior : 1.0f / ior;  // relative IOR seen from the side of wo
        vec3 wm = reflect_ ? wo + wi : wi * etap + wo;
        if (length_squared(wm) < 1e-20f || wo.z == 0 || wi.z == 0) {
            *pdf_out = 0;
            return vec3(0);
        }
        wm = normalize(wm);
        if (wm.z < 0) wm = -wm;
        // Discard back-facing microfacets.
        if (dot(wm, wi) * wi.z < 0 || dot(wm, wo) * wo.z < 0) {
            *pdf_out = 0;
            return vec3(0);
        }
        float F = fresnel_dielectric(dot(wo, wm), ior);
        float R = F, T = 1 - F;
        // Visible normals are sampled from the side of wo.
        vec3 wo_up = wo.z > 0 ? wo : -wo;
        vec3 wm_up = wo.z > 0 ? wm : wm;
        float pdf_m = vndf_pdf(wo_up, wm_up, alpha);
        float D = ggx_d(wm, alpha), G = ggx_g2(wo, wi, alpha);
        if (reflect_) {
            *pdf_out = pdf_m / (4.0f * fabsf(dot(wo, wm))) * R;
            return vec3(D * G * F / (4.0f * fabsf(wi.z * wo.z)));
        }
        float denom = sqr(dot(wi, wm) + dot(wo, wm) / etap);
        float dwm_dwi = fabsf(dot(wi, wm)) / denom;
        *pdf_out = pdf_m * dwm_dwi * T;
        // Radiance is compressed by 1/etap^2 when entering a denser medium.
        float ft = T * D * G * fabsf(dot(wi, wm) * dot(wo, wm) / (wi.z * wo.z * denom)) / (etap * etap);
        return base * ft;
    }

    PHOTON_HD bool glass_sample(vec3 wo, vec2 u, float u_lobe, vec3& wi) const {
        vec3 wo_up = wo.z > 0 ? wo : -wo;
        vec3 wm = sample_vndf(wo_up, alpha, u);
        float F = fresnel_dielectric(dot(wo, wm), ior);
        if (u_lobe < F) {
            wi = reflect(wo, wm);
            return wi.z * wo.z > 0;
        }
        // Refract wo through the microfacet wm.
        float etap = wo.z > 0 ? ior : 1.0f / ior;
        float cos_i = dot(wo, wm);
        vec3 n = wm;
        if (cos_i < 0) {
            n = -wm;
            cos_i = -cos_i;
        }
        float sin2_t = (1 - cos_i * cos_i) / (etap * etap);
        if (sin2_t >= 1) return false;
        float cos_t = safe_sqrt(1 - sin2_t);
        wi = -wo / etap + n * (cos_i / etap - cos_t);
        return wi.z * wo.z < 0;
    }

    // --- the whole mixture --------------------------------------------------------------------

    PHOTON_HD vec3 eval(vec3 wo, vec3 wi) const {
        float pdf;
        return eval_pdf(wo, wi, &pdf);
    }

    PHOTON_HD float pdf(vec3 wo, vec3 wi) const {
        float pdf;
        eval_pdf(wo, wi, &pdf);
        return pdf;
    }

    PHOTON_HD vec3 eval_pdf(vec3 wo, vec3 wi, float* pdf_out) const {
        vec3 f(0);
        float pdf = 0;
        bool same = wo.z * wi.z > 0;
        if (same && wo.z > 0 && (w_metal > 0 || w_opaque > 0)) {
            vec3 h = normalize(wo + wi);
            float D = ggx_d(h, alpha), G = ggx_g2(wo, wi, alpha);
            float spec = D * G / (4.0f * wo.z * wi.z);
            float pdf_spec = vndf_pdf(wo, h, alpha) / (4.0f * dot(wo, h));
            float voh = dot(wo, h);
            if (w_metal > 0) {
                f += fresnel_schlick(base, voh) * (spec * w_metal);
                pdf += w_metal * pdf_spec;
            }
            if (w_opaque > 0) {
                // (An index-matched layer, ior 1, has no interface at all; its Fresnel would still go to 1
                // at exactly grazing angles.)
                bool matched = f0_dielectric < 1e-6f;
                float F = matched ? 0.0f : fresnel_dielectric(voh, ior);
                float base_layer = matched ? kInvPi : (1 - fresnel_dielectric(wo.z, ior)) * (1 - fresnel_dielectric(wi.z, ior)) * diffuse_norm * kInvPi;
                f += (vec3(F * spec) + base * base_layer) * w_opaque;
                float ps = opaque_spec_prob(wo.z);
                pdf += w_opaque * (ps * pdf_spec + (1 - ps) * cosine_hemisphere_pdf(wi.z));
            }
        }
        if (w_glass > 0) {
            float pg;
            f += glass_eval(wo, wi, &pg) * w_glass;
            pdf += w_glass * pg;
        }
        *pdf_out = pdf;
        return f;
    }

    // u = (lobe choice, 2D sample, sub-lobe choice).
    PHOTON_HD BsdfSample sample(vec3 wo, float u_lobe, vec2 u, float u_sub) const {
        BsdfSample s;
        vec3 wi;
        if (u_lobe < w_glass) {
            if (!glass_sample(wo, u, u_sub, wi)) return s;
        } else {
            if (wo.z <= 0) return s;
            float ul = (u_lobe - w_glass) / fmaxf(1 - w_glass, 1e-6f);
            bool spec = true;
            if (ul >= w_metal / fmaxf(w_metal + w_opaque, 1e-6f)) {
                // Opaque component: specular or diffuse.
                spec = u_sub < opaque_spec_prob(wo.z);
            }
            if (spec) {
                vec3 h = sample_vndf(wo, alpha, u);
                wi = reflect(wo, h);
            } else {
                wi = cosine_hemisphere(u);
            }
            if (wi.z <= 0) return s;
        }
        s.wi = wi;
        s.f = eval_pdf(wo, wi, &s.pdf);
        s.transmitted = wi.z * wo.z < 0;
        s.valid = s.pdf > 0 && is_finite(s.f);
        return s;
    }
};

}  // namespace photon
