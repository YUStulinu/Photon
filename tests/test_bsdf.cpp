// BSDF tests: the sampled distribution must match `pdf` (a chi-square test over the sphere), `pdf` must
// integrate to the fraction of valid samples, the estimator f cos / pdf must conserve energy (a white
// furnace per BSDF), and reflection must be reciprocal.
#include <algorithm>
#include <vector>

#include "render/bsdf.h"
#include "test.h"

using namespace photon;

namespace {

struct Case {
    const char* name;
    BsdfParams p;
};

std::vector<Case> cases() {
    auto mk = [](vec3 base, float metallic, float roughness, float transmission) {
        BsdfParams p;
        p.base = base;
        p.metallic = metallic;
        p.roughness = roughness;
        p.transmission = transmission;
        p.ior = 1.5f;
        return p;
    };
    return {
        {"diffuse", mk(vec3(0.8f), 0, 1, 0)},
        {"plastic rough", mk(vec3(0.5f, 0.2f, 0.1f), 0, 0.5f, 0)},
        {"plastic glossy", mk(vec3(0.5f, 0.2f, 0.1f), 0, 0.25f, 0)},
        {"metal rough", mk(vec3(0.95f, 0.7f, 0.3f), 1, 0.6f, 0)},
        {"metal glossy", mk(vec3(0.95f, 0.7f, 0.3f), 1, 0.3f, 0)},
        {"half metal", mk(vec3(0.6f), 0.5f, 0.4f, 0)},
        {"rough glass", mk(vec3(1), 0, 0.5f, 1)},
        {"frosted glass", mk(vec3(1), 0, 0.3f, 1)},
        {"semi transmissive", mk(vec3(0.7f), 0, 0.6f, 0.5f)},
    };
}

vec3 direction(float cos_theta, float phi) {
    float s = safe_sqrt(1 - cos_theta * cos_theta);
    return vec3(s * cosf(phi), s * sinf(phi), cos_theta);
}

// Chi-square test of `sample` against `pdf`, over bins uniform in cos(theta) (the whole sphere) and phi.
// Returns false and reports on failure.
bool chi_square(const Bsdf& b, vec3 wo, const char* name, double* pdf_integral, double* valid_fraction) {
    const int nt = 20, np = 24, n = 400000;
    std::vector<double> observed(nt * np, 0), expected(nt * np, 0);
    Rng rng(1234, 77);
    int valid = 0;
    for (int i = 0; i < n; i++) {
        BsdfSample s = b.sample(wo, rng.next(), rng.next2(), rng.next());
        if (!s.valid) continue;
        valid++;
        vec3 w = normalize(s.wi);
        float phi = atan2f(w.y, w.x);
        if (phi < 0) phi += kTwoPi;
        int it = std::min(nt - 1, int((w.z + 1) * 0.5f * nt));
        int ip = std::min(np - 1, int(phi / kTwoPi * np));
        observed[size_t(it * np + ip)]++;
    }
    // Integrate the pdf over each bin with a stratified grid (the measure in (cos theta, phi) is d omega),
    // refined where the lobe is too peaked for the coarse grid.
    auto bin_integral = [&](int it, int ip, int sub) {
        double sum = 0;
        for (int a = 0; a < sub; a++)
            for (int c = 0; c < sub; c++) {
                float ct = -1 + 2 * (float(it) + (float(a) + 0.5f) / sub) / nt;
                float ph = kTwoPi * (float(ip) + (float(c) + 0.5f) / sub) / np;
                sum += b.pdf(wo, direction(ct, ph));
            }
        return sum / (sub * sub) * (2.0 / nt) * (kTwoPi / np);
    };
    double total = 0;
    for (int it = 0; it < nt; it++)
        for (int ip = 0; ip < np; ip++) {
            double coarse = bin_integral(it, ip, 8), fine = bin_integral(it, ip, 16);
            double integral = fine;
            if (std::fabs(coarse - fine) > 0.01 * fine + 1e-7) integral = bin_integral(it, ip, 128);
            expected[size_t(it * np + ip)] = integral * n;
            total += integral;
        }
    *pdf_integral = total;
    *valid_fraction = double(valid) / n;
    // Pool bins with small expectations (the usual chi-square requirement of >= 5 per bin).
    double chi2 = 0, pool_o = 0, pool_e = 0;
    int dof = 0;
    for (size_t k = 0; k < observed.size(); k++) {
        if (expected[k] < 5) {
            pool_o += observed[k];
            pool_e += expected[k];
            continue;
        }
        chi2 += sqr(float(observed[k] - expected[k])) / expected[k];
        dof++;
    }
    if (pool_e >= 5) {
        chi2 += (pool_o - pool_e) * (pool_o - pool_e) / pool_e;
        dof++;
    }
    // A generous bound (the test is about gross mismatches, and the bin integrals are approximations).
    double bound = dof + 6 * std::sqrt(2.0 * dof) + 20;
    if (chi2 > bound) {
        std::printf("    %s wo.z=%.2f: chi2 %.1f > %.1f (dof %d)\n", name, wo.z, chi2, bound, dof);
        return false;
    }
    if (test::verbose()) std::printf("    %-18s wo.z=%.2f chi2 %7.1f dof %d\n", name, wo.z, chi2, dof);
    return true;
}

// Refraction scales radiance by 1/eta^2 (it is compressed into a smaller solid angle): energy is
// conserved once that factor is undone.
double radiance_scale(const BsdfSample& s, vec3 wo, float ior) {
    if (!s.transmitted) return 1;
    return wo.z > 0 ? double(ior) * ior : 1.0 / (double(ior) * ior);
}

}  // namespace

TEST(bsdf_sampling_matches_pdf) {
    for (const Case& c : cases()) {
        Bsdf b(c.p);
        for (float ct : {0.9f, 0.5f, 0.15f, -0.6f}) {
            if (ct < 0 && c.p.transmission <= 0) continue;  // opaque surfaces are only seen from outside
            vec3 wo = direction(ct, 0.7f);
            double integral = 0, valid = 0;
            CHECK_MSG(chi_square(b, wo, c.name, &integral, &valid), c.name);
            // The pdf integrates to the probability that a sample is produced.
            CHECK_MSG(std::fabs(integral - valid) < 0.02, c.name << " wo.z " << ct << ": pdf integral " << integral << ", valid " << valid);
        }
    }
}

TEST(bsdf_conserves_energy) {
    // A white furnace per BSDF: with a white base color, the reflected (plus transmitted) energy
    // E[f |cos| / pdf] is at most 1, and close to 1 where the model has no energy loss.
    Rng rng(99, 3);
    for (const Case& c : cases()) {
        BsdfParams p = c.p;
        p.base = vec3(1);
        Bsdf b(p);
        for (float ct : {1.0f, 0.7f, 0.3f, 0.05f}) {
            vec3 wo = direction(ct, 1.3f);
            const int n = 200000;
            double e = 0;
            for (int i = 0; i < n; i++) {
                BsdfSample s = b.sample(wo, rng.next(), rng.next2(), rng.next());
                if (s.valid) e += luminance(s.f) * std::fabs(s.wi.z) / s.pdf * radiance_scale(s, wo, p.ior);
            }
            e /= n;
            CHECK_MSG(e <= 1.02, c.name << " wo.z " << ct << ": albedo " << e);
            // Single-scattering microfacet models lose some energy at high roughness and grazing angles.
            CHECK_MSG(e >= (ct > 0.2f ? 0.75 : 0.3), c.name << " wo.z " << ct << ": albedo " << e);
        }
    }
    // Smooth surfaces: a mirror reflects everything, a smooth glass splits it into R + T = 1.
    for (float t : {0.0f, 1.0f}) {
        BsdfParams p;
        p.base = vec3(1);
        p.metallic = 1 - t;
        p.transmission = t;
        p.roughness = 0;
        Bsdf b(p);
        for (float ct : {0.9f, 0.4f, -0.8f}) {
            if (ct < 0 && t == 0) continue;
            vec3 wo = direction(ct, 0.2f);
            double e = 0;
            const int n = 100000;
            for (int i = 0; i < n; i++) {
                BsdfSample s = b.sample(wo, rng.next(), rng.next2(), rng.next());
                if (s.valid) e += luminance(s.f) * std::fabs(s.wi.z) / s.pdf * radiance_scale(s, wo, p.ior);
            }
            e /= n;
            CHECK_MSG(std::fabs(e - 1) < 0.03, (t > 0 ? "smooth glass " : "mirror ") << ct << ": " << e);
        }
    }
}

TEST(bsdf_reflection_is_reciprocal) {
    Rng rng(5, 5);
    for (const Case& c : cases()) {
        if (c.p.transmission > 0) continue;
        Bsdf b(c.p);
        for (int i = 0; i < 2000; i++) {
            vec3 a = cosine_hemisphere(rng.next2()), d = cosine_hemisphere(rng.next2());
            if (a.z < 0.05f || d.z < 0.05f) continue;
            vec3 f1 = b.eval(a, d), f2 = b.eval(d, a);
            float m = fmaxf(max_component(f1), 1e-3f);
            CHECK_MSG(max_component(vmax(f1 - f2, f2 - f1)) <= 1e-3f * m + 1e-5f, c.name);
        }
    }
}

TEST(bsdf_values_are_finite) {
    // Grazing, normal and degenerate directions must never produce NaN or infinity.
    Rng rng(8, 8);
    for (const Case& c : cases()) {
        Bsdf b(c.p);
        for (int i = 0; i < 20000; i++) {
            vec3 wo = uniform_sphere(rng.next2());
            if (i % 100 == 0) wo = vec3(0, 0, 1);
            if (i % 100 == 1) wo = normalize(vec3(1, 0, 1e-7f));
            BsdfSample s = b.sample(wo, rng.next(), rng.next2(), rng.next());
            if (s.valid) {
                CHECK(is_finite(s.f) && std::isfinite(s.pdf) && s.pdf > 0);
                CHECK(min_component(s.f) >= 0);
            }
            float pdf;
            vec3 f = b.eval_pdf(wo, uniform_sphere(rng.next2()), &pdf);
            CHECK(is_finite(f) && std::isfinite(pdf) && pdf >= 0);
        }
    }
}
