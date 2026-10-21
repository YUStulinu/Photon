#include "core/color.h"
#include "core/sampling.h"
#include "render/integrator.h"
#include "scene/scene.h"
#include "test.h"

using namespace photon;

TEST(frame_is_orthonormal) {
    Rng rng(1, 1);
    for (int i = 0; i < 10000; i++) {
        vec3 n = uniform_sphere(rng.next2());
        if (i == 0) n = vec3(0, 0, -1);  // the singular case of the construction
        if (i == 1) n = vec3(0, 0, 1);
        Frame f(n);
        CHECK_NEAR(length(f.t), 1, 1e-5);
        CHECK_NEAR(length(f.b), 1, 1e-5);
        CHECK_NEAR(dot(f.t, f.b), 0, 1e-5);
        CHECK_NEAR(dot(f.t, f.n), 0, 1e-5);
        CHECK_NEAR(dot(f.b, f.n), 0, 1e-5);
        vec3 v = uniform_sphere(rng.next2());
        vec3 r = f.to_world(f.to_local(v));
        CHECK_NEAR(length(r - v), 0, 1e-5);
        CHECK_NEAR(f.to_local(n).z, 1, 1e-5);
    }
}

TEST(rng_is_uniform) {
    Rng rng(42, 7);
    const int bins = 16, n = 160000;
    int hist[bins] = {};
    double sum = 0;
    for (int i = 0; i < n; i++) {
        float u = rng.next();
        CHECK(u >= 0 && u < 1);
        hist[int(u * bins)]++;
        sum += u;
    }
    CHECK_NEAR(sum / n, 0.5, 0.005);
    double chi2 = 0, expected = double(n) / bins;
    for (int b = 0; b < bins; b++) chi2 += sqr(float(hist[b] - expected)) / expected;
    CHECK_MSG(chi2 < 40, "chi2 = " << chi2);  // 15 degrees of freedom; p ~ 0.0005
    // Different streams give different sequences.
    Rng a(1, 1), b(1, 2);
    int same = 0;
    for (int i = 0; i < 100; i++) same += a.next_u32() == b.next_u32();
    CHECK(same < 3);
}

TEST(sampling_densities_integrate_to_one) {
    // E[1 / pdf] over the sampled distribution is the measure of the domain.
    Rng rng(3, 3);
    double hemi = 0;
    const int n = 200000;
    for (int i = 0; i < n; i++) {
        vec3 d = cosine_hemisphere(rng.next2());
        CHECK(d.z >= 0);
        if (d.z > 1e-4f) hemi += 1.0 / cosine_hemisphere_pdf(d.z);
    }
    CHECK_NEAR(hemi / n, 2 * kPi, 0.05);
    // Uniform triangle barycentrics stay inside and average to the centroid.
    double b1 = 0, b2 = 0;
    for (int i = 0; i < n; i++) {
        vec2 b = uniform_triangle(rng.next2());
        CHECK(b.x >= 0 && b.y >= 0 && b.x + b.y <= 1.0001f);
        b1 += b.x;
        b2 += b.y;
    }
    CHECK_NEAR(b1 / n, 1.0 / 3, 0.005);
    CHECK_NEAR(b2 / n, 1.0 / 3, 0.005);
}

TEST(equirect_round_trip) {
    Rng rng(5, 5);
    for (int i = 0; i < 10000; i++) {
        vec3 d = uniform_sphere(rng.next2());
        vec2 uv = direction_to_equirect(d);
        CHECK(uv.x >= 0 && uv.x <= 1 && uv.y >= 0 && uv.y <= 1);
        CHECK_NEAR(length(equirect_to_direction(uv) - d), 0, 1e-4);
    }
    // +y is up: the top row of the image.
    CHECK_NEAR(direction_to_equirect(vec3(0, 1, 0)).y, 0, 1e-6);
}

TEST(alias_table_matches_weights) {
    std::vector<float> w = {1, 0, 3, 0.5f, 5.5f};
    std::vector<AliasEntry> t = build_alias_table(w);
    CHECK(t.size() == w.size());
    double total = 10;
    std::vector<int> hist(w.size(), 0);
    Rng rng(9, 9);
    const int n = 200000;
    for (int i = 0; i < n; i++) {
        float pmf;
        int k = sample_alias(t.data(), int(t.size()), rng.next(), &pmf);
        CHECK_NEAR(pmf, w[size_t(k)] / total, 1e-5);
        hist[size_t(k)]++;
    }
    for (size_t k = 0; k < w.size(); k++) {
        CHECK_NEAR(t[k].pmf, w[k] / total, 1e-6);
        CHECK_NEAR(double(hist[k]) / n, w[k] / total, 0.005);
    }
}

TEST(offset_ray_leaves_the_surface) {
    Rng rng(11, 1);
    for (int i = 0; i < 10000; i++) {
        float scale = std::pow(10.0f, rng.next() * 8 - 4);  // 1e-4 .. 1e4
        vec3 p = (vec3(rng.next(), rng.next(), rng.next()) * 2.0f - vec3(1)) * scale;
        vec3 n = uniform_sphere(rng.next2());
        vec3 q = offset_ray(p, n);
        CHECK(dot(q - p, n) > 0);
        CHECK(length(q - p) < 1e-3f * fmaxf(scale, 1.0f));
    }
}

TEST(color_transfer_round_trip) {
    for (int i = 0; i <= 255; i++) {
        float c = float(i) / 255.0f;
        CHECK_NEAR(linear_to_srgb(srgb_to_linear(c)), c, 1e-5);
    }
    vec3 a = aces_fitted(vec3(0));
    CHECK_NEAR(a.x, 0, 1e-3);
    vec3 b = aces_fitted(vec3(100));
    CHECK(b.x > 0.95f && b.x <= 1.0f);
    CHECK(pack_rgba8(vec3(1)) == 0xFFFFFFFFu);
    CHECK(pack_rgba8(vec3(0)) == 0xFF000000u);
}

TEST(matrix_inverse) {
    mat4 m = translation(vec3(1, 2, 3)) * rotation_quat(vec4(0.1825742f, 0.3651484f, 0.5477226f, 0.7302967f)) * scaling(vec3(2, 3, 0.5f));
    mat4 r = m * inverse(m);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) CHECK_NEAR(r.m[i][j], i == j ? 1 : 0, 1e-5);
}
