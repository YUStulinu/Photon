// Light sampling: the densities returned by the samplers must equal the densities the integrator
// evaluates for BSDF-sampled directions (which multiple importance sampling depends on), and light
// sampling must estimate the same irradiance as plain hemisphere sampling.
#include "render/integrator.h"
#include "scene/builtin.h"
#include "test.h"

using namespace photon;

namespace {

void env_scene(Scene& scene, bool image) {
    Material m;
    m.metallic = 0;
    add_quad(scene, vec3(-1, 0, -1), vec3(0, 0, 2), vec3(2, 0, 0), scene.add_material(m));
    if (image) {
        make_sky(scene.env, normalize(vec3(0.5f, 0.6f, -0.3f)), 3000.0f, 256);
        scene.env.rotation = 0.7f;
    } else {
        scene.env.color = vec3(0.5f, 0.7f, 1.0f);
    }
    scene.build();
}

}  // namespace

TEST(environment_sampling_density_is_consistent) {
    for (bool image : {false, true}) {
        Scene scene;
        env_scene(scene, image);
        SceneView s = scene.view();
        Rng rng(3, 1);
        const int n = 400000;
        double inv_pdf = 0, radiance = 0;
        int mismatches = 0;
        for (int i = 0; i < n; i++) {
            vec3 dir;
            float pdf;
            if (!sample_env(s.env, rng.next2(), dir, pdf)) continue;
            float eval = env_pdf_dir(s.env, dir);
            if (std::fabs(eval - pdf) > 2e-3f * pdf + 1e-6f) mismatches++;
            inv_pdf += 1.0 / pdf;
            radiance += luminance(env_radiance(s.env, dir)) / pdf;
        }
        // Allow a few samples on texel boundaries, where the two lookups may pick neighbors.
        CHECK_MSG(mismatches < n / 1000, mismatches << " density mismatches");
        // E[1/pdf] over the support is the solid angle of the support (all of it, for the sky).
        CHECK_NEAR(inv_pdf / n, 4 * kPi, 4 * kPi * 0.01);
        // E[L/pdf] is the integral of L over the sphere: compare with a sum over a fine direction grid.
        double reference = 0;
        const int gt = 512, gp = 1024;
        for (int y = 0; y < gt; y++)
            for (int x = 0; x < gp; x++) {
                vec3 d = equirect_to_direction(vec2((x + 0.5f) / gp, (y + 0.5f) / gt));
                float sin_t = std::sin((y + 0.5f) / gt * kPi);
                reference += luminance(env_radiance(s.env, d)) * sin_t * (kPi / gt) * (kTwoPi / gp);
            }
        CHECK_MSG(std::fabs(radiance / n - reference) < 0.02 * reference, "E[L/pdf] " << radiance / n << " vs " << reference);
    }
}

TEST(emissive_sampling_density_matches_hit_density) {
    Scene scene;
    make_builtin_scene("cornell", scene);
    scene.build();
    SceneView s = scene.view();
    Rng rng(4, 4);
    int checked = 0;
    for (int i = 0; i < 20000; i++) {
        vec3 p(0.5f + 4.5f * rng.next(), 0.01f + 3 * rng.next(), 0.2f + 5 * rng.next());
        LightSample ls = sample_light(s, p, rng.next(), rng.next2(), rng.next());
        if (ls.pdf <= 0 || ls.delta) continue;
        Ray r;
        r.o = p;
        r.d = ls.wi;
        r.tmax = kInf;
        Hit h;
        if (!intersect(s, r, h) || h.tri != ls.tri) continue;  // occluded from p
        CHECK_NEAR(h.t, ls.dist, 1e-3f * ls.dist);
        float pdf = emissive_pdf(s, h.tri, ls.wi, h.t);
        CHECK_NEAR(pdf, ls.pdf, 1e-3f * ls.pdf);
        checked++;
    }
    CHECK(checked > 5000);
}

TEST(light_sampling_and_hemisphere_sampling_agree) {
    // Irradiance at points of the Cornell floor, estimated by sampling the light and by sampling the
    // cosine-weighted hemisphere: both estimators are unbiased, so they must agree.
    Scene scene;
    make_builtin_scene("cornell", scene);
    scene.build();
    SceneView s = scene.view();
    vec3 points[3] = {vec3(2.7f, 0, 2.8f), vec3(0.6f, 0, 4.9f), vec3(4.5f, 0, 0.8f)};
    for (vec3 p : points) {
        Rng rng(7, 7);
        Frame f(vec3(0, 1, 0));
        vec3 o = offset_ray(p, vec3(0, 1, 0));
        double nee = 0, hemi = 0;
        const int n = 400000;
        for (int i = 0; i < n; i++) {
            LightSample ls = sample_light(s, p, rng.next(), rng.next2(), rng.next());
            if (ls.pdf > 0 && ls.wi.y > 0) {
                Ray sh;
                sh.o = o;
                sh.d = ls.wi;
                sh.tmax = ls.dist * (1 - 1e-4f);
                if (!occluded(s, sh)) nee += luminance(ls.radiance) * ls.wi.y / ls.pdf;
            }
            vec3 d = f.to_world(cosine_hemisphere(rng.next2()));
            Ray r;
            r.o = o;
            r.d = d;
            r.tmax = kInf;
            Hit h;
            if (intersect(s, r, h)) {
                Interaction it = surface_at(s, r, h);
                hemi += luminance(it.emission) * kPi;  // L cos / (cos / pi)
            }
        }
        nee /= n;
        hemi /= n;
        CHECK_MSG(std::fabs(nee - hemi) < 0.03 * hemi + 1e-3, "at (" << p.x << ", " << p.z << "): light sampling " << nee << ", hemisphere " << hemi);
    }
}

TEST(punctual_lights_follow_the_inverse_square_law) {
    Scene scene;
    Material m;
    m.metallic = 0;
    add_quad(scene, vec3(-10, 0, -10), vec3(0, 0, 20), vec3(20, 0, 0), scene.add_material(m));
    PunctualLight l{};
    l.type = LightType::Point;
    l.position = vec3(0, 2, 0);
    l.intensity = vec3(8);
    scene.punctual.push_back(l);
    scene.build();
    SceneView s = scene.view();
    Rng rng(1, 1);
    LightSample ls = sample_light(s, vec3(0, 0, 0), 0.5f, rng.next2(), 0.1f);
    CHECK(ls.delta);
    CHECK_NEAR(ls.dist, 2, 1e-5);
    // Radiance-like contribution intensity / d^2, divided by the selection probability in pdf.
    CHECK_NEAR(luminance(ls.radiance), 8.0 / 4.0, 1e-4);
    CHECK_NEAR(ls.pdf, 1.0, 1e-6);
    LightSample far = sample_light(s, vec3(1.5f, 0, 0), 0.5f, rng.next2(), 0.1f);
    CHECK_NEAR(luminance(far.radiance), 8.0 / 6.25, 1e-4);
}
