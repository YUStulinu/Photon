// Whole-image tests of the integrator on the CPU.
#include <algorithm>

#include "render/cpu_renderer.h"
#include "scene/builtin.h"
#include "test.h"

using namespace photon;

namespace {

double mean_luminance(const std::vector<vec3>& img) {
    double s = 0;
    for (const vec3& c : img) s += luminance(c);
    return s / double(img.size());
}

std::vector<vec3> render(const Scene& scene, int w, int h, int spp, const PathOptions& opt, uint32_t seed = 0) {
    CpuRenderer r(scene);
    r.resize(w, h);
    RenderSettings rs;
    rs.path = opt;
    rs.seed = seed;
    r.render(scene.camera.view(float(w) / float(h)), rs, spp);
    return r.image();
}

}  // namespace

TEST(white_furnace_converges_to_one) {
    // A white Lambertian sphere in a uniform white environment: the radiance leaving every point is
    // exactly 1, so every pixel must be 1 (no energy gained or lost by the path tracer).
    Scene scene;
    make_builtin_scene("furnace", scene);
    scene.normals.clear();  // flat facets: shading normals equal to the geometry, so no energy is lost
    scene.build();
    // BSDF sampling only and no Russian roulette: every path carries weight exactly 1 (a cosine-sampled
    // Lambertian bounce has f cos / pdf = albedo = 1), so every pixel must be 1 up to rounding.
    PathOptions opt;
    opt.max_depth = 64;
    opt.rr_depth = 64;
    opt.nee = 0;
    std::vector<vec3> img = render(scene, 48, 48, 16, opt);
    double worst = 0, mean = 0;
    for (const vec3& c : img) {
        worst = std::max(worst, std::fabs(double(luminance(c)) - 1.0));
        mean += luminance(c);
    }
    mean /= double(img.size());
    CHECK_NEAR(mean, 1.0, 0.002);
    CHECK_MSG(worst < 1e-3, "worst pixel deviates by " << worst);
    // With light sampling (MIS) and Russian roulette the estimate stays unbiased, but noisy.
    std::vector<vec3> rr = render(scene, 48, 48, 64, PathOptions{});
    CHECK_NEAR(mean_luminance(rr), 1.0, 0.005);
    // Smooth shading normals on the tessellated sphere lose a little energy at the silhouette, where
    // sampled directions cross the facets (those paths are dropped rather than leaking).
    Scene smooth;
    make_builtin_scene("furnace", smooth);
    smooth.build();
    double m = mean_luminance(render(smooth, 48, 48, 16, opt));
    if (test::verbose()) std::printf("    smooth-shaded furnace mean %.5f\n", m);
    CHECK_NEAR(m, 1.0, 0.01);
}

TEST(white_furnace_with_glossy_metal) {
    // A white smooth metal sphere reflects all energy: the image is again uniformly 1.
    Scene scene;
    make_builtin_scene("furnace", scene);
    scene.materials[0].metallic = 1;
    scene.materials[0].roughness = 0;
    scene.build();
    PathOptions opt;
    opt.max_depth = 64;
    opt.rr_depth = 64;
    CHECK_NEAR(mean_luminance(render(scene, 32, 32, 16, opt)), 1.0, 0.01);
}

TEST(light_sampling_does_not_change_the_expectation) {
    // MIS with next event estimation, and pure BSDF sampling, converge to the same image.
    for (const char* name : {"cornell", "materials"}) {
        Scene scene;
        make_builtin_scene(name, scene);
        scene.build();
        PathOptions with, without;
        without.nee = 0;
        with.max_depth = without.max_depth = 6;
        double a = mean_luminance(render(scene, 64, 36, 256, with, 1));
        double b = mean_luminance(render(scene, 64, 36, 1024, without, 2));
        CHECK_MSG(std::fabs(a - b) < 0.02 * a, name << ": with NEE " << a << ", without " << b);
    }
}

TEST(next_event_estimation_reduces_noise) {
    Scene scene;
    make_builtin_scene("cornell", scene);
    scene.build();
    PathOptions with, without;
    without.nee = 0;
    std::vector<vec3> ref = render(scene, 32, 32, 512, with, 9);
    auto error = [&](const std::vector<vec3>& img) {
        double e = 0;
        for (size_t i = 0; i < img.size(); i++) e += sqr(luminance(img[i]) - luminance(ref[i]));
        return e / double(img.size());
    };
    double e_with = error(render(scene, 32, 32, 16, with, 3));
    double e_without = error(render(scene, 32, 32, 16, without, 3));
    if (test::verbose()) std::printf("    MSE at 16 spp: %.5f with NEE, %.5f without\n", e_with, e_without);
    CHECK(e_with * 3 < e_without);
}

TEST(rendering_is_deterministic) {
    Scene scene;
    make_builtin_scene("glass", scene);
    scene.build();
    std::vector<vec3> a = render(scene, 40, 24, 4, PathOptions{}, 5);
    std::vector<vec3> b = render(scene, 40, 24, 4, PathOptions{}, 5);
    std::vector<vec3> c = render(scene, 40, 24, 4, PathOptions{}, 6);
    bool same = true, differs = false;
    for (size_t i = 0; i < a.size(); i++) {
        same = same && a[i].x == b[i].x && a[i].y == b[i].y && a[i].z == b[i].z;
        differs = differs || a[i].x != c[i].x;
    }
    CHECK(same);
    CHECK(differs);
}

TEST(cornell_box_brightness) {
    // A regression check on the absolute level of the classic scene: the wall behind the boxes (white,
    // lit directly from above and by interreflection).
    Scene scene;
    make_builtin_scene("cornell", scene);
    scene.build();
    std::vector<vec3> img = render(scene, 64, 64, 256, PathOptions{}, 4);
    // The back wall above the tall box, near the center column: rows 18..24.
    double s = 0;
    int n = 0;
    for (int y = 18; y < 24; y++)
        for (int x = 28; x < 36; x++) {
            s += luminance(img[size_t(y * 64 + x)]);
            n++;
        }
    s /= n;
    if (test::verbose()) std::printf("    back wall luminance %.4f\n", s);
    CHECK(s > 0.05 && s < 0.5);
    // Red on the left, green on the right (the camera looks along +z).
    vec3 left = img[size_t(32 * 64 + 4)], right = img[size_t(32 * 64 + 59)];
    CHECK(left.x > 2 * left.y);
    CHECK(right.y > 1.5f * right.x);
}
