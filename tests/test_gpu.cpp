// The GPU renderer runs the same integrator as the CPU with the same random sequences, so the two
// images must agree to floating-point accuracy (fast-math on the GPU changes the last bits, and a
// few paths then diverge, so the comparison is statistical). The SVGF denoiser must reduce the error of
// one-sample frames.
#include <algorithm>

#include "render/cpu_renderer.h"
#include "scene/builtin.h"
#include "test.h"
#if defined(PHOTON_WITH_CUDA)
#include "render/gpu_renderer.h"
#endif

using namespace photon;

#if defined(PHOTON_WITH_CUDA)

namespace {

void require_gpu() {
    if (!cuda_available()) throw test::Skip{"no CUDA device"};
}

// Mean squared error of the luminance clamped to 1 (as displayed), so that the few pixels on the edge
// of the light source, which no one-sample image can antialias, do not dominate the comparison.
double mse(const std::vector<vec3>& a, const std::vector<vec3>& b) {
    double e = 0;
    for (size_t i = 0; i < a.size(); i++) e += sqr(fminf(luminance(a[i]), 1.0f) - fminf(luminance(b[i]), 1.0f));
    return e / double(a.size());
}

double mean(const std::vector<vec3>& a) {
    double s = 0;
    for (const vec3& c : a) s += luminance(c);
    return s / double(a.size());
}

}  // namespace

TEST(gpu_matches_cpu) {
    require_gpu();
    for (const char* name : {"cornell", "materials", "glass"}) {
        Scene scene;
        make_builtin_scene(name, scene);
        scene.build();
        const int w = 96, h = 54, spp = 64;
        RenderSettings rs;
        CameraView cam = scene.camera.view(float(w) / float(h));
        CpuRenderer cpu(scene);
        cpu.resize(w, h);
        cpu.render(cam, rs, spp);
        GpuRenderer gpu;
        std::string error;
        CHECK_MSG(gpu.upload(scene, error), error);
        gpu.resize(w, h);
        for (int i = 0; i < spp; i += 16) gpu.render(cam, rs, 16);
        std::vector<vec3> a = cpu.image(), b = gpu.download_image();
        // Same sample sequences: most pixels are identical up to rounding.
        int close = 0;
        for (size_t i = 0; i < a.size(); i++) {
            float d = max_component(vmax(a[i] - b[i], b[i] - a[i]));
            if (d <= 1e-3f * fmaxf(1.0f, max_component(a[i]))) close++;
        }
        double frac = double(close) / double(a.size());
        double ma = mean(a), mb = mean(b);
        if (test::verbose()) std::printf("    %-10s %.1f%% of pixels match, means %.5f / %.5f\n", name, 100 * frac, ma, mb);
        CHECK_MSG(frac > 0.9, name << ": only " << 100 * frac << "% of pixels match");
        CHECK_MSG(std::fabs(ma - mb) < 0.01 * ma, name << ": means " << ma << " vs " << mb);
    }
}

TEST(gpu_progressive_accumulation_converges) {
    require_gpu();
    Scene scene;
    make_builtin_scene("furnace", scene);
    scene.normals.clear();
    scene.build();
    GpuRenderer gpu;
    std::string error;
    CHECK(gpu.upload(scene, error));
    gpu.resize(64, 64);
    RenderSettings rs;
    rs.path.max_depth = 64;
    rs.path.rr_depth = 64;
    rs.path.nee = 0;  // deterministic path weights (see white_furnace_converges_to_one)
    CameraView cam = scene.camera.view(1.0f);
    gpu.render(cam, rs, 32);
    std::vector<vec3> img = gpu.download_image();
    double worst = 0;
    for (const vec3& c : img) worst = std::max(worst, std::fabs(double(luminance(c)) - 1.0));
    CHECK_MSG(worst < 2e-3, "furnace worst pixel " << worst);
    CHECK(gpu.samples() == 32);
    gpu.reset();
    CHECK(gpu.samples() == 0);
    // Display buffer: a white image tone-mapped linearly is white.
    rs.tonemap = ToneMap::Linear;
    gpu.render(cam, rs, 4);
    gpu.display(rs, DisplayMode::Color);
    std::vector<uint32_t> px = gpu.download_display();
    CHECK(px[px.size() / 2] == 0xFFFFFFFFu);
    for (DisplayMode m : {DisplayMode::Albedo, DisplayMode::Normal, DisplayMode::Depth, DisplayMode::Variance, DisplayMode::Heatmap}) gpu.display(rs, m);
    CHECK(gpu.download_display().size() == 64 * 64);
}

TEST(svgf_reduces_error) {
    require_gpu();
    Scene scene;
    make_builtin_scene("cornell", scene);
    scene.build();
    const int w = 160, h = 160;
    RenderSettings rs;
    CameraView cam = scene.camera.view(1.0f);
    GpuRenderer gpu;
    std::string error;
    CHECK(gpu.upload(scene, error));
    gpu.resize(w, h);
    // Reference: many samples.
    for (int i = 0; i < 2048; i += 32) gpu.render(cam, rs, 32);
    std::vector<vec3> ref = gpu.download_image();
    // One sample per pixel, noisy.
    gpu.reset();
    gpu.render(cam, rs, 1);
    std::vector<vec3> noisy = gpu.download_image();
    // SVGF on a static camera: the first frame is filtered spatially only; later frames also
    // accumulate temporally.
    SvgfSettings sv;
    gpu.reset();
    gpu.render_denoised(cam, cam, rs, sv);
    std::vector<vec3> first = gpu.download_image();
    for (int f = 1; f < 16; f++) gpu.render_denoised(cam, cam, rs, sv);
    std::vector<vec3> settled = gpu.download_image();
    double e_noisy = mse(noisy, ref), e_first = mse(first, ref), e_settled = mse(settled, ref);
    if (test::verbose()) std::printf("    MSE: 1 spp %.5f, SVGF frame 1 %.5f, frame 16 %.5f\n", e_noisy, e_first, e_settled);
    CHECK(e_first < e_noisy * 0.5);
    CHECK(e_settled < e_first);
    for (const vec3& c : settled) CHECK(is_finite(c) && min_component(c) >= 0);
    // The denoised image keeps the overall brightness (no energy loss from the filter).
    CHECK_MSG(std::fabs(mean(settled) - mean(ref)) < 0.05 * mean(ref), mean(settled) << " vs " << mean(ref));
}

TEST(svgf_handles_camera_motion) {
    require_gpu();
    Scene scene;
    make_builtin_scene("materials", scene);
    scene.build();
    const int w = 128, h = 72;
    RenderSettings rs;
    GpuRenderer gpu;
    std::string error;
    CHECK(gpu.upload(scene, error));
    gpu.resize(w, h);
    SvgfSettings sv;
    Camera c = scene.camera;
    CameraView prev = c.view(float(w) / float(h));
    for (int f = 0; f < 20; f++) {
        c.position = c.position + vec3(0.05f, 0, 0);
        CameraView cur = c.view(float(w) / float(h));
        gpu.render_denoised(cur, prev, rs, sv);
        prev = cur;
    }
    std::vector<vec3> img = gpu.download_image();
    for (const vec3& p : img) CHECK(is_finite(p) && min_component(p) >= 0);
    CHECK(mean(img) > 0.05);
}

#else

TEST(gpu_matches_cpu) { throw test::Skip{"built without CUDA"}; }

#endif
