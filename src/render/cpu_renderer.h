// The reference renderer: the shared integrator on all CPU cores.
#pragma once

#include <vector>

#include "render/pixel.h"
#include "scene/scene.h"

namespace photon {

class CpuRenderer {
public:
    explicit CpuRenderer(const Scene& scene) : scene_(scene), view_(scene.view()) {}

    void resize(int width, int height);
    void reset();
    // Adds `spp` samples per pixel to the accumulation.
    void render(const CameraView& camera, const RenderSettings& settings, int spp, int threads = 0);

    int width() const { return width_; }
    int height() const { return height_; }
    int samples() const { return samples_; }
    // The average radiance per pixel.
    std::vector<vec3> image() const;
    // Tone mapped RGBA8.
    std::vector<uint32_t> display(const RenderSettings& settings) const;

private:
    const Scene& scene_;
    SceneView view_;
    int width_ = 0, height_ = 0;
    int samples_ = 0;
    std::vector<vec3> sum_;
};

}  // namespace photon
