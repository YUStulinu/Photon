// The GPU renderer: the shared integrator as a CUDA kernel, one thread per pixel, with progressive
// accumulation, a real-time mode denoised by SVGF, and tone mapping into an RGBA8 buffer.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "render/pixel.h"
#include "scene/scene.h"

namespace photon {

// What the viewer shows.
enum class DisplayMode : int { Color = 0, Albedo = 1, Normal = 2, Depth = 3, Variance = 4, Heatmap = 5 };

struct SvgfSettings {
    int atrous_iterations = 5;
    float alpha_color = 0.2f;    // temporal blend factor for color
    float alpha_moments = 0.2f;  // and for the luminance moments
    float phi_color = 4.0f;      // edge-stopping on luminance (in units of the standard deviation)
    float phi_normal = 128.0f;   // exponent on the normal similarity
    float phi_depth = 1.0f;      // edge-stopping on depth (relative to the local depth gradient)
};

struct GpuInfo {
    std::string name;
    int compute_major = 0, compute_minor = 0;
    size_t memory = 0;
    int multiprocessors = 0;
};

bool cuda_available(GpuInfo* info = nullptr);

class GpuRenderer {
public:
    GpuRenderer();
    ~GpuRenderer();
    GpuRenderer(const GpuRenderer&) = delete;
    GpuRenderer& operator=(const GpuRenderer&) = delete;

    // Copies the built scene to the device.
    bool upload(const Scene& scene, std::string& error);
    // Re-uploads only what Scene::rebuild_lights changes: materials, light tables and the environment.
    bool update_lights(const Scene& scene, std::string& error);
    void resize(int width, int height);
    void reset();

    // Progressive mode: adds `spp` samples per pixel to the accumulation.
    void render(const CameraView& camera, const RenderSettings& settings, int spp);
    // Real-time mode: one sample per pixel, denoised with SVGF using the previous frame's camera.
    void render_denoised(const CameraView& camera, const CameraView& previous, const RenderSettings& settings, const SvgfSettings& svgf);
    // Tone maps the current image into the RGBA8 display buffer (device memory), or into `target` if
    // given (e.g. a buffer shared with OpenGL).
    void display(const RenderSettings& settings, DisplayMode mode, uint32_t* target = nullptr);

    std::vector<vec3> download_image();      // average radiance (progressive mode)
    std::vector<uint32_t> download_display();  // the last display() output

    int width() const { return width_; }
    int height() const { return height_; }
    int samples() const { return samples_; }
    float last_frame_ms() const { return last_ms_; }
    uint32_t* display_buffer() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int width_ = 0, height_ = 0;
    int samples_ = 0;
    uint32_t frame_ = 0;
    float last_ms_ = 0;
    bool denoised_ = false;  // the current image comes from render_denoised
};

}  // namespace photon
