// Spatiotemporal variance-guided filtering (Schied et al., "Spatiotemporal Variance-Guided Filtering:
// Real-Time Reconstruction for Path-Traced Global Illumination", HPG 2017).
//
// Input per frame: one noisy path-traced sample per pixel and the G-buffer of the first hit.
//  1. Demodulate: divide the illumination by the surface albedo, so the filter does not blur textures.
//  2. Temporal accumulation: reproject each pixel into the previous frame (with the previous camera)
//     and blend with the history where depth, normal and material agree; track the luminance moments
//     and how long the history is.
//  3. Variance: from the moments, or estimated spatially while the history is short.
//  4. A-trous wavelet filter: five passes of a 5x5 B-spline kernel with growing gaps, with edge-stopping
//     weights on luminance (scaled by the variance), normals and depth. The first pass's output is the
//     next frame's history.
//  5. Remodulate by the albedo and add back the emission seen directly.
#pragma once

#include "core/math.h"
#include "render/scene_view.h"

namespace photon {

struct SvgfSettings;

// The G-buffer and noisy color of one frame, in device memory (width * height entries each).
struct SvgfInput {
    const vec4* color;         // radiance of the sample
    const vec4* albedo;        // first-hit albedo
    const vec4* normal_depth;  // first-hit normal and distance (0 for misses)
    const vec4* position;      // first-hit world position, w = material id (-1 for misses)
    const vec4* emission;      // first-hit emission
};

class Svgf {
public:
    Svgf() = default;
    ~Svgf();
    Svgf(const Svgf&) = delete;
    Svgf& operator=(const Svgf&) = delete;

    void resize(int width, int height);
    void reset();
    // Writes the denoised radiance to `output` (device memory).
    void run(const SvgfInput& in, const CameraView& previous_camera, const SvgfSettings& settings, vec4* output);
    // The variance estimate of the last frame (for the debug view).
    const vec4* variance_buffer() const { return filter_[0]; }

private:
    void free_buffers();
    int width_ = 0, height_ = 0;
    bool has_history_ = false;
    // History (previous frame).
    vec4* prev_illum_ = nullptr;      // rgb + history length
    vec4* prev_moments_ = nullptr;    // first and second moment of luminance
    vec4* prev_nd_ = nullptr;
    vec4* prev_pos_ = nullptr;
    // Current frame.
    vec4* illum_ = nullptr;           // temporally accumulated: rgb + variance
    vec4* moments_ = nullptr;
    vec4* filter_[2] = {nullptr, nullptr};  // a-trous ping-pong: rgb + variance
};

}  // namespace photon
