// One sample of one pixel: the unit of work shared by the CPU and GPU renderers. Both seed the random
// generator the same way (from the pixel index and the frame number), so a CPU and a GPU render of a
// scene are statistically identical.
#pragma once

#include "render/integrator.h"

namespace photon {

struct RenderSettings {
    int width = 960, height = 540;
    PathOptions path;
    float exposure = 1.0f;
    ToneMap tonemap = ToneMap::Aces;
    uint32_t seed = 0;
};

PHOTON_HD vec3 render_sample(const SceneView& s, const CameraView& cam, int x, int y, int width, int height, uint32_t frame, uint32_t seed,
                             const PathOptions& opt, PrimaryInfo* primary) {
    uint32_t pixel = uint32_t(y) * uint32_t(width) + uint32_t(x);
    Rng rng(hash_u32(pixel ^ hash_u32(seed)) | (uint64_t(hash_u32(frame + 0x9e3779b9u)) << 32), pixel);
    vec2 jitter = rng.next2();
    vec2 lens = rng.next2();
    Ray ray = camera_ray(cam, float(x) + jitter.x, float(y) + jitter.y, width, height, lens);
    vec3 L = trace_path(s, ray, rng, opt, primary);
    // A rare NaN or infinity (degenerate geometry) must not poison the accumulated image.
    if (!is_finite(L) || L.x < 0 || L.y < 0 || L.z < 0) L = vec3(0);
    return L;
}

}  // namespace photon
