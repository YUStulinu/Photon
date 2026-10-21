// Color encoding and tone mapping.
#pragma once

#include "core/math.h"

namespace photon {

PHOTON_HD float srgb_to_linear(float c) { return c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f); }
PHOTON_HD float linear_to_srgb(float c) {
    c = saturate(c);
    return c <= 0.0031308f ? c * 12.92f : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

// ACES filmic curve fitted by Stephen Hill (with the sRGB -> ACEScg input and output matrices).
PHOTON_HD vec3 aces_fitted(vec3 c) {
    vec3 v(0.59719f * c.x + 0.35458f * c.y + 0.04823f * c.z, 0.07600f * c.x + 0.90834f * c.y + 0.01566f * c.z,
           0.02840f * c.x + 0.13383f * c.y + 0.83777f * c.z);
    vec3 a = v * (v + vec3(0.0245786f)) - vec3(0.000090537f);
    vec3 b = v * (v * 0.983729f + vec3(0.4329510f)) + vec3(0.238081f);
    v = a / b;
    return clamp(vec3(1.60475f * v.x - 0.53108f * v.y - 0.07367f * v.z, -0.10208f * v.x + 1.10813f * v.y - 0.00605f * v.z,
                      -0.00327f * v.x - 0.07276f * v.y + 1.07602f * v.z),
                 0.0f, 1.0f);
}

enum class ToneMap : int { Linear = 0, Aces = 1, Reinhard = 2 };

PHOTON_HD vec3 tone_map(vec3 c, float exposure, ToneMap op) {
    c *= exposure;
    switch (op) {
        case ToneMap::Aces: return aces_fitted(c);
        case ToneMap::Reinhard: return c / (vec3(1.0f) + c);
        default: return clamp(c, 0.0f, 1.0f);
    }
}

// Packs a linear color, tone mapped and sRGB-encoded, into RGBA8.
PHOTON_HD uint32_t pack_rgba8(vec3 c) {
    auto q = [](float v) { return uint32_t(saturate(v) * 255.0f + 0.5f); };
    return q(linear_to_srgb(c.x)) | (q(linear_to_srgb(c.y)) << 8) | (q(linear_to_srgb(c.z)) << 16) | 0xFF000000u;
}

}  // namespace photon
