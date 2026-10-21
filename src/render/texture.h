// Texture sampling, identical on the CPU and the GPU (bilinear filtering of RGBA8 texels).
#pragma once

#include "core/color.h"
#include "render/scene_view.h"

namespace photon {

PHOTON_HD vec4 unpack_texel(uint32_t p, bool srgb, const float* srgb_lut) {
    float r = float(p & 0xFF), g = float((p >> 8) & 0xFF), b = float((p >> 16) & 0xFF), a = float(p >> 24);
    if (srgb) return {srgb_lut[p & 0xFF], srgb_lut[(p >> 8) & 0xFF], srgb_lut[(p >> 16) & 0xFF], a * (1.0f / 255.0f)};
    return vec4(r, g, b, a) * (1.0f / 255.0f);
}

PHOTON_HD int wrap_coord(int i, int n, bool repeat) {
    if (repeat) {
        i %= n;
        return i < 0 ? i + n : i;
    }
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

PHOTON_HD vec4 sample_texture(const SceneView& s, int tex, vec2 uv) {
    const float* srgb_lut = s.srgb_lut;
    const TextureInfo& t = s.textures[tex];
    float x = uv.x * float(t.width) - 0.5f, y = uv.y * float(t.height) - 0.5f;
    float fx = floorf(x), fy = floorf(y);
    // Keep the coordinates in a sane range before converting to int (uv can be huge or NaN).
    if (!(fabsf(fx) < 1e8f) || !(fabsf(fy) < 1e8f)) return vec4(0, 0, 0, 1);
    int x0 = int(fx), y0 = int(fy);
    float tx = x - fx, ty = y - fy;
    bool rep = t.wrap_repeat != 0;
    int xa = wrap_coord(x0, t.width, rep), xb = wrap_coord(x0 + 1, t.width, rep);
    int ya = wrap_coord(y0, t.height, rep), yb = wrap_coord(y0 + 1, t.height, rep);
    const uint32_t* base = s.texels + t.offset;
    bool srgb = t.srgb != 0;
    vec4 a = unpack_texel(base[ya * t.width + xa], srgb, srgb_lut);
    vec4 b = unpack_texel(base[ya * t.width + xb], srgb, srgb_lut);
    vec4 c = unpack_texel(base[yb * t.width + xa], srgb, srgb_lut);
    vec4 d = unpack_texel(base[yb * t.width + xb], srgb, srgb_lut);
    vec4 top = a * (1 - tx) + b * tx;
    vec4 bottom = c * (1 - tx) + d * tx;
    return top * (1 - ty) + bottom * ty;
}

}  // namespace photon
