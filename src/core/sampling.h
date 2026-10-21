// Random numbers and the warps that turn them into directions and points.
#pragma once

#include "core/math.h"

namespace photon {

// PCG32 (O'Neill 2014): a small, fast generator with good statistical quality.
struct Rng {
    uint64_t state = 0x853c49e6748fea9bULL;
    uint64_t inc = 0xda3e39cb94b95bdbULL;

    Rng() = default;
    PHOTON_HD Rng(uint64_t seed, uint64_t stream) {
        state = 0;
        inc = (stream << 1u) | 1u;
        next_u32();
        state += seed;
        next_u32();
    }
    PHOTON_HD uint32_t next_u32() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xorshifted = uint32_t(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = uint32_t(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
    }
    // Uniform in [0, 1).
    PHOTON_HD float next() { return fminf(float(next_u32() >> 8) * (1.0f / 16777216.0f), 0.99999994f); }
    PHOTON_HD vec2 next2() {
        float a = next();
        return {a, next()};
    }
};

// A 32-bit integer hash (for seeding per pixel and frame).
PHOTON_HD uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

PHOTON_HD vec2 concentric_disk(vec2 u) {
    float a = 2 * u.x - 1, b = 2 * u.y - 1;
    if (a == 0 && b == 0) return {0, 0};
    float r, phi;
    if (fabsf(a) > fabsf(b)) {
        r = a;
        phi = (kPi / 4) * (b / a);
    } else {
        r = b;
        phi = (kPi / 2) - (kPi / 4) * (a / b);
    }
    return {r * cosf(phi), r * sinf(phi)};
}

PHOTON_HD vec3 cosine_hemisphere(vec2 u) {
    vec2 d = concentric_disk(u);
    float z = safe_sqrt(1 - d.x * d.x - d.y * d.y);
    return {d.x, d.y, z};
}

PHOTON_HD float cosine_hemisphere_pdf(float cos_theta) { return fmaxf(cos_theta, 0.0f) * kInvPi; }

PHOTON_HD vec3 uniform_sphere(vec2 u) {
    float z = 1 - 2 * u.x;
    float r = safe_sqrt(1 - z * z);
    float phi = kTwoPi * u.y;
    return {r * cosf(phi), r * sinf(phi), z};
}

// Uniform point on a triangle (barycentrics b1, b2; Heitz's low-distortion map would also do).
PHOTON_HD vec2 uniform_triangle(vec2 u) {
    float su = sqrtf(u.x);
    return {1 - su, u.y * su};
}

// The power heuristic (beta = 2) for multiple importance sampling (Veach 1997).
PHOTON_HD float power_heuristic(float pdf_a, float pdf_b) {
    float a = pdf_a * pdf_a, b = pdf_b * pdf_b;
    return a + b > 0 ? a / (a + b) : 0.0f;
}

// Directions on the sphere <-> equirectangular (latitude-longitude) coordinates in [0,1)^2, with +y up.
PHOTON_HD vec2 direction_to_equirect(vec3 d) {
    float u = atan2f(d.x, -d.z) * kInvTwoPi + 0.5f;
    float v = acosf(clampf(d.y, -1.0f, 1.0f)) * kInvPi;
    return {u - floorf(u), v};
}

PHOTON_HD vec3 equirect_to_direction(vec2 uv) {
    float phi = (uv.x - 0.5f) * kTwoPi;
    float theta = uv.y * kPi;
    float s = sinf(theta);
    return {s * sinf(phi), cosf(theta), -s * cosf(phi)};
}

}  // namespace photon
