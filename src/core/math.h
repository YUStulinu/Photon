// Small vector math shared by the CPU and GPU code paths.
//
// Everything here is header-only and marked PHOTON_HD, so the same functions compile with the host
// compiler and with nvcc for device code. The integrator is written once against these types and runs
// on both: the GPU for speed, the CPU as a reference for tests.
#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define PHOTON_HD __host__ __device__ __forceinline__
#define PHOTON_DEVICE __device__
#else
#define PHOTON_HD inline
#define PHOTON_DEVICE
#endif

namespace photon {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;
constexpr float kInvPi = 0.31830988618379067154f;
constexpr float kInvTwoPi = 0.15915494309189533577f;
constexpr float kInf = 1e30f;

struct vec2 {
    float x = 0, y = 0;
    vec2() = default;
    PHOTON_HD constexpr vec2(float x_, float y_) : x(x_), y(y_) {}
    PHOTON_HD explicit constexpr vec2(float s) : x(s), y(s) {}
};

struct vec3 {
    float x = 0, y = 0, z = 0;
    vec3() = default;
    PHOTON_HD constexpr vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    PHOTON_HD explicit constexpr vec3(float s) : x(s), y(s), z(s) {}
    PHOTON_HD float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    PHOTON_HD float& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
};

struct vec4 {
    float x = 0, y = 0, z = 0, w = 0;
    vec4() = default;
    PHOTON_HD constexpr vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    PHOTON_HD constexpr vec4(vec3 v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
    PHOTON_HD vec3 xyz() const { return {x, y, z}; }
};

// vec2
PHOTON_HD vec2 operator+(vec2 a, vec2 b) { return {a.x + b.x, a.y + b.y}; }
PHOTON_HD vec2 operator-(vec2 a, vec2 b) { return {a.x - b.x, a.y - b.y}; }
PHOTON_HD vec2 operator*(vec2 a, float s) { return {a.x * s, a.y * s}; }
PHOTON_HD vec2 operator*(float s, vec2 a) { return {a.x * s, a.y * s}; }

// vec3
PHOTON_HD vec3 operator+(vec3 a, vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
PHOTON_HD vec3 operator-(vec3 a, vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
PHOTON_HD vec3 operator-(vec3 a) { return {-a.x, -a.y, -a.z}; }
PHOTON_HD vec3 operator*(vec3 a, vec3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
PHOTON_HD vec3 operator*(vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
PHOTON_HD vec3 operator*(float s, vec3 a) { return {a.x * s, a.y * s, a.z * s}; }
PHOTON_HD vec3 operator/(vec3 a, vec3 b) { return {a.x / b.x, a.y / b.y, a.z / b.z}; }
PHOTON_HD vec3 operator/(vec3 a, float s) {
    float inv = 1.0f / s;
    return {a.x * inv, a.y * inv, a.z * inv};
}
PHOTON_HD vec3& operator+=(vec3& a, vec3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
PHOTON_HD vec3& operator-=(vec3& a, vec3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}
PHOTON_HD vec3& operator*=(vec3& a, vec3 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}
PHOTON_HD vec3& operator*=(vec3& a, float s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
    return a;
}

PHOTON_HD float dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
PHOTON_HD vec3 cross(vec3 a, vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
PHOTON_HD float length(vec3 a) { return sqrtf(dot(a, a)); }
PHOTON_HD float length_squared(vec3 a) { return dot(a, a); }
PHOTON_HD vec3 normalize(vec3 a) {
    float l2 = dot(a, a);
    return l2 > 0 ? a * (1.0f / sqrtf(l2)) : vec3(0, 0, 1);
}
PHOTON_HD vec3 vmin(vec3 a, vec3 b) { return {fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)}; }
PHOTON_HD vec3 vmax(vec3 a, vec3 b) { return {fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)}; }
PHOTON_HD float max_component(vec3 a) { return fmaxf(a.x, fmaxf(a.y, a.z)); }
PHOTON_HD float min_component(vec3 a) { return fminf(a.x, fminf(a.y, a.z)); }
PHOTON_HD vec3 lerp(vec3 a, vec3 b, float t) { return a + (b - a) * t; }
PHOTON_HD float lerp(float a, float b, float t) { return a + (b - a) * t; }
PHOTON_HD float clampf(float v, float lo, float hi) { return fminf(fmaxf(v, lo), hi); }
PHOTON_HD vec3 clamp(vec3 v, float lo, float hi) { return {clampf(v.x, lo, hi), clampf(v.y, lo, hi), clampf(v.z, lo, hi)}; }
PHOTON_HD float saturate(float v) { return clampf(v, 0.0f, 1.0f); }
PHOTON_HD vec3 reflect(vec3 v, vec3 n) { return n * (2.0f * dot(v, n)) - v; }  // v points away from the surface
PHOTON_HD float luminance(vec3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }
PHOTON_HD bool is_black(vec3 c) { return c.x <= 0 && c.y <= 0 && c.z <= 0; }
#if defined(__CUDA_ARCH__)
PHOTON_HD bool is_finite(vec3 c) { return isfinite(c.x) && isfinite(c.y) && isfinite(c.z); }
#else
PHOTON_HD bool is_finite(vec3 c) { return std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z); }
#endif
PHOTON_HD float sqr(float x) { return x * x; }
PHOTON_HD float safe_sqrt(float x) { return sqrtf(fmaxf(x, 0.0f)); }

// vec4
PHOTON_HD vec4 operator+(vec4 a, vec4 b) { return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
PHOTON_HD vec4 operator*(vec4 a, float s) { return {a.x * s, a.y * s, a.z * s, a.w * s}; }

// An orthonormal basis around a normal (Duff et al., "Building an Orthonormal Basis, Revisited", 2017).
struct Frame {
    vec3 t, b, n;

    Frame() = default;
    PHOTON_HD explicit Frame(vec3 normal) : n(normal) {
        float sign = copysignf(1.0f, n.z);
        float a = -1.0f / (sign + n.z);
        float bb = n.x * n.y * a;
        t = vec3(1.0f + sign * n.x * n.x * a, sign * bb, -sign * n.x);
        b = vec3(bb, sign + n.y * n.y * a, -n.y);
    }
    // A frame with a given tangent direction (orthogonalized against the normal).
    PHOTON_HD Frame(vec3 normal, vec3 tangent) : n(normal) {
        vec3 tt = tangent - n * dot(n, tangent);
        if (length_squared(tt) < 1e-12f) {
            *this = Frame(normal);
            return;
        }
        t = normalize(tt);
        b = cross(n, t);
    }
    PHOTON_HD vec3 to_local(vec3 v) const { return {dot(v, t), dot(v, b), dot(v, n)}; }
    PHOTON_HD vec3 to_world(vec3 v) const { return t * v.x + b * v.y + n * v.z; }
};

// A 4x4 matrix, row-major, for transforming scene geometry (host side).
struct mat4 {
    float m[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};

    PHOTON_HD vec3 transform_point(vec3 p) const {
        float x = m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3];
        float y = m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3];
        float z = m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3];
        float w = m[3][0] * p.x + m[3][1] * p.y + m[3][2] * p.z + m[3][3];
        return w != 0 && w != 1 ? vec3(x / w, y / w, z / w) : vec3(x, y, z);
    }
    PHOTON_HD vec3 transform_vector(vec3 v) const {
        return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z, m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
    }
};

PHOTON_HD mat4 operator*(const mat4& a, const mat4& b) {
    mat4 r;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            float s = 0;
            for (int k = 0; k < 4; k++) s += a.m[i][k] * b.m[k][j];
            r.m[i][j] = s;
        }
    return r;
}

mat4 inverse(const mat4& a);
mat4 transpose(const mat4& a);
mat4 translation(vec3 t);
mat4 scaling(vec3 s);
mat4 rotation_quat(vec4 q);  // (x, y, z, w)

}  // namespace photon
