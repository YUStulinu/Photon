#include "denoise/svgf.h"

#include <cuda_runtime.h>

#include <cstdio>

#include "render/gpu_renderer.h"

namespace photon {

namespace {

#define SVGF_CHECK(x)                                                                              \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    } while (0)

__device__ vec3 demodulate(vec3 color, vec3 emission, vec3 albedo) {
    vec3 illum = vmax(color - emission, vec3(0));
    return vec3(illum.x / fmaxf(albedo.x, 1e-3f), illum.y / fmaxf(albedo.y, 1e-3f), illum.z / fmaxf(albedo.z, 1e-3f));
}

// Projects a world position into the previous frame's pixel coordinates.
__device__ bool project(const CameraView& c, vec3 p, int w, int h, float& px, float& py, float& dist) {
    vec3 d = p - c.position;
    float z = dot(d, c.forward);
    if (z <= 1e-4f) return false;
    float sx = dot(d, c.right) / (z * c.tan_half_fov * c.aspect);
    float sy = dot(d, c.up) / (z * c.tan_half_fov);
    px = (sx + 1.0f) * 0.5f * float(w) - 0.5f;
    py = (1.0f - sy) * 0.5f * float(h) - 0.5f;
    dist = length(d);
    return true;
}

__global__ void temporal_kernel(SvgfInput in, CameraView prev_cam, int w, int h, bool has_history, float alpha_c, float alpha_m, const vec4* prev_illum,
                                const vec4* prev_moments, const vec4* prev_nd, const vec4* prev_pos, vec4* out_illum, vec4* out_moments) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x;
    vec4 pos = in.position[i];
    vec3 cur = demodulate(in.color[i].xyz(), in.emission[i].xyz(), in.albedo[i].xyz());
    float lum = luminance(cur);
    vec3 hist_c(0);
    vec2 hist_m(0, 0);
    float hist_len = 0, wsum = 0;
    if (has_history && pos.w >= 0) {
        float px, py, dist;
        if (project(prev_cam, pos.xyz(), w, h, px, py, dist)) {
            vec3 n = in.normal_depth[i].xyz();
            int x0 = int(floorf(px)), y0 = int(floorf(py));
            float fx = px - float(x0), fy = py - float(y0);
            for (int k = 0; k < 4; k++) {
                int xx = x0 + (k & 1), yy = y0 + (k >> 1);
                if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                int j = yy * w + xx;
                vec4 pp = prev_pos[j];
                vec4 pn = prev_nd[j];
                // Consistent history: the same material, a similar distance and orientation.
                if (pp.w != pos.w) continue;
                if (fabsf(pn.w - dist) > 0.05f * dist + 0.01f) continue;
                if (dot(pn.xyz(), n) < 0.9f) continue;
                float bw = ((k & 1) ? fx : 1 - fx) * ((k >> 1) ? fy : 1 - fy);
                vec4 pi = prev_illum[j];
                vec4 pm = prev_moments[j];
                hist_c += pi.xyz() * bw;
                hist_m = hist_m + vec2(pm.x, pm.y) * bw;
                hist_len += pi.w * bw;
                wsum += bw;
            }
        }
    }
    vec3 c;
    vec2 m;
    float len;
    if (wsum > 1e-3f) {
        hist_c = hist_c / wsum;
        hist_m = hist_m * (1.0f / wsum);
        len = fminf(hist_len / wsum + 1.0f, 64.0f);
        float ac = fmaxf(alpha_c, 1.0f / len), am = fmaxf(alpha_m, 1.0f / len);
        c = lerp(hist_c, cur, ac);
        m = vec2(lerp(hist_m.x, lum, am), lerp(hist_m.y, lum * lum, am));
    } else {
        len = 1.0f;
        c = cur;
        m = vec2(lum, lum * lum);
    }
    float variance = fmaxf(0.0f, m.y - m.x * m.x);
    out_illum[i] = vec4(c, len);
    out_moments[i] = vec4(m.x, m.y, variance, 0);
}

// Variance: from the temporal moments, or (with less than 4 frames of history) a 7x7 bilateral
// estimate from the neighborhood.
__global__ void variance_kernel(SvgfInput in, int w, int h, const vec4* illum, const vec4* moments, float phi_normal, vec4* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x;
    vec4 c = illum[i];
    float len = c.w;
    float variance = moments[i].z;
    if (len < 4.0f && in.position[i].w >= 0) {
        vec3 n = in.normal_depth[i].xyz();
        float z = in.normal_depth[i].w;
        float wsum = 0, m1 = 0, m2 = 0;
        vec3 acc(0);
        for (int dy = -3; dy <= 3; dy++)
            for (int dx = -3; dx <= 3; dx++) {
                int xx = x + dx, yy = y + dy;
                if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                int j = yy * w + xx;
                if (in.position[j].w != in.position[i].w) continue;
                vec4 nd = in.normal_depth[j];
                float wn = powf(fmaxf(0.0f, dot(n, nd.xyz())), phi_normal);
                float wz = expf(-fabsf(nd.w - z) / (0.05f * z + 1e-3f));
                float wt = wn * wz;
                vec4 m = moments[j];
                m1 += m.x * wt;
                m2 += m.y * wt;
                acc += illum[j].xyz() * wt;
                wsum += wt;
            }
        if (wsum > 0) {
            m1 /= wsum;
            m2 /= wsum;
            variance = fmaxf(0.0f, m2 - m1 * m1) * (4.0f / len);
            c = vec4(acc / wsum, len);
        }
    }
    out[i] = vec4(c.xyz(), variance);
}

__global__ void atrous_kernel(SvgfInput in, int w, int h, int step, float phi_color, float phi_normal, float phi_depth, const vec4* src, vec4* dst) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x;
    vec4 center = src[i];
    if (in.position[i].w < 0) {
        dst[i] = center;
        return;
    }
    // Variance, smoothed with a 3x3 Gaussian (more stable edge-stopping).
    float var = 0, vw = 0;
    const float g3[2] = {0.25f, 0.125f};
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            int xx = min(max(x + dx, 0), w - 1), yy = min(max(y + dy, 0), h - 1);
            float k = (dx == 0 && dy == 0) ? 0.25f : ((dx == 0 || dy == 0) ? g3[1] : 0.0625f);
            var += src[yy * w + xx].w * k;
            vw += k;
        }
    var /= vw;
    vec4 ndc = in.normal_depth[i];
    vec3 n = ndc.xyz();
    float z = ndc.w;
    // Depth gradient (for the depth weight, which must tolerate slanted surfaces).
    float zx = in.normal_depth[y * w + min(x + 1, w - 1)].w - in.normal_depth[y * w + max(x - 1, 0)].w;
    float zy = in.normal_depth[min(y + 1, h - 1) * w + x].w - in.normal_depth[max(y - 1, 0) * w + x].w;
    float grad = 0.5f * fmaxf(fabsf(zx), fabsf(zy));
    float lum_c = luminance(center.xyz());
    float sigma_l = phi_color * sqrtf(fmaxf(var, 0.0f)) + 1e-4f;
    const float kernel[3] = {3.0f / 8.0f, 1.0f / 4.0f, 1.0f / 16.0f};
    vec3 sum(0);
    float wsum = 0, vsum = 0;
    for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++) {
            int xx = x + dx * step, yy = y + dy * step;
            if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
            int j = yy * w + xx;
            if (in.position[j].w != in.position[i].w) continue;
            vec4 s = src[j];
            vec4 nd = in.normal_depth[j];
            float k = kernel[abs(dx)] * kernel[abs(dy)];
            float wl = expf(-fabsf(luminance(s.xyz()) - lum_c) / sigma_l);
            float wn = powf(fmaxf(0.0f, dot(n, nd.xyz())), phi_normal);
            float dist = sqrtf(float(dx * dx + dy * dy)) * float(step);
            float wz = expf(-fabsf(nd.w - z) / (phi_depth * grad * dist + 1e-3f * z + 1e-4f));
            float wt = k * wl * wn * wz;
            sum += s.xyz() * wt;
            vsum += wt * wt * s.w;
            wsum += wt;
        }
    if (wsum > 0) dst[i] = vec4(sum / wsum, vsum / (wsum * wsum));
    else dst[i] = center;
}

__global__ void history_kernel(SvgfInput in, int w, int h, const vec4* filtered, const vec4* illum, const vec4* moments, vec4* prev_illum,
                               vec4* prev_moments, vec4* prev_nd, vec4* prev_pos) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x;
    prev_illum[i] = vec4(filtered[i].xyz(), illum[i].w);
    prev_moments[i] = moments[i];
    prev_nd[i] = in.normal_depth[i];
    prev_pos[i] = in.position[i];
}

__global__ void remodulate_kernel(SvgfInput in, int w, int h, const vec4* filtered, vec4* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x;
    if (in.position[i].w < 0) {
        out[i] = in.color[i];
        return;
    }
    vec3 a = in.albedo[i].xyz();
    vec3 c = filtered[i].xyz() * vmax(a, vec3(1e-3f)) + in.emission[i].xyz();
    out[i] = vec4(c, 1);
}

}  // namespace

Svgf::~Svgf() { free_buffers(); }

void Svgf::free_buffers() {
    for (vec4** p : {&prev_illum_, &prev_moments_, &prev_nd_, &prev_pos_, &illum_, &moments_, &filter_[0], &filter_[1]}) {
        if (*p) cudaFree(*p);
        *p = nullptr;
    }
}

void Svgf::resize(int width, int height) {
    if (width == width_ && height == height_ && illum_) return;
    free_buffers();
    width_ = width;
    height_ = height;
    size_t bytes = size_t(width) * size_t(height) * sizeof(vec4);
    for (vec4** p : {&prev_illum_, &prev_moments_, &prev_nd_, &prev_pos_, &illum_, &moments_, &filter_[0], &filter_[1]}) {
        SVGF_CHECK(cudaMalloc(reinterpret_cast<void**>(p), bytes));
        SVGF_CHECK(cudaMemset(*p, 0, bytes));
    }
    has_history_ = false;
}

void Svgf::reset() { has_history_ = false; }

void Svgf::run(const SvgfInput& in, const CameraView& previous_camera, const SvgfSettings& s, vec4* output) {
    dim3 block(16, 8), grid((width_ + 15) / 16, (height_ + 7) / 8);
    temporal_kernel<<<grid, block>>>(in, previous_camera, width_, height_, has_history_, s.alpha_color, s.alpha_moments, prev_illum_, prev_moments_, prev_nd_,
                                     prev_pos_, illum_, moments_);
    variance_kernel<<<grid, block>>>(in, width_, height_, illum_, moments_, s.phi_normal, filter_[0]);
    int src = 0;
    for (int it = 0; it < s.atrous_iterations; it++) {
        atrous_kernel<<<grid, block>>>(in, width_, height_, 1 << it, s.phi_color, s.phi_normal, s.phi_depth, filter_[src], filter_[1 - src]);
        src = 1 - src;
        if (it == 0)
            history_kernel<<<grid, block>>>(in, width_, height_, filter_[src], illum_, moments_, prev_illum_, prev_moments_, prev_nd_, prev_pos_);
    }
    if (s.atrous_iterations == 0)
        history_kernel<<<grid, block>>>(in, width_, height_, filter_[src], illum_, moments_, prev_illum_, prev_moments_, prev_nd_, prev_pos_);
    remodulate_kernel<<<grid, block>>>(in, width_, height_, filter_[src], output);
    SVGF_CHECK(cudaGetLastError());
    has_history_ = true;
}

}  // namespace photon
