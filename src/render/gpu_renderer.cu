#include "render/gpu_renderer.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>

#include "denoise/svgf.h"

namespace photon {

namespace {

#define CUDA_CHECK(x)                                                                                        \
    do {                                                                                                     \
        cudaError_t e_ = (x);                                                                                \
        if (e_ != cudaSuccess)                                                                               \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + " (" #x ")"); \
    } while (0)

// An owning device allocation.
template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count = 0;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { release(); }
    void release() {
        if (ptr) cudaFree(ptr);
        ptr = nullptr;
        count = 0;
    }
    void allocate(size_t n) {
        if (n == count && ptr) return;
        release();
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
        count = n;
    }
    void upload(const std::vector<T>& v) {
        allocate(v.size());
        if (!v.empty()) CUDA_CHECK(cudaMemcpy(ptr, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    void zero() {
        if (ptr) CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(T)));
    }
};

template <typename T>
const T* or_null(const DeviceBuffer<T>& b) {
    return b.count ? b.ptr : nullptr;
}

constexpr int kBlockX = 16, kBlockY = 8;

// Progressive accumulation: rgb += radiance, w += luminance^2 (for the variance view). The first
// sample of each launch also records the G-buffer for the debug views.
__global__ void __launch_bounds__(kBlockX* kBlockY) render_kernel(SceneView s, CameraView cam, int w, int h, uint32_t frame0, int spp, uint32_t seed,
                                                                   PathOptions opt, vec4* accum, vec4* albedo, vec4* normal_depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x;
    vec4 a = accum[i];
    for (int k = 0; k < spp; k++) {
        PrimaryInfo primary;
        vec3 L = render_sample(s, cam, x, y, w, h, frame0 + uint32_t(k), seed, opt, k == 0 ? &primary : nullptr);
        a.x += L.x;
        a.y += L.y;
        a.z += L.z;
        float l = luminance(L);
        a.w += l * l;
        if (k == 0) {
            albedo[i] = vec4(primary.albedo, 1);
            normal_depth[i] = vec4(primary.normal, primary.depth);
        }
    }
    accum[i] = a;
}

// Real-time mode: one sample and the full G-buffer for SVGF.
__global__ void __launch_bounds__(kBlockX* kBlockY) sample_kernel(SceneView s, CameraView cam, int w, int h, uint32_t frame, uint32_t seed, PathOptions opt,
                                                                   vec4* color, vec4* albedo, vec4* normal_depth, vec4* position, vec4* emission) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x;
    PrimaryInfo primary;
    vec3 L = render_sample(s, cam, x, y, w, h, frame, seed, opt, &primary);
    color[i] = vec4(L, 1);
    albedo[i] = vec4(primary.albedo, 1);
    normal_depth[i] = vec4(primary.normal, primary.depth);
    position[i] = vec4(primary.position, float(primary.material));
    emission[i] = vec4(primary.emission, 1);
}

// BVH cost of the primary ray through the pixel center: the number of nodes visited.
__global__ void heatmap_kernel(SceneView s, CameraView cam, int w, int h, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    Ray ray = camera_ray(cam, float(x) + 0.5f, float(y) + 0.5f, w, h, vec2(0.5f, 0.5f));
    Hit hit;
    int visits = 0;
    intersect(s, ray, hit, &visits);
    out[y * w + x] = float(visits);
}

__device__ vec3 heat_color(float t) {
    // A blue -> green -> yellow -> red ramp.
    t = saturate(t);
    vec3 c0(0.05f, 0.05f, 0.35f), c1(0.0f, 0.6f, 0.4f), c2(0.95f, 0.85f, 0.1f), c3(0.85f, 0.1f, 0.05f);
    if (t < 1.0f / 3) return lerp(c0, c1, t * 3);
    if (t < 2.0f / 3) return lerp(c1, c2, t * 3 - 1);
    return lerp(c2, c3, t * 3 - 2);
}

struct DisplayParams {
    int w, h;
    DisplayMode mode;
    float exposure;
    ToneMap tonemap;
    float inv_samples;
    float samples;
    float max_depth;
    float heat_scale;
    bool denoised;
};

__global__ void display_kernel(DisplayParams p, const vec4* accum, const vec4* denoised, const vec4* albedo, const vec4* normal_depth, const vec4* variance,
                               const float* heat, uint32_t* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || y >= p.h) return;
    int i = y * p.w + x;
    vec3 c;
    switch (p.mode) {
        case DisplayMode::Color:
        default:
            c = p.denoised ? denoised[i].xyz() : accum[i].xyz() * p.inv_samples;
            c = tone_map(c, p.exposure, p.tonemap);
            break;
        case DisplayMode::Albedo:
            c = albedo[i].xyz();
            break;
        case DisplayMode::Normal: {
            vec4 nd = normal_depth[i];
            c = nd.w > 0 ? nd.xyz() * 0.5f + vec3(0.5f) : vec3(0);
            break;
        }
        case DisplayMode::Depth: {
            float d = normal_depth[i].w;
            c = d > 0 ? vec3(1.0f - saturate(d / p.max_depth)) : vec3(0);
            break;
        }
        case DisplayMode::Variance: {
            float v;
            if (p.denoised) {
                v = variance[i].w;
            } else {
                vec4 a = accum[i];
                float mean = luminance(a.xyz()) * p.inv_samples;
                // The variance of the mean of the samples so far.
                v = p.samples > 1 ? fmaxf(0.0f, a.w * p.inv_samples - mean * mean) / (p.samples - 1) : 0.0f;
            }
            float t = sqrtf(v);
            c = vec3(t / (1.0f + t));
            break;
        }
        case DisplayMode::Heatmap:
            c = heat_color(heat[i] * p.heat_scale);
            break;
    }
    // pack_rgba8 encodes linear values as sRGB; the false-color debug views are already display values.
    if (p.mode != DisplayMode::Color && p.mode != DisplayMode::Albedo) c = vec3(srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z));
    out[i] = pack_rgba8(c);
}

}  // namespace

bool cuda_available(GpuInfo* info) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        cudaGetLastError();
        return false;
    }
    if (info) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return false;
        info->name = prop.name;
        info->compute_major = prop.major;
        info->compute_minor = prop.minor;
        info->memory = prop.totalGlobalMem;
        info->multiprocessors = prop.multiProcessorCount;
    }
    return true;
}

struct GpuRenderer::Impl {
    // Scene.
    DeviceBuffer<BvhNode> nodes;
    DeviceBuffer<TriGeom> tris;
    DeviceBuffer<int> indices;
    DeviceBuffer<vec3> normals;
    DeviceBuffer<vec2> uvs;
    DeviceBuffer<vec4> tangents;
    DeviceBuffer<Material> materials;
    DeviceBuffer<TextureInfo> textures;
    DeviceBuffer<uint32_t> texels;
    DeviceBuffer<float> srgb_lut;
    DeviceBuffer<EmissiveTri> emissive;
    DeviceBuffer<int> tri_emissive;
    DeviceBuffer<AliasEntry> emissive_alias;
    DeviceBuffer<PunctualLight> punctual;
    DeviceBuffer<vec3> env_texels;
    DeviceBuffer<float> env_marginal, env_conditional;
    SceneView view;
    float scene_extent = 1;
    bool uploaded = false;

    // Frame buffers.
    DeviceBuffer<vec4> accum;
    DeviceBuffer<vec4> color, albedo, normal_depth, position, emission, denoised;
    DeviceBuffer<float> heat;
    DeviceBuffer<uint32_t> display;
    Svgf svgf;
    CameraView last_camera;
    bool heat_valid = false;

    cudaEvent_t start = nullptr, stop = nullptr;

    Impl() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~Impl() {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
    }
};

GpuRenderer::GpuRenderer() : impl_(std::make_unique<Impl>()) {}
GpuRenderer::~GpuRenderer() = default;

bool GpuRenderer::upload(const Scene& scene, std::string& error) {
    if (!scene.built) {
        error = "the scene is not built";
        return false;
    }
    try {
        Impl& m = *impl_;
        m.nodes.upload(scene.bvh.nodes);
        m.tris.upload(scene.tris);
        m.indices.upload(scene.bvh_indices);
        m.normals.upload(scene.normals);
        m.uvs.upload(scene.uvs);
        m.tangents.upload(scene.tangents);
        m.materials.upload(scene.materials);
        m.textures.upload(scene.texture_infos);
        m.texels.upload(scene.texel_pool);
        m.srgb_lut.upload(scene.srgb_lut);
        m.emissive.upload(scene.emissive);
        m.tri_emissive.upload(scene.tri_emissive);
        m.emissive_alias.upload(scene.emissive_alias);
        m.punctual.upload(scene.punctual);
        m.env_texels.upload(scene.env.texels);
        m.env_marginal.upload(scene.env_marginal);
        m.env_conditional.upload(scene.env_conditional);

        // Same scalars as the host view, device pointers.
        SceneView v = scene.view();
        v.nodes = or_null(m.nodes);
        v.tris = or_null(m.tris);
        v.indices = or_null(m.indices);
        v.normals = or_null(m.normals);
        v.uvs = or_null(m.uvs);
        v.tangents = or_null(m.tangents);
        v.materials = or_null(m.materials);
        v.textures = or_null(m.textures);
        v.texels = or_null(m.texels);
        v.srgb_lut = or_null(m.srgb_lut);
        v.emissive = or_null(m.emissive);
        v.tri_emissive = or_null(m.tri_emissive);
        v.emissive_alias = or_null(m.emissive_alias);
        v.punctual = or_null(m.punctual);
        v.env.texels = or_null(m.env_texels);
        v.env.marginal_cdf = or_null(m.env_marginal);
        v.env.conditional_cdf = or_null(m.env_conditional);
        m.view = v;
        m.scene_extent = fmaxf(length(v.bounds_max - v.bounds_min), 1e-3f);
        m.uploaded = true;
        CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
    reset();
    return true;
}

bool GpuRenderer::update_lights(const Scene& scene, std::string& error) {
    if (!impl_->uploaded) return upload(scene, error);
    try {
        Impl& m = *impl_;
        m.tris.upload(scene.tris);
        m.materials.upload(scene.materials);
        m.emissive.upload(scene.emissive);
        m.tri_emissive.upload(scene.tri_emissive);
        m.emissive_alias.upload(scene.emissive_alias);
        m.env_texels.upload(scene.env.texels);
        m.env_marginal.upload(scene.env_marginal);
        m.env_conditional.upload(scene.env_conditional);
        SceneView host = scene.view();
        SceneView& v = m.view;
        v.tris = or_null(m.tris);
        v.materials = or_null(m.materials);
        v.emissive = or_null(m.emissive);
        v.tri_emissive = or_null(m.tri_emissive);
        v.emissive_alias = or_null(m.emissive_alias);
        v.num_emissive = host.num_emissive;
        v.emissive_power = host.emissive_power;
        v.env = host.env;
        v.env.texels = or_null(m.env_texels);
        v.env.marginal_cdf = or_null(m.env_marginal);
        v.env.conditional_cdf = or_null(m.env_conditional);
        v.p_env = host.p_env;
        v.p_emissive = host.p_emissive;
        v.p_punctual = host.p_punctual;
        CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
    reset();
    return true;
}

void GpuRenderer::resize(int width, int height) {
    if (width == width_ && height == height_ && impl_->accum.count) return;
    width_ = width;
    height_ = height;
    size_t n = size_t(width) * size_t(height);
    Impl& m = *impl_;
    for (DeviceBuffer<vec4>* b : {&m.accum, &m.color, &m.albedo, &m.normal_depth, &m.position, &m.emission, &m.denoised}) {
        b->allocate(n);
        b->zero();
    }
    m.heat.allocate(n);
    m.display.allocate(n);
    m.display.zero();
    m.svgf.resize(width, height);
    reset();
}

void GpuRenderer::reset() {
    samples_ = 0;
    frame_ = 0;
    denoised_ = false;
    impl_->heat_valid = false;
    impl_->accum.zero();
    impl_->svgf.reset();
}

void GpuRenderer::render(const CameraView& camera, const RenderSettings& settings, int spp) {
    Impl& m = *impl_;
    if (!m.uploaded || width_ == 0) return;
    if (denoised_) reset();
    dim3 block(kBlockX, kBlockY), grid((width_ + kBlockX - 1) / kBlockX, (height_ + kBlockY - 1) / kBlockY);
    cudaEventRecord(m.start);
    render_kernel<<<grid, block>>>(m.view, camera, width_, height_, uint32_t(samples_), spp, settings.seed, settings.path, m.accum.ptr, m.albedo.ptr,
                                   m.normal_depth.ptr);
    cudaEventRecord(m.stop);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(m.stop));
    cudaEventElapsedTime(&last_ms_, m.start, m.stop);
    samples_ += spp;
    m.last_camera = camera;
    m.heat_valid = false;
}

void GpuRenderer::render_denoised(const CameraView& camera, const CameraView& previous, const RenderSettings& settings, const SvgfSettings& svgf) {
    Impl& m = *impl_;
    if (!m.uploaded || width_ == 0) return;
    if (!denoised_) {
        samples_ = 0;
        m.svgf.reset();
    }
    denoised_ = true;
    dim3 block(kBlockX, kBlockY), grid((width_ + kBlockX - 1) / kBlockX, (height_ + kBlockY - 1) / kBlockY);
    cudaEventRecord(m.start);
    sample_kernel<<<grid, block>>>(m.view, camera, width_, height_, frame_, settings.seed, settings.path, m.color.ptr, m.albedo.ptr, m.normal_depth.ptr,
                                   m.position.ptr, m.emission.ptr);
    SvgfInput in{m.color.ptr, m.albedo.ptr, m.normal_depth.ptr, m.position.ptr, m.emission.ptr};
    m.svgf.run(in, previous, svgf, m.denoised.ptr);
    cudaEventRecord(m.stop);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(m.stop));
    cudaEventElapsedTime(&last_ms_, m.start, m.stop);
    frame_++;
    samples_ = 1;
    m.last_camera = camera;
    m.heat_valid = false;
}

void GpuRenderer::display(const RenderSettings& settings, DisplayMode mode, uint32_t* target) {
    Impl& m = *impl_;
    if (width_ == 0) return;
    dim3 block(kBlockX, kBlockY), grid((width_ + kBlockX - 1) / kBlockX, (height_ + kBlockY - 1) / kBlockY);
    if (mode == DisplayMode::Heatmap && !m.heat_valid && m.uploaded) {
        heatmap_kernel<<<grid, block>>>(m.view, m.last_camera, width_, height_, m.heat.ptr);
        m.heat_valid = true;
    }
    DisplayParams p;
    p.w = width_;
    p.h = height_;
    p.mode = mode;
    p.exposure = settings.exposure;
    p.tonemap = settings.tonemap;
    p.samples = float(samples_);
    p.inv_samples = samples_ > 0 ? 1.0f / float(samples_) : 0.0f;
    p.max_depth = m.scene_extent;
    p.heat_scale = 1.0f / 160.0f;
    p.denoised = denoised_;
    display_kernel<<<grid, block>>>(p, m.accum.ptr, m.denoised.ptr, m.albedo.ptr, m.normal_depth.ptr, m.svgf.variance_buffer(), m.heat.ptr,
                                    target ? target : m.display.ptr);
    CUDA_CHECK(cudaGetLastError());
    if (!target) CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<vec3> GpuRenderer::download_image() {
    Impl& m = *impl_;
    size_t n = size_t(width_) * size_t(height_);
    std::vector<vec4> raw(n);
    std::vector<vec3> out(n);
    if (n == 0) return out;
    if (denoised_) {
        CUDA_CHECK(cudaMemcpy(raw.data(), m.denoised.ptr, n * sizeof(vec4), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; i++) out[i] = raw[i].xyz();
    } else {
        CUDA_CHECK(cudaMemcpy(raw.data(), m.accum.ptr, n * sizeof(vec4), cudaMemcpyDeviceToHost));
        float inv = samples_ > 0 ? 1.0f / float(samples_) : 0.0f;
        for (size_t i = 0; i < n; i++) out[i] = raw[i].xyz() * inv;
    }
    return out;
}

std::vector<uint32_t> GpuRenderer::download_display() {
    size_t n = size_t(width_) * size_t(height_);
    std::vector<uint32_t> out(n);
    if (n) CUDA_CHECK(cudaMemcpy(out.data(), impl_->display.ptr, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return out;
}

uint32_t* GpuRenderer::display_buffer() const { return impl_->display.ptr; }

}  // namespace photon
