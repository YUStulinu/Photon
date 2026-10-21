#include "render/cpu_renderer.h"

#include <algorithm>
#include <atomic>
#include <thread>

namespace photon {

void CpuRenderer::resize(int width, int height) {
    width_ = width;
    height_ = height;
    reset();
}

void CpuRenderer::reset() {
    samples_ = 0;
    sum_.assign(size_t(width_) * size_t(height_), vec3(0));
}

void CpuRenderer::render(const CameraView& camera, const RenderSettings& settings, int spp, int threads) {
    int n = threads > 0 ? threads : int(std::max(1u, std::thread::hardware_concurrency()));
    std::atomic<int> next_row{0};
    auto work = [&] {
        while (true) {
            int y = next_row.fetch_add(1);
            if (y >= height_) break;
            for (int x = 0; x < width_; x++) {
                vec3 acc(0);
                for (int s = 0; s < spp; s++)
                    acc += render_sample(view_, camera, x, y, width_, height_, uint32_t(samples_ + s), settings.seed, settings.path, nullptr);
                sum_[size_t(y) * size_t(width_) + size_t(x)] += acc;
            }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < n; i++) pool.emplace_back(work);
    for (auto& t : pool) t.join();
    samples_ += spp;
}

std::vector<vec3> CpuRenderer::image() const {
    std::vector<vec3> out(sum_.size());
    float inv = samples_ > 0 ? 1.0f / float(samples_) : 0.0f;
    for (size_t i = 0; i < sum_.size(); i++) out[i] = sum_[i] * inv;
    return out;
}

std::vector<uint32_t> CpuRenderer::display(const RenderSettings& settings) const {
    std::vector<vec3> img = image();
    std::vector<uint32_t> out(img.size());
    for (size_t i = 0; i < img.size(); i++) out[i] = pack_rgba8(tone_map(img[i], settings.exposure, settings.tonemap));
    return out;
}

}  // namespace photon
