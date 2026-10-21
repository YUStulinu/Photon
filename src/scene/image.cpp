#include "scene/image.h"

#include <cstdio>
#include <cstring>
#include <fstream>

#include "core/color.h"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_GIF
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace photon {

bool decode_image(const uint8_t* data, size_t size, Texture& out, std::string& error) {
    int w, h, n;
    stbi_uc* px = stbi_load_from_memory(data, int(size), &w, &h, &n, 4);
    if (!px) {
        error = stbi_failure_reason() ? stbi_failure_reason() : "cannot decode the image";
        return false;
    }
    out.width = w;
    out.height = h;
    out.texels.resize(size_t(w) * size_t(h));
    std::memcpy(out.texels.data(), px, out.texels.size() * 4);
    stbi_image_free(px);
    return true;
}

bool load_image(const std::string& path, Texture& out, std::string& error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        error = "cannot open " + path;
        return false;
    }
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (!decode_image(data.data(), data.size(), out, error)) {
        error = path + ": " + error;
        return false;
    }
    out.name = path;
    return true;
}

bool load_environment(const std::string& path, Environment& env, std::string& error) {
    int w, h, n;
    if (stbi_is_hdr(path.c_str())) {
        float* px = stbi_loadf(path.c_str(), &w, &h, &n, 3);
        if (!px) {
            error = path + ": " + (stbi_failure_reason() ? stbi_failure_reason() : "cannot read");
            return false;
        }
        env.texels.resize(size_t(w) * size_t(h));
        for (size_t i = 0; i < env.texels.size(); i++) env.texels[i] = vec3(px[3 * i], px[3 * i + 1], px[3 * i + 2]);
        stbi_image_free(px);
    } else {
        stbi_uc* px = stbi_load(path.c_str(), &w, &h, &n, 3);
        if (!px) {
            error = path + ": " + (stbi_failure_reason() ? stbi_failure_reason() : "cannot read");
            return false;
        }
        env.texels.resize(size_t(w) * size_t(h));
        for (size_t i = 0; i < env.texels.size(); i++)
            env.texels[i] = vec3(srgb_to_linear(px[3 * i] / 255.0f), srgb_to_linear(px[3 * i + 1] / 255.0f), srgb_to_linear(px[3 * i + 2] / 255.0f));
        stbi_image_free(px);
    }
    // Clean up values that would break importance sampling.
    for (vec3& t : env.texels)
        if (!is_finite(t) || t.x < 0 || t.y < 0 || t.z < 0) t = vec3(0);
    env.width = w;
    env.height = h;
    env.name = path;
    return true;
}

bool save_png(const std::string& path, int width, int height, const uint32_t* rgba) {
    return stbi_write_png(path.c_str(), width, height, 4, rgba, width * 4) != 0;
}

bool save_pfm(const std::string& path, int width, int height, const vec3* rgb) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fprintf(f, "PF\n%d %d\n-1.0\n", width, height);
    // PFM rows go bottom to top.
    for (int y = height - 1; y >= 0; y--) std::fwrite(rgb + size_t(y) * size_t(width), sizeof(float) * 3, size_t(width), f);
    std::fclose(f);
    return true;
}

bool load_pfm(const std::string& path, int& width, int& height, std::vector<vec3>& rgb) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[3] = {};
    float scale = 0;
    if (std::fscanf(f, "%2s %d %d %f", magic, &width, &height, &scale) != 4 || std::strcmp(magic, "PF") != 0) {
        std::fclose(f);
        return false;
    }
    std::fgetc(f);
    rgb.resize(size_t(width) * size_t(height));
    for (int y = height - 1; y >= 0; y--)
        if (std::fread(rgb.data() + size_t(y) * size_t(width), sizeof(float) * 3, size_t(width), f) != size_t(width)) {
            std::fclose(f);
            return false;
        }
    std::fclose(f);
    return true;
}

}  // namespace photon
