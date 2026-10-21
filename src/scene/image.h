// Image files: textures (PNG, JPEG, ...), HDR environment maps, and writing renders.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "scene/scene.h"

namespace photon {

// Decodes an 8-bit image (any format stb_image reads) into RGBA8.
bool decode_image(const uint8_t* data, size_t size, Texture& out, std::string& error);
bool load_image(const std::string& path, Texture& out, std::string& error);

// Loads an equirectangular environment map (Radiance .hdr, or an 8-bit image treated as sRGB).
bool load_environment(const std::string& path, Environment& env, std::string& error);

bool save_png(const std::string& path, int width, int height, const uint32_t* rgba);
// Portable float map: linear HDR, lossless (for references and comparisons).
bool save_pfm(const std::string& path, int width, int height, const vec3* rgb);
bool load_pfm(const std::string& path, int& width, int& height, std::vector<vec3>& rgb);

}  // namespace photon
