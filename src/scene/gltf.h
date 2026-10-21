// glTF 2.0 import (.gltf with external or embedded buffers, and binary .glb), and export of scenes.
#pragma once

#include <string>

#include "scene/scene.h"

namespace photon {

// Loads a glTF file into `scene` (appending geometry, materials and lights). Supported: triangle
// meshes (triangles, strips and fans) under the node hierarchy of the default scene, PBR
// metallic-roughness materials with KHR_materials_transmission, KHR_materials_ior and
// KHR_materials_emissive_strength, textures with samplers, perspective cameras and
// KHR_lights_punctual lights.
bool load_gltf(const std::string& path, Scene& scene, std::string& error);

// Writes the scene's meshes, materials, camera and punctual lights as a .gltf file with an embedded
// buffer (textures and the environment are not written).
bool save_gltf(const std::string& path, const Scene& scene, std::string& error);

}  // namespace photon
