// Built-in scenes (procedural geometry) and a procedural sky.
#pragma once

#include <string>
#include <vector>

#include "scene/scene.h"

namespace photon {

// Scene names: "cornell", "materials", "glass", "furnace".
std::vector<std::string> builtin_scene_names();
bool make_builtin_scene(const std::string& name, Scene& scene);

// An equirectangular sky: a gradient from the horizon to the zenith, ground below, and a sun disk.
void make_sky(Environment& env, vec3 sun_direction, float sun_intensity = 30000.0f, int width = 1024);

// Geometry helpers.
void add_sphere(Scene& scene, vec3 center, float radius, int material, int segments = 48);
void add_quad(Scene& scene, vec3 corner, vec3 edge_u, vec3 edge_v, int material);
void add_box(Scene& scene, vec3 center, vec3 half_size, float rotation_y, int material);

// Frames the camera on the scene's bounds (after loading files without a camera).
void frame_camera(Scene& scene);

}  // namespace photon
