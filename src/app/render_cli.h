// The command-line side of the program: loading scenes from a name or a file, offline renders,
// benchmarks and scene export.
#pragma once

#include <string>
#include <vector>

#include "scene/scene.h"

namespace photon {

// Options shared by the commands that take a scene.
struct SceneOptions {
    std::string scene = "cornell";  // a built-in scene name or a .gltf/.glb file
    std::string env;                // an environment map (.hdr, .png, ...), or "sky"
    float env_intensity = 1.0f;
    float env_rotation = 0.0f;  // degrees
    int camera = 0;             // which of the file's cameras
    // Camera overrides.
    bool look_from_set = false, look_at_set = false;
    vec3 look_from, look_at;
    float fov = 0;       // degrees, 0: keep
    float aperture = -1;  // lens radius, < 0: keep
    float focus = 0;      // focus distance, 0: keep
    // The procedural sky ("--env sky", or files without lights).
    vec3 sun = vec3(0.4f, 0.8f, 0.3f);
    float sun_intensity = 30000.0f;
};

// Loads and builds the scene. Prints the reason and returns false on failure.
bool load_scene(const SceneOptions& opt, Scene& scene, std::string& error);

// Parses `--name value` style options common to the scene commands; returns false if `arg` is not one.
bool parse_scene_option(const std::vector<std::string>& args, size_t& i, SceneOptions& opt);

int cmd_render(const std::vector<std::string>& args);
int cmd_bench(const std::vector<std::string>& args);
int cmd_export(const std::vector<std::string>& args);
int cmd_info(const std::vector<std::string>& args);
#if defined(PHOTON_WITH_VIEWER)
int cmd_view(const std::vector<std::string>& args);
#endif

}  // namespace photon
