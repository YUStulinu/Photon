#include "render_cli.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <thread>

#include "render/cpu_renderer.h"
#include "scene/builtin.h"
#include "scene/gltf.h"
#include "scene/image.h"
#if defined(PHOTON_WITH_CUDA)
#include "render/gpu_renderer.h"
#endif

namespace photon {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

std::string lower(std::string s) {
    for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

const std::string& value_of(const std::vector<std::string>& args, size_t& i) {
    if (i + 1 >= args.size()) throw std::runtime_error("option " + args[i] + " needs a value");
    return args[++i];
}

int int_of(const std::vector<std::string>& args, size_t& i) { return std::stoi(value_of(args, i)); }
float float_of(const std::vector<std::string>& args, size_t& i) { return std::stof(value_of(args, i)); }

vec3 vec3_of(const std::vector<std::string>& args, size_t& i) {
    std::string v = value_of(args, i);
    vec3 r;
    if (std::sscanf(v.c_str(), "%f,%f,%f", &r.x, &r.y, &r.z) != 3) throw std::runtime_error(args[i - 1] + " needs x,y,z");
    return r;
}

bool gpu_present() {
#if defined(PHOTON_WITH_CUDA)
    return cuda_available();
#else
    return false;
#endif
}

void print_scene_summary(const Scene& scene) {
    const BvhStats& b = scene.bvh.stats;
    std::printf("scene %s: %d triangles, %zu materials, %zu textures, %zu emissive triangles, %zu punctual lights%s\n", scene.name.c_str(), scene.num_tris(),
                scene.materials.size(), scene.textures.size(), scene.emissive.size(), scene.punctual.size(),
                scene.env.width > 0 ? ", environment map" : (scene.env.color.x + scene.env.color.y + scene.env.color.z > 0 ? ", constant environment" : ""));
    std::printf("BVH: %d nodes, %d leaves, depth %d, SAH cost %.2f, built in %.1f ms\n", b.nodes, b.leaves, b.max_depth, b.sah_cost, b.build_ms);
    const Aabb& a = scene.bvh.bounds;
    std::printf("bounds (%.2f, %.2f, %.2f) - (%.2f, %.2f, %.2f)\n", a.min.x, a.min.y, a.min.z, a.max.x, a.max.y, a.max.z);
}

}  // namespace

bool parse_scene_option(const std::vector<std::string>& args, size_t& i, SceneOptions& opt) {
    const std::string& a = args[i];
    if (a == "--scene") opt.scene = value_of(args, i);
    else if (a == "--env") opt.env = value_of(args, i);
    else if (a == "--env-intensity") opt.env_intensity = float_of(args, i);
    else if (a == "--env-rotation") opt.env_rotation = float_of(args, i);
    else if (a == "--camera") opt.camera = int_of(args, i);
    else if (a == "--look-from") {
        opt.look_from = vec3_of(args, i);
        opt.look_from_set = true;
    } else if (a == "--look-at") {
        opt.look_at = vec3_of(args, i);
        opt.look_at_set = true;
    } else if (a == "--fov") opt.fov = float_of(args, i);
    else if (a == "--aperture") opt.aperture = float_of(args, i);
    else if (a == "--focus") opt.focus = float_of(args, i);
    else if (a == "--sun") opt.sun = vec3_of(args, i);
    else if (a == "--sun-intensity") opt.sun_intensity = float_of(args, i);
    else return false;
    return true;
}

bool load_scene(const SceneOptions& opt, Scene& scene, std::string& error) {
    auto t0 = Clock::now();
    bool builtin = false;
    for (const std::string& n : builtin_scene_names())
        if (n == opt.scene) builtin = true;
    if (builtin) {
        make_builtin_scene(opt.scene, scene);
    } else {
        std::string ext = lower(std::filesystem::path(opt.scene).extension().string());
        if (ext != ".gltf" && ext != ".glb") {
            error = "unknown scene '" + opt.scene + "' (a built-in name or a .gltf/.glb file)";
            return false;
        }
        scene = Scene();
        if (!load_gltf(opt.scene, scene, error)) return false;
        scene.name = std::filesystem::path(opt.scene).stem().string();
        if (scene.cameras.empty()) frame_camera(scene);
        // A file without lights would render black: light it with the sky.
        bool lit = !scene.punctual.empty();
        for (const Material& m : scene.materials)
            if (m.emission.x + m.emission.y + m.emission.z > 0) lit = true;
        if (!lit && opt.env.empty()) make_sky(scene.env, normalize(opt.sun), opt.sun_intensity);
    }
    if (opt.camera > 0) {
        if (size_t(opt.camera) >= scene.cameras.size()) {
            error = "the scene has " + std::to_string(scene.cameras.size()) + " cameras";
            return false;
        }
        scene.camera = scene.cameras[size_t(opt.camera)];
    }
    if (opt.look_from_set) scene.camera.position = opt.look_from;
    if (opt.look_at_set) scene.camera.target = opt.look_at;
    if (opt.fov > 0) scene.camera.vfov = opt.fov;
    if (opt.aperture >= 0) scene.camera.aperture = opt.aperture;
    if (opt.focus > 0) scene.camera.focus_distance = opt.focus;
    if (opt.env == "sky") {
        make_sky(scene.env, normalize(opt.sun), opt.sun_intensity);
    } else if (!opt.env.empty()) {
        if (!load_environment(opt.env, scene.env, error)) return false;
    }
    scene.env.intensity *= opt.env_intensity;
    scene.env.rotation += opt.env_rotation * kPi / 180.0f;
    std::printf("loaded %s in %.2f s\n", opt.scene.c_str(), seconds_since(t0));
    scene.build();
    return true;
}

int cmd_render(const std::vector<std::string>& args) {
    SceneOptions so;
    RenderSettings rs;
    std::string out = "render.png";
    int spp = 256, threads = 0, svgf_frames = 0;
#if defined(PHOTON_WITH_CUDA)
    DisplayMode display_mode = DisplayMode::Color;
#endif
    bool force_cpu = false, force_gpu = false;
    for (size_t i = 0; i < args.size(); i++) {
        const std::string& a = args[i];
        if (parse_scene_option(args, i, so)) continue;
        if (a == "--out" || a == "-o") out = value_of(args, i);
        else if (a == "--width") rs.width = int_of(args, i);
        else if (a == "--height") rs.height = int_of(args, i);
        else if (a == "--spp") spp = int_of(args, i);
        else if (a == "--cpu") force_cpu = true;
        else if (a == "--gpu") force_gpu = true;
        else if (a == "--max-depth") rs.path.max_depth = int_of(args, i);
        else if (a == "--exposure") rs.exposure = float_of(args, i);
        else if (a == "--clamp") rs.path.clamp = float_of(args, i);
        else if (a == "--no-nee") rs.path.nee = 0;
        else if (a == "--threads") threads = int_of(args, i);
        else if (a == "--svgf") svgf_frames = int_of(args, i);
        else if (a == "--view") {
#if defined(PHOTON_WITH_CUDA)
            std::string v = value_of(args, i);
            const char* names[] = {"color", "albedo", "normal", "depth", "variance", "heatmap"};
            bool found = false;
            for (int k = 0; k < 6; k++)
                if (v == names[k]) {
                    display_mode = DisplayMode(k);
                    found = true;
                }
            if (!found) throw std::runtime_error("unknown view " + v);
#else
            throw std::runtime_error("--view needs the GPU renderer");
#endif
        }
        else if (a == "--seed") rs.seed = uint32_t(std::stoul(value_of(args, i)));
        else if (a == "--tonemap") {
            std::string t = value_of(args, i);
            if (t == "aces") rs.tonemap = ToneMap::Aces;
            else if (t == "reinhard") rs.tonemap = ToneMap::Reinhard;
            else if (t == "linear") rs.tonemap = ToneMap::Linear;
            else throw std::runtime_error("unknown tone map " + t);
        } else if (!a.empty() && a[0] != '-') so.scene = a;
        else throw std::runtime_error("unknown option " + a);
    }
    if (rs.width <= 0 || rs.height <= 0 || spp <= 0) throw std::runtime_error("the resolution and the sample count must be positive");
    bool use_gpu = !force_cpu && gpu_present();
    if (svgf_frames > 0 && !use_gpu) {
        std::fprintf(stderr, "error: the SVGF denoiser runs on the GPU only\n");
        return 1;
    }
    if (force_gpu && !use_gpu) {
        std::fprintf(stderr, "error: no CUDA device (or Photon was built without CUDA)\n");
        return 1;
    }

    Scene scene;
    std::string error;
    if (!load_scene(so, scene, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    print_scene_summary(scene);
    CameraView cam = scene.camera.view(float(rs.width) / float(rs.height));

    std::vector<vec3> image;
    std::vector<uint32_t> ldr;
    auto t0 = Clock::now();
    if (use_gpu) {
#if defined(PHOTON_WITH_CUDA)
        GpuInfo info;
        cuda_available(&info);
        std::printf("rendering %dx%d, %d spp on %s\n", rs.width, rs.height, spp, info.name.c_str());
        GpuRenderer gpu;
        if (!gpu.upload(scene, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }
        gpu.resize(rs.width, rs.height);
        t0 = Clock::now();
        if (svgf_frames > 0) {
            // The real-time path: one-sample frames through the SVGF denoiser, on a still camera.
            SvgfSettings sv;
            for (int f = 0; f < svgf_frames; f++) gpu.render_denoised(cam, cam, rs, sv);
            spp = svgf_frames;
            std::printf("  %d SVGF frames (1 spp each), last frame %.2f ms", svgf_frames, gpu.last_frame_ms());
        } else {
            // Batches short enough to stay well under the display driver's watchdog.
            int done = 0;
            while (done < spp) {
                int n = std::min(spp - done, 8);
                gpu.render(cam, rs, n);
                done += n;
                std::printf("\r  %d / %d spp", done, spp);
                std::fflush(stdout);
            }
        }
        gpu.display(rs, display_mode);
        image = gpu.download_image();
        ldr = gpu.download_display();
#endif
    } else {
        int n_threads = threads > 0 ? threads : int(std::max(1u, std::thread::hardware_concurrency()));
        std::printf("rendering %dx%d, %d spp on the CPU (%d threads)\n", rs.width, rs.height, spp, n_threads);
        CpuRenderer cpu(scene);
        cpu.resize(rs.width, rs.height);
        int done = 0;
        while (done < spp) {
            int n = std::min(spp - done, 4);
            cpu.render(cam, rs, n, threads);
            done += n;
            std::printf("\r  %d / %d spp", done, spp);
            std::fflush(stdout);
        }
        image = cpu.image();
        ldr = cpu.display(rs);
    }
    double secs = seconds_since(t0);
    double samples = double(rs.width) * double(rs.height) * double(spp);
    std::printf("\nrendered in %.2f s (%.1f M samples/s)\n", secs, samples / secs * 1e-6);

    std::string ext = lower(std::filesystem::path(out).extension().string());
    bool ok = ext == ".pfm" ? save_pfm(out, rs.width, rs.height, image.data()) : save_png(out, rs.width, rs.height, ldr.data());
    if (!ok) {
        std::fprintf(stderr, "error: could not write %s\n", out.c_str());
        return 1;
    }
    std::printf("wrote %s\n", out.c_str());
    return 0;
}

int cmd_bench(const std::vector<std::string>& args) {
    int width = 960, height = 540, gpu_spp = 64, cpu_spp = 4;
    std::vector<std::string> scenes = {"cornell", "materials", "glass"};
    bool cpu = true;
    std::vector<std::string> files;
    SceneOptions base;  // camera and environment options apply to every scene
    for (size_t i = 0; i < args.size(); i++) {
        const std::string& a = args[i];
        if (a == "--scene") files.push_back(value_of(args, i));
        else if (parse_scene_option(args, i, base)) continue;
        else if (a == "--width") width = int_of(args, i);
        else if (a == "--height") height = int_of(args, i);
        else if (a == "--gpu-spp") gpu_spp = int_of(args, i);
        else if (a == "--cpu-spp") cpu_spp = int_of(args, i);
        else if (a == "--no-cpu") cpu = false;
        else throw std::runtime_error("unknown option " + a);
    }
    if (!files.empty()) scenes = files;
    [[maybe_unused]] bool gpu = gpu_present();
    std::vector<std::string> rows;
    for (const std::string& name : scenes) {
        SceneOptions so = base;
        so.scene = name;
        Scene scene;
        std::string error;
        if (!load_scene(so, scene, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }
        RenderSettings rs;
        rs.width = width;
        rs.height = height;
        CameraView cam = scene.camera.view(float(width) / float(height));
        double px = double(width) * double(height);
        double cpu_rate = 0, gpu_rate = 0;
        // Rays per sample (camera, bounce and shadow rays), measured on every 7th pixel.
        double rays_per_sample = 0;
        {
            SceneView sv = scene.view();
            uint32_t rays = 0;
            PathOptions po = rs.path;
            po.ray_count = &rays;
            int n = 0;
            for (int y = 0; y < height; y += 7)
                for (int x = (y / 7) % 7; x < width; x += 7, n++) render_sample(sv, cam, x, y, width, height, 0, 0, po, nullptr);
            rays_per_sample = double(rays) / std::max(n, 1);
        }
        if (cpu) {
            CpuRenderer r(scene);
            r.resize(width, height);
            r.render(cam, rs, 1);  // warm-up
            r.reset();
            auto t0 = Clock::now();
            r.render(cam, rs, cpu_spp);
            cpu_rate = px * cpu_spp / seconds_since(t0) * 1e-6;
        }
#if defined(PHOTON_WITH_CUDA)
        if (gpu) {
            GpuRenderer r;
            if (!r.upload(scene, error)) {
                std::fprintf(stderr, "error: %s\n", error.c_str());
                return 1;
            }
            r.resize(width, height);
            r.render(cam, rs, 4);  // warm-up
            r.reset();
            auto t0 = Clock::now();
            for (int done = 0; done < gpu_spp; done += 8) r.render(cam, rs, std::min(8, gpu_spp - done));
            gpu_rate = px * gpu_spp / seconds_since(t0) * 1e-6;
        }
#endif
        char speed[32] = "-";
        if (cpu_rate > 0 && gpu_rate > 0) std::snprintf(speed, sizeof(speed), "%.1fx", gpu_rate / cpu_rate);
        char row[256];
        std::snprintf(row, sizeof(row), "%-14s %10d %8.1f %9.2f %11.2f %11.2f %12.1f %8s", scene.name.c_str(), scene.num_tris(), scene.bvh.stats.build_ms,
                      rays_per_sample, cpu_rate, gpu_rate, gpu_rate * rays_per_sample, speed);
        rows.push_back(row);
    }
    std::printf("\n%-14s %10s %8s %9s %11s %11s %12s %8s\n", "scene", "triangles", "BVH ms", "rays/path", "CPU Msam/s", "GPU Msam/s", "GPU Mrays/s", "speedup");
    for (const std::string& r : rows) std::printf("%s\n", r.c_str());
    std::printf("(a sample is one path of up to %d bounces with a shadow ray at each, at %dx%d)\n", PathOptions{}.max_depth, width, height);
    return 0;
}

int cmd_export(const std::vector<std::string>& args) {
    SceneOptions so;
    std::vector<std::string> positional;
    for (size_t i = 0; i < args.size(); i++) {
        if (parse_scene_option(args, i, so)) continue;
        positional.push_back(args[i]);
    }
    if (positional.size() != 2) throw std::runtime_error("usage: photon export <scene> <out.gltf>");
    so.scene = positional[0];
    Scene scene;
    std::string error;
    if (!load_scene(so, scene, error) || !save_gltf(positional[1], scene, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    std::printf("wrote %s (%d triangles)\n", positional[1].c_str(), scene.num_tris());
    return 0;
}

int cmd_info(const std::vector<std::string>&) {
#if defined(PHOTON_WITH_CUDA)
    GpuInfo info;
    if (cuda_available(&info))
        std::printf("GPU: %s (compute %d.%d, %d SMs, %.1f GB)\n", info.name.c_str(), info.compute_major, info.compute_minor, info.multiprocessors,
                    double(info.memory) / (1024.0 * 1024.0 * 1024.0));
    else
        std::printf("GPU: no CUDA device found\n");
#else
    std::printf("GPU: built without CUDA\n");
#endif
    std::printf("CPU: %u threads\n", std::thread::hardware_concurrency());
    std::printf("built-in scenes:");
    for (const std::string& n : builtin_scene_names()) std::printf(" %s", n.c_str());
    std::printf("\n");
    return 0;
}

}  // namespace photon
