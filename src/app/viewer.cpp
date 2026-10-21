// The interactive viewer: GLFW window, Dear ImGui panels, and the path tracer drawing into a texture.
//
// Two modes:
//  - progressive: samples accumulate while the camera stands still, converging to the reference image;
//  - real-time: one sample per pixel per frame, denoised with SVGF (temporal reprojection with the
//    previous frame's camera plus an edge-aware a-trous filter), for moving the camera around.
//
// Camera: left drag orbits around the target, right drag (or middle drag) pans, the wheel zooms; in fly
// mode, WASD/QE move and right drag looks around. Clicking a surface with Ctrl selects its material for
// the editor. glTF files and environment maps can be dropped onto the window.
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "render/cpu_renderer.h"
#include "render_cli.h"
#include "scene/builtin.h"
#include "scene/gltf.h"
#include "scene/image.h"
#if defined(PHOTON_WITH_CUDA)
#include "render/gpu_renderer.h"
#else
namespace photon {
enum class DisplayMode : int { Color = 0, Albedo = 1, Normal = 2, Depth = 3, Variance = 4, Heatmap = 5 };
struct SvgfSettings {
    int atrous_iterations = 5;
    float alpha_color = 0.2f, alpha_moments = 0.2f, phi_color = 4.0f, phi_normal = 128.0f, phi_depth = 1.0f;
};
}  // namespace photon
#endif

namespace photon {

namespace {

using Clock = std::chrono::steady_clock;

// --- the camera controller ----------------------------------------------------------------------

struct CameraController {
    vec3 target;
    float yaw = 0, pitch = 0, distance = 5;
    float vfov = 45, aperture = 0, focus_distance = 0;
    bool fly = false;
    vec3 position;  // fly mode

    void from(const Camera& c) {
        target = c.target;
        vec3 d = c.position - c.target;
        distance = std::max(length(d), 1e-3f);
        d = d / distance;
        pitch = std::asin(clampf(d.y, -1, 1));
        yaw = std::atan2(d.x, d.z);
        vfov = c.vfov;
        aperture = c.aperture;
        focus_distance = c.focus_distance;
        position = c.position;
    }
    vec3 offset() const { return vec3(std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw)); }
    Camera camera() const {
        Camera c;
        if (fly) {
            c.position = position;
            c.target = position - offset();
        } else {
            c.position = target + offset() * distance;
            c.target = target;
        }
        c.vfov = vfov;
        c.aperture = aperture;
        c.focus_distance = focus_distance;
        return c;
    }
};

// --- the renderer behind the viewer: the GPU when present, else the CPU ---------------------------

struct Backend {
    bool gpu = false;
#if defined(PHOTON_WITH_CUDA)
    std::unique_ptr<GpuRenderer> g;
#endif
    std::unique_ptr<CpuRenderer> c;
    std::vector<uint32_t> pixels;
    int width = 0, height = 0;

    bool init(const Scene& scene, [[maybe_unused]] std::string& error) {
#if defined(PHOTON_WITH_CUDA)
        if (cuda_available()) {
            if (!g) g = std::make_unique<GpuRenderer>();
            gpu = g->upload(scene, error);
            if (gpu) {
                if (width) g->resize(width, height);
                return true;
            }
            return false;
        }
#endif
        c = std::make_unique<CpuRenderer>(scene);
        if (width) c->resize(width, height);
        return true;
    }
    void update_lights(const Scene& scene) {
        std::string error;
#if defined(PHOTON_WITH_CUDA)
        if (gpu) {
            if (!g->update_lights(scene, error)) std::fprintf(stderr, "error: %s\n", error.c_str());
            return;
        }
#endif
        c = std::make_unique<CpuRenderer>(scene);
        c->resize(width, height);
    }
    void resize(int w, int h) {
        width = w;
        height = h;
#if defined(PHOTON_WITH_CUDA)
        if (gpu) g->resize(w, h);
#endif
        if (c) c->resize(w, h);
    }
    void reset() {
#if defined(PHOTON_WITH_CUDA)
        if (gpu) g->reset();
#endif
        if (c) c->reset();
    }
    int samples() const {
#if defined(PHOTON_WITH_CUDA)
        if (gpu) return g->samples();
#endif
        return c ? c->samples() : 0;
    }
};

// --- the viewer state ------------------------------------------------------------------------------

struct Viewer {
    GLFWwindow* window = nullptr;
    Scene scene;
    SceneOptions scene_opt;
    std::string scene_label;
    Backend backend;
    CameraController cam;
    CameraView prev_view;
    bool have_prev = false;
    RenderSettings rs;
    SvgfSettings svgf;
    bool realtime = false;
    DisplayMode display = DisplayMode::Color;
    float render_scale = 1.0f;
    int max_spp = 4096;
    int spp_per_frame = 1;
    float target_ms = 30.0f;
    GLuint texture = 0;
    int tex_w = 0, tex_h = 0;
    int selected_material = -1;
    std::string status;
    double last_frame_ms = 0, last_render_ms = 0;
    bool dirty = true;  // the camera or settings changed: restart accumulation
    bool show_ui = true;
    std::vector<std::string> pending_drops;
    Clock::time_point status_time;

    void set_status(const std::string& s) {
        status = s;
        status_time = Clock::now();
        std::printf("%s\n", s.c_str());
    }

    bool load(const SceneOptions& opt) {
        Scene next;
        std::string error;
        auto t0 = Clock::now();
        if (!load_scene(opt, next, error)) {
            set_status("error: " + error);
            return false;
        }
        scene = std::move(next);
        scene_opt = opt;
        scene_label = scene.name.empty() ? opt.scene : scene.name;
        if (!backend.init(scene, error)) {
            set_status("error: " + error);
            return false;
        }
        cam = CameraController();
        cam.from(scene.camera);
        selected_material = -1;
        have_prev = false;
        dirty = true;
        double secs = std::chrono::duration<double>(Clock::now() - t0).count();
        char buf[256];
        std::snprintf(buf, sizeof(buf), "loaded %s: %d triangles in %.2f s (BVH %.0f ms)", scene_label.c_str(), scene.num_tris(), secs, scene.bvh.stats.build_ms);
        set_status(buf);
        return true;
    }

    void load_environment_file(const std::string& path) {
        std::string error;
        Environment env;
        if (path == "sky") {
            make_sky(env, normalize(vec3(0.4f, 0.8f, 0.3f)));
        } else if (!photon::load_environment(path, env, error)) {
            set_status("error: " + error);
            return;
        }
        env.intensity = scene.env.intensity;
        env.rotation = scene.env.rotation;
        scene.env = std::move(env);
        scene.rebuild_lights();
        backend.update_lights(scene);
        dirty = true;
        set_status("environment " + path);
    }

    void handle_drops() {
        for (const std::string& path : pending_drops) {
            std::string ext = std::filesystem::path(path).extension().string();
            for (char& ch : ext) ch = char(std::tolower(static_cast<unsigned char>(ch)));
            if (ext == ".gltf" || ext == ".glb") {
                SceneOptions o;
                o.scene = path;
                load(o);
            } else if (ext == ".hdr" || ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".exr") {
                load_environment_file(path);
            } else {
                set_status("cannot open " + path + " (drop a .gltf/.glb scene or an environment map)");
            }
        }
        pending_drops.clear();
    }

    // --- input ---

    void camera_input(float dt) {
        ImGuiIO& io = ImGui::GetIO();
        bool moved = false;
        if (!io.WantCaptureMouse) {
            ImVec2 d = io.MouseDelta;
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && !io.KeyCtrl && !cam.fly && (d.x != 0 || d.y != 0)) {
                cam.yaw -= d.x * 0.005f;
                cam.pitch = clampf(cam.pitch + d.y * 0.005f, -1.55f, 1.55f);
                moved = true;
            }
            bool look = cam.fly ? ImGui::IsMouseDown(ImGuiMouseButton_Right) : false;
            if (look && (d.x != 0 || d.y != 0)) {
                cam.yaw -= d.x * 0.004f;
                cam.pitch = clampf(cam.pitch + d.y * 0.004f, -1.55f, 1.55f);
                moved = true;
            }
            bool pan = !cam.fly && (ImGui::IsMouseDown(ImGuiMouseButton_Right) || ImGui::IsMouseDown(ImGuiMouseButton_Middle));
            if (pan && (d.x != 0 || d.y != 0)) {
                CameraView v = cam.camera().view(1.0f);
                float scale = cam.distance * 0.0015f;
                cam.target = cam.target - v.right * (d.x * scale) + v.up * (d.y * scale);
                moved = true;
            }
            if (io.MouseWheel != 0) {
                if (cam.fly) cam.vfov = clampf(cam.vfov - io.MouseWheel * 2.0f, 5.0f, 120.0f);
                else cam.distance *= std::pow(0.9f, io.MouseWheel);
                moved = true;
            }
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && io.KeyCtrl) pick(io.MousePos);
        }
        if (cam.fly && !io.WantCaptureKeyboard) {
            CameraView v = cam.camera().view(1.0f);
            float extent = length(scene.bvh.bounds.max - scene.bvh.bounds.min);
            float speed = extent * 0.25f * dt * (io.KeyShift ? 4.0f : 1.0f);
            vec3 m(0);
            if (ImGui::IsKeyDown(ImGuiKey_W)) m += v.forward;
            if (ImGui::IsKeyDown(ImGuiKey_S)) m -= v.forward;
            if (ImGui::IsKeyDown(ImGuiKey_D)) m += v.right;
            if (ImGui::IsKeyDown(ImGuiKey_A)) m -= v.right;
            if (ImGui::IsKeyDown(ImGuiKey_E)) m += vec3(0, 1, 0);
            if (ImGui::IsKeyDown(ImGuiKey_Q)) m -= vec3(0, 1, 0);
            if (length_squared(m) > 0) {
                cam.position += normalize(m) * speed;
                moved = true;
            }
        }
        if (!io.WantCaptureKeyboard) {
            if (ImGui::IsKeyPressed(ImGuiKey_Tab)) show_ui = !show_ui;
            if (ImGui::IsKeyPressed(ImGuiKey_Space)) {
                realtime = !realtime;
                dirty = true;
            }
            if (ImGui::IsKeyPressed(ImGuiKey_F)) toggle_fly();
            if (ImGui::IsKeyPressed(ImGuiKey_R)) {
                cam.from(scene.camera);
                moved = true;
            }
            if (ImGui::IsKeyPressed(ImGuiKey_F12)) screenshot();
        }
        if (moved) dirty = true;
    }

    void toggle_fly() {
        Camera c = cam.camera();
        cam.fly = !cam.fly;
        if (cam.fly) {
            cam.position = c.position;
        } else {
            // Orbit around the point in front of the camera.
            cam.target = c.position - cam.offset() * cam.distance;
        }
        dirty = true;
    }

    // Selects the material under the cursor.
    void pick(ImVec2 mouse) {
        ImGuiIO& io = ImGui::GetIO();
        float fx = mouse.x / io.DisplaySize.x, fy = mouse.y / io.DisplaySize.y;
        if (fx < 0 || fy < 0 || fx > 1 || fy > 1) return;
        CameraView v = cam.camera().view(io.DisplaySize.x / std::max(1.0f, io.DisplaySize.y));
        Ray r = camera_ray(v, fx * 1000.0f, fy * 1000.0f, 1000, 1000, vec2(0.5f, 0.5f));
        Hit h;
        SceneView sv = scene.view();
        if (intersect(sv, r, h)) {
            selected_material = sv.tris[h.tri].material;
            set_status("selected material " + std::to_string(selected_material));
        } else {
            selected_material = -1;
        }
    }

    void screenshot() {
        std::string base = "photon_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count() / 1000000);
        if (backend.width == 0) return;
        save_png(base + ".png", backend.width, backend.height, backend.pixels.data());
#if defined(PHOTON_WITH_CUDA)
        if (backend.gpu) {
            std::vector<vec3> img = backend.g->download_image();
            save_pfm(base + ".pfm", backend.width, backend.height, img.data());
        }
#endif
        if (backend.c) {
            std::vector<vec3> img = backend.c->image();
            save_pfm(base + ".pfm", backend.width, backend.height, img.data());
        }
        set_status("saved " + base + ".png and .pfm (" + std::to_string(backend.samples()) + " spp)");
    }

    // --- rendering ---

    void render_frame() {
        int fbw, fbh;
        glfwGetFramebufferSize(window, &fbw, &fbh);
        if (fbw <= 0 || fbh <= 0) return;
        float scale = backend.gpu ? render_scale : std::min(render_scale, 0.5f);
        int w = std::max(16, int(float(fbw) * scale)), h = std::max(16, int(float(fbh) * scale));
        if (w != backend.width || h != backend.height) {
            backend.resize(w, h);
            dirty = true;
        }
        CameraView view = cam.camera().view(float(w) / float(h));
        if (dirty) {
            // Progressive mode restarts; the real-time mode keeps its history and reprojects it.
            if (!realtime) backend.reset();
            spp_per_frame = 1;
            dirty = false;
        }
        auto t0 = Clock::now();
#if defined(PHOTON_WITH_CUDA)
        if (backend.gpu) {
            GpuRenderer& g = *backend.g;
            if (realtime) {
                g.render_denoised(view, have_prev ? prev_view : view, rs, svgf);
            } else if (g.samples() < max_spp) {
                int n = std::min(spp_per_frame, max_spp - g.samples());
                g.render(view, rs, n);
                // Adapt the batch to the frame time budget.
                float ms = g.last_frame_ms();
                if (ms < target_ms * 0.5f) spp_per_frame = std::min(spp_per_frame * 2, 64);
                else if (ms > target_ms && spp_per_frame > 1) spp_per_frame /= 2;
            }
            g.display(rs, display);
            backend.pixels = g.download_display();
        }
#endif
        if (!backend.gpu && backend.c) {
            CpuRenderer& c = *backend.c;
            if (c.samples() < max_spp) c.render(view, rs, 1);
            backend.pixels = c.display(rs);
        }
        last_render_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        prev_view = view;
        have_prev = true;
        upload_texture();
    }

    void upload_texture() {
        if (backend.pixels.empty()) return;
        if (!texture) glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        if (tex_w != backend.width || tex_h != backend.height) {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, backend.width, backend.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, backend.pixels.data());
            tex_w = backend.width;
            tex_h = backend.height;
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tex_w, tex_h, GL_RGBA, GL_UNSIGNED_BYTE, backend.pixels.data());
        }
    }

    // --- the user interface ---

    void ui() {
        ImGuiIO& io = ImGui::GetIO();
        if (texture)
            ImGui::GetBackgroundDrawList()->AddImage(ImTextureID(intptr_t(texture)), ImVec2(0, 0), io.DisplaySize);
        if (!show_ui) return;
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(390, 0), ImGuiCond_FirstUseEver);
        ImGui::Begin("Photon", nullptr);
        double msps = last_frame_ms > 0 ? double(backend.width) * backend.height * (realtime ? 1 : spp_per_frame) / (last_render_ms * 1e3) : 0;
        ImGui::Text("%s", backend.gpu ? "GPU (CUDA)" : "CPU (no CUDA device)");
        ImGui::Text("%dx%d  %d spp  %.1f ms/frame (%.0f fps)", backend.width, backend.height, backend.samples(), last_frame_ms, 1000.0 / std::max(last_frame_ms, 1e-3));
        ImGui::Text("render %.1f ms  ~%.0f M samples/s", last_render_ms, msps);
        if (!status.empty() && std::chrono::duration<double>(Clock::now() - status_time).count() < 8) ImGui::TextWrapped("%s", status.c_str());

        if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("%s: %d triangles, %zu materials", scene_label.c_str(), scene.num_tris(), scene.materials.size());
            ImGui::Text("%zu emissive tris, %zu punctual lights", scene.emissive.size(), scene.punctual.size());
            ImGui::Text("BVH %d nodes, depth %d, built in %.0f ms", scene.bvh.stats.nodes, scene.bvh.stats.max_depth, scene.bvh.stats.build_ms);
            const std::vector<std::string> builtins = builtin_scene_names();
            bool is_builtin = std::find(builtins.begin(), builtins.end(), scene_opt.scene) != builtins.end();
            if (ImGui::BeginCombo("Built-in", is_builtin ? scene_opt.scene.c_str() : "(file)")) {
                for (const std::string& n : builtins)
                    if (ImGui::Selectable(n.c_str(), n == scene_opt.scene)) {
                        SceneOptions o;
                        o.scene = n;
                        load(o);
                    }
                ImGui::EndCombo();
            }
            if (scene.cameras.size() > 1) {
                for (size_t i = 0; i < scene.cameras.size(); i++) {
                    ImGui::SameLine();
                    if (ImGui::SmallButton(("cam " + std::to_string(i)).c_str())) {
                        cam.from(scene.cameras[i]);
                        dirty = true;
                    }
                }
            }
            ImGui::TextDisabled("drop a .gltf/.glb or an .hdr onto the window");
            float intensity = scene.env.intensity;
            if (ImGui::SliderFloat("Env intensity", &intensity, 0.0f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic)) {
                scene.env.intensity = intensity;
                scene.rebuild_lights();
                backend.update_lights(scene);
                dirty = true;
            }
            float rot = scene.env.rotation * 180.0f / kPi;
            if (ImGui::SliderFloat("Env rotation", &rot, -180.0f, 180.0f, "%.0f deg")) {
                scene.env.rotation = rot * kPi / 180.0f;
                scene.rebuild_lights();
                backend.update_lights(scene);
                dirty = true;
            }
            if (ImGui::Button("Procedural sky")) load_environment_file("sky");
        }

        if (ImGui::CollapsingHeader("Render", ImGuiTreeNodeFlags_DefaultOpen)) {
            int mode = realtime ? 1 : 0;
            if (ImGui::RadioButton("Progressive", &mode, 0)) dirty = true;
            ImGui::SameLine();
            if (ImGui::RadioButton("Real-time + SVGF", &mode, 1)) dirty = true;
            realtime = mode == 1;
            const char* views[] = {"Color", "Albedo", "Normal", "Depth", "Variance", "BVH heatmap"};
            int dv = int(display);
            if (ImGui::Combo("View", &dv, views, IM_ARRAYSIZE(views))) display = DisplayMode(dv);
            if (ImGui::SliderFloat("Render scale", &render_scale, 0.25f, 1.0f, "%.2f")) dirty = true;
            if (ImGui::SliderInt("Max depth", &rs.path.max_depth, 1, 32)) dirty = true;
            bool nee = rs.path.nee != 0;
            if (ImGui::Checkbox("Light sampling (NEE + MIS)", &nee)) {
                rs.path.nee = nee ? 1 : 0;
                dirty = true;
            }
            if (ImGui::SliderFloat("Clamp indirect", &rs.path.clamp, 0.0f, 100.0f, rs.path.clamp > 0 ? "%.1f" : "off", ImGuiSliderFlags_Logarithmic)) dirty = true;
            ImGui::SliderInt("Max spp", &max_spp, 1, 65536, "%d", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Exposure", &rs.exposure, 0.05f, 20.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            int tm = int(rs.tonemap);
            const char* tms[] = {"Linear", "ACES", "Reinhard"};
            if (ImGui::Combo("Tone map", &tm, tms, 3)) rs.tonemap = ToneMap(tm);
            if (ImGui::Button("Screenshot (F12)")) screenshot();
        }

        if (realtime && ImGui::CollapsingHeader("SVGF", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderInt("A-trous passes", &svgf.atrous_iterations, 0, 6);
            ImGui::SliderFloat("Temporal alpha", &svgf.alpha_color, 0.02f, 1.0f);
            ImGui::SliderFloat("Moments alpha", &svgf.alpha_moments, 0.02f, 1.0f);
            ImGui::SliderFloat("Sigma luminance", &svgf.phi_color, 0.5f, 32.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Normal power", &svgf.phi_normal, 1.0f, 256.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Sigma depth", &svgf.phi_depth, 0.1f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        }

        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            bool fly = cam.fly;
            if (ImGui::Checkbox("Fly mode (F)", &fly)) toggle_fly();
            if (ImGui::SliderFloat("Field of view", &cam.vfov, 5.0f, 120.0f, "%.1f deg")) dirty = true;
            float extent = std::max(1e-3f, length(scene.bvh.bounds.max - scene.bvh.bounds.min));
            if (ImGui::SliderFloat("Aperture", &cam.aperture, 0.0f, extent * 0.05f, "%.4f")) dirty = true;
            if (cam.aperture > 0 && ImGui::SliderFloat("Focus distance", &cam.focus_distance, 0.0f, extent * 2.0f, cam.focus_distance > 0 ? "%.2f" : "target"))
                dirty = true;
            if (ImGui::Button("Reset (R)")) {
                cam.from(scene.camera);
                dirty = true;
            }
            ImGui::TextDisabled(cam.fly ? "WASD/QE move, right drag looks, shift: faster" : "left drag orbits, right drag pans, wheel zooms");
            ImGui::TextDisabled("ctrl+click: pick a material  Tab: hide UI  Space: mode");
        }

        if (selected_material >= 0 && size_t(selected_material) < scene.materials.size() &&
            ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
            Material& m = scene.materials[size_t(selected_material)];
            ImGui::Text("material %d%s", selected_material, m.base_color_tex >= 0 ? " (textured)" : "");
            bool changed = false;
            changed |= ImGui::ColorEdit3("Base color", &m.base_color.x);
            changed |= ImGui::SliderFloat("Metallic", &m.metallic, 0.0f, 1.0f);
            changed |= ImGui::SliderFloat("Roughness", &m.roughness, 0.0f, 1.0f);
            changed |= ImGui::SliderFloat("Transmission", &m.transmission, 0.0f, 1.0f);
            changed |= ImGui::SliderFloat("IOR", &m.ior, 1.0f, 3.0f);
            changed |= ImGui::ColorEdit3("Emission", &m.emission.x, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
            if (changed) {
                scene.rebuild_lights();
                backend.update_lights(scene);
                dirty = true;
            }
        }
        ImGui::End();
    }
};

}  // namespace

int cmd_view(const std::vector<std::string>& args) {
    SceneOptions so;
    int width = 1600, height = 900;
    bool realtime = false;
    // Automation: render a number of frames (orbiting the camera by `orbit` radians per frame), save the
    // window with its UI, and quit.
    int capture_frames = 0;
    std::string capture_path;
    float orbit = 0;
    int view_mode = 0;
    for (size_t i = 0; i < args.size(); i++) {
        if (parse_scene_option(args, i, so)) continue;
        if (args[i] == "--size" && i + 2 < args.size()) {
            width = std::stoi(args[++i]);
            height = std::stoi(args[++i]);
        } else if (args[i] == "--realtime") {
            realtime = true;
        } else if (args[i] == "--capture" && i + 2 < args.size()) {
            capture_frames = std::stoi(args[++i]);
            capture_path = args[++i];
        } else if (args[i] == "--orbit") {
            orbit = std::stof(args[++i]);
        } else if (args[i] == "--view-mode") {
            view_mode = std::stoi(args[++i]);
        } else if (!args[i].empty() && args[i][0] != '-') {
            so.scene = args[i];
        } else {
            throw std::runtime_error("unknown option " + args[i]);
        }
    }

    if (!glfwInit()) throw std::runtime_error("cannot initialize GLFW");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(width, height, "Photon", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("cannot create a window (OpenGL 3.3 needed)");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::StyleColorsDark();
    ImGui::GetStyle().WindowRounding = 4;
    ImGui::GetStyle().Colors[ImGuiCol_WindowBg].w = 0.85f;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    {
        Viewer v;
        v.window = window;
        v.realtime = realtime;
        glfwSetWindowUserPointer(window, &v);
        glfwSetDropCallback(window, [](GLFWwindow* w, int count, const char** paths) {
            Viewer* self = static_cast<Viewer*>(glfwGetWindowUserPointer(w));
            for (int i = 0; i < count; i++) self->pending_drops.emplace_back(paths[i]);
        });
        if (!v.load(so)) {
            SceneOptions fallback;
            v.load(fallback);
        }
        v.display = DisplayMode(view_mode);
        auto last = Clock::now();
        int frame = 0;
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            if (glfwGetWindowAttrib(window, GLFW_ICONIFIED)) {
                glfwWaitEvents();
                continue;
            }
            auto now = Clock::now();
            float dt = std::chrono::duration<float>(now - last).count();
            v.last_frame_ms = dt * 1000.0;
            last = now;
            v.handle_drops();
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            v.camera_input(dt);
            if (orbit != 0) {
                v.cam.yaw += orbit;
                v.dirty = true;
            }
            v.render_frame();
            v.ui();
            ImGui::Render();
            int fbw, fbh;
            glfwGetFramebufferSize(window, &fbw, &fbh);
            glViewport(0, 0, fbw, fbh);
            glClearColor(0.05f, 0.05f, 0.06f, 1);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            if (capture_frames > 0 && ++frame >= capture_frames) {
                std::vector<uint32_t> px(size_t(fbw) * size_t(fbh)), flipped(px.size());
                glPixelStorei(GL_PACK_ALIGNMENT, 4);
                glReadPixels(0, 0, fbw, fbh, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
                for (int y = 0; y < fbh; y++)
                    std::copy_n(px.begin() + ptrdiff_t(size_t(fbh - 1 - y) * size_t(fbw)), fbw, flipped.begin() + ptrdiff_t(size_t(y) * size_t(fbw)));
                for (uint32_t& p : flipped) p |= 0xFF000000u;
                save_png(capture_path, fbw, fbh, flipped.data());
                std::printf("captured %s after %d frames (%d spp, last frame %.1f ms)\n", capture_path.c_str(), frame, v.backend.samples(), v.last_render_ms);
                break;
            }
            glfwSwapBuffers(window);
        }
        if (v.texture) glDeleteTextures(1, &v.texture);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

}  // namespace photon
