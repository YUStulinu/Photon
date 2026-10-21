#include "scene/builtin.h"

#include <algorithm>
#include <cmath>

#include "core/sampling.h"

namespace photon {

std::vector<std::string> builtin_scene_names() { return {"cornell", "materials", "glass", "furnace"}; }

void add_sphere(Scene& scene, vec3 center, float radius, int material, int segments) {
    int rings = segments / 2;
    std::vector<vec3> pos, nrm;
    std::vector<vec2> uv;
    std::vector<int> idx;
    for (int r = 0; r <= rings; r++) {
        float theta = kPi * float(r) / float(rings);
        for (int s = 0; s <= segments; s++) {
            float phi = kTwoPi * float(s) / float(segments);
            vec3 n(std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi));
            pos.push_back(center + n * radius);
            nrm.push_back(n);
            uv.push_back(vec2(float(s) / float(segments), float(r) / float(rings)));
        }
    }
    int row = segments + 1;
    for (int r = 0; r < rings; r++)
        for (int s = 0; s < segments; s++) {
            int a = r * row + s, b = a + row;
            if (r > 0) idx.insert(idx.end(), {a, a + 1, b});
            if (r < rings - 1) idx.insert(idx.end(), {a + 1, b + 1, b});
        }
    scene.add_mesh(pos, idx, material, nrm, uv);
}

void add_quad(Scene& scene, vec3 corner, vec3 edge_u, vec3 edge_v, int material) {
    vec3 n = normalize(cross(edge_u, edge_v));
    std::vector<vec3> pos = {corner, corner + edge_u, corner + edge_u + edge_v, corner + edge_v};
    std::vector<vec3> nrm(4, n);
    std::vector<vec2> uv = {vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)};
    scene.add_mesh(pos, {0, 1, 2, 0, 2, 3}, material, nrm, uv);
}

void add_box(Scene& scene, vec3 center, vec3 h, float rot, int material) {
    float c = std::cos(rot), s = std::sin(rot);
    auto tr = [&](vec3 p) { return center + vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z); };
    auto dir = [&](vec3 d) { return vec3(c * d.x + s * d.z, d.y, -s * d.x + c * d.z); };
    // Six faces, outward normals.
    struct Face {
        vec3 o, u, v;
    };
    Face faces[6] = {
        {vec3(-h.x, -h.y, h.z), vec3(2 * h.x, 0, 0), vec3(0, 2 * h.y, 0)},    // +z
        {vec3(h.x, -h.y, -h.z), vec3(-2 * h.x, 0, 0), vec3(0, 2 * h.y, 0)},   // -z
        {vec3(h.x, -h.y, h.z), vec3(0, 0, -2 * h.z), vec3(0, 2 * h.y, 0)},    // +x
        {vec3(-h.x, -h.y, -h.z), vec3(0, 0, 2 * h.z), vec3(0, 2 * h.y, 0)},   // -x
        {vec3(-h.x, h.y, h.z), vec3(2 * h.x, 0, 0), vec3(0, 0, -2 * h.z)},    // +y
        {vec3(-h.x, -h.y, -h.z), vec3(2 * h.x, 0, 0), vec3(0, 0, 2 * h.z)},   // -y
    };
    for (const Face& f : faces) add_quad(scene, tr(f.o), dir(f.u), dir(f.v), material);
}

void make_sky(Environment& env, vec3 sun, float sun_intensity, int width) {
    int height = width / 2;
    env.width = width;
    env.height = height;
    env.texels.resize(size_t(width) * size_t(height));
    sun = normalize(sun);
    const float sun_cos = std::cos(0.6f * kPi / 180.0f);  // a 1.2 degree disk
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            vec3 d = equirect_to_direction(vec2((float(x) + 0.5f) / float(width), (float(y) + 0.5f) / float(height)));
            vec3 c;
            if (d.y >= 0) {
                float t = std::pow(d.y, 0.45f);
                vec3 horizon(0.85f, 0.9f, 1.0f), zenith(0.22f, 0.42f, 0.85f);
                c = lerp(horizon, zenith, t) * 1.2f;
                // A glow around the sun.
                float cs = dot(d, sun);
                c += vec3(1.0f, 0.85f, 0.6f) * (0.35f * std::pow(std::max(cs, 0.0f), 32.0f));
                if (cs > sun_cos) c = vec3(1.0f, 0.95f, 0.85f) * sun_intensity;
            } else {
                c = vec3(0.25f, 0.23f, 0.21f) * (0.5f + 0.5f * std::exp(8.0f * d.y));
            }
            env.texels[size_t(y) * size_t(width) + size_t(x)] = c;
        }
    env.name = "sky";
}

static Material diffuse(vec3 c) {
    Material m;
    m.base_color = vec4(c, 1);
    m.metallic = 0;
    m.roughness = 1;
    return m;
}

bool make_builtin_scene(const std::string& name, Scene& scene) {
    scene = Scene();
    scene.name = name;
    if (name == "cornell") {
        // The Cornell box (dimensions and colors of the classic scene), in units of 100 mm.
        int white = scene.add_material(diffuse(vec3(0.725f, 0.71f, 0.68f)));
        int red = scene.add_material(diffuse(vec3(0.63f, 0.065f, 0.05f)));
        int green = scene.add_material(diffuse(vec3(0.14f, 0.45f, 0.091f)));
        Material light = diffuse(vec3(0.78f));
        light.emission = vec3(17.0f, 12.0f, 4.0f);
        int lamp = scene.add_material(light);
        float S = 5.5f;
        add_quad(scene, vec3(0, 0, 0), vec3(0, 0, S), vec3(S, 0, 0), white);         // floor (+y)
        add_quad(scene, vec3(0, S, 0), vec3(S, 0, 0), vec3(0, 0, S), white);         // ceiling (-y)
        // The camera looks along +z, so the screen's right is -x: red on the left (x = S), green on the right.
        add_quad(scene, vec3(0, 0, S), vec3(0, S, 0), vec3(S, 0, 0), white);         // back wall (-z)
        add_quad(scene, vec3(S, 0, 0), vec3(0, 0, S), vec3(0, S, 0), red);           // left wall (-x)
        add_quad(scene, vec3(0, 0, S), vec3(0, 0, -S), vec3(0, S, 0), green);        // right wall (+x)
        add_quad(scene, vec3(2.13f, S - 0.01f, 2.27f), vec3(1.3f, 0, 0), vec3(0, 0, 1.05f), lamp);  // light (-y)
        add_box(scene, vec3(1.85f, 0.825f, 1.69f), vec3(0.825f), -0.29f, white);      // short box, front right
        add_box(scene, vec3(3.68f, 1.65f, 3.51f), vec3(0.825f, 1.65f, 0.825f), 0.29f, white);  // tall box, back left
        scene.camera.position = vec3(2.78f, 2.73f, -8.0f);
        scene.camera.target = vec3(2.78f, 2.73f, 0.0f);
        scene.camera.vfov = 39.3f;
        return true;
    }
    if (name == "materials" || name == "glass") {
        Material ground = diffuse(vec3(0.4f, 0.4f, 0.42f));
        ground.roughness = 0.9f;
        int g = scene.add_material(ground);
        add_quad(scene, vec3(-20, 0, 20), vec3(40, 0, 0), vec3(0, 0, -40), g);
        if (name == "materials") {
            // Rows: rough dielectric, metal, glass; columns: roughness 0 .. 1.
            for (int row = 0; row < 3; row++)
                for (int col = 0; col < 6; col++) {
                    Material m;
                    float rough = float(col) / 5.0f;
                    m.roughness = rough;
                    if (row == 0) {
                        m.base_color = vec4(0.8f, 0.15f, 0.1f, 1);
                        m.metallic = 0;
                    } else if (row == 1) {
                        m.base_color = vec4(0.95f, 0.75f, 0.35f, 1);  // gold-like
                        m.metallic = 1;
                    } else {
                        m.base_color = vec4(1, 1, 1, 1);
                        m.metallic = 0;
                        m.transmission = 1;
                        m.ior = 1.5f;
                    }
                    add_sphere(scene, vec3(-5.0f + 2.0f * float(col), 0.8f, 2.4f * float(row) - 2.4f), 0.8f, scene.add_material(m));
                }
            scene.camera.position = vec3(0, 6.5f, 11.0f);
            scene.camera.target = vec3(0, 0.3f, 0);
            scene.camera.vfov = 40;
        } else {
            Material glass;
            glass.base_color = vec4(1, 1, 1, 1);
            glass.metallic = 0;
            glass.roughness = 0;
            glass.transmission = 1;
            glass.ior = 1.5f;
            add_sphere(scene, vec3(-1.2f, 1.0f, 0), 1.0f, scene.add_material(glass), 96);
            Material frosted = glass;
            frosted.roughness = 0.35f;
            add_sphere(scene, vec3(1.2f, 1.0f, 0), 1.0f, scene.add_material(frosted), 96);
            Material checker = diffuse(vec3(0.9f, 0.3f, 0.2f));
            add_box(scene, vec3(0, 0.5f, -2.5f), vec3(2.5f, 0.5f, 0.3f), 0, scene.add_material(checker));
            scene.camera.position = vec3(0, 2.2f, 6.0f);
            scene.camera.target = vec3(0, 0.9f, 0);
            scene.camera.vfov = 40;
        }
        make_sky(scene.env, vec3(0.5f, 0.7f, 0.4f));
        return true;
    }
    if (name == "furnace") {
        // A white diffuse sphere inside a uniform white environment: every pixel must converge to 1.
        Material m = diffuse(vec3(1.0f));
        m.ior = 1.0f;  // no specular layer: a pure Lambertian
        add_sphere(scene, vec3(0), 1.0f, scene.add_material(m), 64);
        scene.env.color = vec3(1.0f);
        scene.camera.position = vec3(0, 0, 4);
        scene.camera.target = vec3(0);
        scene.camera.vfov = 40;
        return true;
    }
    return false;
}

void frame_camera(Scene& scene) {
    Aabb b;
    for (const vec3& p : scene.positions) b.grow(p);
    if (b.empty()) return;
    vec3 c = b.center();
    float r = 0.5f * length(b.max - b.min);
    scene.camera.target = c;
    scene.camera.position = c + normalize(vec3(0.3f, 0.25f, 1.0f)) * (r / std::tan(0.5f * scene.camera.vfov * kPi / 180.0f));
}

}  // namespace photon
