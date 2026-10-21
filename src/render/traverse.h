// Ray-scene intersection: stack-based BVH traversal, nearest child first, with Moller-Trumbore
// triangle tests and alpha testing for masked materials (foliage, fences).
#pragma once

#include "render/scene_view.h"
#include "render/texture.h"

namespace photon {

struct Ray {
    vec3 o;
    vec3 d;
    float tmax = kInf;
};

struct Hit {
    float t = kInf;
    int tri = -1;
    float u = 0, v = 0;  // barycentrics of vertices 1 and 2
};

constexpr int kStackSize = 64;

PHOTON_HD bool intersect_triangle(const TriGeom& g, vec3 o, vec3 d, float tmax, float& t, float& u, float& v) {
    vec3 p = cross(d, g.e2);
    float det = dot(g.e1, p);
    if (fabsf(det) < 1e-14f) return false;
    float inv = 1.0f / det;
    vec3 s = o - g.v0;
    u = dot(s, p) * inv;
    if (u < 0.0f || u > 1.0f) return false;
    vec3 q = cross(s, g.e1);
    v = dot(d, q) * inv;
    if (v < 0.0f || u + v > 1.0f) return false;
    t = dot(g.e2, q) * inv;
    return t > 0.0f && t < tmax;
}

// Slab test; returns the entry distance or kInf when the box is missed.
PHOTON_HD float intersect_box(vec3 bmin, vec3 bmax, vec3 o, vec3 inv_d, float tmax) {
    float tx0 = (bmin.x - o.x) * inv_d.x, tx1 = (bmax.x - o.x) * inv_d.x;
    float ty0 = (bmin.y - o.y) * inv_d.y, ty1 = (bmax.y - o.y) * inv_d.y;
    float tz0 = (bmin.z - o.z) * inv_d.z, tz1 = (bmax.z - o.z) * inv_d.z;
    float tnear = fmaxf(fmaxf(fminf(tx0, tx1), fminf(ty0, ty1)), fmaxf(fminf(tz0, tz1), 0.0f));
    float tfar = fminf(fminf(fmaxf(tx0, tx1), fmaxf(ty0, ty1)), fminf(fmaxf(tz0, tz1), tmax));
    return tnear <= tfar ? tnear : kInf;
}

PHOTON_HD vec3 safe_inverse(vec3 d) {
    auto inv = [](float x) { return fabsf(x) > 1e-20f ? 1.0f / x : copysignf(1e20f, x); };
    return {inv(d.x), inv(d.y), inv(d.z)};
}

// Does the alpha-tested triangle let the ray through at barycentrics (u, v)?
PHOTON_HD bool alpha_passes(const SceneView& s, int tri, float u, float v) {
    const Material& m = s.materials[s.tris[tri].material];
    float alpha = m.base_color.w;
    if (m.base_color_tex >= 0 && s.uvs) {
        const int* idx = s.indices + 3 * tri;
        vec2 uv = s.uvs[idx[0]] * (1 - u - v) + s.uvs[idx[1]] * u + s.uvs[idx[2]] * v;
        alpha *= sample_texture(s, m.base_color_tex, uv).w;
    }
    return alpha < m.alpha_cutoff;
}

template <bool AnyHit>
PHOTON_HD bool traverse(const SceneView& s, const Ray& ray, Hit& hit, int* node_visits) {
    vec3 inv_d = safe_inverse(ray.d);
    float tmax = ray.tmax;
    int stack[kStackSize];
    int sp = 0;
    int node = 0;
    bool found = false;
    int visits = 0;
    while (true) {
        const BvhNode& n = s.nodes[node];
        visits++;
        float tl = intersect_box(n.lmin, n.lmax, ray.o, inv_d, tmax);
        float tr = intersect_box(n.rmin, n.rmax, ray.o, inv_d, tmax);
        // Leaves are handled in place; interior children are visited nearest first.
        int next[2];
        float tn[2];
        int nn = 0;
        for (int side = 0; side < 2; side++) {
            float tc = side == 0 ? tl : tr;
            if (tc == kInf) continue;
            int count = side == 0 ? n.lcount : n.rcount;
            int idx = side == 0 ? n.left : n.right;
            if (count > 0) {
                for (int i = idx; i < idx + count; i++) {
                    float t, u, v;
                    if (intersect_triangle(s.tris[i], ray.o, ray.d, tmax, t, u, v)) {
                        if ((s.tris[i].flags & kTriAlphaTested) && alpha_passes(s, i, u, v)) continue;
                        if (AnyHit) {
                            if (node_visits) *node_visits += visits;
                            hit.t = t;
                            hit.tri = i;
                            return true;
                        }
                        tmax = t;
                        hit.t = t;
                        hit.tri = i;
                        hit.u = u;
                        hit.v = v;
                        found = true;
                    }
                }
            } else if (idx >= 0) {
                next[nn] = idx;
                tn[nn] = tc;
                nn++;
            }
        }
        if (nn == 2) {
            // Re-test against the possibly shortened tmax, nearest first.
            if (tn[0] > tn[1]) {
                int a = next[0];
                next[0] = next[1];
                next[1] = a;
                float b = tn[0];
                tn[0] = tn[1];
                tn[1] = b;
            }
            if (tn[1] < tmax && sp < kStackSize) stack[sp++] = next[1];
            if (tn[0] < tmax) {
                node = next[0];
                continue;
            }
        } else if (nn == 1 && tn[0] < tmax) {
            node = next[0];
            continue;
        }
        if (sp == 0) break;
        node = stack[--sp];
    }
    if (node_visits) *node_visits += visits;
    return found;
}

PHOTON_HD bool intersect(const SceneView& s, const Ray& ray, Hit& hit, int* visits = nullptr) {
    return traverse<false>(s, ray, hit, visits);
}

PHOTON_HD bool occluded(const SceneView& s, const Ray& ray) {
    Hit h;
    return traverse<true>(s, ray, h, nullptr);
}

}  // namespace photon
