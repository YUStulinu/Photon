// Bounding volume hierarchy construction with the surface area heuristic.
#pragma once

#include <vector>

#include "core/math.h"
#include "render/scene_view.h"

namespace photon {

struct Aabb {
    vec3 min = vec3(kInf);
    vec3 max = vec3(-kInf);

    void grow(vec3 p) {
        min = vmin(min, p);
        max = vmax(max, p);
    }
    void grow(const Aabb& b) {
        min = vmin(min, b.min);
        max = vmax(max, b.max);
    }
    bool empty() const { return min.x > max.x; }
    float area() const {
        if (empty()) return 0.0f;
        vec3 d = max - min;
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }
    vec3 center() const { return (min + max) * 0.5f; }
};

struct BvhBuildOptions {
    int bins = 32;           // SAH candidate planes per axis
    int max_leaf_size = 8;   // leaves never hold more triangles than this
    float traversal_cost = 1.0f;  // relative to the cost of one triangle test
    int threads = 0;         // 0: use the hardware's concurrency
};

struct BvhStats {
    int nodes = 0;
    int leaves = 0;
    int max_depth = 0;
    float sah_cost = 0.0f;  // expected cost of a random ray, in triangle tests
    double build_ms = 0.0;
};

struct Bvh {
    std::vector<BvhNode> nodes;
    std::vector<int> order;  // BVH position -> original triangle index
    Aabb bounds;
    BvhStats stats;
};

// Builds a BVH over triangles given by their three vertices (3 * n positions).
Bvh build_bvh(const std::vector<vec3>& tri_vertices, const BvhBuildOptions& options = {});

}  // namespace photon
