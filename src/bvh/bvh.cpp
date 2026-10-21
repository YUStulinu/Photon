// Binned SAH BVH construction (Wald, "On fast Construction of SAH-based Bounding Volume Hierarchies",
// 2007): at each node, triangle centroids are binned along each axis, and the split plane between
// bins that minimizes the surface area heuristic
//
//     C = C_trav + (A_left * N_left + A_right * N_right) / A_node
//
// wins, unless making a leaf is cheaper. Large subtrees are built on separate threads.
#include "bvh/bvh.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <thread>

namespace photon {

namespace {

struct BuildNode {
    Aabb bounds;
    int left = -1, right = -1;  // children (build nodes), or -1 for a leaf
    int first = 0, count = 0;   // triangle range of a leaf, in `order`
};

struct Builder {
    const BvhBuildOptions& opt;
    std::vector<Aabb> tri_bounds;
    std::vector<vec3> centroids;
    std::vector<int>& order;
    std::vector<BuildNode> nodes;
    std::mutex nodes_mutex;
    std::atomic<int> active_threads{0};
    int max_threads = 1;

    Builder(const BvhBuildOptions& o, std::vector<int>& ord) : opt(o), order(ord) {}

    int add_node(const BuildNode& n) {
        std::lock_guard<std::mutex> lock(nodes_mutex);
        nodes.push_back(n);
        return int(nodes.size()) - 1;
    }

    void set_node(int i, const BuildNode& n) {
        std::lock_guard<std::mutex> lock(nodes_mutex);
        nodes[i] = n;
    }

    // Builds the subtree over order[first, first + count) and returns its node index.
    int build(int first, int count, int depth) {
        BuildNode node;
        Aabb centroid_bounds;
        for (int i = first; i < first + count; i++) {
            node.bounds.grow(tri_bounds[order[i]]);
            centroid_bounds.grow(centroids[order[i]]);
        }
        node.first = first;
        node.count = count;
        int index = add_node(node);
        if (count <= 1 || depth > 60) return index;

        // Find the best split over all axes.
        const int nb = opt.bins;
        float best_cost = kInf;
        int best_axis = -1, best_bin = -1;
        float leaf_cost = float(count);
        for (int axis = 0; axis < 3; axis++) {
            float lo = centroid_bounds.min[axis], hi = centroid_bounds.max[axis];
            if (hi - lo < 1e-12f) continue;
            float scale = float(nb) / (hi - lo);
            Aabb bin_bounds[64];
            int bin_count[64] = {};
            for (int i = first; i < first + count; i++) {
                int t = order[i];
                int b = std::min(nb - 1, int((centroids[t][axis] - lo) * scale));
                bin_count[b]++;
                bin_bounds[b].grow(tri_bounds[t]);
            }
            // Sweep from the left and the right to get the cost of every split plane.
            float right_area[64];
            int right_count[64];
            Aabb acc;
            int n = 0;
            for (int b = nb - 1; b > 0; b--) {
                acc.grow(bin_bounds[b]);
                n += bin_count[b];
                right_area[b] = acc.area();
                right_count[b] = n;
            }
            acc = Aabb();
            n = 0;
            for (int b = 0; b < nb - 1; b++) {
                acc.grow(bin_bounds[b]);
                n += bin_count[b];
                int nr = right_count[b + 1];
                if (n == 0 || nr == 0) continue;
                float cost = opt.traversal_cost + (acc.area() * n + right_area[b + 1] * nr) / node.bounds.area();
                if (cost < best_cost) {
                    best_cost = cost;
                    best_axis = axis;
                    best_bin = b;
                }
            }
        }
        bool must_split = count > opt.max_leaf_size;
        if (best_axis < 0) {
            if (!must_split) return index;
            // All centroids coincide: split the range in the middle.
            best_axis = -1;
        } else if (best_cost >= leaf_cost && !must_split) {
            return index;
        }

        int mid;
        if (best_axis < 0) {
            mid = first + count / 2;
        } else {
            float lo = centroid_bounds.min[best_axis], hi = centroid_bounds.max[best_axis];
            float scale = float(nb) / (hi - lo);
            int* p = std::partition(order.data() + first, order.data() + first + count, [&](int t) {
                return std::min(nb - 1, int((centroids[t][best_axis] - lo) * scale)) <= best_bin;
            });
            mid = int(p - order.data());
            if (mid == first || mid == first + count) mid = first + count / 2;
        }

        int lc = mid - first, rc = count - lc;
        int left, right;
        // Build the larger subtrees in parallel while there are threads to spare.
        if (count > 20000 && active_threads.fetch_add(1) < max_threads - 1) {
            auto fut = std::async(std::launch::async, [&] { return build(first, lc, depth + 1); });
            right = build(mid, rc, depth + 1);
            left = fut.get();
            active_threads.fetch_sub(1);
        } else {
            if (count > 20000) active_threads.fetch_sub(1);
            left = build(first, lc, depth + 1);
            right = build(mid, rc, depth + 1);
        }
        BuildNode interior = node;
        interior.left = left;
        interior.right = right;
        interior.count = 0;
        set_node(index, interior);
        return index;
    }
};

}  // namespace

Bvh build_bvh(const std::vector<vec3>& tri_vertices, const BvhBuildOptions& options) {
    auto t0 = std::chrono::steady_clock::now();
    Bvh bvh;
    int n = int(tri_vertices.size() / 3);
    bvh.order.resize(n);
    for (int i = 0; i < n; i++) bvh.order[i] = i;
    Builder b(options, bvh.order);
    b.max_threads = options.threads > 0 ? options.threads : std::max(1u, std::thread::hardware_concurrency());
    b.tri_bounds.resize(n);
    b.centroids.resize(n);
    for (int i = 0; i < n; i++) {
        Aabb box;
        box.grow(tri_vertices[3 * i]);
        box.grow(tri_vertices[3 * i + 1]);
        box.grow(tri_vertices[3 * i + 2]);
        b.tri_bounds[i] = box;
        b.centroids[i] = box.center();
    }
    b.nodes.reserve(size_t(std::max(1, 2 * n / std::max(1, options.max_leaf_size / 2))));

    auto empty_child = [](BvhNode& node, bool left_side) {
        if (left_side) {
            node.lmin = vec3(kInf);
            node.lmax = vec3(-kInf);
            node.left = -1;
            node.lcount = 0;
        } else {
            node.rmin = vec3(kInf);
            node.rmax = vec3(-kInf);
            node.right = -1;
            node.rcount = 0;
        }
    };

    if (n == 0) {
        BvhNode root{};
        empty_child(root, true);
        empty_child(root, false);
        bvh.nodes.push_back(root);
        return bvh;
    }
    int root = b.build(0, n, 0);
    bvh.bounds = b.nodes[root].bounds;

    // Flatten: each output node describes the two children of a build node. Depth-first order puts a
    // node's subtree next to it in memory.
    std::vector<BvhNode>& out = bvh.nodes;
    BvhStats& st = bvh.stats;
    float root_area = std::max(bvh.bounds.area(), 1e-20f);
    struct Item {
        int build;
        int out;
        int depth;
    };
    auto set_child = [&](BvhNode& node, bool left_side, const BuildNode& c, int child_out) {
        vec3 mn = c.bounds.min, mx = c.bounds.max;
        int idx = c.left < 0 ? c.first : child_out;
        int cnt = c.left < 0 ? c.count : 0;
        if (left_side) {
            node.lmin = mn;
            node.lmax = mx;
            node.left = idx;
            node.lcount = cnt;
        } else {
            node.rmin = mn;
            node.rmax = mx;
            node.right = idx;
            node.rcount = cnt;
        }
    };
    const BuildNode& r = b.nodes[root];
    out.push_back(BvhNode{});
    if (r.left < 0) {
        // The whole scene is one leaf.
        set_child(out[0], true, r, 0);
        empty_child(out[0], false);
        st.nodes = 1;
        st.leaves = 1;
        st.sah_cost = float(r.count);
    } else {
        std::vector<Item> stack{{root, 0, 0}};
        st.sah_cost = options.traversal_cost;
        while (!stack.empty()) {
            Item it = stack.back();
            stack.pop_back();
            const BuildNode& bn = b.nodes[it.build];
            st.nodes++;
            st.max_depth = std::max(st.max_depth, it.depth);
            for (int side = 0; side < 2; side++) {
                const BuildNode& c = b.nodes[side == 0 ? bn.left : bn.right];
                int child_out = -1;
                if (c.left >= 0) {
                    child_out = int(out.size());
                    out.push_back(BvhNode{});
                    stack.push_back({side == 0 ? bn.left : bn.right, child_out, it.depth + 1});
                    st.sah_cost += options.traversal_cost * c.bounds.area() / root_area;
                } else {
                    st.leaves++;
                    st.sah_cost += float(c.count) * c.bounds.area() / root_area;
                }
                set_child(out[it.out], side == 0, c, child_out);
            }
        }
    }
    st.build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return bvh;
}

}  // namespace photon
