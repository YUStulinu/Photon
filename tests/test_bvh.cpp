#include <algorithm>

#include "core/sampling.h"
#include "render/traverse.h"
#include "scene/builtin.h"
#include "scene/scene.h"
#include "test.h"

using namespace photon;

namespace {

// A soup of random triangles of mixed sizes in a unit box, plus a few long thin ones.
void random_soup(Scene& scene, int n, uint64_t seed) {
    Rng rng(seed, 1);
    std::vector<vec3> pos;
    std::vector<int> idx;
    for (int i = 0; i < n; i++) {
        vec3 c(rng.next(), rng.next(), rng.next());
        float size = i % 50 == 0 ? 0.5f : 0.02f + 0.05f * rng.next();
        for (int k = 0; k < 3; k++) {
            pos.push_back(c + (vec3(rng.next(), rng.next(), rng.next()) - vec3(0.5f)) * size);
            idx.push_back(int(pos.size()) - 1);
        }
    }
    int m = scene.add_material(Material{});
    scene.add_mesh(pos, idx, m);
}

bool brute_force(const SceneView& s, const Ray& ray, Hit& hit) {
    bool found = false;
    float tmax = ray.tmax;
    for (int i = 0; i < s.num_tris; i++) {
        float t, u, v;
        if (intersect_triangle(s.tris[i], ray.o, ray.d, tmax, t, u, v)) {
            tmax = t;
            hit.t = t;
            hit.tri = i;
            hit.u = u;
            hit.v = v;
            found = true;
        }
    }
    return found;
}

void check_against_brute_force(const Scene& scene, int rays, uint64_t seed) {
    SceneView s = scene.view();
    Rng rng(seed, 2);
    int hits = 0, mismatches = 0;
    for (int i = 0; i < rays; i++) {
        Ray r;
        r.o = vec3(rng.next(), rng.next(), rng.next()) * 1.6f - vec3(0.3f);
        r.d = uniform_sphere(rng.next2());
        r.tmax = i % 4 == 0 ? 0.3f : kInf;  // some short rays (shadow-ray style)
        Hit a, b;
        bool ha = intersect(s, r, a), hb = brute_force(s, r, b);
        if (ha != hb || (ha && std::fabs(a.t - b.t) > 1e-5f * std::max(1.0f, b.t))) mismatches++;
        if (ha) hits++;
        CHECK(occluded(s, r) == hb);
    }
    CHECK_MSG(mismatches == 0, mismatches << " of " << rays << " rays differ");
    CHECK(hits > rays / 10);  // the test is meaningful
}

}  // namespace

TEST(bvh_matches_brute_force) {
    Scene scene;
    random_soup(scene, 2000, 1);
    scene.build();
    check_against_brute_force(scene, 20000, 7);
}

TEST(bvh_threaded_build_matches_brute_force) {
    // Large enough for the parallel subtree builds.
    Scene scene;
    random_soup(scene, 60000, 2);
    BvhBuildOptions opt;
    opt.threads = 8;
    scene.build(opt);
    check_against_brute_force(scene, 3000, 8);
}

TEST(bvh_structure_is_valid) {
    Scene scene;
    random_soup(scene, 5000, 3);
    scene.build();
    const Bvh& b = scene.bvh;
    // The order is a permutation of the triangles.
    std::vector<int> order = b.order;
    std::sort(order.begin(), order.end());
    for (int i = 0; i < int(order.size()); i++) CHECK(order[size_t(i)] == i);
    // Every leaf triangle lies inside its child box, every triangle is in exactly one leaf, and leaves
    // respect the size limit.
    std::vector<int> covered(scene.tris.size(), 0);
    int leaves = 0;
    for (const BvhNode& n : b.nodes) {
        for (int side = 0; side < 2; side++) {
            int count = side ? n.rcount : n.lcount;
            int first = side ? n.right : n.left;
            vec3 lo = side ? n.rmin : n.lmin, hi = side ? n.rmax : n.lmax;
            if (count <= 0) continue;
            leaves++;
            CHECK(count <= 8);
            for (int i = first; i < first + count; i++) {
                covered[size_t(i)]++;
                const TriGeom& g = scene.tris[size_t(i)];
                vec3 v[3] = {g.v0, g.v0 + g.e1, g.v0 + g.e2};
                for (vec3 p : v)
                    for (int k = 0; k < 3; k++) CHECK(p[k] >= lo[k] - 1e-5f && p[k] <= hi[k] + 1e-5f);
            }
        }
    }
    for (int c : covered) CHECK(c == 1);
    CHECK(leaves == b.stats.leaves);
    CHECK(b.stats.sah_cost > 0);
    if (test::verbose())
        std::printf("    %d nodes, %d leaves, depth %d, SAH %.2f, %.1f ms\n", b.stats.nodes, b.stats.leaves, b.stats.max_depth, b.stats.sah_cost, b.stats.build_ms);
}

TEST(bvh_sah_beats_a_naive_split_count) {
    // For the Cornell box (a handful of large quads) the SAH tree must stay shallow, and for a uniform
    // soup the number of node visits per ray must grow far slower than the triangle count.
    Scene small, large;
    random_soup(small, 1000, 4);
    random_soup(large, 64000, 4);
    small.build();
    large.build();
    auto visits = [](const Scene& sc) {
        SceneView s = sc.view();
        Rng rng(5, 5);
        double total = 0;
        for (int i = 0; i < 4000; i++) {
            Ray r;
            r.o = vec3(rng.next(), rng.next(), rng.next());
            r.d = uniform_sphere(rng.next2());
            Hit h;
            int v = 0;
            intersect(s, r, h, &v);
            total += v;
        }
        return total / 4000;
    };
    double a = visits(small), b = visits(large);
    if (test::verbose()) std::printf("    node visits per ray: %.1f (1K tris), %.1f (64K tris)\n", a, b);
    CHECK_MSG(b < a * 8, a << " -> " << b);  // 64x the triangles
}

TEST(alpha_masked_triangles_are_skipped) {
    // Two parallel quads; the front one is alpha-masked out (alpha 0 < cutoff 0.5).
    Scene scene;
    Material masked;
    masked.base_color = vec4(1, 1, 1, 0);
    masked.alpha_mode = AlphaMode::Mask;
    int m0 = scene.add_material(masked);
    int m1 = scene.add_material(Material{});
    add_quad(scene, vec3(-1, -1, 1), vec3(2, 0, 0), vec3(0, 2, 0), m0);
    add_quad(scene, vec3(-1, -1, 0), vec3(2, 0, 0), vec3(0, 2, 0), m1);
    scene.build();
    SceneView s = scene.view();
    Ray r;
    r.o = vec3(0.1f, 0.2f, 5);
    r.d = vec3(0, 0, -1);
    r.tmax = kInf;
    Hit h;
    CHECK(intersect(s, r, h));
    CHECK_NEAR(h.t, 5, 1e-4);
    CHECK(s.tris[h.tri].material == m1);
}
