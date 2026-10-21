#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "test.h"

namespace photon::test {

namespace {

struct Entry {
    const char* name;
    std::function<void()> fn;
};

std::vector<Entry>& registry() {
    static std::vector<Entry> r;
    return r;
}

int g_failures = 0;
bool g_verbose = false;

}  // namespace

void add(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }

void fail(const char* file, int line, const std::string& message) {
    g_failures++;
    std::printf("    %s:%d: %s\n", file, line, message.c_str());
}

std::string source_dir() {
    if (const char* env = std::getenv("PHOTON_SOURCE_DIR")) return env;
    return PHOTON_SOURCE_DIR;
}

bool verbose() { return g_verbose; }

}  // namespace photon::test

// usage: photon_tests [-v] [filter...]   (a test runs if its name contains any filter)
int main(int argc, char** argv) {
    using namespace photon::test;
    std::vector<std::string> filters;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-v") == 0) g_verbose = true;
        else filters.emplace_back(argv[i]);
    }
    int passed = 0, failed = 0, skipped = 0;
    auto t_all = std::chrono::steady_clock::now();
    for (const Entry& e : registry()) {
        if (!filters.empty()) {
            bool match = false;
            for (const std::string& f : filters)
                if (std::string(e.name).find(f) != std::string::npos) match = true;
            if (!match) continue;
        }
        int before = g_failures;
        auto t0 = std::chrono::steady_clock::now();
        std::string note;
        bool skip = false;
        try {
            e.fn();
        } catch (const Skip& s) {
            skip = true;
            note = s.reason;
        } catch (const std::exception& ex) {
            fail("<exception>", 0, ex.what());
        }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (skip) {
            skipped++;
            std::printf("SKIP %-40s (%s)\n", e.name, note.c_str());
        } else if (g_failures > before) {
            failed++;
            std::printf("FAIL %-40s %8.1f ms\n", e.name, ms);
        } else {
            passed++;
            std::printf("ok   %-40s %8.1f ms\n", e.name, ms);
        }
        std::fflush(stdout);
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all).count();
    std::printf("\n%d passed, %d failed, %d skipped in %.1f s\n", passed, failed, skipped, secs);
    return failed > 0 ? 1 : 0;
}
