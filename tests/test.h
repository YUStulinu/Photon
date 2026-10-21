// A minimal test framework: TEST(name) registers a function; CHECK and CHECK_NEAR record failures
// with their location; a test may throw Skip when it cannot run here (no GPU, missing assets).
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <sstream>
#include <string>

namespace photon::test {

struct Skip {
    std::string reason;
};

void add(const char* name, std::function<void()> fn);
void fail(const char* file, int line, const std::string& message);
std::string source_dir();
bool verbose();

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) { add(name, std::move(fn)); }
};

}  // namespace photon::test

#define PHOTON_CONCAT2(a, b) a##b
#define PHOTON_CONCAT(a, b) PHOTON_CONCAT2(a, b)
#define TEST(name)                                                                                    \
    static void name();                                                                               \
    static ::photon::test::Registrar PHOTON_CONCAT(registrar_, name)(#name, name);                   \
    static void name()

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) ::photon::test::fail(__FILE__, __LINE__, "CHECK(" #cond ")");  \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                                    \
    do {                                                                                                         \
        double a_ = double(a), b_ = double(b), t_ = double(tol);                                                 \
        if (!(std::fabs(a_ - b_) <= t_)) {                                                                       \
            std::ostringstream os_;                                                                              \
            os_ << #a " = " << a_ << ", " #b " = " << b_ << " (tolerance " << t_ << ")";                         \
            ::photon::test::fail(__FILE__, __LINE__, os_.str());                                                 \
        }                                                                                                        \
    } while (0)

#define CHECK_MSG(cond, msg)                                                        \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::ostringstream os_;                                                 \
            os_ << "CHECK(" #cond "): " << msg;                                     \
            ::photon::test::fail(__FILE__, __LINE__, os_.str());                    \
        }                                                                           \
    } while (0)
