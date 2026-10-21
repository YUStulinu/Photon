cmake_minimum_required(VERSION 3.24)
project(Photon VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(PHOTON_CUDA "Build the CUDA renderer (needs the CUDA toolkit)" ON)
option(PHOTON_VIEWER "Build the interactive viewer (GLFW + Dear ImGui + OpenGL)" ON)
option(PHOTON_TESTS "Build the tests" ON)
option(PHOTON_WERROR "Treat compiler warnings as errors" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(PHOTON_WITH_CUDA OFF)
if(PHOTON_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      # The GPU in this machine; pass -DCMAKE_CUDA_ARCHITECTURES=75;86;89 to build for others.
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(PHOTON_WITH_CUDA ON)
    message(STATUS "Photon: CUDA renderer enabled (CUDA ${CUDAToolkit_VERSION}, architectures ${CMAKE_CUDA_ARCHITECTURES})")
  else()
    message(STATUS "Photon: no CUDA compiler found; building the CPU renderer only")
  endif()
endif()

# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------
if(MSVC)
  set(PHOTON_CXX_WARNINGS /W4 /permissive- /utf-8 /Zc:__cplusplus /wd4324)
  if(PHOTON_WERROR)
    list(APPEND PHOTON_CXX_WARNINGS /WX)
  endif()
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS WIN32_LEAN_AND_MEAN NOMINMAX)
  set(PHOTON_CXX_OPT /O2 /fp:fast)
else()
  set(PHOTON_CXX_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wno-missing-field-initializers)
  if(PHOTON_WERROR)
    list(APPEND PHOTON_CXX_WARNINGS -Werror)
  endif()
  set(PHOTON_CXX_OPT -O3)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
include(FetchContent)
set(FETCHCONTENT_QUIET ON)
FetchContent_Declare(stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG f58f558c120e9b32c217290b80bad1a0729fbb2c)
FetchContent_MakeAvailable(stb)

# ---------------------------------------------------------------------------
# The renderer library (CPU)
# ---------------------------------------------------------------------------
add_library(photon_core STATIC
  src/core/math.cpp
  src/bvh/bvh.cpp
  src/scene/json.cpp
  src/scene/scene.cpp
  src/scene/image.cpp
  src/scene/gltf.cpp
  src/scene/builtin.cpp
  src/render/cpu_renderer.cpp
)
target_include_directories(photon_core PUBLIC src)
target_include_directories(photon_core SYSTEM PRIVATE ${stb_SOURCE_DIR})
target_compile_options(photon_core PRIVATE ${PHOTON_CXX_WARNINGS} $<$<CONFIG:Release,RelWithDebInfo>:${PHOTON_CXX_OPT}>)
find_package(Threads REQUIRED)
target_link_libraries(photon_core PUBLIC Threads::Threads)

# The CUDA renderer and the SVGF denoiser.
if(PHOTON_WITH_CUDA)
  add_library(photon_gpu STATIC
    src/render/gpu_renderer.cu
    src/denoise/svgf.cu
  )
  target_link_libraries(photon_gpu PUBLIC photon_core CUDA::cudart_static)
  target_compile_definitions(photon_gpu PUBLIC PHOTON_WITH_CUDA)
  # (--expt-relaxed-constexpr is avoided: it triggers an nvcc 13.4 internal error with MSVC 14.5x.)
  set(PHOTON_CUDA_FLAGS -lineinfo --use_fast_math)
  if(MSVC)
    # C4211 comes from the stub files nvcc generates for kernels in anonymous namespaces.
    list(APPEND PHOTON_CUDA_FLAGS -Xcompiler=/utf-8 -Xcompiler=/Zc:__cplusplus -Xcompiler=/wd4324 -Xcompiler=/wd4211)
  endif()
  target_compile_options(photon_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${PHOTON_CUDA_FLAGS}>)
  set(PHOTON_RENDER_LIB photon_gpu)
else()
  set(PHOTON_RENDER_LIB photon_core)
endif()

# ---------------------------------------------------------------------------
# The program: offline renders, benchmarks and the viewer
# ---------------------------------------------------------------------------
set(PHOTON_APP_SOURCES src/app/main.cpp src/app/render_cli.cpp)
if(PHOTON_VIEWER)
  FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
  # X11 only on Linux (it also runs under XWayland), so the build needs no Wayland development packages.
  set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "")
  FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.9b
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(glfw imgui)
  find_package(OpenGL REQUIRED)
  add_library(photon_imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
  )
  target_include_directories(photon_imgui SYSTEM PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
  target_link_libraries(photon_imgui PUBLIC glfw OpenGL::GL)
  list(APPEND PHOTON_APP_SOURCES src/app/viewer.cpp)
endif()

add_executable(photon ${PHOTON_APP_SOURCES})
target_link_libraries(photon PRIVATE ${PHOTON_RENDER_LIB})
target_include_directories(photon PRIVATE src/app)
target_include_directories(photon SYSTEM PRIVATE ${stb_SOURCE_DIR})
target_compile_options(photon PRIVATE ${PHOTON_CXX_WARNINGS} $<$<CONFIG:Release,RelWithDebInfo>:${PHOTON_CXX_OPT}>)
if(PHOTON_VIEWER)
  target_link_libraries(photon PRIVATE photon_imgui)
  target_compile_definitions(photon PRIVATE PHOTON_WITH_VIEWER)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(PHOTON_TESTS)
enable_testing()
add_executable(photon_tests
  tests/test_main.cpp
  tests/test_math.cpp
  tests/test_bvh.cpp
  tests/test_bsdf.cpp
  tests/test_lights.cpp
  tests/test_render.cpp
  tests/test_gltf.cpp
  tests/test_gpu.cpp
)
target_link_libraries(photon_tests PRIVATE ${PHOTON_RENDER_LIB})
target_include_directories(photon_tests PRIVATE tests)
target_compile_options(photon_tests PRIVATE ${PHOTON_CXX_WARNINGS} $<$<CONFIG:Release,RelWithDebInfo>:${PHOTON_CXX_OPT}>)
target_compile_definitions(photon_tests PRIVATE PHOTON_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME unit COMMAND photon_tests)
endif()
