// Photon: a physically based path tracer on the GPU.
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "render_cli.h"

namespace {

void usage() {
    std::printf(
        "Photon: a physically based GPU path tracer\n"
        "\n"
        "usage: photon <command> [options]\n"
        "\n"
        "commands:\n"
#if defined(PHOTON_WITH_VIEWER)
        "  view     [scene]       interactive viewer (progressive and SVGF real-time modes)\n"
#endif
        "  render   [scene]       offline render to PNG (and .pfm for linear HDR)\n"
        "  bench                  rendering throughput of the built-in scenes, CPU and GPU\n"
        "  export   <scene> <out.gltf>  write a scene as glTF\n"
        "  info                   the devices and the built-in scenes\n"
        "\n"
        "scene options:\n"
        "  --scene <name|file>    built-in scene (cornell, materials, glass, furnace) or .gltf/.glb\n"
        "  --env <file|sky>       environment map (.hdr, .png, ...) or the procedural sky\n"
        "  --env-intensity <f>    environment multiplier\n"
        "  --env-rotation <deg>   environment rotation around +y\n"
        "  --camera <i>           which of the file's cameras to use\n"
        "  --look-from <x,y,z>    camera position       --look-at <x,y,z>  camera target\n"
        "  --fov <deg>            vertical field of view\n"
        "  --aperture <r>         lens radius (depth of field)   --focus <d>  focus distance\n"
        "\n"
        "render options:\n"
        "  --out <file>           output .png (default render.png); .pfm writes linear HDR\n"
        "  --width/--height <n>   resolution (default 960x540)\n"
        "  --spp <n>              samples per pixel (default 256)\n"
        "  --cpu | --gpu          device (default: the GPU when available)\n"
        "  --max-depth <n>        path length (default 8)\n"
        "  --exposure <f>         exposure multiplier\n"
        "  --tonemap <aces|reinhard|linear>\n"
        "  --clamp <f>            clamp indirect contributions (0: off)\n"
        "  --no-nee               BSDF sampling only (no light sampling)\n"
        "  --threads <n>          CPU threads\n"
        "  --svgf <frames>        the real-time path: one-sample frames denoised by SVGF (GPU)\n"
        "  --view <name>          color, albedo, normal, depth, variance or heatmap (GPU)\n"
        "  --sun <x,y,z>          sun direction of the procedural sky; --sun-intensity <f>\n"
        "  --seed <n>             random seed\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty() || args[0] == "-h" || args[0] == "--help" || args[0] == "help") {
        usage();
        return args.empty() ? 1 : 0;
    }
    std::string cmd = args[0];
    args.erase(args.begin());
    try {
        if (cmd == "render") return photon::cmd_render(args);
        if (cmd == "bench") return photon::cmd_bench(args);
        if (cmd == "export") return photon::cmd_export(args);
        if (cmd == "info") return photon::cmd_info(args);
#if defined(PHOTON_WITH_VIEWER)
        if (cmd == "view") return photon::cmd_view(args);
#endif
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "unknown command '%s' (photon --help lists them)\n", cmd.c_str());
    return 1;
}
