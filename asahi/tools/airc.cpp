// airc: compiles a .air / .metallib for AGX the way libmetal_asahi does and
// prints each kernel's launch record (and optionally writes its code).
//   airc file.air [--cpu agx-g14c] [--pipeline P] [--dump-dir DIR]
#include "../src/compiler.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

using namespace masahi;

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: airc file.air [--cpu CPU] [--pipeline P] [--dump-dir DIR] [--obj FILE]\n");
        return 2;
    }
    CompileOptions o;
    const char* dumpDir = nullptr;
    const char* objOut = nullptr;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--cpu")) o.cpu = argv[i + 1];
        else if (!strcmp(argv[i], "--pipeline")) o.irPipeline = argv[i + 1];
        else if (!strcmp(argv[i], "--features")) o.features = argv[i + 1];
        else if (!strcmp(argv[i], "--dump-dir")) dumpDir = argv[i + 1];
        else if (!strcmp(argv[i], "--obj")) objOut = argv[i + 1];
    }
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::string err;
    auto mods = splitAIRContainer(bytes.data(), bytes.size(), err);
    if (mods.empty()) {
        fprintf(stderr, "airc: %s\n", err.c_str());
        return 1;
    }
    int rc = 0;
    for (auto& m : mods) {
        CompiledModule cm = compileAIR(m, o);
        if (!cm.error.empty()) {
            fprintf(stderr, "airc: %s\n", cm.error.c_str());
            rc = 1;
            continue;
        }
        if (objOut) {
            std::ofstream of(objOut, std::ios::binary);
            of.write((const char*)cm.object.data(), (std::streamsize)cm.object.size());
        }
        printf("module: %zu kernel(s), %zu bytes of globals, %zu relocations, %.1f ms\n", cm.kernels.size(), cm.globals.size(),
               cm.relocs.size(), cm.compileSeconds * 1e3);
        for (auto& k : cm.kernels) {
            printf("kernel %s: %zu code bytes, %u uniform halves (tgpg u%u), %u gprs, %u tg bytes, %u stack bytes, globals %s\n",
                   k.name.c_str(), k.code.size(), k.uniformHalves, k.tgpgHalf / 2, k.gprs, k.threadgroupBytes, k.stackBytes,
                   k.globalDataHalf == ~0u ? "-" : ("u" + std::to_string(k.globalDataHalf / 2)).c_str());
            for (size_t a = 0; a < k.args.size(); a++) {
                auto& x = k.args[a];
                printf("  arg %zu: buffer(%u) as%u %s kind %u, %u bytes at half %d\n", a, x.location, x.airAddrSpace,
                       x.byref ? "byref" : "pointer", x.kind, x.size, (int)x.firstHalf);
            }
            if (k.threadgroupBufferLocation >= 0) printf("  threadgroup(%d): dynamic threadgroup memory\n", k.threadgroupBufferLocation);
            if (k.hiddenTgBytesHalf != ~0u) printf("  hidden threadgroup bytes at half %u\n", k.hiddenTgBytesHalf);
            for (auto& lg : k.launchGrid) printf("  launch grid value %u at half %u\n", lg.second, lg.first);
            if (dumpDir) {
                std::ofstream of(std::string(dumpDir) + "/" + k.name + ".bin", std::ios::binary);
                of.write((const char*)k.code.data(), (std::streamsize)k.code.size());
            }
        }
    }
    return rc;
}
