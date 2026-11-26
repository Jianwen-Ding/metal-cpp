// libmetal_asahi: AIR (.air / .metallib) -> AGX machine code through the
// in-tree LLVM AGX backend (agx-lower-air + AGX codegen). No MSL.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace masahi {

struct KernelArg {
    uint32_t kind = 0;      // 0 = bytes in uniforms, 1 = pointer to the bytes in uniforms
    uint32_t size = 0;      // bytes
    uint32_t firstHalf = 0; // 0xffffffff if unused
    // Metal binding of the argument (from !air.kernel): buffer index.
    uint32_t location = 0;
    uint32_t airAddrSpace = 1; // 1 device, 2 constant
    bool byref = false;        // the argument is the bytes of a constant buffer, not its address
};

struct LaunchDivisor {
    uint32_t half, sourceHalf, kind;
};

struct CompiledKernel {
    std::string name;
    std::vector<uint8_t> code; // the kernel's instructions (ends with stop + traps)
    uint32_t uniformHalves = 0;
    uint32_t tgpgHalf = 0;            // threadgroups_per_grid.x
    uint32_t globalDataHalf = ~0u;    // global data buffer address, ~0u if none
    uint32_t gprs = 0;                // 32-bit general purpose registers
    uint32_t threadgroupBytes = 0;    // static threadgroup memory
    uint32_t stackBytes = 0;          // per thread
    std::vector<KernelArg> args;
    // Record extensions.
    uint32_t clusterDims[3] = {1, 1, 1};
    uint32_t hiddenTgBytesHalf = ~0u; // ext 4
    uint32_t textureHeapHalf = ~0u;   // ext 5
    std::vector<std::pair<uint32_t, uint32_t>> uniformConstants; // ext 6 (half, value)
    std::vector<LaunchDivisor> divisors;                         // ext 7
    uint32_t perCtaHalf = ~0u, perCtaSize = 0;                   // ext 8
    uint32_t address32Half = ~0u;                                // ext 9
    std::vector<std::pair<uint32_t, uint32_t>> launchGrid;       // ext 10 (half, kind): 0-2 grid threads, 3-5 dispatch threadgroup size
    bool checked = false;                                        // ext 2/3 (not driven yet)
    // Metal-level facts.
    int threadgroupBufferLocation = -1; // [[threadgroup(n)]] index of the dynamic threadgroup buffer
    uint32_t threadgroupBufferAlign = 16;
    bool usesSimdgroupBarrierOnly = false;
};

struct GlobalReloc {
    uint64_t offset;
    uint32_t type; // 1 = ABS64, 2 = ABS32
    uint64_t symbolOffset;
    int64_t addend;
};

struct CompileOptions {
    std::string cpu = "agx-g14c";
    std::string features;
    int optLevel = 3;
    bool fastMath = true;
    // New-PM IR pipeline run before codegen; agx-lower-air runs at its start.
    std::string irPipeline = "default<O2>";
};

struct CompiledModule {
    std::vector<CompiledKernel> kernels;
    std::vector<uint8_t> globals;          // initial global data buffer
    uint32_t globalsAlign = 16;
    std::vector<GlobalReloc> relocs;
    std::vector<uint8_t> object;           // the ELF (kept for the simulator's --lines)
    std::string error;
    double compileSeconds = 0;
    bool fromCache = false;
};

// Splits a file's bytes into the wrapped AIR bitcode modules it contains
// (one for a .air, one per entry for a .metallib). Empty with error set if none.
std::vector<std::vector<uint8_t>> splitAIRContainer(const uint8_t* data, size_t size, std::string& error);

// Compiles one AIR bitcode module for AGX.
CompiledModule compileAIR(const std::vector<uint8_t>& bitcode, const CompileOptions& opts);

// Function names (kernels) of an AIR module, without compiling it.
std::vector<std::string> kernelNames(const std::vector<uint8_t>& bitcode, std::string& error);

} // namespace masahi
