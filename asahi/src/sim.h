// libmetal_asahi: runs a dispatch on the AGX Performance Simulator's agx_run
// (functional simulation) before (or instead of) the GPU, to catch faults and
// hangs ahead of time and to check the hardware's results.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace masahi {

struct SimRegion {
    uint64_t va = 0;
    const uint8_t* data = nullptr; // initial contents
    size_t size = 0;
    bool dump = false;             // read back after the run into `out` (the whole region)
    uint8_t* patch = nullptr;      // else, if set: what the run changed is written here (data's layout)
    std::vector<uint8_t> out;
};

struct SimDispatch {
    const std::vector<uint8_t>* code = nullptr;
    uint32_t grid[3] = {1, 1, 1}; // threads
    uint32_t wg[3] = {1, 1, 1};
    uint32_t sharedBytes = 0;
    std::vector<std::pair<uint32_t, uint32_t>> uniforms; // 32-bit uniform index, value
    std::vector<SimRegion> regions;
    uint64_t maxSteps = 0;     // 0 = agx_run's default
    const std::vector<uint8_t>* object = nullptr; // for --lines
    uint64_t linesOffset = 0;
};

struct SimResult {
    bool ok = false;
    bool hang = false;       // step budget exhausted or wall-clock timeout
    std::string error;
    uint64_t instructions = 0;
    double seconds = 0;
};

// The agx_run binary: METAL_ASAHI_AGX_RUN, else the build-time default.
const std::string& simulatorPath();
bool simulatorAvailable();
SimResult simulate(SimDispatch& d);

} // namespace masahi
