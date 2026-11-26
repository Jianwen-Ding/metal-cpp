// libmetal_asahi: the Asahi DRM uAPI (asahi_drm.h) used directly: one VM per
// MTLDevice, GEM buffers bound at addresses we choose, compute commands built
// by hand (USC pipeline words + CDM control stream), syncobj waits.
// Register layouts are those of Mesa's open genxml/cmdbuf.xml (MIT).
#pragma once
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace masahi {

struct Bo {
    uint32_t handle = 0;
    uint64_t size = 0;
    uint64_t va = 0;
    uint8_t* map = nullptr;
    uint64_t mmapOffset = 0;
    bool usc = false;
};

// First-fit range allocator over a virtual address window.
class VaHeap {
public:
    void init(uint64_t base, uint64_t size);
    uint64_t alloc(uint64_t size, uint64_t align);
    void free(uint64_t va, uint64_t size);
    uint64_t base() const { return base_; }

private:
    uint64_t base_ = 0;
    std::map<uint64_t, uint64_t> free_; // start -> size
    std::mutex m_;
};

struct GpuInfo {
    uint32_t generation = 0;
    char variant = 0;
    uint32_t revision = 0;
    uint32_t clusters = 0;
    uint32_t cores = 0;
    uint64_t vmStart = 0, vmEnd = 0;
    uint64_t timestampHz = 24000000;
    uint32_t maxKhz = 0;
    uint32_t maxCommandsPerSubmission = 64;
    uint64_t features = 0;
    std::string chipName;  // e.g. "agx-g14c"
    std::string marketing; // e.g. "Apple M2 Max (G14C B1, Asahi DRM)"
    bool g14x = false;
};

struct ComputeCommand {
    uint64_t cdmBase = 0, cdmEnd = 0;
    uint32_t helperBinary = 0, helperCfg = 0;
    uint64_t helperData = 0;
    uint64_t samplerHeap = 0;
    uint32_t samplerCount = 0;
    bool barrierOnPrevious = true; // wait for the command before it in the submission
};

class DrmDevice {
public:
    static DrmDevice* open(std::string& error); // the first Asahi render node
    ~DrmDevice();

    const GpuInfo& info() const { return info_; }
    int fd() const { return fd_; }

    Bo* alloc(uint64_t size, bool usc, bool writable = true);
    void free(Bo* bo);
    uint64_t uscBase() const { return uscBase_; }
    uint64_t allocatedBytes() const { return allocated_; }

    uint32_t createQueue(uint32_t priority);
    void destroyQueue(uint32_t queue);

    // Submits up to maxCommandsPerSubmission compute commands as one submission.
    // inSync (0 = none) is waited for; outSync is signalled. Timestamps (start,
    // end ticks) of each command land in tsOut when non-null.
    bool submit(uint32_t queue, const std::vector<ComputeCommand>& cmds, uint32_t inSync, uint32_t outSync,
                std::vector<std::pair<uint64_t, uint64_t>>* tsOut, std::string& error);
    uint32_t createSyncobj();
    void destroySyncobj(uint32_t h);
    // Returns 0 on success, ETIME on timeout, another errno on failure.
    int waitSyncobj(uint32_t h, uint64_t timeoutNs);
    void resetSyncobj(uint32_t h);
    uint64_t gpuTimeNs();

private:
    DrmDevice() = default;
    bool init(int fd, std::string& error);
    int fd_ = -1;
    GpuInfo info_;
    uint32_t vm_ = 0;
    uint64_t uscBase_ = 0;
    VaHeap uscHeap_, dataHeap_;
    uint64_t allocated_ = 0;
    std::mutex m_;
    // Timestamp slots (one BO bound as a timestamp object).
    Bo* tsBo_ = nullptr;
    uint32_t tsObject_ = 0;
    std::mutex tsMutex_;
};

} // namespace masahi
