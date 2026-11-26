// libmetal_asahi: Metal objects (MTLDevice, MTLBuffer, ...) over the Asahi DRM device.
#pragma once
#include "compiler.h"
#include "drm.h"
#include "rt.h"

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace masahi {

enum class ExecMode { Hardware, Simulator, Preflight, Verify };
ExecMode execMode();

struct MDevice;

// Bump allocator over BO chunks, for per-command-buffer upload data
// (uniforms, CDM streams, pipeline words, setBytes data).
class Arena {
public:
    Arena(MDevice* dev, bool usc) : dev_(dev), usc_(usc) {}
    ~Arena();
    // Returns CPU pointer and VA.
    uint8_t* alloc(size_t size, size_t align, uint64_t& va);
    void reset(); // keeps the first chunk
private:
    MDevice* dev_;
    bool usc_;
    std::vector<Bo*> chunks_;
    size_t used_ = 0;
};

struct MBuffer : Object {
    MDevice* dev = nullptr;
    Bo* bo = nullptr;
    uint64_t length = 0;
    uint64_t options = 0;
    std::string label;
    // newBufferWithBytesNoCopy: the client's pages, remapped onto the BO.
    void* noCopyPtr = nullptr;
    void* deallocator = nullptr; // block
    ~MBuffer();
};

struct MLibrary;
struct MFunction : Object {
    Ref<Object> library; // MLibrary
    MLibrary* lib = nullptr;
    size_t module = 0;
    const CompiledKernel* kernel = nullptr;
    std::string name;
    std::string label;
};

struct LibModule {
    std::vector<uint8_t> bitcode;
    CompiledModule compiled;
    Bo* globals = nullptr; // placed global data buffer, relocated
};

struct MLibrary : Object {
    MDevice* dev = nullptr;
    std::vector<std::unique_ptr<LibModule>> modules;
    std::vector<std::pair<std::string, std::pair<size_t, size_t>>> functions; // name -> (module, kernel)
    std::string label;
    ~MLibrary();
};

struct MPipeline : Object {
    MDevice* dev = nullptr;
    Ref<Object> function; // MFunction
    MFunction* fn = nullptr;
    uint32_t codeOffset = 0; // USC-relative
    uint32_t maxThreads = 1024;
    std::string label;
    bool simVerified = false;
};

// One dispatch as recorded: everything needed to run it later.
struct Binding {
    Ref<Object> buffer; // MBuffer, or empty
    uint64_t offset = 0;
    std::vector<uint8_t> bytes; // setBytes
    bool bound = false;
};

struct DispatchCmd {
    Ref<Object> pipeline; // MPipeline
    MPipeline* pso = nullptr;
    std::vector<std::pair<uint32_t, Binding>> bindings; // location -> binding (only the kernel's)
    std::vector<uint32_t> tgLengths;                    // by threadgroup index
    uint32_t grid[3] = {1, 1, 1};                       // threads
    uint32_t local[3] = {1, 1, 1};
    std::vector<Ref<Object>> used;                      // useResource(s): mapped for the simulator
    std::string label;
};

struct BlitCmd {
    enum Kind { Copy, Fill } kind = Copy;
    Ref<Object> src, dst;
    uint64_t srcOff = 0, dstOff = 0, size = 0;
    uint8_t value = 0;
};

struct EventCmd {
    Ref<Object> event;
    uint64_t value = 0;
    bool wait = false;
};

struct Command {
    enum Kind { Dispatch, Blit, Event, Barrier } kind;
    std::unique_ptr<DispatchCmd> dispatch;
    std::unique_ptr<BlitCmd> blit;
    std::unique_ptr<EventCmd> event;
};

struct MQueue;
struct MCommandBuffer : Object {
    MQueue* queue = nullptr;
    Ref<Object> queueRef;
    std::vector<Command> commands;
    std::vector<void*> completedHandlers, scheduledHandlers; // copied blocks
    std::mutex m;
    std::condition_variable cv;
    int status = 0; // MTL::CommandBufferStatus
    Ref<Object> error;
    double gpuStart = 0, gpuEnd = 0, kernelStart = 0, kernelEnd = 0;
    std::string label;
    bool retainedReferences = true;
    Ref<Object> currentEncoder;
    ~MCommandBuffer();
};

struct MEncoder : Object {
    MCommandBuffer* cb = nullptr;
    bool ended = false;
    std::string label;
    // compute state
    Ref<Object> pipeline;
    std::map<uint32_t, Binding> bindings;
    std::map<uint32_t, uint32_t> tgLengths;
    std::vector<Ref<Object>> used;
    bool concurrent = false;
};

struct MEvent : Object {
    MDevice* dev = nullptr;
    std::mutex m;
    std::condition_variable cv;
    uint64_t value = 0;
    std::string label;
    bool shared = false;
};

struct MQueue : Object {
    MDevice* dev = nullptr;
    Ref<Object> devRef;
    uint32_t drmQueue = ~0u;
    uint32_t syncobj = 0;
    std::string label;
    std::mutex m;
    std::condition_variable cv;
    std::deque<MCommandBuffer*> pending; // retained
    std::thread worker;
    bool stop = false;
    std::unique_ptr<Arena> data, usc;
    ~MQueue();
    void run();
};

struct MDevice : Object {
    DrmDevice* drm = nullptr;
    std::string error;
    std::string name;
    CompileOptions compileOpts;
    std::mutex uscMutex;
    std::unique_ptr<Arena> codeArena; // pipeline code (USC)
    uint64_t registryID = 0x4153414849ull;
};

MDevice* defaultDevice(); // the process-wide device (+0)

} // namespace masahi
