// libmetal_asahi: MTLDevice, MTLBuffer, MTLLibrary, MTLFunction,
// MTLComputePipelineState, descriptors and events.
#include "metal.h"

#include <thread>

#include <atomic>
#include "sim.h"

#include <dispatch/dispatch.h>

#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

namespace masahi {

// METAL_ASAHI_DEVICE=sim: a device without a GPU (every dispatch runs on the simulator).
bool simulatedDeviceRequested()
{
    const char* v = getenv("METAL_ASAHI_DEVICE");
    return v && (!strcmp(v, "sim") || !strcmp(v, "simulator"));
}

ExecMode execMode()
{
    static ExecMode m = [] {
        const char* v = getenv("METAL_ASAHI_MODE");
        std::string s = v ? v : "";
        if (simulatedDeviceRequested() && s != "sim" && s != "simulator")
            warn("METAL_ASAHI_DEVICE=sim has no GPU: METAL_ASAHI_MODE=%s runs as sim", s.c_str());
        if (simulatedDeviceRequested()) return ExecMode::Simulator;
        if (s == "hw" || s == "hardware") return ExecMode::Hardware;
        if (s == "sim" || s == "simulator") return ExecMode::Simulator;
        if (s == "verify") return ExecMode::Verify;
        if (s.empty() || s == "preflight") {
            if (simulatedDeviceRequested()) return ExecMode::Simulator; // no GPU to run on
            return ExecMode::Preflight;
        }
        warn("unknown METAL_ASAHI_MODE '%s' (hw, sim, preflight, verify); using preflight", s.c_str());
        return ExecMode::Preflight;
    }();
    return m;
}

// ---------------------------------------------------------------- Arena

Arena::~Arena()
{
    for (Bo* b : chunks_) dev_->drm ? dev_->drm->free(b) : (void)0;
}

uint8_t* Arena::alloc(size_t size, size_t align, uint64_t& va)
{
    const size_t chunk = 4u << 20;
    if (!chunks_.empty()) {
        Bo* b = chunks_.back();
        size_t off = (used_ + align - 1) & ~(align - 1);
        if (off + size <= b->size) {
            used_ = off + size;
            va = b->va + off;
            return b->map + off;
        }
    }
    Bo* b = dev_->drm->alloc(std::max(chunk, size + align), usc_);
    if (!b) return nullptr;
    chunks_.push_back(b);
    used_ = size;
    va = b->va;
    return b->map;
}

void Arena::reset()
{
    while (chunks_.size() > 1) {
        dev_->drm->free(chunks_.back());
        chunks_.pop_back();
    }
    used_ = 0;
}

// ---------------------------------------------------------------- objects

MBuffer::~MBuffer()
{
    if (noCopyPtr && bo && dev && dev->drm && dev->drm->simulated()) {
        // the buffer is the client's memory itself: nothing to unmap or copy back
        if (deallocator) {
            callBlock(deallocator, noCopyPtr, (uintptr_t)length);
            _Block_release(deallocator);
        }
        bo->map = nullptr;
    } else if (noCopyPtr && bo) {
        // Give the client back ordinary memory with the final contents at its address.
        std::vector<uint8_t> tmp(bo->map, bo->map + length);
        void* p = mmap(noCopyPtr, (length + 16383) & ~16383ull, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
        if (p == noCopyPtr) memcpy(noCopyPtr, tmp.data(), length);
        if (deallocator) {
            callBlock(deallocator, noCopyPtr, (uintptr_t)length);
            _Block_release(deallocator);
        }
    }
    if (bo && dev && dev->drm) dev->drm->free(bo);
}

MLibrary::~MLibrary()
{
    for (auto& m : modules)
        if (m->globals && dev && dev->drm) dev->drm->free(m->globals);
}

static Class C(const char* n) { return lookupClass(n); }

// Generic property-bag descriptor: "foo" / "setFoo:" for integer, object and double properties.
struct Desc : Object {
    std::map<std::string, uint64_t> ints;
    std::map<std::string, double> dbls;
    std::map<std::string, Ref<Object>> objs;
    std::map<uint64_t, Ref<Object>> indexed; // objectAtIndexedSubscript:
};

static std::string setterName(const std::string& p)
{
    std::string s = "set" + p + ":";
    s[3] = (char)toupper((unsigned char)s[3]);
    return s;
}

enum PropKind { PInt, PObj, PDbl };
struct PropDef {
    const char* name;
    PropKind kind;
    uint64_t def;
};

// Per-class defaults, looked up when a property was never set.
static std::map<Class, std::map<std::string, uint64_t>>& defaults()
{
    static auto* d = new std::map<Class, std::map<std::string, uint64_t>>();
    return *d;
}

static uint64_t descInt(Desc* d, const char* name)
{
    auto it = d->ints.find(name);
    if (it != d->ints.end()) return it->second;
    for (Class c = d->isa; c; c = c->superclass) {
        auto dc = defaults().find(c);
        if (dc != defaults().end()) {
            auto v = dc->second.find(name);
            if (v != dc->second.end()) return v->second;
        }
        if (c->superclass == c) break;
    }
    return 0;
}

static Object* descObj(Desc* d, const char* name)
{
    auto it = d->objs.find(name);
    return it == d->objs.end() ? nullptr : it->second.get();
}

static void defineProps(Class cls, std::initializer_list<PropDef> props)
{
    for (auto& p : props) {
        std::string name = p.name, setter = setterName(name);
        SEL getSel = sel_registerName(name.c_str());
        SEL setSel = sel_registerName(setter.c_str());
        const char* key = strdup(name.c_str());
        defaults()[cls][key] = p.def;
        // The selector string is the key; the IMPs find it from _cmd.
        (void)key;
        switch (p.kind) {
        case PInt:
            (*cls->methods)[getSel] = reinterpret_cast<IMP>(+[](Desc* d, SEL s) -> uint64_t { return descInt(d, sel_getName(s)); });
            (*cls->methods)[setSel] = reinterpret_cast<IMP>(+[](Desc* d, SEL s, uint64_t v) {
                std::string n = sel_getName(s) + 3;
                n.pop_back();
                n[0] = (char)tolower((unsigned char)n[0]);
                d->ints[n] = v;
            });
            break;
        case PDbl:
            (*cls->methods)[getSel] = reinterpret_cast<IMP>(+[](Desc* d, SEL s) -> double {
                auto it = d->dbls.find(sel_getName(s));
                return it == d->dbls.end() ? 0.0 : it->second;
            });
            (*cls->methods)[setSel] = reinterpret_cast<IMP>(+[](Desc* d, SEL s, double v) {
                std::string n = sel_getName(s) + 3;
                n.pop_back();
                n[0] = (char)tolower((unsigned char)n[0]);
                d->dbls[n] = v;
            });
            break;
        case PObj:
            (*cls->methods)[getSel] = reinterpret_cast<IMP>(+[](Desc* d, SEL s) -> Object* { return descObj(d, sel_getName(s)); });
            (*cls->methods)[setSel] = reinterpret_cast<IMP>(+[](Desc* d, SEL s, Object* v) {
                std::string n = sel_getName(s) + 3;
                n.pop_back();
                n[0] = (char)tolower((unsigned char)n[0]);
                d->objs[n] = Ref<Object>(v);
            });
            break;
        }
    }
}

static Class defineDescriptor(const char* name, std::initializer_list<PropDef> props)
{
    Class c = defineClass(name, C("NSObject"));
    defClass(c, "alloc", +[](Class k, SEL) -> Object* { return make<Desc>(k); });
    defClass(c, "new", +[](Class k, SEL) -> Object* { return make<Desc>(k); });
    defineProps(c, props);
    def(c, "copy", +[](Desc* d, SEL) -> Object* {
        Desc* n = make<Desc>(d->isa);
        n->ints = d->ints;
        n->dbls = d->dbls;
        n->objs = d->objs;
        n->indexed = d->indexed;
        return n;
    });
    def(c, "reset", +[](Desc* d, SEL) {
        d->ints.clear();
        d->dbls.clear();
        d->objs.clear();
        d->indexed.clear();
    });
    return c;
}

// ---------------------------------------------------------------- device

struct Size3 {
    uintptr_t w, h, d;
};

MDevice* defaultDevice()
{
    static MDevice* dev = [] {
        MDevice* d = make<MDevice>("MTLAsahiDevice");
        std::string err;
        const char* simGpu = getenv("METAL_ASAHI_SIM_GPU");
        std::string chip = simGpu && *simGpu ? simGpu : "agx-g14c";
        if (simulatedDeviceRequested()) {
            d->drm = DrmDevice::openSimulated(chip, err);
        } else {
            d->drm = DrmDevice::open(err);
            if (!d->drm && execMode() == ExecMode::Simulator) {
                // sim mode needs no GPU: fall back to the simulated device
                if (debugLevel() > 0) warn("%s; METAL_ASAHI_MODE=sim continues on the simulated device", err.c_str());
                err.clear();
                d->drm = DrmDevice::openSimulated(chip, err);
            }
        }
        if (!d->drm) {
            d->error = err;
            warn("%s", err.c_str());
            return (MDevice*)nullptr;
        }
        if (d->drm->simulated() && !simulatorAvailable()) {
            warn("the simulated device needs the AGX Performance Simulator's agx_run (METAL_ASAHI_AGX_RUN=%s is not executable)",
                 simulatorPath().c_str());
            return (MDevice*)nullptr;
        }
        const GpuInfo& gi = d->drm->info();
        d->name = gi.marketing;
        d->compileOpts.cpu = gi.chipName;
        if (const char* cpu = getenv("METAL_ASAHI_CPU")) d->compileOpts.cpu = cpu;
        if (const char* f = getenv("METAL_ASAHI_FEATURES")) d->compileOpts.features = f;
        if (const char* p = getenv("METAL_ASAHI_IR_PIPELINE")) d->compileOpts.irPipeline = p;
        d->codeArena = std::make_unique<Arena>(d, true);
        if (debugLevel() > 0)
            warn("device %s: %s, %u cores, %u clusters, %u MHz, mode %d", d->name.c_str(), gi.chipName.c_str(), gi.cores, gi.clusters,
                 gi.maxKhz / 1000, (int)execMode());
        return d;
    }();
    return dev;
}

static uint64_t physicalMemory() { return (uint64_t)sysconf(_SC_PHYS_PAGES) * (uint64_t)sysconf(_SC_PAGESIZE); }

static MBuffer* newBuffer(MDevice* d, uint64_t length, uint64_t options)
{
    MBuffer* b = make<MBuffer>("MTLAsahiBuffer");
    b->dev = d;
    b->length = length;
    b->options = options;
    b->bo = d->drm->alloc(length ? length : 1, false);
    if (!b->bo) {
        b->destroy(b);
        return nullptr;
    }
    return b;
}

// Loads every AIR module of a file's bytes into a library and compiles it.
static MLibrary* newLibraryFromBytes(MDevice* d, const uint8_t* data, size_t size, void** error)
{
    std::string err;
    auto mods = splitAIRContainer(data, size, err);
    if (mods.empty()) {
        setError(error, "MTLLibraryErrorDomain", 1, "%s", err.c_str());
        return nullptr;
    }
    // The modules of a .metallib (one per kernel) compile in parallel, each in its own LLVMContext.
    std::vector<CompiledModule> compiled(mods.size());
    {
        unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        unsigned n = (unsigned)std::min<size_t>(mods.size(), std::min(8u, hw));
        if (const char* j = getenv("METAL_ASAHI_COMPILE_THREADS")) n = (unsigned)std::max(1, atoi(j));
        std::atomic<size_t> next{0};
        auto work = [&] {
            for (size_t i; (i = next++) < mods.size();) compiled[i] = compileAIRCached(mods[i], d->compileOpts);
        };
        std::vector<std::thread> ts;
        for (unsigned t = 1; t < n; t++) ts.emplace_back(work);
        work();
        for (auto& t : ts) t.join();
    }
    MLibrary* lib = make<MLibrary>("MTLAsahiLibrary");
    lib->dev = d;
    for (size_t ci = 0; ci < mods.size(); ci++) {
        auto m = std::make_unique<LibModule>();
        m->bitcode = std::move(mods[ci]);
        m->compiled = std::move(compiled[ci]);
        if (!m->compiled.error.empty()) {
            setError(error, "MTLLibraryErrorDomain", 3, "%s", m->compiled.error.c_str());
            lib->destroy(lib);
            return nullptr;
        }
        if (debugLevel() > 0)
            warn("%s %zu kernel(s) in %.1f ms", m->compiled.fromCache ? "loaded (cache)" : "compiled", m->compiled.kernels.size(),
                 m->compiled.compileSeconds * 1e3);
        auto& cm = m->compiled;
        if (!cm.globals.empty()) {
            m->globals = d->drm->alloc(cm.globals.size(), false);
            if (!m->globals) {
                setError(error, "MTLLibraryErrorDomain", 1, "cannot allocate %zu bytes of global data", cm.globals.size());
                lib->destroy(lib);
                return nullptr;
            }
            memcpy(m->globals->map, cm.globals.data(), cm.globals.size());
            for (auto& r : cm.relocs) {
                uint64_t v = m->globals->va + r.symbolOffset + (uint64_t)r.addend;
                if (r.type == 1 && r.offset + 8 <= cm.globals.size()) memcpy(m->globals->map + r.offset, &v, 8);
                else if (r.type == 2 && r.offset + 4 <= cm.globals.size()) {
                    uint32_t v32 = (uint32_t)v;
                    memcpy(m->globals->map + r.offset, &v32, 4);
                }
            }
        }
        size_t mi = lib->modules.size();
        for (size_t k = 0; k < cm.kernels.size(); k++) lib->functions.push_back({cm.kernels[k].name, {mi, k}});
        lib->modules.push_back(std::move(m));
    }
    return lib;
}

static bool readWholeFile(const std::string& path, std::vector<uint8_t>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

static MLibrary* newLibraryFromPath(MDevice* d, const std::string& path, void** error)
{
    std::vector<uint8_t> bytes;
    if (!readWholeFile(path, bytes)) {
        setError(error, "MTLLibraryErrorDomain", 6, "cannot read library file %s", path.c_str());
        return nullptr;
    }
    MLibrary* lib = newLibraryFromBytes(d, bytes.data(), bytes.size(), error);
    if (lib) lib->label = path;
    return lib;
}

static uint32_t maxThreadsFor(const CompiledKernel& k)
{
    // Resident SIMD-groups per core: min(96, 2 * floor(3328 / (2 * H))), H = 16-bit register halves (>= 8).
    uint32_t h = std::max<uint32_t>(8, (2 * k.gprs + 7) & ~7u);
    uint32_t groups = std::min<uint32_t>(96, 2 * (3328 / (2 * h)));
    return std::min<uint32_t>(1024, groups * 32);
}

static MPipeline* newPipeline(MDevice* d, MFunction* fn, void** error)
{
    if (!fn || !fn->kernel) {
        setError(error, "MTLLibraryErrorDomain", 2, "no compute function");
        return nullptr;
    }
    const CompiledKernel& k = *fn->kernel;
    if (k.stackBytes) {
        setError(error, "MTLLibraryErrorDomain", 3,
                 "kernel %s needs %u bytes of stack per thread; stack (scratch) launches are not supported yet", k.name.c_str(),
                 k.stackBytes);
        return nullptr;
    }
    if (k.textureHeapHalf != ~0u) {
        setError(error, "MTLLibraryErrorDomain", 3, "kernel %s samples textures; textures are not supported yet", k.name.c_str());
        return nullptr;
    }
    MPipeline* p = make<MPipeline>("MTLAsahiComputePipelineState");
    p->dev = d;
    p->function = Ref<Object>(fn);
    p->fn = fn;
    p->maxThreads = maxThreadsFor(k);
    {
        std::lock_guard<std::mutex> l(d->uscMutex);
        uint64_t va;
        uint8_t* dst = d->codeArena->alloc(k.code.size() + 64, 64, va);
        if (!dst) {
            p->destroy(p);
            setError(error, "MTLLibraryErrorDomain", 1, "out of shader memory");
            return nullptr;
        }
        memcpy(dst, k.code.data(), k.code.size());
        memset(dst + k.code.size(), 0, 64);
        p->codeOffset = (uint32_t)(va - d->drm->uscBase());
    }
    return p;
}

// ---------------------------------------------------------------- registration

void registerDevice()
{
    Class obj = C("NSObject");
    Class dev = defineClass("MTLAsahiDevice", obj);
    def(dev, "name", +[](MDevice* d, SEL) -> Object* { return tempString(d->name); });
    def(dev, "registryID", +[](MDevice* d, SEL) -> uint64_t { return d->registryID; });
    def(dev, "architecture", +[](MDevice*, SEL) -> Object* { return nullptr; });
    def(dev, "maxThreadsPerThreadgroup", +[](MDevice*, SEL) -> Size3 { return {1024, 1024, 1024}; });
    def(dev, "maxThreadgroupMemoryLength", +[](MDevice*, SEL) -> uintptr_t { return 32768; });
    def(dev, "maxBufferLength", +[](MDevice*, SEL) -> uintptr_t { return physicalMemory() / 4 * 3; });
    def(dev, "recommendedMaxWorkingSetSize", +[](MDevice*, SEL) -> uint64_t { return physicalMemory() / 4 * 3; });
    def(dev, "currentAllocatedSize", +[](MDevice* d, SEL) -> uintptr_t { return d->drm->allocatedBytes(); });
    def(dev, "hasUnifiedMemory", +[](MDevice*, SEL) -> bool { return true; });
    def(dev, "isLowPower", +[](MDevice*, SEL) -> bool { return false; });
    def(dev, "isHeadless", +[](MDevice*, SEL) -> bool { return false; });
    def(dev, "isRemovable", +[](MDevice*, SEL) -> bool { return false; });
    def(dev, "location", +[](MDevice*, SEL) -> uintptr_t { return 0; });
    def(dev, "locationNumber", +[](MDevice*, SEL) -> uintptr_t { return 0; });
    def(dev, "maxTransferRate", +[](MDevice*, SEL) -> uint64_t { return 0; });
    def(dev, "peerGroupID", +[](MDevice*, SEL) -> uint64_t { return 0; });
    def(dev, "peerCount", +[](MDevice*, SEL) -> uint32_t { return 0; });
    def(dev, "peerIndex", +[](MDevice*, SEL) -> uint32_t { return 0; });
    def(dev, "argumentBuffersSupport", +[](MDevice*, SEL) -> uintptr_t { return 1; });
    def(dev, "readWriteTextureSupport", +[](MDevice*, SEL) -> uintptr_t { return 0; });
    def(dev, "supportsRaytracing", +[](MDevice*, SEL) -> bool { return false; });
    def(dev, "supportsFunctionPointers", +[](MDevice*, SEL) -> bool { return false; });
    def(dev, "supportsDynamicLibraries", +[](MDevice*, SEL) -> bool { return false; });
    def(dev, "supports32BitFloatFiltering", +[](MDevice*, SEL) -> bool { return false; });
    def(dev, "supportsBCTextureCompression", +[](MDevice*, SEL) -> bool { return false; });
    def(dev, "supportsShaderBarycentricCoordinates", +[](MDevice*, SEL) -> bool { return false; });
    def(dev, "maximumConcurrentCompilationTaskCount", +[](MDevice*, SEL) -> uintptr_t { return 1; });
    def(dev, "shouldMaximizeConcurrentCompilation", +[](MDevice*, SEL) -> bool { return false; });
    def(dev, "setShouldMaximizeConcurrentCompilation:", +[](MDevice*, SEL, bool) {});
    def(dev, "queryTimestampFrequency", +[](MDevice* d, SEL) -> uint64_t { return d->drm->info().timestampHz; });
    def(dev, "sampleTimestamps:gpuTimestamp:", +[](MDevice* d, SEL, uint64_t* cpu, uint64_t* gpu) {
        if (cpu) *cpu = (uint64_t)(CACurrentMediaTime() * 1e9);
        if (gpu) *gpu = d->drm->gpuTimeNs();
    });
    def(dev, "supportsFamily:", +[](MDevice*, SEL, intptr_t family) -> bool {
        // Apple1..Apple8 (M2 is Apple8), Common1..3, Mac2, Metal3.
        return (family >= 1001 && family <= 1008) || (family >= 3001 && family <= 3003) || family == 2002 || family == 5001;
    });
    def(dev, "supportsFeatureSet:", +[](MDevice*, SEL, uintptr_t) -> bool { return true; });
    def(dev, "supportsTextureSampleCount:", +[](MDevice*, SEL, uintptr_t n) -> bool { return n == 1; });
    def(dev, "supportsCounterSampling:", +[](MDevice*, SEL, uintptr_t) -> bool { return false; });
    def(dev, "counterSets", +[](MDevice*, SEL) -> Object* { return nullptr; });

    def(dev, "newBufferWithLength:options:", +[](MDevice* d, SEL, uintptr_t len, uintptr_t opts) -> Object* {
        return newBuffer(d, len, opts);
    });
    def(dev, "newBufferWithBytes:length:options:", +[](MDevice* d, SEL, const void* bytes, uintptr_t len, uintptr_t opts) -> Object* {
        MBuffer* b = newBuffer(d, len, opts);
        if (b && bytes && len) memcpy(b->bo->map, bytes, len);
        return b;
    });
    def(dev, "newBufferWithBytesNoCopy:length:options:deallocator:",
        +[](MDevice* d, SEL, void* ptr, uintptr_t len, uintptr_t opts, void* dealloc) -> Object* {
            MBuffer* b = newBuffer(d, len, opts);
            if (!b) return nullptr;
            if (ptr && len) memcpy(b->bo->map, ptr, len);
            // Alias the client's pages to the GPU buffer, so both see one memory. Metal requires a
            // page-aligned pointer and a whole number of pages; anything else is copied (not shared).
            if (ptr && len && d->drm->simulated()) {
                // simulated device: the buffer's memory is the client's (any alignment)
                munmap(b->bo->map, b->bo->size);
                b->bo->map = (uint8_t*)ptr;
                b->noCopyPtr = ptr;
            } else if (ptr && ((uintptr_t)ptr & 16383) == 0 && (len & 16383) == 0 && len) {
                void* p = mmap(ptr, len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, d->drm->fd(), (off_t)b->bo->mmapOffset);
                if (p == ptr) b->noCopyPtr = ptr;
                else warn("newBufferWithBytesNoCopy: cannot alias %p (%s); the buffer is a copy", ptr, strerror(errno));
            } else if (ptr) {
                warn("newBufferWithBytesNoCopy: %p/%zu is not page aligned (16 KiB); the buffer is a copy", ptr, (size_t)len);
            }
            if (dealloc) {
                b->deallocator = _Block_copy(dealloc);
            }
            return b;
        });
    def(dev, "newCommandQueue", +[](MDevice* d, SEL) -> Object* {
        extern Object* newQueue(MDevice*, uintptr_t);
        return newQueue(d, 64);
    });
    def(dev, "newCommandQueueWithMaxCommandBufferCount:", +[](MDevice* d, SEL, uintptr_t n) -> Object* {
        extern Object* newQueue(MDevice*, uintptr_t);
        return newQueue(d, n);
    });
    def(dev, "newCommandQueueWithDescriptor:", +[](MDevice* d, SEL, Desc* desc) -> Object* {
        extern Object* newQueue(MDevice*, uintptr_t);
        return newQueue(d, desc ? descInt(desc, "maxCommandBufferCount") : 64);
    });
    def(dev, "newLibraryWithURL:error:", +[](MDevice* d, SEL, Object* url, void** error) -> Object* {
        auto get = reinterpret_cast<const char* (*)(Object*, SEL)>(objc_msgSend);
        const char* path = url ? get(url, sel_registerName("fileSystemRepresentation")) : nullptr;
        if (!path) {
            setError(error, "MTLLibraryErrorDomain", 6, "no URL");
            return nullptr;
        }
        return newLibraryFromPath(d, path, error);
    });
    def(dev, "newLibraryWithFile:error:", +[](MDevice* d, SEL, Object* path, void** error) -> Object* {
        return newLibraryFromPath(d, str(path), error);
    });
    def(dev, "newLibraryWithData:error:", +[](MDevice* d, SEL, dispatch_data_t data, void** error) -> Object* {
        if (!data) {
            setError(error, "MTLLibraryErrorDomain", 1, "no data");
            return nullptr;
        }
        return newLibraryFromBytes(d, (const uint8_t*)metal_asahi_dispatch_data_bytes(data), dispatch_data_get_size(data), error);
    });
    def(dev, "newLibraryWithSource:options:error:", +[](MDevice*, SEL, Object*, Object*, void** error) -> Object* {
        setError(error, "MTLLibraryErrorDomain", 3,
                 "Metal Shading Language source is not supported on Asahi Linux: compile kernels to AIR (.air or .metallib) "
                 "and load them with newLibrary(URL)");
        return nullptr;
    });
    def(dev, "newDefaultLibrary", +[](MDevice* d, SEL) -> Object* {
        char exe[4096];
        ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
        std::string dir = n > 0 ? std::string(exe, (size_t)n) : ".";
        dir = dir.substr(0, dir.find_last_of('/'));
        for (const char* f : {"/default.metallib", "/default.air"}) {
            if (access((dir + f).c_str(), R_OK) == 0) return newLibraryFromPath(d, dir + f, nullptr);
        }
        if (debugLevel() > 0) warn("newDefaultLibrary: no default.metallib or default.air next to the executable");
        return nullptr;
    });
    def(dev, "newComputePipelineStateWithFunction:error:", +[](MDevice* d, SEL, MFunction* fn, void** error) -> Object* {
        return newPipeline(d, fn, error);
    });
    def(dev, "newComputePipelineStateWithFunction:options:reflection:error:",
        +[](MDevice* d, SEL, MFunction* fn, uintptr_t, void** reflection, void** error) -> Object* {
            if (reflection) *reflection = nullptr;
            return newPipeline(d, fn, error);
        });
    def(dev, "newComputePipelineStateWithDescriptor:options:reflection:error:",
        +[](MDevice* d, SEL, Desc* desc, uintptr_t, void** reflection, void** error) -> Object* {
            if (reflection) *reflection = nullptr;
            MPipeline* p = newPipeline(d, desc ? (MFunction*)descObj(desc, "computeFunction") : nullptr, error);
            if (p && desc) {
                uint64_t mt = descInt(desc, "maxTotalThreadsPerThreadgroup");
                if (mt && mt < p->maxThreads) p->maxThreads = (uint32_t)mt;
                if (Object* l = descObj(desc, "label")) p->label = str(l);
            }
            return p;
        });
    def(dev, "newComputePipelineStateWithFunction:completionHandler:", +[](MDevice* d, SEL, MFunction* fn, void* block) {
        void* err = nullptr;
        MPipeline* p = newPipeline(d, fn, &err);
        callBlock(block, (Object*)p, (Object*)err);
        release(p);
    });
    def(dev, "newEvent", +[](MDevice* d, SEL) -> Object* {
        MEvent* e = make<MEvent>("MTLAsahiEvent");
        e->dev = d;
        return e;
    });
    def(dev, "newSharedEvent", +[](MDevice* d, SEL) -> Object* {
        MEvent* e = make<MEvent>("MTLAsahiEvent");
        e->dev = d;
        e->shared = true;
        return e;
    });
    def(dev, "newFence", +[](MDevice* d, SEL) -> Object* {
        MEvent* e = make<MEvent>("MTLAsahiEvent");
        e->dev = d;
        return e;
    });
    def(dev, "heapBufferSizeAndAlignWithLength:options:", +[](MDevice*, SEL, uintptr_t len, uintptr_t) -> Size3 {
        return {(len + 16383) & ~16383ul, 16384, 0};
    });

    // MTLBuffer
    Class buf = defineClass("MTLAsahiBuffer", obj);
    def(buf, "contents", +[](MBuffer* b, SEL) -> void* {
        return (b->options & 0xf0) == 0x20 ? nullptr : b->bo->map; // private storage has no CPU view (as Metal)
    });
    def(buf, "length", +[](MBuffer* b, SEL) -> uintptr_t { return b->length; });
    def(buf, "allocatedSize", +[](MBuffer* b, SEL) -> uintptr_t { return b->bo->size; });
    def(buf, "gpuAddress", +[](MBuffer* b, SEL) -> uint64_t { return b->bo->va; });
    def(buf, "didModifyRange:", +[](MBuffer*, SEL, uintptr_t, uintptr_t) {});
    def(buf, "device", +[](MBuffer* b, SEL) -> Object* { return b->dev; });
    def(buf, "label", +[](MBuffer* b, SEL) -> Object* { return b->label.empty() ? nullptr : tempString(b->label); });
    def(buf, "setLabel:", +[](MBuffer* b, SEL, Object* l) { b->label = str(l); });
    def(buf, "storageMode", +[](MBuffer* b, SEL) -> uintptr_t { return (b->options >> 4) & 0xf; });
    def(buf, "cpuCacheMode", +[](MBuffer* b, SEL) -> uintptr_t { return b->options & 0xf; });
    def(buf, "hazardTrackingMode", +[](MBuffer* b, SEL) -> uintptr_t { return (b->options >> 8) & 0x3; });
    def(buf, "resourceOptions", +[](MBuffer* b, SEL) -> uintptr_t { return b->options; });
    def(buf, "setPurgeableState:", +[](MBuffer*, SEL, uintptr_t) -> uintptr_t { return 2; });
    def(buf, "heap", +[](MBuffer*, SEL) -> Object* { return nullptr; });
    def(buf, "heapOffset", +[](MBuffer*, SEL) -> uintptr_t { return 0; });
    def(buf, "isAliasable", +[](MBuffer*, SEL) -> bool { return false; });
    def(buf, "makeAliasable", +[](MBuffer*, SEL) {});
    def(buf, "addDebugMarker:range:", +[](MBuffer*, SEL, Object*, uintptr_t, uintptr_t) {});
    def(buf, "removeAllDebugMarkers", +[](MBuffer*, SEL) {});

    // MTLLibrary
    Class lib = defineClass("MTLAsahiLibrary", obj);
    def(lib, "functionNames", +[](MLibrary* l, SEL) -> Object* {
        std::vector<Object*> names;
        for (auto& f : l->functions) names.push_back(autorelease(newString(f.first)));
        return autorelease(newArray(names));
    });
    auto newFn = +[](MLibrary* l, SEL, Object* name) -> Object* {
        std::string n = str(name);
        for (auto& f : l->functions) {
            if (f.first != n) continue;
            MFunction* fn = make<MFunction>("MTLAsahiFunction");
            fn->library = Ref<Object>(l);
            fn->lib = l;
            fn->module = f.second.first;
            fn->kernel = &l->modules[f.second.first]->compiled.kernels[f.second.second];
            fn->name = n;
            return fn;
        }
        if (debugLevel() > 0) warn("newFunctionWithName: no kernel named %s", n.c_str());
        return nullptr;
    };
    def(lib, "newFunctionWithName:", newFn);
    def(lib, "newFunctionWithName:constantValues:error:", +[](MLibrary* l, SEL s, Object* name, Object* values, void** error) -> Object* {
        auto fn = reinterpret_cast<Object* (*)(MLibrary*, SEL, Object*)>(
            class_getMethodImplementation(l->isa, sel_registerName("newFunctionWithName:")))(l, s, name);
        if (!fn) setError(error, "MTLLibraryErrorDomain", 5, "no function named %s", utf8(name));
        else if (values && debugLevel() > 0) warn("function constants are ignored (not supported yet)");
        return fn;
    });
    def(lib, "device", +[](MLibrary* l, SEL) -> Object* { return l->dev; });
    def(lib, "label", +[](MLibrary* l, SEL) -> Object* { return l->label.empty() ? nullptr : tempString(l->label); });
    def(lib, "setLabel:", +[](MLibrary* l, SEL, Object* s) { l->label = str(s); });
    def(lib, "type", +[](MLibrary*, SEL) -> intptr_t { return 0; });
    def(lib, "installName", +[](MLibrary*, SEL) -> Object* { return nullptr; });

    Class fn = defineClass("MTLAsahiFunction", obj);
    def(fn, "name", +[](MFunction* f, SEL) -> Object* { return tempString(f->name); });
    def(fn, "functionType", +[](MFunction*, SEL) -> uintptr_t { return 3; }); // MTLFunctionTypeKernel
    def(fn, "device", +[](MFunction* f, SEL) -> Object* { return f->lib->dev; });
    def(fn, "label", +[](MFunction* f, SEL) -> Object* { return f->label.empty() ? nullptr : tempString(f->label); });
    def(fn, "setLabel:", +[](MFunction* f, SEL, Object* s) { f->label = str(s); });
    def(fn, "options", +[](MFunction*, SEL) -> uintptr_t { return 0; });
    def(fn, "functionConstantsDictionary", +[](MFunction*, SEL) -> Object* { return nullptr; });

    Class pso = defineClass("MTLAsahiComputePipelineState", obj);
    def(pso, "maxTotalThreadsPerThreadgroup", +[](MPipeline* p, SEL) -> uintptr_t { return p->maxThreads; });
    def(pso, "threadExecutionWidth", +[](MPipeline*, SEL) -> uintptr_t { return 32; });
    def(pso, "staticThreadgroupMemoryLength", +[](MPipeline* p, SEL) -> uintptr_t { return p->fn->kernel->threadgroupBytes; });
    def(pso, "device", +[](MPipeline* p, SEL) -> Object* { return p->dev; });
    def(pso, "label", +[](MPipeline* p, SEL) -> Object* { return p->label.empty() ? nullptr : tempString(p->label); });
    def(pso, "supportIndirectCommandBuffers", +[](MPipeline*, SEL) -> bool { return false; });
    def(pso, "imageblockMemoryLengthForDimensions:", +[](MPipeline*, SEL, uintptr_t, uintptr_t, uintptr_t) -> uintptr_t { return 0; });
    struct ResourceID {
        uint64_t v;
    };
    def(pso, "gpuResourceID", +[](MPipeline* p, SEL) -> ResourceID { return {(uint64_t)p->codeOffset}; });

    Class ev = defineClass("MTLAsahiEvent", obj);
    def(ev, "device", +[](MEvent* e, SEL) -> Object* { return e->dev; });
    def(ev, "label", +[](MEvent* e, SEL) -> Object* { return e->label.empty() ? nullptr : tempString(e->label); });
    def(ev, "setLabel:", +[](MEvent* e, SEL, Object* s) { e->label = str(s); });
    def(ev, "signaledValue", +[](MEvent* e, SEL) -> uint64_t {
        std::lock_guard<std::mutex> l(e->m);
        return e->value;
    });
    def(ev, "setSignaledValue:", +[](MEvent* e, SEL, uint64_t v) {
        {
            std::lock_guard<std::mutex> l(e->m);
            e->value = v;
        }
        e->cv.notify_all();
    });
    def(ev, "waitUntilSignaledValue:timeoutMS:", +[](MEvent* e, SEL, uint64_t v, uint64_t ms) -> bool {
        std::unique_lock<std::mutex> l(e->m);
        return e->cv.wait_for(l, std::chrono::milliseconds(ms), [&] { return e->value >= v; });
    });

    // Descriptors.
    defineDescriptor("MTLComputePipelineDescriptor",
                     {{"computeFunction", PObj, 0},
                      {"label", PObj, 0},
                      {"threadGroupSizeIsMultipleOfThreadExecutionWidth", PInt, 0},
                      {"maxTotalThreadsPerThreadgroup", PInt, 0},
                      {"maxCallStackDepth", PInt, 1},
                      {"supportIndirectCommandBuffers", PInt, 0},
                      {"supportAddingBinaryFunctions", PInt, 0},
                      {"stageInputDescriptor", PObj, 0},
                      {"linkedFunctions", PObj, 0},
                      {"preloadedLibraries", PObj, 0},
                      {"binaryArchives", PObj, 0},
                      {"shaderValidation", PInt, 0}});
    defineDescriptor("MTLCompileOptions",
                     {{"fastMathEnabled", PInt, 1},
                      {"mathMode", PInt, 2},
                      {"mathFloatingPointFunctions", PInt, 0},
                      {"languageVersion", PInt, 0x30001},
                      {"libraryType", PInt, 0},
                      {"installName", PObj, 0},
                      {"libraries", PObj, 0},
                      {"preprocessorMacros", PObj, 0},
                      {"preserveInvariance", PInt, 0},
                      {"optimizationLevel", PInt, 0},
                      {"maxTotalThreadsPerThreadgroup", PInt, 0},
                      {"enableLogging", PInt, 0}});
    Class cpd = defineDescriptor("MTLComputePassDescriptor", {{"dispatchType", PInt, 0}, {"sampleBufferAttachments", PObj, 0}});
    defClass(cpd, "computePassDescriptor", +[](Class k, SEL) -> Object* { return autorelease(make<Desc>(k)); });
    defineDescriptor("MTLCommandBufferDescriptor",
                     {{"retainedReferences", PInt, 1}, {"errorOptions", PInt, 0}, {"logState", PObj, 0}});
    defineDescriptor("MTLCommandQueueDescriptor", {{"maxCommandBufferCount", PInt, 64}, {"logState", PObj, 0}});
    defineDescriptor("MTLHeapDescriptor", {{"size", PInt, 0},
                                           {"storageMode", PInt, 0},
                                           {"cpuCacheMode", PInt, 0},
                                           {"hazardTrackingMode", PInt, 0},
                                           {"resourceOptions", PInt, 0},
                                           {"type", PInt, 0}});
    Class fcv = defineDescriptor("MTLFunctionConstantValues", {});
    def(fcv, "setConstantValue:type:atIndex:", +[](Desc*, SEL, const void*, uintptr_t, uintptr_t) {});
    def(fcv, "setConstantValue:type:withName:", +[](Desc*, SEL, const void*, uintptr_t, Object*) {});
    def(fcv, "setConstantValues:type:withRange:", +[](Desc*, SEL, const void*, uintptr_t, uintptr_t, uintptr_t) {});
    defineDescriptor("MTLCaptureDescriptor", {{"captureObject", PObj, 0}, {"destination", PInt, 1}, {"outputURL", PObj, 0}});
    Class cm = defineClass("MTLCaptureManager", obj);
    defClass(cm, "sharedCaptureManager", +[](Class k, SEL) -> Object* {
        static Object* m = make<Object>(k);
        return m;
    });
    def(cm, "supportsDestination:", +[](Object*, SEL, intptr_t) -> bool { return false; });
    def(cm, "startCaptureWithDescriptor:error:", +[](Object*, SEL, Object*, void** e) -> bool {
        setError(e, "MTLCaptureErrorDomain", 1, "GPU capture is not supported on Asahi Linux");
        return false;
    });
    def(cm, "stopCapture", +[](Object*, SEL) {});
    def(cm, "isCapturing", +[](Object*, SEL) -> bool { return false; });
}

} // namespace masahi

// ---------------------------------------------------------------- C entry points

using namespace masahi;

extern "C" {

__attribute__((visibility("default"))) Object* MTLCreateSystemDefaultDevice(void)
{
    MDevice* d = defaultDevice();
    return d ? retain(d) : nullptr;
}

__attribute__((visibility("default"))) Object* MTLCopyAllDevices(void)
{
    MDevice* d = defaultDevice();
    std::vector<Object*> v;
    if (d) v.push_back(d);
    return newArray(v);
}

__attribute__((visibility("default"))) Object* MTLCopyAllDevicesWithObserver(Object** observer, void*)
{
    if (observer) *observer = nullptr;
    return MTLCopyAllDevices();
}

__attribute__((visibility("default"))) void MTLRemoveDeviceObserver(Object*) {}

struct dispatch_data_s {
    std::vector<uint8_t> bytes;
};

__attribute__((visibility("default"))) dispatch_data_t dispatch_data_create(const void* buffer, size_t size, dispatch_queue_t, void* destructor)
{
    auto* d = new dispatch_data_s();
    d->bytes.assign((const uint8_t*)buffer, (const uint8_t*)buffer + size);
    if (destructor) {
        callBlock(destructor);
    }
    return d;
}
__attribute__((visibility("default"))) size_t dispatch_data_get_size(dispatch_data_t d) { return d ? d->bytes.size() : 0; }
__attribute__((visibility("default"))) const void* metal_asahi_dispatch_data_bytes(dispatch_data_t d) { return d ? d->bytes.data() : nullptr; }
__attribute__((visibility("default"))) void dispatch_release(void* o) { delete (dispatch_data_s*)o; }

} // extern "C"
