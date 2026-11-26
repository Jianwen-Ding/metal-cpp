// Test harness for metal-cpp on Asahi Linux. Each test program is an ordinary
// metal-cpp program (the same code would run on macOS with a .metallib built
// from the same kernels); TEST() cases run in order and report PASS / FAIL.
#pragma once

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace t {

struct Case {
    const char* name;
    void (*fn)();
};
inline std::vector<Case>& cases()
{
    static std::vector<Case> c;
    return c;
}
struct Reg {
    Reg(const char* n, void (*f)()) { cases().push_back({n, f}); }
};
inline int& failures()
{
    static int f = 0;
    return f;
}
inline bool& caseFailed()
{
    static bool f = false;
    return f;
}

#define TEST(name)                                     \
    static void test_##name();                         \
    static t::Reg reg_##name(#name, test_##name);      \
    static void test_##name()

#define CHECK(cond)                                                                                 \
    do {                                                                                            \
        if (!(cond)) {                                                                              \
            fprintf(stderr, "  CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
            t::caseFailed() = true;                                                                 \
            return;                                                                                 \
        }                                                                                           \
    } while (0)

#define CHECK_MSG(cond, ...)                                                                        \
    do {                                                                                            \
        if (!(cond)) {                                                                              \
            fprintf(stderr, "  CHECK failed at %s:%d: %s: ", __FILE__, __LINE__, #cond);            \
            fprintf(stderr, __VA_ARGS__);                                                           \
            fputc('\n', stderr);                                                                    \
            t::caseFailed() = true;                                                                 \
            return;                                                                                 \
        }                                                                                           \
    } while (0)

inline MTL::Device* device()
{
    static MTL::Device* d = MTL::CreateSystemDefaultDevice();
    return d;
}

inline MTL::CommandQueue* queue()
{
    static MTL::CommandQueue* q = device()->newCommandQueue();
    return q;
}

inline std::string kernelPath(const char* file)
{
    const char* dir = getenv("KERNEL_DIR");
    return std::string(dir ? dir : "build/kernels") + "/" + file;
}

// Loads (and caches) the library built from kernels/<file>.
inline MTL::Library* library(const char* file)
{
    static std::vector<std::pair<std::string, MTL::Library*>> cache;
    for (auto& c : cache)
        if (c.first == file) return c.second;
    NS::Error* err = nullptr;
    NS::String* path = NS::String::string(kernelPath(file).c_str(), NS::UTF8StringEncoding);
    NS::URL* url = NS::URL::fileURLWithPath(path);
    MTL::Library* lib = device()->newLibrary(url, &err);
    if (!lib) fprintf(stderr, "  newLibrary(%s): %s\n", file, err ? err->localizedDescription()->utf8String() : "?");
    cache.push_back({file, lib});
    return lib;
}

inline MTL::ComputePipelineState* pipeline(const char* file, const char* fn)
{
    static std::vector<std::pair<std::string, MTL::ComputePipelineState*>> cache;
    std::string key = std::string(file) + ":" + fn;
    for (auto& c : cache)
        if (c.first == key) return c.second;
    MTL::Library* lib = library(file);
    if (!lib) return nullptr;
    MTL::Function* f = lib->newFunction(NS::String::string(fn, NS::UTF8StringEncoding));
    if (!f) {
        fprintf(stderr, "  no function %s in %s\n", fn, file);
        return nullptr;
    }
    NS::Error* err = nullptr;
    MTL::ComputePipelineState* p = device()->newComputePipelineState(f, &err);
    if (!p) fprintf(stderr, "  pipeline %s: %s\n", fn, err ? err->localizedDescription()->utf8String() : "?");
    f->release();
    cache.push_back({key, p});
    return p;
}

template <typename T> inline MTL::Buffer* buffer(const std::vector<T>& v)
{
    return device()->newBuffer(v.data(), v.size() * sizeof(T), MTL::ResourceStorageModeShared);
}
inline MTL::Buffer* zeros(size_t bytes)
{
    MTL::Buffer* b = device()->newBuffer(bytes, MTL::ResourceStorageModeShared);
    memset(b->contents(), 0, bytes);
    return b;
}
template <typename T> inline T* data(MTL::Buffer* b) { return static_cast<T*>(b->contents()); }

// Runs a command buffer to completion; returns false (and prints the error) if it failed.
inline bool finish(MTL::CommandBuffer* cb)
{
    cb->commit();
    cb->waitUntilCompleted();
    if (cb->status() != MTL::CommandBufferStatusCompleted) {
        NS::Error* e = cb->error();
        fprintf(stderr, "  command buffer status %d: %s\n", (int)cb->status(), e ? e->localizedDescription()->utf8String() : "?");
        return false;
    }
    return true;
}

// One-dispatch helper: binds buffers (and inline bytes) at consecutive indices from 0.
struct Arg {
    MTL::Buffer* buf = nullptr;
    std::vector<uint8_t> bytes;
    size_t offset = 0;
    Arg(MTL::Buffer* b, size_t off = 0) : buf(b), offset(off) {}
    template <typename T> static Arg value(const T& v)
    {
        Arg a(nullptr);
        a.bytes.assign((const uint8_t*)&v, (const uint8_t*)&v + sizeof(T));
        return a;
    }
};

inline bool dispatch(MTL::ComputePipelineState* p, const std::vector<Arg>& args, MTL::Size grid, MTL::Size tg,
                     size_t tgMemory = 0, bool groups = false)
{
    if (!p) return false;
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::CommandBuffer* cb = queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(p);
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i].buf) enc->setBuffer(args[i].buf, args[i].offset, i);
        else enc->setBytes(args[i].bytes.data(), args[i].bytes.size(), i);
    }
    if (tgMemory) enc->setThreadgroupMemoryLength(tgMemory, 0);
    if (groups) enc->dispatchThreadgroups(grid, tg);
    else enc->dispatchThreads(grid, tg);
    enc->endEncoding();
    bool ok = finish(cb);
    pool->release();
    return ok;
}

inline bool near(float a, float b, float rel = 1e-5f, float abs_ = 1e-6f)
{
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    return std::fabs(a - b) <= abs_ + rel * std::fabs(b);
}

inline int runAll(int argc, char** argv)
{
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    if (!device()) {
        printf("FAIL: no Metal device\n");
        return 1;
    }
    const char* only = argc > 1 ? argv[1] : nullptr;
    int run = 0;
    for (auto& c : cases()) {
        if (only && !strstr(c.name, only)) continue;
        caseFailed() = false;
        auto t0 = std::chrono::steady_clock::now();
        NS::AutoreleasePool* p = NS::AutoreleasePool::alloc()->init();
        c.fn();
        p->release();
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        printf("%s: %s (%.1f ms)\n", caseFailed() ? "FAIL" : "PASS", c.name, ms);
        fflush(stdout);
        if (caseFailed()) failures()++;
        run++;
    }
    pool->release();
    printf("%d/%d passed\n", run - failures(), run);
    return failures() ? 1 : 0;
}

} // namespace t

#define TEST_MAIN \
    int main(int argc, char** argv) { return t::runAll(argc, argv); }
