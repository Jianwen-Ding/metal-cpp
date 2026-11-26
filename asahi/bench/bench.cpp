// Benchmarks for libmetal_asahi: compile time, dispatch overhead, bandwidth, GEMM/GEMV.
// Run with METAL_ASAHI_MODE=hw for timings (sizes are above the simulator's limits anyway);
// `bench --small` runs scaled-down problems suitable for METAL_ASAHI_MODE=verify.
#include "../tests/harness.h"
#include <sys/resource.h>

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }
static bool small = false;

static double gpuTime(MTL::CommandBuffer* cb) { return cb->GPUEndTime() - cb->GPUStartTime(); }

struct Enc {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    ~Enc() { pool->release(); }
};

static double runOnce(MTL::ComputePipelineState* p, const std::vector<t::Arg>& args, MTL::Size grid, MTL::Size tg,
                      bool groups = false, int reps = 1, double* wall = nullptr)
{
    Enc E;
    for (int r = 0; r < reps; r++) {
        E.e->setComputePipelineState(p);
        for (size_t i = 0; i < args.size(); i++) {
            if (args[i].buf) E.e->setBuffer(args[i].buf, args[i].offset, i);
            else E.e->setBytes(args[i].bytes.data(), args[i].bytes.size(), i);
        }
        if (groups) E.e->dispatchThreadgroups(grid, tg);
        else E.e->dispatchThreads(grid, tg);
    }
    E.e->endEncoding();
    auto t0 = clk::now();
    E.cb->commit();
    E.cb->waitUntilCompleted();
    if (wall) *wall = secs(t0);
    if (E.cb->status() != MTL::CommandBufferStatusCompleted) {
        fprintf(stderr, "command buffer failed\n");
        exit(1);
    }
    return gpuTime(E.cb);
}

static void benchCompile()
{
    const char* files[] = {"basic.air", "math.air", "memory.air", "workloads.air", "bench.air"};
    for (auto f : files) {
        auto t0 = clk::now();
        MTL::Library* lib = t::library(f);
        double load = secs(t0);
        NS::Array* names = lib->functionNames();
        t0 = clk::now();
        for (NS::UInteger i = 0; i < names->count(); i++) {
            auto* n = static_cast<NS::String*>(names->object(i));
            t::pipeline(f, n->utf8String());
        }
        printf("compile %-14s load %7.1f ms, %2lu pipelines %7.1f ms\n", f, load * 1e3, (unsigned long)names->count(),
               secs(t0) * 1e3);
    }
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    printf("compile max RSS %.1f MB\n", ru.ru_maxrss / 1024.0);
}

static void benchDispatch()
{
    auto* p = t::pipeline("bench.air", "empty");
    MTL::Buffer* b = t::zeros(4096);
    double wall;
    runOnce(p, {b}, MTL::Size(32, 1, 1), MTL::Size(32, 1, 1), false, 1, &wall); // warm
    const int N = small ? 5 : 200;
    auto t0 = clk::now();
    for (int i = 0; i < N; i++) runOnce(p, {b}, MTL::Size(32, 1, 1), MTL::Size(32, 1, 1));
    printf("dispatch: 1 dispatch/command buffer round trip %.1f us\n", secs(t0) / N * 1e6);
    for (int reps : {16, 256, 1024}) {
        if (small && reps > 16) break;
        double g = runOnce(p, {b}, MTL::Size(32, 1, 1), MTL::Size(32, 1, 1), false, reps, &wall);
        printf("dispatch: %4d dispatches/command buffer: wall %.2f us/dispatch, gpu %.2f us/dispatch\n", reps,
               wall / reps * 1e6, g / reps * 1e6);
    }
    // Many command buffers in flight: committed back to back, waiting on the last only.
    for (int ncb : {16, 128}) {
        if (small && ncb > 16) break;
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        std::vector<MTL::CommandBuffer*> cbs;
        auto t1 = clk::now();
        for (int i = 0; i < ncb; i++) {
            MTL::CommandBuffer* cb = t::queue()->commandBuffer();
            auto* e = cb->computeCommandEncoder();
            e->setComputePipelineState(p);
            e->setBuffer(b, 0, 0);
            e->dispatchThreads(MTL::Size(32, 1, 1), MTL::Size(32, 1, 1));
            e->endEncoding();
            cb->commit();
            cbs.push_back(cb);
        }
        cbs.back()->waitUntilCompleted();
        double w = secs(t1);
        for (auto* cb : cbs)
            if (cb->status() != MTL::CommandBufferStatusCompleted) fprintf(stderr, "command buffer failed\n");
        printf("dispatch: %4d command buffers in flight (1 dispatch each): %.1f us/command buffer\n", ncb, w / ncb * 1e6);
        pool->release();
    }
    b->release();
}

static void benchBandwidth()
{
    size_t n = small ? (1 << 16) : (64u << 20); // float4s: 1 GiB in each direction
    MTL::Buffer* in = t::zeros(n * 16);
    MTL::Buffer* out = t::zeros(n * 16);
    auto* p = t::pipeline("workloads.air", "stream_copy");
    uint32_t nn = (uint32_t)n;
    for (size_t threads : {size_t(1) << 14, size_t(1) << 16, size_t(1) << 18}) {
        if (small) threads = 1024;
        runOnce(p, {in, out, t::Arg::value(nn)}, MTL::Size(threads, 1, 1), MTL::Size(256, 1, 1));
        double g = runOnce(p, {in, out, t::Arg::value(nn)}, MTL::Size(threads, 1, 1), MTL::Size(256, 1, 1));
        printf("stream_copy %zu threads: %.1f GB/s\n", threads, 2.0 * n * 16 / g / 1e9);
        if (small) break;
    }
    auto* c = t::pipeline("bench.air", "copy4");
    runOnce(c, {in, out}, MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    double g = runOnce(c, {in, out}, MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    printf("copy4 (1 float4/thread): %.1f GB/s\n", 2.0 * n * 16 / g / 1e9);
    in->release();
    out->release();
}

static void benchGemm(const char* fn, uint32_t M, uint32_t N, uint32_t K, MTL::Size tg, MTL::Size groups)
{
    std::vector<float> A(size_t(M) * K), B(size_t(K) * N);
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> d(-1, 1);
    for (auto& x : A) x = d(rng);
    for (auto& x : B) x = d(rng);
    MTL::Buffer *a = t::buffer(A), *b = t::buffer(B), *c = t::zeros(size_t(M) * N * 4);
    auto* p = t::pipeline("bench.air", fn);
    if (!p) p = t::pipeline("workloads.air", fn);
    std::vector<t::Arg> args = {a, b, c, t::Arg::value(M), t::Arg::value(N), t::Arg::value(K)};
    runOnce(p, args, groups, tg, true);
    double g = 1e9;
    for (int i = 0; i < 3; i++) g = std::min(g, runOnce(p, args, groups, tg, true));
    // spot check
    float* C = t::data<float>(c);
    double maxErr = 0;
    for (int s = 0; s < 64; s++) {
        uint32_t i = rng() % M, j = rng() % N;
        double ref = 0;
        for (uint32_t k = 0; k < K; k++) ref += double(A[size_t(i) * K + k]) * B[size_t(k) * N + j];
        maxErr = std::max(maxErr, std::fabs(ref - C[size_t(i) * N + j]));
    }
    printf("%-14s %ux%ux%u: %.3f ms, %.1f GFLOP/s (max err %.2g)\n", fn, M, N, K, g * 1e3, 2.0 * M * N * K / g / 1e9,
           maxErr);
    a->release();
    b->release();
    c->release();
}

int main(int argc, char** argv)
{
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    std::string only;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--small")) small = true;
        else only = argv[i];
    }
    auto want = [&](const char* s) { return only.empty() || only == s; };
    if (want("compile")) benchCompile();
    if (want("dispatch")) benchDispatch();
    if (want("bandwidth")) benchBandwidth();
    if (want("gemm")) {
        uint32_t S = small ? 64 : 1024;
        benchGemm("sgemm_naive", S, S, S, MTL::Size(16, 16, 1), MTL::Size(S / 16, S / 16, 1));
        benchGemm("sgemm_tiled", S, S, S, MTL::Size(16, 16, 1), MTL::Size(S / 16, S / 16, 1));
        benchGemm("sgemm_reg4x4", S, S, S, MTL::Size(16, 16, 1), MTL::Size(S / 64, S / 64, 1));
        benchGemm("sgemm_simd", S, S, S, MTL::Size(128, 1, 1), MTL::Size(S / 64, S / 64, 1));
        if (!small) {
            benchGemm("sgemm_simd", 2048, 2048, 2048, MTL::Size(128, 1, 1), MTL::Size(32, 32, 1));
            benchGemm("sgemm_simd", 4096, 4096, 4096, MTL::Size(128, 1, 1), MTL::Size(64, 64, 1));
        }
        benchGemm("sgemm_reg4x4v", S, S, S, MTL::Size(16, 16, 1), MTL::Size(S / 64, S / 64, 1));
        if (!small) {
            benchGemm("sgemm_reg8x8", S, S, S, MTL::Size(16, 16, 1), MTL::Size(S / 128, S / 128, 1));
            benchGemm("sgemm_reg4x4v", 2048, 2048, 2048, MTL::Size(16, 16, 1), MTL::Size(32, 32, 1));
            benchGemm("sgemm_reg8x8", 2048, 2048, 2048, MTL::Size(16, 16, 1), MTL::Size(16, 16, 1));
            benchGemm("sgemm_reg8x8", 4096, 4096, 4096, MTL::Size(16, 16, 1), MTL::Size(32, 32, 1));
        }
        if (!small) benchGemm("sgemm_reg4x4", 2048, 2048, 2048, MTL::Size(16, 16, 1), MTL::Size(32, 32, 1));
    }
    pool->release();
    return 0;
}
