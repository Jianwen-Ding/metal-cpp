// Realistic kernels checked against CPU references.
#include "harness.h"

#include <algorithm>

static void sgemm(const char* fn, uint32_t M, uint32_t N, uint32_t K, uint32_t tile)
{
    std::vector<float> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N);
    for (size_t i = 0; i < A.size(); i++) A[i] = (float)((i * 7) % 11) * 0.125f - 0.5f;
    for (size_t i = 0; i < B.size(); i++) B[i] = (float)((i * 5) % 13) * 0.0625f - 0.25f;
    for (uint32_t r = 0; r < M; r++)
        for (uint32_t c = 0; c < N; c++) {
            float acc = 0;
            for (uint32_t k = 0; k < K; k++) acc = std::fma(A[r * K + k], B[k * N + c], acc);
            C[r * N + c] = acc;
        }
    MTL::Buffer *bA = t::buffer(A), *bB = t::buffer(B), *bC = t::zeros(C.size() * 4);
    bool groups = tile != 0;
    MTL::Size grid = groups ? MTL::Size((N + tile - 1) / tile, (M + tile - 1) / tile, 1) : MTL::Size(N, M, 1);
    MTL::Size tg = groups ? MTL::Size(tile, tile, 1) : MTL::Size(16, 8, 1);
    CHECK(t::dispatch(t::pipeline("workloads.air", fn), {bA, bB, bC, t::Arg::value(M), t::Arg::value(N), t::Arg::value(K)}, grid, tg, 0, groups));
    for (size_t i = 0; i < C.size(); i++)
        CHECK_MSG(t::near(t::data<float>(bC)[i], C[i], 1e-4f, 1e-4f), "C[%zu] = %f want %f", i, t::data<float>(bC)[i], C[i]);
    bA->release();
    bB->release();
    bC->release();
}

TEST(sgemm_naive_64) { sgemm("sgemm_naive", 64, 64, 64, 0); }
TEST(sgemm_naive_ragged) { sgemm("sgemm_naive", 37, 53, 29, 0); }
TEST(sgemm_tiled_128) { sgemm("sgemm_tiled", 128, 128, 128, 16); }
TEST(sgemm_tiled_ragged) { sgemm("sgemm_tiled", 100, 70, 45, 16); }

TEST(transpose)
{
    const uint32_t w = 300, h = 170;
    std::vector<float> in(w * h);
    for (uint32_t i = 0; i < w * h; i++) in[i] = (float)i;
    MTL::Buffer *bi = t::buffer(in), *bo = t::zeros(w * h * 4);
    CHECK(t::dispatch(t::pipeline("workloads.air", "transpose"), {bi, bo, t::Arg::value(w), t::Arg::value(h)},
                      MTL::Size((w + 15) / 16, (h + 15) / 16, 1), MTL::Size(16, 16, 1), 0, true));
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) CHECK(t::data<float>(bo)[x * h + y] == in[y * w + x]);
    bi->release();
    bo->release();
}

TEST(prefix_scan_two_pass)
{
    // Scan each group, scan the group sums on the CPU... no: on the GPU with one group, then add.
    const uint32_t tg = 256, groups = 256, n = tg * groups;
    std::vector<uint32_t> in(n);
    for (uint32_t i = 0; i < n; i++) in[i] = (i * 2654435761u) >> 28;
    MTL::Buffer *bi = t::buffer(in), *bo = t::zeros(n * 4), *sums = t::zeros(groups * 4), *sums2 = t::zeros(groups * 4),
                *dummy = t::zeros(16);
    MTL::ComputePipelineState* scan = t::pipeline("workloads.air", "scan_group");
    MTL::ComputePipelineState* add = t::pipeline("workloads.air", "add_offsets");
    CHECK(scan && add);
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    e->setComputePipelineState(scan);
    e->setBuffer(bi, 0, 0);
    e->setBuffer(bo, 0, 1);
    e->setBuffer(sums, 0, 2);
    e->setThreadgroupMemoryLength(tg * 2 * 4, 0);
    e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(tg, 1, 1));
    e->setBuffer(sums, 0, 0);
    e->setBuffer(sums2, 0, 1);
    e->setBuffer(dummy, 0, 2);
    e->dispatchThreads(MTL::Size(groups, 1, 1), MTL::Size(groups, 1, 1));
    e->setComputePipelineState(add);
    e->setBuffer(bo, 0, 0);
    e->setBuffer(sums2, 0, 1);
    e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(tg, 1, 1));
    e->endEncoding();
    CHECK(t::finish(cb));
    uint32_t acc = 0;
    for (uint32_t i = 0; i < n; i++) {
        acc += in[i];
        CHECK_MSG(t::data<uint32_t>(bo)[i] == acc, "i=%u got %u want %u", i, t::data<uint32_t>(bo)[i], acc);
    }
    bi->release();
    bo->release();
    sums->release();
    sums2->release();
    dummy->release();
}

TEST(bitonic_sort)
{
    const uint32_t n = 1 << 14;
    std::vector<uint32_t> v(n);
    uint32_t s = 99;
    for (uint32_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        v[i] = s;
    }
    MTL::Buffer* b = t::buffer(v);
    MTL::ComputePipelineState* p = t::pipeline("workloads.air", "bitonic_step");
    CHECK(p);
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    e->setComputePipelineState(p);
    e->setBuffer(b, 0, 0);
    int steps = 0;
    for (uint32_t k = 2; k <= n; k <<= 1)
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            e->setBytes(&j, 4, 1);
            e->setBytes(&k, 4, 2);
            e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
            steps++;
        }
    e->endEncoding();
    CHECK(t::finish(cb));
    std::sort(v.begin(), v.end());
    for (uint32_t i = 0; i < n; i++) CHECK_MSG(t::data<uint32_t>(b)[i] == v[i], "i=%u after %d steps", i, steps);
    b->release();
}

TEST(blur3x3)
{
    const uint32_t w = 257, h = 129;
    std::vector<float> in(w * h);
    for (uint32_t i = 0; i < w * h; i++) in[i] = (float)((i * 13) % 255);
    MTL::Buffer *bi = t::buffer(in), *bo = t::zeros(w * h * 4);
    CHECK(t::dispatch(t::pipeline("workloads.air", "blur3x3"), {bi, bo, t::Arg::value(w), t::Arg::value(h)}, MTL::Size(w, h, 1),
                      MTL::Size(16, 16, 1)));
    for (int y = 0; y < (int)h; y++)
        for (int x = 0; x < (int)w; x++) {
            float s = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) s += in[std::clamp(y + dy, 0, (int)h - 1) * w + std::clamp(x + dx, 0, (int)w - 1)];
            CHECK(t::near(t::data<float>(bo)[y * w + x], s / 9.0f, 1e-5f, 1e-4f));
        }
    bi->release();
    bo->release();
}

TEST(mandelbrot)
{
    const uint32_t w = 256, h = 192, it = 256;
    MTL::Buffer* bo = t::zeros(w * h * 4);
    CHECK(t::dispatch(t::pipeline("workloads.air", "mandelbrot"), {bo, t::Arg::value(w), t::Arg::value(h), t::Arg::value(it)},
                      MTL::Size(w, h, 1), MTL::Size(16, 16, 1)));
    // Float order of operations may differ near the boundary: compare with a small mismatch budget.
    uint32_t mism = 0;
    for (uint32_t py = 0; py < h; py++)
        for (uint32_t px = 0; px < w; px++) {
            float cx = -2.0f + 3.0f * px / w, cy = -1.25f + 2.5f * py / h, x = 0, y = 0;
            uint32_t k = 0;
            while (x * x + y * y <= 4.0f && k < it) {
                float tt = x * x - y * y + cx;
                y = 2.0f * x * y + cy;
                x = tt;
                k++;
            }
            mism += t::data<uint32_t>(bo)[py * w + px] != k;
        }
    CHECK_MSG(mism < w * h / 100, "%u of %u pixels differ", mism, w * h);
    bo->release();
}

TEST(nbody)
{
    const uint32_t n = 512;
    std::vector<float> pos(n * 4), vel(n * 4, 0.0f);
    for (uint32_t i = 0; i < n; i++) {
        pos[4 * i] = std::sin(i * 0.1f) * 10;
        pos[4 * i + 1] = std::cos(i * 0.13f) * 10;
        pos[4 * i + 2] = (i % 17) * 0.5f;
        pos[4 * i + 3] = 1.0f + (i % 3);
    }
    MTL::Buffer *bp = t::buffer(pos), *bv = t::buffer(vel), *bn = t::zeros(n * 16);
    float dt = 0.01f;
    CHECK(t::dispatch(t::pipeline("workloads.air", "nbody_step"), {bp, bv, bn, t::Arg::value(n), t::Arg::value(dt)},
                      MTL::Size(n, 1, 1), MTL::Size(128, 1, 1)));
    for (uint32_t i = 0; i < n; i++) {
        float ax = 0, ay = 0, az = 0;
        for (uint32_t j = 0; j < n; j++) {
            float dx = pos[4 * j] - pos[4 * i], dy = pos[4 * j + 1] - pos[4 * i + 1], dz = pos[4 * j + 2] - pos[4 * i + 2];
            float r2 = dx * dx + dy * dy + dz * dz + 0.01f, inv = 1.0f / std::sqrt(r2);
            float f = pos[4 * j + 3] * inv * inv * inv;
            ax += dx * f;
            ay += dy * f;
            az += dz * f;
        }
        CHECK_MSG(t::near(t::data<float>(bv)[4 * i], ax * dt, 1e-3f, 1e-5f), "body %u vx %g want %g", i, t::data<float>(bv)[4 * i], ax * dt);
        CHECK(t::near(t::data<float>(bv)[4 * i + 1], ay * dt, 1e-3f, 1e-5f));
        CHECK(t::near(t::data<float>(bv)[4 * i + 2], az * dt, 1e-3f, 1e-5f));
    }
    bp->release();
    bv->release();
    bn->release();
}

TEST(softmax_rows)
{
    const uint32_t rows = 64, cols = 1000, tg = 256;
    std::vector<float> in(rows * cols);
    for (uint32_t i = 0; i < rows * cols; i++) in[i] = std::sin(i * 0.37f) * 4.0f;
    MTL::Buffer *bi = t::buffer(in), *bo = t::zeros(rows * cols * 4);
    CHECK(t::dispatch(t::pipeline("workloads.air", "softmax_rows"), {bi, bo, t::Arg::value(cols)}, MTL::Size(rows, 1, 1),
                      MTL::Size(tg, 1, 1), tg * 4, true));
    for (uint32_t r = 0; r < rows; r++) {
        float m = -1e30f, s = 0;
        for (uint32_t c = 0; c < cols; c++) m = std::max(m, in[r * cols + c]);
        for (uint32_t c = 0; c < cols; c++) s += std::exp(in[r * cols + c] - m);
        float rowsum = 0;
        for (uint32_t c = 0; c < cols; c++) {
            float want = std::exp(in[r * cols + c] - m) / s, got = t::data<float>(bo)[r * cols + c];
            CHECK_MSG(t::near(got, want, 3e-3f, 1e-7f), "row %u col %u got %g want %g", r, c, got, want);
            rowsum += got;
        }
        CHECK(std::fabs(rowsum - 1.0f) < 1e-3f);
    }
    bi->release();
    bo->release();
}

TEST(gemv)
{
    const uint32_t rows = 1024, cols = 4096, tg = 128;
    std::vector<float> A((size_t)rows * cols), x(cols);
    for (size_t i = 0; i < A.size(); i++) A[i] = (float)((i * 2654435761u) % 1000) * 0.001f - 0.5f;
    for (uint32_t i = 0; i < cols; i++) x[i] = std::cos(i * 0.01f);
    MTL::Buffer *bA = t::buffer(A), *bx = t::buffer(x), *by = t::zeros(rows * 4);
    CHECK(t::dispatch(t::pipeline("workloads.air", "gemv"), {bA, bx, by, t::Arg::value(rows), t::Arg::value(cols)}, MTL::Size(rows, 1, 1),
                      MTL::Size(tg, 1, 1), tg * 4, true));
    for (uint32_t r = 0; r < rows; r++) {
        double acc = 0;
        for (uint32_t c = 0; c < cols; c++) acc += (double)A[(size_t)r * cols + c] * x[c];
        CHECK_MSG(std::fabs(t::data<float>(by)[r] - acc) < 1e-2, "row %u: %f want %f", r, t::data<float>(by)[r], acc);
    }
    bA->release();
    bx->release();
    by->release();
}

static float cndRef(float d)
{
    const float A1 = 0.31938153f, A2 = -0.356563782f, A3 = 1.781477937f, A4 = -1.821255978f, A5 = 1.330274429f;
    float K = 1.0f / (1.0f + 0.2316419f * std::fabs(d));
    float c = 0.3989422804f * std::exp(-0.5f * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
    return d > 0 ? 1.0f - c : c;
}

TEST(black_scholes)
{
    const uint32_t n = 10000;
    std::vector<float> S(n), X(n), T(n);
    for (uint32_t i = 0; i < n; i++) {
        S[i] = 5.0f + (i % 100) * 0.3f;
        X[i] = 1.0f + (i % 77) * 0.4f;
        T[i] = 0.25f + (i % 10) * 0.9f;
    }
    float r = 0.02f, v = 0.3f;
    MTL::Buffer *bS = t::buffer(S), *bX = t::buffer(X), *bT = t::buffer(T), *bc = t::zeros(n * 4), *bp = t::zeros(n * 4);
    CHECK(t::dispatch(t::pipeline("workloads.air", "black_scholes"),
                      {bS, bX, bT, bc, bp, t::Arg::value(r), t::Arg::value(v), t::Arg::value(n)}, MTL::Size(n, 1, 1), MTL::Size(128, 1, 1)));
    for (uint32_t i = 0; i < n; i++) {
        float sq = std::sqrt(T[i]), d1 = (std::log(S[i] / X[i]) + (r + 0.5f * v * v) * T[i]) / (v * sq), d2 = d1 - v * sq;
        float e = std::exp(-r * T[i]);
        float call = S[i] * cndRef(d1) - X[i] * e * cndRef(d2), put = X[i] * e * (1 - cndRef(d2)) - S[i] * (1 - cndRef(d1));
        CHECK_MSG(std::fabs(t::data<float>(bc)[i] - call) < 2e-3f * (1 + std::fabs(call)), "i=%u call %f want %f", i, t::data<float>(bc)[i], call);
        CHECK(std::fabs(t::data<float>(bp)[i] - put) < 2e-3f * (1 + std::fabs(put)));
    }
    bS->release();
    bX->release();
    bT->release();
    bc->release();
    bp->release();
}

TEST(stream_copy_triad)
{
    const uint32_t n = 1 << 20;
    std::vector<float> a(n), b(n);
    for (uint32_t i = 0; i < n; i++) {
        a[i] = (float)i;
        b[i] = 1.0f;
    }
    MTL::Buffer *ba = t::buffer(a), *bb = t::buffer(b), *bc = t::zeros(n * 4), *bd = t::zeros(n * 4);
    uint32_t n4 = n / 4;
    CHECK(t::dispatch(t::pipeline("workloads.air", "stream_copy"), {ba, bc, t::Arg::value(n4)}, MTL::Size(38 * 1024, 1, 1),
                      MTL::Size(256, 1, 1)));
    for (uint32_t i = 0; i < n; i += 1013) CHECK(t::data<float>(bc)[i] == a[i]);
    float s = 3.0f;
    CHECK(t::dispatch(t::pipeline("workloads.air", "stream_triad"), {ba, bb, bd, t::Arg::value(s), t::Arg::value(n)},
                      MTL::Size(38 * 1024, 1, 1), MTL::Size(256, 1, 1)));
    for (uint32_t i = 0; i < n; i += 1013) CHECK(t::data<float>(bd)[i] == a[i] + 3.0f);
    ba->release();
    bb->release();
    bc->release();
    bd->release();
}

TEST_MAIN
