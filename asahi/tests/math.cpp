// Arithmetic: float functions (fast-math accuracy, as Metal's default),
// integers of 32 and 64 bits, conversions, half, vectors, control flow.
#include "harness.h"

#include <algorithm>

static float tolOf(int k)
{
    switch (k) {
    case 0: case 1: case 22: return 2e-3f;              // sin, cos, tan
    case 2: case 14: case 6: return 2e-3f;              // exp, exp2, pow
    case 3: case 15: case 23: return 2e-3f;             // logs
    case 16: case 17: return 2e-3f;                     // tanh, atan
    case 5: case 19: case 20: return 1e-5f;             // rsqrt, divides
    default: return 1e-6f;
    }
}

TEST(float_funcs)
{
    const uint32_t n = 2048;
    std::vector<float> x(n);
    for (uint32_t i = 0; i < n; i++) x[i] = -8.0f + 16.0f * i / n + 0.013f;
    MTL::Buffer *bx = t::buffer(x), *bo = t::zeros(n * 24 * 4);
    CHECK(t::dispatch(t::pipeline("math.air", "float_funcs"), {bx, bo, t::Arg::value(n)}, MTL::Size(n, 1, 1), MTL::Size(128, 1, 1)));
    float* o = t::data<float>(bo);
    for (uint32_t i = 0; i < n; i++) {
        float v = x[i], a = std::fabs(v) + 0.001f;
        float want[24] = {std::sin(v), std::cos(v), std::exp(v * 0.1f), std::log(a), std::sqrt(a), 1.0f / std::sqrt(a),
                          std::pow(a, 1.7f), std::floor(v), std::ceil(v), std::trunc(v), std::rint(v), std::fma(v, 1.5f, -2.0f),
                          std::fmin(v, 0.25f), std::fmax(v, 0.25f), std::exp2(v * 0.1f), std::log2(a), std::tanh(v * 0.3f),
                          std::atan(v), std::copysign(1.0f, v), v / 3.0f, 1.0f / (v + 10.0f), std::round(v), std::tan(v * 0.2f),
                          std::log10(a)};
        for (int k = 0; k < 24; k++) {
            float got = o[24 * i + k];
            float tol = tolOf(k);
            CHECK_MSG(t::near(got, want[k], tol, tol), "x=%g func %d: got %.9g want %.9g", v, k, got, want[k]);
        }
    }
    bx->release();
    bo->release();
}

TEST(int_ops)
{
    const uint32_t n = 4096;
    std::vector<uint32_t> a(n), b(n);
    uint32_t s = 7;
    for (uint32_t i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        a[i] = s ^ (i << 7);
        s = s * 1103515245u + 12345u;
        b[i] = (i % 5 == 0) ? (s & 0xff) : s;
    }
    a[0] = 0x80000000u;
    a[1] = 0xffffffffu;
    MTL::Buffer *ba = t::buffer(a), *bb = t::buffer(b), *bo = t::zeros(n * 16 * 4);
    CHECK(t::dispatch(t::pipeline("math.air", "int_ops"), {ba, bb, bo, t::Arg::value(n)}, MTL::Size(n, 1, 1), MTL::Size(64, 1, 1)));
    uint32_t* o = t::data<uint32_t>(bo);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t x = a[i], y = b[i] | 1;
        int32_t sx = (int32_t)x, sy = (int32_t)(y >> 1) + 1;
        uint32_t rev = 0;
        for (int k = 0; k < 32; k++) rev |= ((x >> k) & 1u) << (31 - k);
        uint32_t r = y & 31;
        uint32_t want[16] = {x * y, x / y, x % y, (uint32_t)(sx / sy), (uint32_t)(sx % sy), (uint32_t)(((uint64_t)x * y) >> 32),
                             (uint32_t)(((int64_t)sx * (int32_t)y) >> 32), (uint32_t)__builtin_popcount(x),
                             (uint32_t)__builtin_clz(x | 1), (uint32_t)__builtin_ctz(x | 0x80000000u),
                             r ? (x << r) | (x >> (32 - r)) : x, x >> r, (uint32_t)(sx >> r), rev, std::min(x, y),
                             (uint32_t)std::max(sx, -sy)};
        for (int k = 0; k < 16; k++) CHECK_MSG(o[16 * i + k] == want[k], "i=%u x=%#x y=%#x op %d: got %#x want %#x", i, x, y, k, o[16 * i + k], want[k]);
    }
    ba->release();
    bb->release();
    bo->release();
}

TEST(int64_ops)
{
    const uint32_t n = 1024;
    std::vector<uint64_t> a(n), b(n);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (uint32_t i = 0; i < n; i++) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        a[i] = s;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        b[i] = (i % 3 == 0) ? (s & 0xffff) : s;
    }
    MTL::Buffer *ba = t::buffer(a), *bb = t::buffer(b), *bo = t::zeros(n * 8 * 8);
    CHECK(t::dispatch(t::pipeline("math.air", "int64_ops"), {ba, bb, bo, t::Arg::value(n)}, MTL::Size(n, 1, 1), MTL::Size(64, 1, 1)));
    uint64_t* o = t::data<uint64_t>(bo);
    for (uint32_t i = 0; i < n; i++) {
        uint64_t x = a[i], y = b[i] | 1;
        uint64_t want[8] = {x + y, x * y, x / y, x % y, x << (y & 63), x >> (y & 63), (uint64_t)((int64_t)x >> (y & 63)),
                            x ^ (y * 0x9E3779B97F4A7C15ull)};
        for (int k = 0; k < 8; k++) CHECK_MSG(o[8 * i + k] == want[k], "i=%u op %d: got %#llx want %#llx", i, k, (unsigned long long)o[8 * i + k], (unsigned long long)want[k]);
    }
    ba->release();
    bb->release();
    bo->release();
}

TEST(conversions)
{
    const uint32_t n = 1000;
    std::vector<float> f(n);
    for (uint32_t i = 0; i < n; i++) f[i] = -500.0f + i * 1.37f;
    MTL::Buffer *bf = t::buffer(f), *bi = t::zeros(n * 4), *bu = t::zeros(n * 4), *bo = t::zeros(n * 4);
    CHECK(t::dispatch(t::pipeline("math.air", "conversions"), {bf, bi, bu, bo, t::Arg::value(n)}, MTL::Size(n, 1, 1), MTL::Size(64, 1, 1)));
    for (uint32_t i = 0; i < n; i++) {
        CHECK(t::data<int32_t>(bi)[i] == (int32_t)f[i]);
        CHECK(t::data<uint32_t>(bu)[i] == (uint32_t)std::fmax(f[i], 0.0f));
        CHECK(t::data<float>(bo)[i] == (float)(int32_t)(f[i] * 3.0f) + (float)i);
    }
    bf->release();
    bi->release();
    bu->release();
    bo->release();
}

TEST(half_math)
{
    const uint32_t n = 512;
    std::vector<_Float16> a(n), b(n);
    for (uint32_t i = 0; i < n; i++) {
        a[i] = (_Float16)(i * 0.125f);
        b[i] = (_Float16)0.5f;
    }
    MTL::Buffer *ba = t::buffer(a), *bb = t::buffer(b), *bo = t::zeros(n * 2);
    CHECK(t::dispatch(t::pipeline("math.air", "half_math"), {ba, bb, bo, t::Arg::value(n)}, MTL::Size(n, 1, 1), MTL::Size(64, 1, 1)));
    for (uint32_t i = 0; i < n; i++) {
        float want = (float)(_Float16)((float)a[i] * 0.5f + 1.0f);
        CHECK_MSG(t::near((float)t::data<_Float16>(bo)[i], want, 2e-3f, 1e-3f), "i=%u got %f want %f", i, (float)t::data<_Float16>(bo)[i], want);
    }
    ba->release();
    bb->release();
    bo->release();
}

TEST(vec_math)
{
    const uint32_t n = 256;
    std::vector<float> a(n * 4), b(n * 4);
    for (uint32_t i = 0; i < n * 4; i++) {
        a[i] = i * 0.5f;
        b[i] = 3.0f - i * 0.25f;
    }
    MTL::Buffer *ba = t::buffer(a), *bb = t::buffer(b), *bo = t::zeros(n * 16), *bd = t::zeros(n * 4);
    CHECK(t::dispatch(t::pipeline("math.air", "vec_math"), {ba, bb, bo, bd, t::Arg::value(n)}, MTL::Size(n, 1, 1), MTL::Size(64, 1, 1)));
    for (uint32_t i = 0; i < n; i++) {
        float dot = 0;
        for (int k = 0; k < 4; k++) {
            float want = a[4 * i + k] * b[4 * i + k] + (k + 1) - b[4 * i + 3 - k];
            CHECK(t::near(t::data<float>(bo)[4 * i + k], want));
            dot += a[4 * i + k] * b[4 * i + k];
        }
        CHECK(t::near(t::data<float>(bd)[i], dot, 1e-5f, 1e-4f));
    }
    ba->release();
    bb->release();
    bo->release();
    bd->release();
}

TEST(collatz_divergent_loops)
{
    const uint32_t n = 4096;
    std::vector<uint32_t> st(n);
    for (uint32_t i = 0; i < n; i++) st[i] = i + 1;
    MTL::Buffer *bs = t::buffer(st), *bo = t::zeros(n * 4);
    CHECK(t::dispatch(t::pipeline("math.air", "collatz"), {bs, bo, t::Arg::value(n)}, MTL::Size(n, 1, 1), MTL::Size(64, 1, 1)));
    for (uint32_t i = 0; i < n; i++) {
        uint32_t x = st[i], s = 0;
        while (x != 1 && s < 10000) {
            x = (x & 1) ? 3 * x + 1 : x / 2;
            s++;
        }
        CHECK(t::data<uint32_t>(bo)[i] == s);
    }
    bs->release();
    bo->release();
}

TEST(switch_divergence)
{
    const uint32_t n = 1000;
    std::vector<uint32_t> in(n);
    for (uint32_t i = 0; i < n; i++) in[i] = i * 13 + 5;
    MTL::Buffer *bi = t::buffer(in), *bo = t::zeros(n * 4);
    CHECK(t::dispatch(t::pipeline("math.air", "switchy"), {bi, bo, t::Arg::value(n)}, MTL::Size(n, 1, 1), MTL::Size(32, 1, 1)));
    for (uint32_t i = 0; i < n; i++) {
        uint32_t v = in[i], r;
        switch (v % 7) {
        case 0: r = v * 2; break;
        case 1: r = v + 100; break;
        case 2: r = v ^ 0xffff; break;
        case 3: r = v >> 3; break;
        case 4: r = v * v; break;
        case 5: r = 7; break;
        default: r = v - 1; break;
        }
        CHECK(t::data<uint32_t>(bo)[i] == r);
    }
    bi->release();
    bo->release();
}

TEST(globals_constants)
{
    const uint32_t n = 300;
    std::vector<uint32_t> idx(n);
    for (uint32_t i = 0; i < n; i++) idx[i] = i * 5;
    static const float coeffs[8] = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f};
    static const uint32_t primes[16] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};
    MTL::Buffer *bi = t::buffer(idx), *bo = t::zeros(n * 4);
    CHECK(t::dispatch(t::pipeline("globals.air", "use_constants"), {bi, bo, t::Arg::value(n)}, MTL::Size(n, 1, 1), MTL::Size(64, 1, 1)));
    for (uint32_t i = 0; i < n; i++) CHECK(t::data<float>(bo)[i] == coeffs[idx[i] & 7] * (float)primes[idx[i] & 15]);
    bi->release();
    bo->release();
}

TEST(globals_filter)
{
    const uint32_t n = 777;
    std::vector<float> in(n);
    for (uint32_t i = 0; i < n; i++) in[i] = (float)((i * 31) % 17);
    MTL::Buffer *bi = t::buffer(in), *bo = t::zeros(n * 4);
    CHECK(t::dispatch(t::pipeline("globals.air", "filter5"), {bi, bo, t::Arg::value(n)}, MTL::Size(n, 1, 1), MTL::Size(64, 1, 1)));
    const float w[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
    for (int i = 0; i < (int)n; i++) {
        float s = 0;
        for (int k = -2; k <= 2; k++) s += w[k + 2] * in[std::clamp(i + k, 0, (int)n - 1)];
        CHECK(t::near(t::data<float>(bo)[i], s, 1e-5f, 1e-5f));
    }
    bi->release();
    bo->release();
}

TEST_MAIN
