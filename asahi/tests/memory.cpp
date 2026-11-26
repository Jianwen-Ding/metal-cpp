// Threadgroup memory, barriers and atomics.
#include "harness.h"

#include <algorithm>
#include <set>

static void reverseTest(const char* fn, uint32_t tg, bool dynamic)
{
    const uint32_t n = tg * 16;
    std::vector<uint32_t> in(n);
    for (uint32_t i = 0; i < n; i++) in[i] = i * 11 + 3;
    MTL::Buffer *bi = t::buffer(in), *bo = t::zeros(n * 4);
    CHECK(t::dispatch(t::pipeline("memory.air", fn), {bi, bo}, MTL::Size(n, 1, 1), MTL::Size(tg, 1, 1), dynamic ? tg * 4 : 0));
    for (uint32_t i = 0; i < n; i++) {
        uint32_t g = i / tg, l = i % tg;
        CHECK_MSG(t::data<uint32_t>(bo)[i] == in[g * tg + tg - 1 - l], "i=%u", i);
    }
    bi->release();
    bo->release();
}

TEST(tg_static_64) { reverseTest("tg_reverse_static", 64, false); }
TEST(tg_static_256) { reverseTest("tg_reverse_static", 256, false); }
TEST(tg_dynamic_32) { reverseTest("tg_reverse_dynamic", 32, true); }
TEST(tg_dynamic_512) { reverseTest("tg_reverse_dynamic", 512, true); }
TEST(tg_dynamic_1024) { reverseTest("tg_reverse_dynamic", 1024, true); }

TEST(tg_two_arrays)
{
    const uint32_t tg = 128, n = tg * 8;
    std::vector<float> in(n);
    for (uint32_t i = 0; i < n; i++) in[i] = (float)(i % 50);
    MTL::Buffer *bi = t::buffer(in), *bo = t::zeros(n * 4);
    CHECK(t::dispatch(t::pipeline("memory.air", "tg_two_arrays"), {bi, bo}, MTL::Size(n, 1, 1), MTL::Size(tg, 1, 1)));
    for (uint32_t i = 0; i < n; i++) {
        uint32_t g = i / tg, l = i % tg;
        float want = (in[g * tg + (l + 1) % tg] + in[i]) * 0.5f;
        CHECK(t::data<float>(bo)[i] == want);
    }
    bi->release();
    bo->release();
}

static void reduceTest(uint32_t n, uint32_t tg)
{
    std::vector<float> in(n);
    double total = 0;
    for (uint32_t i = 0; i < n; i++) {
        in[i] = (float)((i * 37) % 101) * 0.25f;
        total += in[i];
    }
    uint32_t groups = (n + tg - 1) / tg;
    MTL::Buffer *bi = t::buffer(in), *bp = t::zeros(groups * 4);
    CHECK(t::dispatch(t::pipeline("memory.air", "reduce_sum"), {bi, bp, t::Arg::value(n)},
                      MTL::Size(groups * tg, 1, 1), MTL::Size(tg, 1, 1), tg * 4));
    double sum = 0;
    for (uint32_t g = 0; g < groups; g++) sum += t::data<float>(bp)[g];
    CHECK_MSG(std::fabs(sum - total) <= 1e-4 * total + 1e-3, "sum %f want %f", sum, total);
    bi->release();
    bp->release();
}
TEST(reduce_sum_small) { reduceTest(1000, 64); }
TEST(reduce_sum_256) { reduceTest(1 << 16, 256); }
TEST(reduce_sum_1024) { reduceTest(1 << 20, 1024); }

TEST(atomic_ops)
{
    const uint32_t n = 4096;
    std::vector<uint32_t> v(n);
    for (uint32_t i = 0; i < n; i++) v[i] = (i * 2654435761u) % 1000;
    MTL::Buffer* bc = t::zeros(64);
    uint32_t* c = t::data<uint32_t>(bc);
    c[3] = 0xffffffffu;
    c[6] = 0xffffffffu;
    MTL::Buffer* bs = t::zeros(64);
    int32_t* s = t::data<int32_t>(bs);
    s[1] = INT32_MIN;
    s[2] = INT32_MAX;
    MTL::Buffer* bv = t::buffer(v);
    CHECK(t::dispatch(t::pipeline("memory.air", "atomic_ops"), {bc, bs, bv, t::Arg::value(n)}, MTL::Size(n, 1, 1), MTL::Size(128, 1, 1)));
    uint32_t sum = 0, mx = 0, mn = ~0u, orv = 0, xr = 0, andv = ~0u;
    int32_t ssub = 0, smax = INT32_MIN, smin = INT32_MAX;
    for (uint32_t i = 0; i < n; i++) {
        sum += v[i];
        mx = std::max(mx, v[i]);
        mn = std::min(mn, v[i]);
        orv |= 1u << (v[i] & 31);
        xr ^= v[i];
        andv &= ~(1u << (i & 31));
        ssub -= (int32_t)(v[i] & 7);
        smax = std::max(smax, (int32_t)v[i] - 500);
        smin = std::min(smin, (int32_t)v[i] - 500);
    }
    CHECK(c[0] == n);
    CHECK(c[1] == sum);
    CHECK(c[2] == mx);
    CHECK(c[3] == mn);
    CHECK(c[4] == orv);
    CHECK(c[5] == xr);
    CHECK(c[6] == andv);
    CHECK(s[0] == ssub);
    CHECK(s[1] == smax);
    CHECK(s[2] == smin);
    bc->release();
    bs->release();
    bv->release();
}

TEST(atomic_slots_unique)
{
    t::OrderDependent order;
    const uint32_t n = 20000;
    MTL::Buffer *counter = t::zeros(4), *slots = t::zeros(n * 4);
    CHECK(t::dispatch(t::pipeline("memory.air", "atomic_slots"), {counter, slots, t::Arg::value(n)}, MTL::Size(n, 1, 1),
                      MTL::Size(256, 1, 1)));
    CHECK(t::data<uint32_t>(counter)[0] == n);
    std::vector<uint32_t> v(t::data<uint32_t>(slots), t::data<uint32_t>(slots) + n);
    std::sort(v.begin(), v.end());
    for (uint32_t i = 0; i < n; i++) CHECK(v[i] == i);
    counter->release();
    slots->release();
}

TEST(cas_increment)
{
    const uint32_t n = 5000;
    MTL::Buffer* w = t::zeros(4);
    CHECK(t::dispatch(t::pipeline("memory.air", "cas_increment"), {w, t::Arg::value(n)}, MTL::Size(n, 1, 1), MTL::Size(128, 1, 1)));
    CHECK_MSG(t::data<uint32_t>(w)[0] == n, "got %u", t::data<uint32_t>(w)[0]);
    w->release();
}

TEST(histogram)
{
    const uint32_t n = 1 << 18;
    std::vector<uint8_t> d(n);
    std::vector<uint32_t> want(256, 0);
    uint32_t x = 12345;
    for (uint32_t i = 0; i < n; i++) {
        x = x * 1664525u + 1013904223u;
        d[i] = (uint8_t)((x >> 24) ^ (x >> 9));
        want[d[i]]++;
    }
    MTL::Buffer *bd = t::buffer(d), *bh = t::zeros(1024);
    CHECK(t::dispatch(t::pipeline("memory.air", "histogram"), {bd, bh, t::Arg::value(n)}, MTL::Size(64 * 256, 1, 1),
                      MTL::Size(256, 1, 1)));
    for (int b = 0; b < 256; b++) CHECK_MSG(t::data<uint32_t>(bh)[b] == want[b], "bin %d: %u want %u", b, t::data<uint32_t>(bh)[b], want[b]);
    bd->release();
    bh->release();
}

TEST(atomic_xchg_permutation)
{
    t::OrderDependent order;
    // Every thread swaps its id+1 into one cell: what they got back, plus the final cell, is 0..n each once.
    const uint32_t n = 1000;
    MTL::Buffer *cell = t::zeros(4), *got = t::zeros(n * 4);
    CHECK(t::dispatch(t::pipeline("memory.air", "xchg"), {cell, got, t::Arg::value(n)}, MTL::Size(n, 1, 1), MTL::Size(64, 1, 1)));
    std::vector<uint32_t> v(t::data<uint32_t>(got), t::data<uint32_t>(got) + n);
    v.push_back(t::data<uint32_t>(cell)[0]);
    std::sort(v.begin(), v.end());
    for (uint32_t i = 0; i <= n; i++) CHECK(v[i] == i);
    cell->release();
    got->release();
}

TEST(float_atomic_sum_cas)
{
    const uint32_t n = 2048;
    std::vector<float> in(n, 0.5f);
    MTL::Buffer *bi = t::buffer(in), *bs = t::zeros(4);
    CHECK(t::dispatch(t::pipeline("memory.air", "float_atomic_sum"), {bi, bs, t::Arg::value(n)}, MTL::Size(n, 1, 1),
                      MTL::Size(128, 1, 1)));
    CHECK(t::data<float>(bs)[0] == 1024.0f);
    bi->release();
    bs->release();
}

TEST_MAIN
