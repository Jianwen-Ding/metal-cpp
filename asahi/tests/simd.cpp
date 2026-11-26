// SIMD-group functions (AIR's simd_* written as AIR).
#include "harness.h"

TEST(simd_ops)
{
    const uint32_t n = 256;
    std::vector<uint32_t> in(n);
    for (uint32_t i = 0; i < n; i++) in[i] = (i * 37 + 11) % 97;
    MTL::Buffer *bi = t::buffer(in), *bo = t::zeros(n * 16 * 4);
    CHECK(t::dispatch(t::pipeline("simd.air", "simd_ops"), {bo, bi}, MTL::Size(n, 1, 1), MTL::Size(64, 1, 1)));
    uint32_t* o = t::data<uint32_t>(bo);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t base = i & ~31u, lane = i & 31;
        auto at = [&](uint32_t l) { return in[base + (l & 31)]; };
        uint32_t sum = 0, mn = ~0u, mx = 0, incl = 0, excl = 0, ballot = 0;
        for (uint32_t l = 0; l < 32; l++) {
            uint32_t v = at(l);
            sum += v;
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            if (l <= lane) incl += v;
            if (l < lane) excl += v;
            if (v & 1) ballot |= 1u << l;
        }
        uint32_t want[16] = {at(31 - lane), at(lane ^ 5), sum, sum, mn, mx, incl, excl, ballot, ballot != 0, ballot == 0xffffffffu,
                             at(3), lane + 1 < 32 ? at(lane + 1) : at(lane), lane >= 2 ? at(lane - 2) : at(lane),
                             (lane & 3) + 1 < 4 ? at(lane + 1) : at(lane), excl};
        for (int k = 0; k < 16; k++) {
            // shuffle_down / up past the edge leave the lane's own value in Metal; quad_shuffle_down likewise.
            CHECK_MSG(o[16 * i + k] == want[k], "thread %u op %d: got %u want %u", i, k, o[16 * i + k], want[k]);
        }
    }
    bi->release();
    bo->release();
}

TEST(simd_reduce_atomic)
{
    const uint32_t n = 1 << 16;
    std::vector<uint32_t> in(n);
    uint64_t total = 0;
    for (uint32_t i = 0; i < n; i++) {
        in[i] = i % 13;
        total += in[i];
    }
    MTL::Buffer *bi = t::buffer(in), *bt = t::zeros(4);
    CHECK(t::dispatch(t::pipeline("simd.air", "simd_reduce_atomic"), {bi, bt}, MTL::Size(n, 1, 1), MTL::Size(256, 1, 1)));
    CHECK_MSG(t::data<uint32_t>(bt)[0] == (uint32_t)total, "got %u want %llu", t::data<uint32_t>(bt)[0], (unsigned long long)total);
    bi->release();
    bt->release();
}

TEST_MAIN
