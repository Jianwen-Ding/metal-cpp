// Dispatch geometry, system values and argument passing.
#include "harness.h"

static bool runSysvals(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t lx, uint32_t ly, uint32_t lz, bool groups)
{
    uint32_t tx = groups ? gx * lx : gx, ty = groups ? gy * ly : gy, tz = groups ? gz * lz : gz;
    size_t n = (size_t)tx * ty * tz;
    MTL::Buffer* out = t::zeros(n * 12 * 4);
    bool ok = t::dispatch(t::pipeline("sysvals.air", "sysvals3d"), {out}, MTL::Size(gx, gy, gz), MTL::Size(lx, ly, lz), 0, groups);
    if (!ok) {
        out->release();
        return false;
    }
    uint32_t* r = t::data<uint32_t>(out);
    uint32_t ngx = (tx + lx - 1) / lx, ngy = (ty + ly - 1) / ly, ngz = (tz + lz - 1) / lz;
    for (uint32_t z = 0; z < tz; z++)
        for (uint32_t y = 0; y < ty; y++)
            for (uint32_t x = 0; x < tx; x++) {
                // get_global_size is groups * local size: the sysvals3d index assumes whole groups in x/y.
                uint32_t sx = ngx * lx, sy = ngy * ly;
                size_t idx = ((size_t)z * sy + y) * sx + x;
                if (idx >= n) continue;
                uint32_t* e = r + idx * 12;
                uint32_t want[12] = {x, y, z, x % lx, y % ly, z % lz, x / lx, y / ly, z / lz,
                                     lx * 10000 + ly * 100 + lz, ngx * 10000 + ngy * 100 + ngz, 0xC0FFEE};
                for (int k = 0; k < 12; k++)
                    if (e[k] != want[k]) {
                        fprintf(stderr, "  (%u,%u,%u) field %d: got %u want %u\n", x, y, z, k, e[k], want[k]);
                        out->release();
                        return false;
                    }
            }
    out->release();
    return true;
}

TEST(sysvals_1d) { CHECK(runSysvals(256, 1, 1, 64, 1, 1, false)); }
TEST(sysvals_2d) { CHECK(runSysvals(64, 16, 1, 16, 4, 1, false)); }
TEST(sysvals_3d) { CHECK(runSysvals(16, 8, 4, 4, 4, 2, false)); }
TEST(sysvals_groups_3d) { CHECK(runSysvals(3, 5, 2, 8, 2, 4, true)); }
TEST(sysvals_tg_1) { CHECK(runSysvals(37, 1, 1, 1, 1, 1, false)); }
TEST(sysvals_tg_1024) { CHECK(runSysvals(4096, 1, 1, 1024, 1, 1, false)); }
TEST(sysvals_tg_odd) { CHECK(runSysvals(21 * 3, 7, 1, 21, 7, 1, false)); }

TEST(simd_ids)
{
    const uint32_t n = 256, tg = 128;
    MTL::Buffer* out = t::zeros(n * 16);
    CHECK(t::dispatch(t::pipeline("sysvals.air", "simd_ids"), {out}, MTL::Size(n, 1, 1), MTL::Size(tg, 1, 1)));
    uint32_t* r = t::data<uint32_t>(out);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t l = i % tg;
        CHECK_MSG(r[4 * i] == l % 32, "lane of %u = %u", i, r[4 * i]);
        CHECK_MSG(r[4 * i + 1] == l / 32, "simd group of %u = %u", i, r[4 * i + 1]);
        CHECK(r[4 * i + 2] == 32);
        CHECK(r[4 * i + 3] == tg / 32);
    }
    out->release();
}

// Non-uniform threadgroups (dispatchThreads with a ragged last group): exactly the grid runs.
static bool countThreads(uint32_t w, uint32_t h, uint32_t lx, uint32_t ly)
{
    uint32_t gw = (w + lx - 1) / lx * lx;
    MTL::Buffer* out = t::zeros((size_t)gw * ((h + ly - 1) / ly * ly) * 4 + 64);
    MTL::Buffer* counter = t::zeros(4);
    bool ok = t::dispatch(t::pipeline("sysvals.air", "count_threads"), {out, counter}, MTL::Size(w, h, 1), MTL::Size(lx, ly, 1));
    if (ok) {
        ok = t::data<uint32_t>(counter)[0] == w * h;
        if (!ok) fprintf(stderr, "  ran %u threads for a %ux%u grid\n", t::data<uint32_t>(counter)[0], w, h);
    }
    out->release();
    counter->release();
    return ok;
}
TEST(partial_threadgroups_1d) { CHECK(countThreads(1000, 1, 64, 1)); }
TEST(partial_threadgroups_small) { CHECK(countThreads(5, 1, 32, 1)); }
TEST(partial_threadgroups_2d) { CHECK(countThreads(33, 17, 8, 8)); }
TEST(partial_threadgroups_large) { CHECK(countThreads(100003, 1, 256, 1)); }

TEST(index2d)
{
    const uint32_t w = 300, h = 77;
    MTL::Buffer* out = t::zeros(w * h * 4);
    CHECK(t::dispatch(t::pipeline("sysvals.air", "index2d"), {out, t::Arg::value(w), t::Arg::value(h)}, MTL::Size(w, h, 1),
                      MTL::Size(16, 16, 1)));
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) CHECK(t::data<uint32_t>(out)[y * w + x] == x * 1000 + y);
    out->release();
}

TEST(fill_index_large)
{
    const uint32_t n = 1u << 22;
    MTL::Buffer* out = t::zeros((size_t)n * 4);
    CHECK(t::dispatch(t::pipeline("basic.air", "fill_index"), {out}, MTL::Size(n, 1, 1), MTL::Size(256, 1, 1)));
    for (uint32_t i = 0; i < n; i += 997) CHECK(t::data<uint32_t>(out)[i] == i);
    CHECK(t::data<uint32_t>(out)[n - 1] == n - 1);
    out->release();
}

TEST(saxpy)
{
    const uint32_t n = 12345;
    std::vector<float> x(n), y(n);
    for (uint32_t i = 0; i < n; i++) {
        x[i] = (float)i;
        y[i] = 1.0f;
    }
    MTL::Buffer *bx = t::buffer(x), *by = t::buffer(y);
    float a = 0.5f;
    CHECK(t::dispatch(t::pipeline("basic.air", "saxpy"), {bx, by, t::Arg::value(a), t::Arg::value(n)}, MTL::Size(n, 1, 1),
                      MTL::Size(128, 1, 1)));
    for (uint32_t i = 0; i < n; i++) CHECK(t::data<float>(by)[i] == 0.5f * i + 1.0f);
    bx->release();
    by->release();
}

struct Params {
    float scale, bias;
    uint32_t count;
    int32_t shift;
    float color[4];
};

TEST(struct_params_setbytes)
{
    const uint32_t n = 500;
    std::vector<float> in(n);
    for (uint32_t i = 0; i < n; i++) in[i] = (float)i;
    MTL::Buffer *bi = t::buffer(in), *bo = t::zeros(n * 4);
    Params p{2.0f, -1.0f, 400, 3, {0, 0.5f, 0, 0}};
    CHECK(t::dispatch(t::pipeline("args.air", "apply_params"), {bi, bo, t::Arg::value(p)}, MTL::Size(n, 1, 1), MTL::Size(64, 1, 1)));
    for (uint32_t i = 0; i < n; i++) {
        float want = i < 400 ? in[i] * 2.0f - 1.0f + 0.5f * 3 : 0.0f;
        CHECK_MSG(t::data<float>(bo)[i] == want, "i=%u got %f want %f", i, t::data<float>(bo)[i], want);
    }
    bi->release();
    bo->release();
}

TEST(struct_params_buffer)
{
    // The same constant structure bound as an MTLBuffer at an offset.
    const uint32_t n = 64;
    std::vector<float> in(n, 1.0f);
    MTL::Buffer *bi = t::buffer(in), *bo = t::zeros(n * 4);
    MTL::Buffer* pb = t::zeros(4096);
    Params p{3.0f, 1.0f, n, 0, {0, 0, 0, 0}};
    memcpy(t::data<uint8_t>(pb) + 256, &p, sizeof p);
    CHECK(t::dispatch(t::pipeline("args.air", "apply_params"), {bi, bo, t::Arg(pb, 256)}, MTL::Size(n, 1, 1), MTL::Size(32, 1, 1)));
    for (uint32_t i = 0; i < n; i++) CHECK(t::data<float>(bo)[i] == 4.0f);
    bi->release();
    bo->release();
    pb->release();
}

TEST(constant_table_pointer)
{
    const uint32_t n = 1000;
    std::vector<uint32_t> idx(n);
    std::vector<float> table(256);
    for (uint32_t i = 0; i < 256; i++) table[i] = i * 1.25f;
    for (uint32_t i = 0; i < n; i++) idx[i] = (i * 7919u) & 255;
    MTL::Buffer *bi = t::buffer(idx), *bt = t::buffer(table), *bo = t::zeros(n * 4);
    CHECK(t::dispatch(t::pipeline("args.air", "lookup"), {bi, bt, bo, t::Arg::value(n)}, MTL::Size(n, 1, 1), MTL::Size(64, 1, 1)));
    for (uint32_t i = 0; i < n; i++) CHECK(t::data<float>(bo)[i] == table[idx[i]]);
    bi->release();
    bt->release();
    bo->release();
}

TEST(many_buffers)
{
    const uint32_t n = 256;
    std::vector<MTL::Buffer*> bufs;
    for (uint32_t k = 0; k < 7; k++) {
        std::vector<uint32_t> v(n);
        for (uint32_t i = 0; i < n; i++) v[i] = i + k * 1000;
        bufs.push_back(t::buffer(v));
    }
    MTL::Buffer* out = t::zeros(n * 4);
    std::vector<t::Arg> args;
    for (auto* b : bufs) args.push_back(b);
    args.push_back(out);
    CHECK(t::dispatch(t::pipeline("args.air", "many_buffers"), args, MTL::Size(n, 1, 1), MTL::Size(64, 1, 1)));
    for (uint32_t i = 0; i < n; i++) {
        uint32_t want = 0;
        for (uint32_t k = 0; k < 7; k++) want += (k + 1) * (i + k * 1000);
        CHECK(t::data<uint32_t>(out)[i] == want);
    }
    for (auto* b : bufs) b->release();
    out->release();
}

TEST(scalar_types)
{
    MTL::Buffer* out = t::zeros(64);
    uint8_t a = 200;
    uint16_t b = 60000;
    uint32_t c = 0xdeadbeef;
    uint64_t d = 0x123456789abcdef0ull;
    float e = -1.5f;
    int32_t f = -77;
    int16_t g = -300;
    CHECK(t::dispatch(t::pipeline("args.air", "scalar_types"),
                      {out, t::Arg::value(a), t::Arg::value(b), t::Arg::value(c), t::Arg::value(d), t::Arg::value(e),
                       t::Arg::value(f), t::Arg::value(g)},
                      MTL::Size(1, 1, 1), MTL::Size(1, 1, 1)));
    uint32_t* r = t::data<uint32_t>(out);
    uint32_t ebits;
    memcpy(&ebits, &e, 4);
    CHECK(r[0] == 200);
    CHECK(r[1] == 60000);
    CHECK(r[2] == 0xdeadbeef);
    CHECK(r[3] == 0x9abcdef0u);
    CHECK(r[4] == 0x12345678u);
    CHECK(r[5] == ebits);
    CHECK((int32_t)r[6] == -77);
    CHECK((int32_t)r[7] == -300);
    out->release();
}

TEST(big_params)
{
    float w[48];
    for (int i = 0; i < 48; i++) w[i] = i * 0.5f;
    MTL::Buffer* out = t::zeros(48 * 4);
    t::Arg pa = t::Arg::value(w);
    CHECK(t::dispatch(t::pipeline("args.air", "big_params"), {out, pa}, MTL::Size(48, 1, 1), MTL::Size(48, 1, 1)));
    for (int i = 0; i < 48; i++) CHECK(t::data<float>(out)[i] == w[i] * 2.0f + w[47 - i]);
    out->release();
}

TEST(buffer_offsets)
{
    const uint32_t n = 1024;
    std::vector<uint32_t> in(n * 2);
    for (uint32_t i = 0; i < n * 2; i++) in[i] = i * 3;
    MTL::Buffer *bi = t::buffer(in), *bo = t::zeros(n * 8);
    CHECK(t::dispatch(t::pipeline("args.air", "copy_u32"), {t::Arg(bi, 400 * 4), t::Arg(bo, 16)}, MTL::Size(n, 1, 1),
                      MTL::Size(64, 1, 1)));
    for (uint32_t i = 0; i < n; i++) CHECK(t::data<uint32_t>(bo)[4 + i] == (400 + i) * 3);
    CHECK(t::data<uint32_t>(bo)[0] == 0);
    bi->release();
    bo->release();
}

TEST(set_buffer_offset_rebind)
{
    // setBufferOffset between dispatches in one encoder.
    const uint32_t n = 64;
    std::vector<uint32_t> in(n * 4);
    for (uint32_t i = 0; i < n * 4; i++) in[i] = i;
    MTL::Buffer *bi = t::buffer(in), *bo = t::zeros(n * 16);
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(t::pipeline("args.air", "copy_u32"));
    enc->setBuffer(bi, 0, 0);
    enc->setBuffer(bo, 0, 1);
    for (uint32_t k = 0; k < 4; k++) {
        enc->setBufferOffset(k * n * 4, 0);
        enc->setBufferOffset(k * n * 4, 1);
        enc->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(32, 1, 1));
    }
    enc->endEncoding();
    CHECK(t::finish(cb));
    for (uint32_t i = 0; i < n * 4; i++) CHECK(t::data<uint32_t>(bo)[i] == i);
    bi->release();
    bo->release();
}

struct Particle {
    float x, y;
    uint32_t id;
    float pad;
};

TEST(struct_buffer)
{
    const uint32_t n = 300;
    std::vector<Particle> p(n);
    for (uint32_t i = 0; i < n; i++) p[i] = {(float)i, 2.0f, 0, 0};
    MTL::Buffer* b = t::buffer(p);
    float dt = 0.25f;
    CHECK(t::dispatch(t::pipeline("args.air", "particles"), {b, t::Arg::value(dt), t::Arg::value(n)}, MTL::Size(n, 1, 1),
                      MTL::Size(64, 1, 1)));
    Particle* r = t::data<Particle>(b);
    for (uint32_t i = 0; i < n; i++) {
        CHECK(r[i].x == i + 0.5f);
        CHECK(r[i].id == i * 3);
    }
    b->release();
}

TEST(vector_arg)
{
    const uint32_t n = 128;
    std::vector<float> v(n * 4, 2.0f);
    MTL::Buffer* b = t::buffer(v);
    float s[4] = {1, 2, 3, 4};
    CHECK(t::dispatch(t::pipeline("args.air", "vec4_scale"), {b, t::Arg::value(s)}, MTL::Size(n, 1, 1), MTL::Size(32, 1, 1)));
    for (uint32_t i = 0; i < n; i++)
        for (int k = 0; k < 4; k++) CHECK(t::data<float>(b)[4 * i + k] == 2.0f * s[k]);
    b->release();
}

TEST(empty_grid)
{
    MTL::Buffer* out = t::zeros(16);
    CHECK(t::dispatch(t::pipeline("basic.air", "fill_index"), {out}, MTL::Size(0, 1, 1), MTL::Size(32, 1, 1)));
    CHECK(t::data<uint32_t>(out)[0] == 0);
    out->release();
}

TEST(pipeline_properties)
{
    MTL::ComputePipelineState* p = t::pipeline("basic.air", "vadd");
    CHECK(p != nullptr);
    CHECK(p->threadExecutionWidth() == 32);
    CHECK(p->maxTotalThreadsPerThreadgroup() >= 256 && p->maxTotalThreadsPerThreadgroup() <= 1024);
    CHECK(p->device() == t::device());
    MTL::ComputePipelineState* tg = t::pipeline("memory.air", "tg_reverse_static");
    CHECK(tg->staticThreadgroupMemoryLength() == 1024);
}

TEST_MAIN
