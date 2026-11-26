#include "harness.h"

TEST(device_exists)
{
    MTL::Device* d = t::device();
    CHECK(d != nullptr);
    CHECK(d->name() != nullptr);
    CHECK(strlen(d->name()->utf8String()) > 0);
    CHECK(d->hasUnifiedMemory());
    CHECK(d->maxThreadsPerThreadgroup().width == 1024);
}

TEST(buffer_contents)
{
    MTL::Buffer* b = t::zeros(4096);
    CHECK(b->length() == 4096);
    CHECK(b->contents() != nullptr);
    CHECK(b->gpuAddress() != 0);
    t::data<uint32_t>(b)[5] = 7;
    CHECK(t::data<uint32_t>(b)[5] == 7);
    b->release();
}

TEST(vadd)
{
    const uint32_t n = 1000;
    std::vector<float> a(n), b(n);
    for (uint32_t i = 0; i < n; i++) {
        a[i] = i * 0.5f;
        b[i] = 3.0f - i;
    }
    MTL::Buffer *ba = t::buffer(a), *bb = t::buffer(b), *bc = t::zeros(n * 4);
    CHECK(t::dispatch(t::pipeline("basic.air", "vadd"), {ba, bb, bc, t::Arg::value(n)}, MTL::Size(n, 1, 1), MTL::Size(64, 1, 1)));
    for (uint32_t i = 0; i < n; i++) CHECK_MSG(t::data<float>(bc)[i] == a[i] + b[i], "i=%u got %f", i, t::data<float>(bc)[i]);
    ba->release();
    bb->release();
    bc->release();
}

TEST_MAIN
