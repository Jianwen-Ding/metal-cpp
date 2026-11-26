// The simulator preflight: a kernel that hangs or faults in the AGX
// Performance Simulator is never sent to the GPU, and its command buffer
// fails with an error instead (preflight and verify modes).
#include "harness.h"

static bool simulatorMode()
{
    const char* m = getenv("METAL_ASAHI_MODE");
    std::string s = m ? m : "preflight";
    return s == "preflight" || s == "verify" || s == "sim" || s == "simulator";
}

TEST(hang_is_refused)
{
    if (!simulatorMode()) {
        printf("  (skipped: hardware-only mode would hang the GPU)\n");
        return;
    }
    // A small step budget so the hang is found quickly.
    setenv("METAL_ASAHI_SIM_MAX_STEPS", "200000", 1);
    MTL::Buffer *flag = t::zeros(16), *out = t::zeros(64);
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    e->setComputePipelineState(t::pipeline("faults.air", "spin_forever"));
    e->setBuffer(flag, 0, 0);
    e->setBuffer(out, 0, 1);
    e->dispatchThreads(MTL::Size(32, 1, 1), MTL::Size(32, 1, 1));
    e->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    unsetenv("METAL_ASAHI_SIM_MAX_STEPS");
    CHECK(cb->status() == MTL::CommandBufferStatusError);
    CHECK(cb->error() != nullptr);
    CHECK(strstr(cb->error()->localizedDescription()->utf8String(), "hang") != nullptr);
    flag->release();
    out->release();
}

TEST(wild_store_is_refused)
{
    if (!simulatorMode()) {
        printf("  (skipped in hardware-only mode)\n");
        return;
    }
    MTL::Buffer* out = t::zeros(64);
    uint32_t stride = 4096;
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    e->setComputePipelineState(t::pipeline("faults.air", "wild_store"));
    e->setBuffer(out, 0, 0);
    e->setBytes(&stride, 4, 1);
    e->dispatchThreads(MTL::Size(32, 1, 1), MTL::Size(32, 1, 1));
    e->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    CHECK(cb->status() == MTL::CommandBufferStatusError);
    CHECK(cb->error() != nullptr);
    out->release();
}

TEST(queue_recovers_after_refusal)
{
    // Work after a refused command buffer still runs.
    MTL::Buffer* b = t::zeros(1024);
    CHECK(t::dispatch(t::pipeline("faults.air", "fine"), {b}, MTL::Size(256, 1, 1), MTL::Size(64, 1, 1)));
    CHECK(t::data<uint32_t>(b)[255] == 42);
    b->release();
}

TEST(failed_buffer_skips_later_work)
{
    if (!simulatorMode()) {
        printf("  (skipped in hardware-only mode)\n");
        return;
    }
    setenv("METAL_ASAHI_SIM_MAX_STEPS", "200000", 1);
    MTL::Buffer *flag = t::zeros(16), *out = t::zeros(64), *b = t::zeros(1024);
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    e->setComputePipelineState(t::pipeline("faults.air", "spin_forever"));
    e->setBuffer(flag, 0, 0);
    e->setBuffer(out, 0, 1);
    e->dispatchThreads(MTL::Size(32, 1, 1), MTL::Size(32, 1, 1));
    e->setComputePipelineState(t::pipeline("faults.air", "fine"));
    e->setBuffer(b, 0, 0);
    e->dispatchThreads(MTL::Size(256, 1, 1), MTL::Size(64, 1, 1));
    e->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    unsetenv("METAL_ASAHI_SIM_MAX_STEPS");
    CHECK(cb->status() == MTL::CommandBufferStatusError);
    CHECK(t::data<uint32_t>(b)[0] == 0);
    flag->release();
    out->release();
    b->release();
}

TEST(threadgroup_too_large_is_error)
{
    MTL::ComputePipelineState* p = t::pipeline("basic.air", "fill_index");
    MTL::Buffer* b = t::zeros(1 << 16);
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    e->setComputePipelineState(p);
    e->setBuffer(b, 0, 0);
    e->dispatchThreads(MTL::Size(4096, 1, 1), MTL::Size(2048, 1, 1));
    e->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    CHECK(cb->status() == MTL::CommandBufferStatusError);
    b->release();
}

TEST_MAIN
