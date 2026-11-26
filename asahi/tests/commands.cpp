// Command queues, command buffers and encoders: ordering, blits, handlers, events, timing.
#include "harness.h"

#include <atomic>
#include <mutex>
#include <thread>

static MTL::ComputePipelineState* addOne()
{
    return t::pipeline("args.air", "particles"); // x += y * dt, id = i * 3
}

TEST(dependent_dispatches_in_order)
{
    // Ten dispatches in one encoder, each reading the previous one's output.
    const uint32_t n = 4096;
    std::vector<float> a(n), b(n, 1.0f);
    for (uint32_t i = 0; i < n; i++) a[i] = (float)i;
    MTL::Buffer *ba = t::buffer(a), *bb = t::buffer(b);
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    e->setComputePipelineState(t::pipeline("basic.air", "vadd"));
    for (int k = 0; k < 10; k++) {
        e->setBuffer(k % 2 ? bb : ba, 0, 0);
        e->setBuffer(k % 2 ? ba : bb, 0, 1);
        e->setBuffer(k % 2 ? ba : bb, 0, 2);
        e->setBytes(&n, 4, 3);
        e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
    }
    e->endEncoding();
    CHECK(t::finish(cb));
    // Fibonacci-like: track on the CPU.
    std::vector<float> x = a, y = b;
    for (int k = 0; k < 10; k++) {
        if (k % 2 == 0)
            for (uint32_t i = 0; i < n; i++) y[i] = x[i] + y[i];
        else
            for (uint32_t i = 0; i < n; i++) x[i] = y[i] + x[i];
    }
    for (uint32_t i = 0; i < n; i++) {
        CHECK(t::data<float>(ba)[i] == x[i]);
        CHECK(t::data<float>(bb)[i] == y[i]);
    }
    ba->release();
    bb->release();
}

TEST(many_dispatches_batching)
{
    // More dispatches than one submission holds: 200 increments of one counter buffer.
    const uint32_t n = 1024;
    std::vector<float> p(n * 4, 0.0f);
    for (uint32_t i = 0; i < n; i++) p[4 * i + 1] = 1.0f;
    MTL::Buffer* b = t::buffer(p);
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    e->setComputePipelineState(addOne());
    e->setBuffer(b, 0, 0);
    float dt = 1.0f;
    e->setBytes(&dt, 4, 1);
    e->setBytes(&n, 4, 2);
    for (int k = 0; k < 200; k++) e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(64, 1, 1));
    e->endEncoding();
    CHECK(t::finish(cb));
    for (uint32_t i = 0; i < n; i++) CHECK_MSG(t::data<float>(b)[4 * i] == 200.0f, "i=%u: %f", i, t::data<float>(b)[4 * i]);
    b->release();
}

TEST(multiple_encoders)
{
    const uint32_t n = 512;
    MTL::Buffer* b = t::zeros(n * 4);
    MTL::Buffer* c = t::zeros(n * 4);
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    e->setComputePipelineState(t::pipeline("basic.air", "fill_index"));
    e->setBuffer(b, 0, 0);
    e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(64, 1, 1));
    e->endEncoding();
    MTL::ComputeCommandEncoder* e2 = cb->computeCommandEncoder();
    e2->setComputePipelineState(t::pipeline("args.air", "copy_u32"));
    e2->setBuffer(b, 0, 0);
    e2->setBuffer(c, 0, 1);
    e2->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(64, 1, 1));
    e2->endEncoding();
    CHECK(t::finish(cb));
    for (uint32_t i = 0; i < n; i++) CHECK(t::data<uint32_t>(c)[i] == i);
    b->release();
    c->release();
}

TEST(blit_copy_and_fill)
{
    const uint32_t n = 10000;
    MTL::Buffer* a = t::zeros(n * 4);
    MTL::Buffer* b = t::zeros(n * 4);
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    e->setComputePipelineState(t::pipeline("basic.air", "fill_index"));
    e->setBuffer(a, 0, 0);
    e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(100, 1, 1));
    e->endEncoding();
    MTL::BlitCommandEncoder* bl = cb->blitCommandEncoder();
    bl->fillBuffer(b, NS::Range(0, n * 4), 0xab);
    bl->copyFromBuffer(a, 400, b, 800, 4000);
    bl->endEncoding();
    MTL::ComputeCommandEncoder* e2 = cb->computeCommandEncoder();
    e2->setComputePipelineState(t::pipeline("args.air", "copy_u32"));
    e2->setBuffer(b, 0, 0);
    e2->setBuffer(a, 0, 1);
    e2->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(100, 1, 1));
    e2->endEncoding();
    CHECK(t::finish(cb));
    for (uint32_t i = 0; i < n; i++) {
        uint32_t want = (i >= 200 && i < 1200) ? i - 100 : 0xabababab;
        CHECK_MSG(t::data<uint32_t>(a)[i] == want, "i=%u: %#x want %#x", i, t::data<uint32_t>(a)[i], want);
    }
    a->release();
    b->release();
}

TEST(completed_handler)
{
    std::atomic<int> calls{0};
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    MTL::Buffer* b = t::zeros(256);
    MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    e->setComputePipelineState(t::pipeline("basic.air", "fill_index"));
    e->setBuffer(b, 0, 0);
    e->dispatchThreads(MTL::Size(64, 1, 1), MTL::Size(64, 1, 1));
    e->endEncoding();
    std::atomic<uint32_t> seen{0};
    cb->addCompletedHandler([&](MTL::CommandBuffer* c) {
        calls++;
        seen = t::data<uint32_t>(b)[63];
        (void)c;
    });
    cb->addScheduledHandler([&](MTL::CommandBuffer*) { calls += 10; });
    CHECK(t::finish(cb));
    // The completed handler runs before waitUntilCompleted returns.
    CHECK_MSG(calls == 11, "calls=%d", calls.load());
    CHECK(seen == 63);
    b->release();
}

TEST(status_transitions)
{
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    CHECK(cb->status() == MTL::CommandBufferStatusNotEnqueued);
    cb->enqueue();
    CHECK(cb->status() == MTL::CommandBufferStatusEnqueued);
    cb->commit();
    cb->waitUntilCompleted();
    CHECK(cb->status() == MTL::CommandBufferStatusCompleted);
    CHECK(cb->error() == nullptr);
}

TEST(gpu_times)
{
    MTL::Buffer* b = t::zeros(1 << 22);
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    e->setComputePipelineState(t::pipeline("basic.air", "fill_index"));
    e->setBuffer(b, 0, 0);
    e->dispatchThreads(MTL::Size(1 << 20, 1, 1), MTL::Size(256, 1, 1));
    e->endEncoding();
    CHECK(t::finish(cb));
    CHECK(cb->GPUEndTime() >= cb->GPUStartTime());
    CHECK(cb->GPUStartTime() > 0);
    CHECK(cb->kernelEndTime() >= cb->kernelStartTime());
    b->release();
}

TEST(many_command_buffers_in_flight)
{
    const int k = 32;
    const uint32_t n = 1024;
    std::vector<MTL::Buffer*> bufs;
    std::vector<MTL::CommandBuffer*> cbs;
    for (int i = 0; i < k; i++) {
        MTL::Buffer* b = t::zeros(n * 4);
        MTL::CommandBuffer* cb = t::queue()->commandBuffer();
        MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
        e->setComputePipelineState(t::pipeline("basic.air", "fill_index"));
        e->setBuffer(b, 0, 0);
        e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(128, 1, 1));
        e->endEncoding();
        cb->commit();
        bufs.push_back(b);
        cbs.push_back(cb);
    }
    cbs.back()->waitUntilCompleted();
    // Command buffers of one queue complete in order: all are done.
    for (int i = 0; i < k; i++) CHECK(cbs[i]->status() == MTL::CommandBufferStatusCompleted);
    for (int i = 0; i < k; i++) {
        CHECK(t::data<uint32_t>(bufs[i])[n - 1] == n - 1);
        bufs[i]->release();
    }
}

TEST(pipelined_dependent_command_buffers)
{
    // 64 command buffers in flight, each adding one to the same buffer; every 16th
    // also blits a snapshot (the CPU copy must see the GPU work of the ones before).
    // Completed handlers run in commit order.
    const int k = 64;
    const uint32_t n = 2048;
    std::vector<float> zero(n, 0.0f), one(n, 1.0f);
    MTL::Buffer *acc = t::buffer(zero), *ones = t::buffer(one);
    std::vector<MTL::Buffer*> snaps;
    std::vector<MTL::CommandBuffer*> cbs;
    std::vector<int> order;
    std::mutex om;
    for (int i = 0; i < k; i++) {
        MTL::CommandBuffer* cb = t::queue()->commandBuffer();
        MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
        e->setComputePipelineState(t::pipeline("basic.air", "vadd"));
        e->setBuffer(acc, 0, 0);
        e->setBuffer(ones, 0, 1);
        e->setBuffer(acc, 0, 2);
        e->setBytes(&n, 4, 3);
        e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
        e->endEncoding();
        if (i % 16 == 15) {
            MTL::Buffer* s = t::zeros(n * 4);
            MTL::BlitCommandEncoder* b = cb->blitCommandEncoder();
            b->copyFromBuffer(acc, 0, s, 0, n * 4);
            b->endEncoding();
            snaps.push_back(s);
        }
        cb->addCompletedHandler([&order, &om, i](MTL::CommandBuffer*) {
            std::lock_guard<std::mutex> l(om);
            order.push_back(i);
        });
        cb->commit();
        cbs.push_back(cb);
    }
    cbs.back()->waitUntilCompleted();
    for (int i = 0; i < k; i++) CHECK(cbs[i]->status() == MTL::CommandBufferStatusCompleted);
    {
        std::lock_guard<std::mutex> l(om);
        CHECK(order.size() == (size_t)k);
        for (int i = 0; i < k; i++) CHECK_MSG(order[i] == i, "handler %d ran as %d", i, order[i]);
    }
    for (uint32_t j = 0; j < n; j++) CHECK(t::data<float>(acc)[j] == (float)k);
    for (size_t s = 0; s < snaps.size(); s++) {
        for (uint32_t j = 0; j < n; j++) CHECK_MSG(t::data<float>(snaps[s])[j] == 16.0f * (s + 1), "snapshot %zu", s);
        snaps[s]->release();
    }
    // GPU intervals are ordered: each starts after the one before it started.
    for (int i = 1; i < k; i++) CHECK(cbs[i]->GPUStartTime() >= cbs[i - 1]->GPUStartTime());
    acc->release();
    ones->release();
}

TEST(two_queues)
{
    MTL::CommandQueue* q2 = t::device()->newCommandQueue();
    CHECK(q2 != nullptr);
    MTL::Buffer* a = t::zeros(4096);
    MTL::Buffer* b = t::zeros(4096);
    MTL::CommandBuffer* c1 = t::queue()->commandBuffer();
    MTL::CommandBuffer* c2 = q2->commandBuffer();
    for (auto [cb, buf] : {std::pair{c1, a}, std::pair{c2, b}}) {
        MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
        e->setComputePipelineState(t::pipeline("basic.air", "fill_index"));
        e->setBuffer(buf, 0, 0);
        e->dispatchThreads(MTL::Size(1024, 1, 1), MTL::Size(64, 1, 1));
        e->endEncoding();
        cb->commit();
    }
    c1->waitUntilCompleted();
    c2->waitUntilCompleted();
    CHECK(t::data<uint32_t>(a)[1000] == 1000);
    CHECK(t::data<uint32_t>(b)[1000] == 1000);
    a->release();
    b->release();
    q2->release();
}

TEST(shared_event_signal_wait)
{
    MTL::SharedEvent* ev = t::device()->newSharedEvent();
    CHECK(ev->signaledValue() == 0);
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    cb->encodeSignalEvent(ev, 5);
    CHECK(t::finish(cb));
    CHECK(ev->signaledValue() == 5);
    // A command buffer that waits for the CPU to signal.
    MTL::Buffer* b = t::zeros(256);
    MTL::CommandBuffer* cb2 = t::queue()->commandBuffer();
    cb2->encodeWait(ev, 7);
    MTL::ComputeCommandEncoder* e = cb2->computeCommandEncoder();
    e->setComputePipelineState(t::pipeline("basic.air", "fill_index"));
    e->setBuffer(b, 0, 0);
    e->dispatchThreads(MTL::Size(64, 1, 1), MTL::Size(64, 1, 1));
    e->endEncoding();
    cb2->commit();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(cb2->status() != MTL::CommandBufferStatusCompleted);
    ev->setSignaledValue(7);
    cb2->waitUntilCompleted();
    CHECK(cb2->status() == MTL::CommandBufferStatusCompleted);
    CHECK(t::data<uint32_t>(b)[63] == 63);
    b->release();
    ev->release();
}

TEST(concurrent_dispatch_type)
{
    // Independent dispatches in a concurrent encoder, then a barrier and a dependent one.
    const uint32_t n = 2048;
    MTL::Buffer *a = t::zeros(n * 4), *b = t::zeros(n * 4), *c = t::zeros(n * 4);
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
    CHECK(e->dispatchType() == MTL::DispatchTypeConcurrent);
    e->setComputePipelineState(t::pipeline("basic.air", "fill_index"));
    e->setBuffer(a, 0, 0);
    e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(64, 1, 1));
    e->setBuffer(b, 0, 0);
    e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(64, 1, 1));
    e->memoryBarrier(MTL::BarrierScopeBuffers);
    e->setComputePipelineState(t::pipeline("args.air", "copy_u32"));
    e->setBuffer(b, 0, 0);
    e->setBuffer(c, 0, 1);
    e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(64, 1, 1));
    e->endEncoding();
    CHECK(t::finish(cb));
    for (uint32_t i = 0; i < n; i++) {
        CHECK(t::data<uint32_t>(a)[i] == i);
        CHECK(t::data<uint32_t>(c)[i] == i);
    }
    a->release();
    b->release();
    c->release();
}

TEST(compute_pass_descriptor)
{
    MTL::ComputePassDescriptor* d = MTL::ComputePassDescriptor::computePassDescriptor();
    d->setDispatchType(MTL::DispatchTypeSerial);
    MTL::Buffer* b = t::zeros(256);
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder(d);
    e->setComputePipelineState(t::pipeline("basic.air", "fill_index"));
    e->setBuffer(b, 0, 0);
    e->dispatchThreads(MTL::Size(64, 1, 1), MTL::Size(32, 1, 1));
    e->endEncoding();
    CHECK(t::finish(cb));
    CHECK(t::data<uint32_t>(b)[40] == 40);
    b->release();
}

TEST(empty_command_buffer)
{
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    CHECK(t::finish(cb));
}

TEST(labels)
{
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    cb->setLabel(MTLSTR("my command buffer"));
    CHECK(strcmp(cb->label()->utf8String(), "my command buffer") == 0);
    MTL::Buffer* b = t::zeros(64);
    b->setLabel(MTLSTR("buf"));
    CHECK(strcmp(b->label()->utf8String(), "buf") == 0);
    b->release();
}

TEST(unretained_references)
{
    MTL::Buffer* b = t::zeros(4096);
    MTL::CommandBuffer* cb = t::queue()->commandBufferWithUnretainedReferences();
    MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    e->setComputePipelineState(t::pipeline("basic.air", "fill_index"));
    e->setBuffer(b, 0, 0);
    e->dispatchThreads(MTL::Size(1024, 1, 1), MTL::Size(64, 1, 1));
    e->endEncoding();
    CHECK(t::finish(cb));
    CHECK(t::data<uint32_t>(b)[1023] == 1023);
    b->release();
}

TEST_MAIN
