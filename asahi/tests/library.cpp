// MTLLibrary: .air, .metallib, data, errors, functions, buffers' memory.
#include "harness.h"

#include <fstream>
#include <iterator>
#include <dirent.h>
#include <sys/mman.h>

TEST(function_names)
{
    MTL::Library* lib = t::library("basic.air");
    CHECK(lib);
    NS::Array* names = lib->functionNames();
    CHECK(names->count() == 3);
    bool vadd = false;
    for (NS::UInteger i = 0; i < names->count(); i++) vadd |= !strcmp(names->object<NS::String>(i)->utf8String(), "vadd");
    CHECK(vadd);
}

TEST(missing_function_is_nil)
{
    MTL::Library* lib = t::library("basic.air");
    CHECK(lib->newFunction(MTLSTR("no_such_kernel")) == nullptr);
    MTL::Function* f = lib->newFunction(MTLSTR("vadd"));
    CHECK(f);
    CHECK(strcmp(f->name()->utf8String(), "vadd") == 0);
    CHECK(f->functionType() == MTL::FunctionTypeKernel);
    f->release();
}

TEST(missing_file_error)
{
    NS::Error* err = nullptr;
    MTL::Library* lib = t::device()->newLibrary(NS::URL::fileURLWithPath(MTLSTR("/nonexistent/x.air")), &err);
    CHECK(lib == nullptr);
    CHECK(err != nullptr);
}

TEST(not_air_error)
{
    NS::Error* err = nullptr;
    const char junk[] = "this is not bitcode";
    dispatch_data_t d = dispatch_data_create(junk, sizeof junk, nullptr, nullptr);
    MTL::Library* lib = t::device()->newLibrary(d, &err);
    CHECK(lib == nullptr);
    CHECK(err != nullptr);
    CHECK(strstr(err->localizedDescription()->utf8String(), "AIR") != nullptr);
}

TEST(msl_source_rejected)
{
    NS::Error* err = nullptr;
    MTL::Library* lib = t::device()->newLibrary(MTLSTR("kernel void k() {}"), nullptr, &err);
    CHECK(lib == nullptr);
    CHECK(err != nullptr);
}

TEST(library_from_data)
{
    std::ifstream f(t::kernelPath("basic.air"), std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK(!bytes.empty());
    dispatch_data_t d = dispatch_data_create(bytes.data(), bytes.size(), nullptr, nullptr);
    NS::Error* err = nullptr;
    MTL::Library* lib = t::device()->newLibrary(d, &err);
    CHECK(lib != nullptr);
    MTL::Function* fn = lib->newFunction(MTLSTR("fill_index"));
    CHECK(fn);
    MTL::ComputePipelineState* p = t::device()->newComputePipelineState(fn, &err);
    CHECK(p);
    MTL::Buffer* b = t::zeros(1024);
    CHECK(t::dispatch(p, {b}, MTL::Size(256, 1, 1), MTL::Size(64, 1, 1)));
    CHECK(t::data<uint32_t>(b)[255] == 255);
    b->release();
    p->release();
    fn->release();
    lib->release();
}

TEST(metallib_container)
{
    // basic.metallib: the same kernels in a .metallib (one module per kernel), built by tools/air-build.sh.
    NS::Error* err = nullptr;
    MTL::Library* lib = t::device()->newLibrary(NS::URL::fileURLWithPath(NS::String::string(t::kernelPath("basic.metallib").c_str(), NS::UTF8StringEncoding)), &err);
    CHECK_MSG(lib != nullptr, "%s", err ? err->localizedDescription()->utf8String() : "?");
    CHECK(lib->functionNames()->count() == 3);
    MTL::Function* fn = lib->newFunction(MTLSTR("vadd"));
    CHECK(fn);
    MTL::ComputePipelineState* p = t::device()->newComputePipelineState(fn, &err);
    CHECK(p);
    const uint32_t n = 100;
    std::vector<float> a(n, 1.0f), b(n, 2.0f);
    MTL::Buffer *ba = t::buffer(a), *bb = t::buffer(b), *bc = t::zeros(n * 4);
    CHECK(t::dispatch(p, {ba, bb, bc, t::Arg::value(n)}, MTL::Size(n, 1, 1), MTL::Size(32, 1, 1)));
    CHECK(t::data<float>(bc)[99] == 3.0f);
    ba->release();
    bb->release();
    bc->release();
    p->release();
    fn->release();
    lib->release();
}

TEST(pipeline_descriptor)
{
    MTL::Function* fn = t::library("basic.air")->newFunction(MTLSTR("fill_index"));
    MTL::ComputePipelineDescriptor* d = MTL::ComputePipelineDescriptor::alloc()->init();
    d->setComputeFunction(fn);
    d->setMaxTotalThreadsPerThreadgroup(128);
    d->setLabel(MTLSTR("fill"));
    CHECK(d->computeFunction() == fn);
    NS::Error* err = nullptr;
    MTL::ComputePipelineState* p = t::device()->newComputePipelineState(d, MTL::PipelineOptionNone, nullptr, &err);
    CHECK(p);
    CHECK(p->maxTotalThreadsPerThreadgroup() == 128);
    MTL::Buffer* b = t::zeros(1024);
    CHECK(t::dispatch(p, {b}, MTL::Size(256, 1, 1), MTL::Size(128, 1, 1)));
    CHECK(t::data<uint32_t>(b)[200] == 200);
    b->release();
    p->release();
    d->release();
    fn->release();
}

TEST(pipeline_async)
{
    MTL::Function* fn = t::library("basic.air")->newFunction(MTLSTR("fill_index"));
    MTL::ComputePipelineState* got = nullptr;
    t::device()->newComputePipelineState(fn, [&](MTL::ComputePipelineState* p, NS::Error*) {
        got = p;
        if (p) p->retain();
    });
    CHECK(got != nullptr);
    got->release();
    fn->release();
}

TEST(two_libraries_same_kernel_names)
{
    MTL::Library* a = t::library("basic.air");
    MTL::Library* b = t::library("args.air");
    CHECK(a && b && a != b);
    MTL::Function* fa = a->newFunction(MTLSTR("vadd"));
    MTL::Function* fb = b->newFunction(MTLSTR("vadd"));
    CHECK(fa && !fb);
    fa->release();
}

TEST(buffer_with_bytes)
{
    std::vector<uint32_t> v(1000);
    for (uint32_t i = 0; i < 1000; i++) v[i] = i * i;
    MTL::Buffer* b = t::device()->newBuffer(v.data(), v.size() * 4, MTL::ResourceStorageModeShared);
    CHECK(b->length() == 4000);
    CHECK(memcmp(b->contents(), v.data(), 4000) == 0);
    CHECK(b->allocatedSize() >= 4000);
    CHECK(b->storageMode() == MTL::StorageModeShared);
    b->release();
}

TEST(buffer_private_storage)
{
    MTL::Buffer* b = t::device()->newBuffer(4096, MTL::ResourceStorageModePrivate);
    CHECK(b != nullptr);
    CHECK(b->contents() == nullptr);
    CHECK(b->storageMode() == MTL::StorageModePrivate);
    // The GPU can still write it, and a blit can copy it out.
    CHECK(t::dispatch(t::pipeline("basic.air", "fill_index"), {b}, MTL::Size(1024, 1, 1), MTL::Size(64, 1, 1)));
    MTL::Buffer* out = t::zeros(4096);
    MTL::CommandBuffer* cb = t::queue()->commandBuffer();
    MTL::BlitCommandEncoder* bl = cb->blitCommandEncoder();
    bl->copyFromBuffer(b, 0, out, 0, 4096);
    bl->endEncoding();
    CHECK(t::finish(cb));
    CHECK(t::data<uint32_t>(out)[1023] == 1023);
    out->release();
    b->release();
}

TEST(buffer_no_copy_shared_pages)
{
    const size_t len = 4 * 16384;
    void* mem = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(mem != MAP_FAILED);
    uint32_t* u = (uint32_t*)mem;
    u[7] = 77;
    __block bool deallocated = false;
    MTL::Buffer* b = t::device()->newBuffer(mem, len, MTL::ResourceStorageModeShared, ^(void*, NS::UInteger) { deallocated = true; });
    CHECK(b);
    CHECK(((uint32_t*)b->contents())[7] == 77);
    CHECK(t::dispatch(t::pipeline("basic.air", "fill_index"), {b}, MTL::Size(len / 4, 1, 1), MTL::Size(256, 1, 1)));
    // The GPU's writes are visible through the client's own pointer.
    CHECK(u[7] == 7);
    CHECK(u[len / 4 - 1] == len / 4 - 1);
    b->release();
    CHECK(deallocated);
    CHECK(u[100] == 100); // still readable afterwards
    munmap(mem, len);
}

TEST(big_buffer)
{
    const size_t len = 1ull << 30; // 1 GiB
    MTL::Buffer* b = t::device()->newBuffer(len, MTL::ResourceStorageModeShared);
    CHECK(b != nullptr);
    CHECK(b->length() == len);
    ((uint8_t*)b->contents())[len - 1] = 9;
    CHECK(((uint8_t*)b->contents())[len - 1] == 9);
    b->release();
}

// The compile cache (make check points METAL_ASAHI_CACHE_DIR at build/cache): a library loads from
// it, and entries that are corrupt or truncated are compiled again, with the same results.
TEST(compile_cache_bad_entries)
{
    const char* dir = getenv("METAL_ASAHI_CACHE_DIR");
    if (!dir) return; // not hermetic: leave the user's cache alone
    auto load = [] {
        NS::Error* err = nullptr;
        MTL::Library* lib = t::device()->newLibrary(NS::URL::fileURLWithPath(NS::String::string(t::kernelPath("basic.air").c_str(),
                                                                                                  NS::UTF8StringEncoding)),
                                                    &err);
        return lib;
    };
    auto run = [](MTL::Library* lib) {
        MTL::Function* f = lib->newFunction(MTLSTR("fill_index"));
        NS::Error* err = nullptr;
        MTL::ComputePipelineState* p = t::device()->newComputePipelineState(f, &err);
        f->release();
        MTL::Buffer* b = t::zeros(4096 * 4);
        bool ok = t::dispatch(p, {b}, MTL::Size(4096, 1, 1), MTL::Size(64, 1, 1));
        for (uint32_t i = 0; ok && i < 4096; i++) ok = t::data<uint32_t>(b)[i] == i;
        b->release();
        p->release();
        return ok;
    };
    MTL::Library* lib = load(); // writes (or reads) the entry
    CHECK(lib && run(lib));
    lib->release();
    std::vector<std::string> entries;
    if (DIR* d = opendir(dir)) {
        while (dirent* e = readdir(d))
            if (strstr(e->d_name, ".agx")) entries.push_back(std::string(dir) + "/" + e->d_name);
        closedir(d);
    }
    CHECK(!entries.empty());
    for (int variant = 0; variant < 3; variant++) {
        for (size_t i = 0; i < entries.size(); i++) {
            std::ifstream in(entries[i], std::ios::binary);
            std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (variant == 0) bytes.resize(bytes.size() / 2);          // truncated
            else if (variant == 1) for (size_t k = 8; k < bytes.size(); k += 7) bytes[k] ^= 0x5a; // corrupted after the header
            else bytes[bytes.size() / 2] ^= 0x01;                       // one bit (in the code): caught by the payload hash
            std::ofstream(entries[i], std::ios::binary | std::ios::trunc).write(bytes.data(), (std::streamsize)bytes.size());
        }
        lib = load();
        CHECK_MSG(lib && run(lib), "variant %d", variant);
        lib->release();
    }
}

TEST(device_properties)
{
    MTL::Device* d = t::device();
    CHECK(d->supportsFamily(MTL::GPUFamilyApple7));
    CHECK(d->supportsFamily(MTL::GPUFamilyMetal3));
    CHECK(!d->supportsFamily(MTL::GPUFamilyApple9));
    CHECK(d->maxThreadgroupMemoryLength() == 32768);
    CHECK(d->recommendedMaxWorkingSetSize() > (1ull << 30));
    CHECK(d->registryID() != 0);
    NS::Array* all = MTL::CopyAllDevices();
    CHECK(all->count() == 1);
    all->release();
}

TEST_MAIN
