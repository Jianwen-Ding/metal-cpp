// libmetal_asahi: direct Asahi DRM ioctls. Patterned on the repository's own
// raw-ioctl harness (Apple-M2-Playground tests/drm_compute.c).
#include "drm.h"

#include <drm/asahi_drm.h>
#include <drm/drm.h>

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

namespace masahi {

static const uint64_t kPage = 16384;
static const uint64_t kTsBytes = 16 * kPage; // 16384 timestamp slots of 16 bytes

void VaHeap::init(uint64_t base, uint64_t size)
{
    base_ = base;
    free_.clear();
    free_[base] = size;
}

uint64_t VaHeap::alloc(uint64_t size, uint64_t align)
{
    std::lock_guard<std::mutex> l(m_);
    for (auto it = free_.begin(); it != free_.end(); ++it) {
        uint64_t start = (it->first + align - 1) & ~(align - 1);
        uint64_t end = it->first + it->second;
        if (start + size > end) continue;
        uint64_t s0 = it->first;
        free_.erase(it);
        if (start > s0) free_[s0] = start - s0;
        if (start + size < end) free_[start + size] = end - (start + size);
        return start;
    }
    return 0;
}

void VaHeap::free(uint64_t va, uint64_t size)
{
    std::lock_guard<std::mutex> l(m_);
    auto it = free_.emplace(va, size).first;
    auto next = std::next(it);
    if (next != free_.end() && it->first + it->second == next->first) {
        it->second += next->second;
        free_.erase(next);
    }
    if (it != free_.begin()) {
        auto prev = std::prev(it);
        if (prev->first + prev->second == it->first) {
            prev->second += it->second;
            free_.erase(it);
        }
    }
}

static int xioctl(int fd, unsigned long req, void* arg)
{
    int r;
    do r = ioctl(fd, req, arg);
    while (r < 0 && (errno == EINTR || errno == EAGAIN));
    return r;
}

DrmDevice* DrmDevice::open(std::string& error)
{
    const char* forced = getenv("METAL_ASAHI_DRM_NODE");
    std::vector<std::string> nodes;
    if (forced && *forced) nodes.push_back(forced);
    else
        for (int i = 128; i < 192; i++) nodes.push_back("/dev/dri/renderD" + std::to_string(i));
    for (auto& path : nodes) {
        int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
        if (fd < 0) continue;
        char name[64] = {0};
        drm_version v{};
        v.name = name;
        v.name_len = sizeof name - 1;
        if (xioctl(fd, DRM_IOCTL_VERSION, &v) != 0 || strcmp(name, "asahi") != 0) {
            close(fd);
            continue;
        }
        auto* d = new DrmDevice();
        if (!d->init(fd, error)) {
            delete d;
            return nullptr;
        }
        return d;
    }
    error = "no Asahi DRM render node (/dev/dri/renderD*) could be opened";
    return nullptr;
}

DrmDevice* DrmDevice::openSimulated(const std::string& chip, std::string& error)
{
    // agx-g<generation><variant>; the core counts are the full parts' (M2 Max: 38 cores in 4 clusters)
    unsigned gen = 0;
    char v = 0;
    if (sscanf(chip.c_str(), "agx-g%u%c", &gen, &v) != 2 || gen < 13 || gen > 17) {
        error = "METAL_ASAHI_SIM_GPU: expected agx-g13g ... agx-g16x style name, got '" + chip + "'";
        return nullptr;
    }
    auto* d = new DrmDevice();
    d->simulated_ = true;
    GpuInfo& i = d->info_;
    i.generation = gen;
    i.variant = (char)toupper((unsigned char)v);
    i.revision = 0x11;
    i.clusters = v == 'c' ? 4 : v == 'd' ? 8 : v == 's' ? 2 : 1;
    i.cores = v == 'c' ? 38 : v == 'd' ? 76 : v == 's' ? 19 : 10;
    i.maxKhz = 1398000;
    i.maxCommandsPerSubmission = 64;
    i.g14x = gen >= 14 && i.clusters > 1;
    i.chipName = chip;
    static const char* names[] = {"", "", "", "", "", "", "", "", "", "", "", "", "", "M1", "M2", "M3", "M4", "M5"};
    std::string tier = v == 'g' ? "" : v == 's' ? " Pro" : v == 'c' ? " Max" : v == 'd' ? " Ultra" : "";
    i.marketing = std::string("Apple ") + names[gen] + tier + " (AGX Performance Simulator)";
    // The VM a G14 gives (Asahi 6.19): 4 KiB .. 2^39, the top 32 GiB the kernel's.
    i.vmStart = 0x4000;
    i.vmEnd = 1ull << 39;
    d->initLayout(i.vmStart, i.vmEnd, i.vmEnd - (32ull << 30));
    d->tsBo_ = d->alloc(kTsBytes, false);
    d->tsFree_.push_back({1, (uint32_t)(kTsBytes / 16) - 1});
    return d;
}

void DrmDevice::initLayout(uint64_t vmStart, uint64_t vmEnd, uint64_t kernelStart)
{
    uint64_t shaderBase = vmStart > (1ull << 36) ? vmStart : (1ull << 36);
    shaderBase = (shaderBase + 0xffffffffull) & ~0xffffffffull;
    uint64_t dataBase = shaderBase + 0x100000000ull;
    uscBase_ = shaderBase;
    uscHeap_.init(shaderBase + kPage, 0x100000000ull - kPage);
    dataHeap_.init(dataBase + kPage, kernelStart - dataBase - kPage);
}

bool DrmDevice::init(int fd, std::string& error)
{
    fd_ = fd;
    drm_asahi_params_global p{};
    drm_asahi_get_params gp{};
    gp.param_group = 0;
    gp.pointer = (uint64_t)(uintptr_t)&p;
    gp.size = sizeof p;
    if (xioctl(fd_, DRM_IOCTL_ASAHI_GET_PARAMS, &gp) != 0) {
        error = "DRM_IOCTL_ASAHI_GET_PARAMS failed: " + std::string(strerror(errno));
        return false;
    }
    info_.generation = p.gpu_generation;
    info_.variant = (char)p.gpu_variant;
    info_.revision = p.gpu_revision;
    info_.clusters = p.num_clusters_total;
    info_.vmStart = p.vm_start;
    info_.vmEnd = p.vm_end;
    info_.timestampHz = p.command_timestamp_frequency_hz ? p.command_timestamp_frequency_hz : 24000000;
    info_.maxKhz = p.max_frequency_khz;
    info_.maxCommandsPerSubmission = p.max_commands_per_submission ? p.max_commands_per_submission : 1;
    info_.features = p.features;
    for (uint32_t c = 0; c < p.num_clusters_total && c < DRM_ASAHI_MAX_CLUSTERS; c++)
        info_.cores += (uint32_t)__builtin_popcountll(p.core_masks[c]);
    info_.g14x = p.gpu_generation >= 14 && p.num_clusters_total > 1;
    char v = (char)tolower((unsigned char)p.gpu_variant);
    info_.chipName = "agx-g" + std::to_string(p.gpu_generation) + v;
    static const char* names[] = {"", "", "", "", "", "", "", "", "", "", "", "", "", "M1", "M2", "M3", "M4", "M5"};
    std::string chip = p.gpu_generation < 18 ? names[p.gpu_generation] : "Apple";
    std::string tier = v == 'g' ? "" : v == 's' ? " Pro" : v == 'c' ? " Max" : v == 'd' ? " Ultra" : "";
    char rev[32];
    snprintf(rev, sizeof rev, "%c%u", 'A' + (p.gpu_revision >> 4), p.gpu_revision & 15);
    info_.marketing = "Apple " + chip + tier;

    // Mesa's layout: USC heap at the 4 GiB-aligned bottom above 2^36, data after it, kernel range on top.
    uint64_t shaderBase = p.vm_start > (1ull << 36) ? p.vm_start : (1ull << 36);
    shaderBase = (shaderBase + 0xffffffffull) & ~0xffffffffull;
    uint64_t dataBase = shaderBase + 0x100000000ull;
    uint64_t ksize = p.vm_kernel_min_size > (32ull << 30) ? p.vm_kernel_min_size : (32ull << 30);
    drm_asahi_vm_create vc{};
    vc.kernel_start = p.vm_end - ksize;
    vc.kernel_end = p.vm_end;
    if (dataBase + (1ull << 30) >= vc.kernel_start) {
        error = "unexpected GPU VM layout";
        return false;
    }
    if (xioctl(fd_, DRM_IOCTL_ASAHI_VM_CREATE, &vc) != 0) {
        error = "DRM_IOCTL_ASAHI_VM_CREATE failed: " + std::string(strerror(errno));
        return false;
    }
    vm_ = vc.vm_id;
    uscBase_ = shaderBase;
    // Leave the first page of each heap unused, so no object sits at offset 0.
    uscHeap_.init(shaderBase + kPage, 0x100000000ull - kPage);
    dataHeap_.init(dataBase + kPage, vc.kernel_start - dataBase - kPage);

    tsBo_ = alloc(kTsBytes, false);
    if (!tsBo_) {
        error = "cannot allocate the timestamp buffer";
        return false;
    }
    drm_asahi_gem_bind_object bo{};
    bo.op = DRM_ASAHI_BIND_OBJECT_OP_BIND;
    bo.flags = DRM_ASAHI_BIND_OBJECT_USAGE_TIMESTAMPS;
    bo.handle = tsBo_->handle;
    bo.range = kTsBytes;
    if (xioctl(fd_, DRM_IOCTL_ASAHI_GEM_BIND_OBJECT, &bo) != 0) {
        error = "binding the timestamp buffer failed: " + std::string(strerror(errno));
        return false;
    }
    tsObject_ = bo.object_handle;
    tsFree_.push_back({1, (uint32_t)(kTsBytes / 16) - 1}); // slot 0 unused
    return true;
}

DrmDevice::~DrmDevice()
{
    if (simulated_) {
        if (tsBo_) free(tsBo_);
        return;
    }
    if (fd_ >= 0) {
        if (vm_) {
            drm_asahi_vm_destroy vd{};
            vd.vm_id = vm_;
            xioctl(fd_, DRM_IOCTL_ASAHI_VM_DESTROY, &vd);
        }
        close(fd_);
    }
}

Bo* DrmDevice::alloc(uint64_t size, bool usc, bool writable)
{
    size = (size + kPage - 1) & ~(kPage - 1);
    if (!size) size = kPage;
    if (simulated_) {
        void* m = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) return nullptr;
        uint64_t align = size >= (2ull << 20) ? (2ull << 20) : kPage;
        uint64_t va = (usc ? uscHeap_ : dataHeap_).alloc(size, align);
        if (!va) {
            munmap(m, size);
            return nullptr;
        }
        auto* b = new Bo();
        std::lock_guard<std::mutex> l(m_);
        b->handle = nextHandle_++;
        b->size = size;
        b->usc = usc;
        b->map = (uint8_t*)m;
        b->va = va;
        allocated_ += size;
        return b;
    }
    drm_asahi_gem_create gc{};
    gc.size = size;
    gc.flags = DRM_ASAHI_GEM_WRITEBACK;
    gc.vm_id = 0;
    if (xioctl(fd_, DRM_IOCTL_ASAHI_GEM_CREATE, &gc) != 0) return nullptr;
    auto* b = new Bo();
    b->handle = gc.handle;
    b->size = size;
    b->usc = usc;
    drm_asahi_gem_mmap_offset mo{};
    mo.handle = b->handle;
    if (xioctl(fd_, DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET, &mo) != 0) goto fail;
    b->mmapOffset = mo.offset;
    b->map = (uint8_t*)mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, (off_t)mo.offset);
    if (b->map == MAP_FAILED) {
        b->map = nullptr;
        goto fail;
    }
    {
        uint64_t align = size >= (2ull << 20) ? (2ull << 20) : kPage;
        b->va = (usc ? uscHeap_ : dataHeap_).alloc(size, align);
        if (!b->va) goto fail;
        drm_asahi_gem_bind_op op{};
        op.flags = DRM_ASAHI_BIND_READ | (writable && !usc ? DRM_ASAHI_BIND_WRITE : 0);
        op.handle = b->handle;
        op.range = size;
        op.addr = b->va;
        drm_asahi_vm_bind vb{};
        vb.vm_id = vm_;
        vb.num_binds = 1;
        vb.stride = sizeof op;
        vb.userptr = (uint64_t)(uintptr_t)&op;
        if (xioctl(fd_, DRM_IOCTL_ASAHI_VM_BIND, &vb) != 0) {
            (usc ? uscHeap_ : dataHeap_).free(b->va, size);
            b->va = 0;
            goto fail;
        }
    }
    {
        std::lock_guard<std::mutex> l(m_);
        allocated_ += size;
    }
    return b;
fail:
    if (b->map) munmap(b->map, size);
    drm_gem_close gcl{};
    gcl.handle = b->handle;
    xioctl(fd_, DRM_IOCTL_GEM_CLOSE, &gcl);
    delete b;
    return nullptr;
}

void DrmDevice::free(Bo* b)
{
    if (!b) return;
    if (simulated_) {
        if (b->va) (b->usc ? uscHeap_ : dataHeap_).free(b->va, b->size);
        if (b->map) munmap(b->map, b->size);
        std::lock_guard<std::mutex> l(m_);
        allocated_ -= b->size;
        delete b;
        return;
    }
    if (b->va) {
        drm_asahi_gem_bind_op op{};
        op.flags = DRM_ASAHI_BIND_UNBIND;
        op.range = b->size;
        op.addr = b->va;
        drm_asahi_vm_bind vb{};
        vb.vm_id = vm_;
        vb.num_binds = 1;
        vb.stride = sizeof op;
        vb.userptr = (uint64_t)(uintptr_t)&op;
        xioctl(fd_, DRM_IOCTL_ASAHI_VM_BIND, &vb);
        (b->usc ? uscHeap_ : dataHeap_).free(b->va, b->size);
    }
    if (b->map) munmap(b->map, b->size);
    drm_gem_close gcl{};
    gcl.handle = b->handle;
    xioctl(fd_, DRM_IOCTL_GEM_CLOSE, &gcl);
    {
        std::lock_guard<std::mutex> l(m_);
        allocated_ -= b->size;
    }
    delete b;
}

uint32_t DrmDevice::createQueue(uint32_t priority)
{
    if (simulated_) return 1;

    drm_asahi_queue_create qc{};
    qc.vm_id = vm_;
    qc.priority = priority;
    qc.usc_exec_base = uscBase_;
    if (xioctl(fd_, DRM_IOCTL_ASAHI_QUEUE_CREATE, &qc) != 0) return ~0u;
    return qc.queue_id;
}

void DrmDevice::destroyQueue(uint32_t q)
{
    if (simulated_) return;

    drm_asahi_queue_destroy qd{};
    qd.queue_id = q;
    xioctl(fd_, DRM_IOCTL_ASAHI_QUEUE_DESTROY, &qd);
}

uint32_t DrmDevice::createSyncobj()
{
    if (simulated_) {
        std::lock_guard<std::mutex> l(m_);
        return nextHandle_++;
    }

    drm_syncobj_create sc{};
    if (xioctl(fd_, DRM_IOCTL_SYNCOBJ_CREATE, &sc) != 0) return 0;
    return sc.handle;
}

void DrmDevice::destroySyncobj(uint32_t h)
{
    if (simulated_) return;

    drm_syncobj_destroy sd{};
    sd.handle = h;
    xioctl(fd_, DRM_IOCTL_SYNCOBJ_DESTROY, &sd);
}

static uint64_t monoNs()
{
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

int DrmDevice::waitSyncobj(uint32_t h, uint64_t timeoutNs)
{
    if (simulated_) return 0;

    drm_syncobj_wait sw{};
    sw.handles = (uint64_t)(uintptr_t)&h;
    sw.count_handles = 1;
    sw.timeout_nsec = (int64_t)(monoNs() + timeoutNs);
    int r = ioctl(fd_, DRM_IOCTL_SYNCOBJ_WAIT, &sw);
    return r == 0 ? 0 : errno;
}

void DrmDevice::resetSyncobj(uint32_t h)
{
    if (simulated_) return;

    drm_syncobj_array a{};
    a.handles = (uint64_t)(uintptr_t)&h;
    a.count_handles = 1;
    xioctl(fd_, DRM_IOCTL_SYNCOBJ_RESET, &a);
}

uint64_t DrmDevice::gpuTimeNs()
{
    if (simulated_) return monoNs();

    drm_asahi_get_time t{};
    if (xioctl(fd_, DRM_IOCTL_ASAHI_GET_TIME, &t) != 0) return 0;
    return t.gpu_timestamp;
}

uint32_t DrmDevice::allocTimestamps(uint32_t n)
{
    std::lock_guard<std::mutex> l(tsMutex_);
    for (auto& r : tsFree_) {
        if (r.second < n) continue;
        uint32_t base = r.first;
        r.first += n;
        r.second -= n;
        return base;
    }
    return 0;
}

void DrmDevice::freeTimestamps(uint32_t base, uint32_t n)
{
    if (!base) return;
    std::lock_guard<std::mutex> l(tsMutex_);
    tsFree_.push_back({base, n});
}

std::pair<uint64_t, uint64_t> DrmDevice::timestamp(uint32_t slot) const
{
    std::pair<uint64_t, uint64_t> t{0, 0};
    if (slot) memcpy(&t, tsBo_->map + 16ull * slot, 16);
    return t;
}

bool DrmDevice::submit(uint32_t queue, const std::vector<ComputeCommand>& cmds, uint32_t inSync, uint32_t outSync,
                       uint32_t tsBase, bool wait, std::string& error)
{
    struct Cmd {
        drm_asahi_cmd_header h;
        drm_asahi_cmd_compute c;
    };
    if (simulated_) {
        error = "the simulated device has no GPU to submit to";
        return false;
    }
    static constexpr size_t kMaxCmds = 64;
    if (cmds.empty() || cmds.size() > info_.maxCommandsPerSubmission || cmds.size() > kMaxCmds) {
        error = "bad command count";
        return false;
    }
    Cmd buf[kMaxCmds];
    if (tsBase) memset(tsBo_->map + 16ull * tsBase, 0, 16 * cmds.size());
    for (size_t k = 0; k < cmds.size(); k++) {
        Cmd& c = buf[k];
        memset(&c, 0, sizeof c);
        c.h.cmd_type = DRM_ASAHI_CMD_COMPUTE;
        c.h.size = sizeof(drm_asahi_cmd_compute);
        c.h.vdm_barrier = DRM_ASAHI_BARRIER_NONE;
        c.h.cdm_barrier = k == 0 ? 0 : (cmds[k].barrierOnPrevious ? (uint16_t)k : DRM_ASAHI_BARRIER_NONE);
        c.c.cdm_ctrl_stream_base = cmds[k].cdmBase;
        c.c.cdm_ctrl_stream_end = cmds[k].cdmEnd;
        c.c.sampler_heap = cmds[k].samplerHeap;
        c.c.sampler_count = cmds[k].samplerCount;
        c.c.helper.binary = cmds[k].helperBinary;
        c.c.helper.cfg = cmds[k].helperCfg;
        c.c.helper.data = cmds[k].helperData;
        if (tsBase) {
            c.c.ts.start.handle = tsObject_;
            c.c.ts.start.offset = (uint32_t)(16 * (tsBase + k));
            c.c.ts.end.handle = tsObject_;
            c.c.ts.end.offset = (uint32_t)(16 * (tsBase + k) + 8);
        }
    }
    drm_asahi_sync syncs[2]{};
    uint32_t nin = 0;
    if (inSync) {
        syncs[0].sync_type = DRM_ASAHI_SYNC_SYNCOBJ;
        syncs[0].handle = inSync;
        nin = 1;
    }
    syncs[nin].sync_type = DRM_ASAHI_SYNC_SYNCOBJ;
    syncs[nin].handle = outSync;
    drm_asahi_submit sub{};
    sub.syncs = (uint64_t)(uintptr_t)syncs;
    sub.cmdbuf = (uint64_t)(uintptr_t)buf;
    sub.queue_id = queue;
    sub.in_sync_count = nin;
    sub.out_sync_count = 1;
    sub.cmdbuf_size = (uint32_t)(sizeof(Cmd) * cmds.size());
    if (xioctl(fd_, DRM_IOCTL_ASAHI_SUBMIT, &sub) != 0) {
        error = "DRM_IOCTL_ASAHI_SUBMIT failed: " + std::string(strerror(errno));
        return false;
    }
    if (wait) {
        int r = waitSyncobj(outSync, 10ull * 1000000000ull);
        if (r != 0) {
            error = r == ETIME ? "GPU timeout (10 s): the command did not finish" : "syncobj wait failed: " + std::string(strerror(r));
            return false;
        }
    }
    return true;
}

} // namespace masahi
