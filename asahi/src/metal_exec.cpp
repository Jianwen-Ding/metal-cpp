// libmetal_asahi: MTLCommandQueue, MTLCommandBuffer, compute and blit encoders,
// and the engine that turns recorded dispatches into Asahi compute commands.
//
// Every command buffer runs on its queue's worker thread, in commit order.
// Dispatches become CDM launches (one compute command each, chained by CDM
// barriers within a submission). In the simulator modes each dispatch first
// runs on the AGX Performance Simulator's agx_run: preflight refuses to send
// a kernel that faults or hangs there to the GPU, verify also compares the
// GPU's results with the simulator's, sim runs on the simulator only.
#include "metal.h"
#include "sim.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>


namespace masahi {

static std::atomic<uint64_t> gStats[8]; // dispatches, sim runs, sim failures, verify mismatches, hw submits, sim skipped

enum CbStatus { NotEnqueued = 0, Enqueued = 1, Committed = 2, Scheduled = 3, Completed = 4, StatusError = 5 };

static const char* kCbErrorDomain = "MTLCommandBufferErrorDomain";

// ---------------------------------------------------------------- dispatch preparation

static void put32(std::vector<uint8_t>& u, uint32_t half, uint32_t v)
{
    if (half == ~0u) return;
    size_t off = (size_t)half * 2;
    if (off + 4 > u.size()) u.resize(off + 4);
    memcpy(&u[off], &v, 4);
}
static void put64(std::vector<uint8_t>& u, uint32_t half, uint64_t v)
{
    put32(u, half, (uint32_t)v);
    if (half != ~0u) put32(u, half + 2, (uint32_t)(v >> 32));
}

struct Upload {
    uint64_t va;
    uint8_t* ptr;
    size_t size;
    bool writable;
};

struct Prepared {
    std::vector<uint8_t> launch; // CDM Launch words (the stream around them is the batcher's)
    std::vector<uint8_t> uniforms;
    uint32_t tgBytes = 0;
    std::vector<Upload> uploads;         // arena pieces this dispatch reads (and the per-CTA region)
    std::vector<MBuffer*> buffers;       // buffers it can reach (bound + used)
    Bo* globals = nullptr;
};

static uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

static void divisorValues(uint32_t d, uint32_t kind, uint32_t& out)
{
    auto calc = [&](uint32_t dd, uint32_t k) -> uint32_t {
        if (dd == 0) return 0;
        uint32_t l = 0;
        while (l < 32 && (1ull << l) < dd) l++;
        switch (k) {
        case 0: return (uint32_t)(((1ull << 32) * ((1ull << l) - dd)) / dd + 1);
        case 1: return std::min<uint32_t>(l, 1);
        default: return std::max<uint32_t>(l, 1) - 1;
        }
    };
    if (kind <= 2) out = calc(d, kind);
    else if (kind <= 5) out = calc((uint32_t)std::abs((int32_t)d), kind - 3);
    else out = calc(std::max<uint32_t>(d, 1), kind - 6);
}

// Builds the uniforms, pipeline words and CDM stream of one dispatch.
static bool prepare(MQueue* q, DispatchCmd& dc, Prepared& p, std::string& error)
{
    MDevice* dev = q->dev;
    MPipeline* pso = dc.pso;
    const CompiledKernel& k = *pso->fn->kernel;
    LibModule& mod = *pso->fn->lib->modules[pso->fn->module];
    Arena& data = *q->data;
    Arena& usc = *q->usc;
    auto upload = [&](const void* src, size_t n, size_t align, bool writable) -> uint64_t {
        uint64_t va;
        uint8_t* dst = data.alloc(std::max<size_t>(n, 4), align, va);
        if (!dst) return 0;
        if (src) memcpy(dst, src, n);
        else memset(dst, 0, n);
        p.uploads.push_back({va, dst, std::max<size_t>(n, 4), writable});
        return va;
    };
    uint32_t local = dc.local[0] * dc.local[1] * dc.local[2];
    if (local == 0 || local > pso->maxThreads) {
        error = "threadgroup of " + std::to_string(local) + " threads exceeds the pipeline's maxTotalThreadsPerThreadgroup (" +
                std::to_string(pso->maxThreads) + ") for " + k.name;
        return false;
    }
    if (dc.grid[0] == 0 || dc.grid[1] == 0 || dc.grid[2] == 0) return true; // empty dispatch: nothing to run
    p.uniforms.assign(((size_t)k.uniformHalves * 2 + 7) & ~7ull, 0);
    std::map<uint32_t, Binding*> byLoc;
    for (auto& b : dc.bindings) byLoc[b.first] = &b.second;
    for (const KernelArg& a : k.args) {
        Binding* b = byLoc.count(a.location) ? byLoc[a.location] : nullptr;
        MBuffer* buf = b && b->buffer ? static_cast<MBuffer*>(b->buffer.get()) : nullptr;
        if (!b || !b->bound) {
            if (debugLevel() > 0) warn("%s: nothing bound at buffer index %u (reads as null)", k.name.c_str(), a.location);
            continue;
        }
        if (buf) p.buffers.push_back(buf);
        if (!a.byref) {
            uint64_t va = buf ? buf->bo->va + b->offset : upload(b->bytes.data(), b->bytes.size(), 16, false);
            put64(p.uniforms, a.firstHalf, va);
            continue;
        }
        // A constant buffer passed as its bytes.
        if (a.kind == 0) {
            if (a.firstHalf == ~0u) continue;
            size_t off = (size_t)a.firstHalf * 2;
            if (off + a.size > p.uniforms.size()) p.uniforms.resize((off + a.size + 7) & ~7ull);
            const uint8_t* src = nullptr;
            size_t avail = 0;
            if (buf) {
                src = buf->bo->map + b->offset;
                avail = b->offset < buf->length ? buf->length - b->offset : 0;
            } else {
                src = b->bytes.data();
                avail = b->bytes.size();
            }
            memcpy(&p.uniforms[off], src, std::min<size_t>(avail, a.size));
        } else {
            uint64_t va = buf ? buf->bo->va + b->offset : upload(b->bytes.data(), b->bytes.size(), 16, false);
            put64(p.uniforms, a.firstHalf, va);
        }
    }
    for (auto& u : dc.used)
        if (u) p.buffers.push_back(static_cast<MBuffer*>(u.get()));
    uint32_t groups[3];
    for (int d = 0; d < 3; d++) groups[d] = ceilDiv(dc.grid[d], dc.local[d]);
    for (int d = 0; d < 3; d++) put32(p.uniforms, k.tgpgHalf + 2 * d, groups[d]);
    if (k.globalDataHalf != ~0u) {
        p.globals = mod.globals;
        put64(p.uniforms, k.globalDataHalf, mod.globals ? mod.globals->va : 0);
    }
    uint32_t dynamicTg = 0;
    if (k.threadgroupBufferLocation >= 0 && (size_t)k.threadgroupBufferLocation < dc.tgLengths.size())
        dynamicTg = dc.tgLengths[k.threadgroupBufferLocation];
    p.tgBytes = dynamicTg ? ((k.threadgroupBytes + 15) & ~15u) + dynamicTg : k.threadgroupBytes;
    if (p.tgBytes > 65536) {
        error = k.name + ": " + std::to_string(p.tgBytes) + " bytes of threadgroup memory (the limit is 32768)";
        return false;
    }
    put32(p.uniforms, k.hiddenTgBytesHalf, p.tgBytes);
    for (auto& lg : k.launchGrid) put32(p.uniforms, lg.first, lg.second < 3 ? dc.grid[lg.second] : lg.second < 6 ? dc.local[lg.second - 3] : 0);
    for (auto& c : k.uniformConstants) put32(p.uniforms, c.first, c.second);
    for (auto& dv : k.divisors) {
        uint32_t src = 0, v = 0;
        if ((size_t)dv.sourceHalf * 2 + 4 <= p.uniforms.size()) memcpy(&src, &p.uniforms[(size_t)dv.sourceHalf * 2], 4);
        divisorValues(src, dv.kind, v);
        put32(p.uniforms, dv.half, v);
    }
    if (k.perCtaHalf != ~0u) {
        uint64_t n = (uint64_t)groups[0] * groups[1] * groups[2] * k.perCtaSize;
        uint64_t va = upload(nullptr, n, 1024, true);
        if (!va) {
            error = "cannot allocate per-threadgroup memory";
            return false;
        }
        put64(p.uniforms, k.perCtaHalf, va);
    }
    if (k.address32Half != ~0u) {
        error = k.name + " uses 32-bit addressing (PTX .address_size 32), which Metal kernels do not";
        return false;
    }
    uint32_t halves = k.uniformHalves;
    if (halves > 448) {
        error = k.name + ": " + std::to_string(halves) + " uniform halves (at most 448 supported)";
        return false;
    }
    uint64_t unifVa = 0;
    if (halves) {
        unifVa = upload(p.uniforms.data(), p.uniforms.size(), 64, false);
        if (!unifVa) {
            error = "out of upload memory";
            return false;
        }
    }

    // USC pipeline words (genxml/cmdbuf.xml): Uniform x n, Shared, Shader, Registers, No preshader.
    std::vector<uint8_t> w;
    auto put = [&](uint64_t v, int n) {
        for (int i = 0; i < n; i++) w.push_back((uint8_t)(v >> (8 * i)));
    };
    for (uint32_t s = 0; s < halves; s += 32) {
        uint32_t n = std::min<uint32_t>(32, halves - s);
        uint64_t tag = s < 256 ? 0x1d : 0x3d;
        uint64_t start = s < 256 ? s : s - 256;
        put(tag | (start << 8) | ((uint64_t)(n & 63) << 20) | (((unifVa + 2ull * s) >> 2) << 26), 8);
    }
    if (p.tgBytes) {
        uint32_t g = ceilDiv(p.tgBytes, 256) & 0xff; // groups(256), 0 = 65536
        put(0x4du | (1u << 8) | (0x24u << 10) | (g << 24), 4);
    } else {
        put(0x4du | (0x24u << 10), 4);
    }
    put(0x0dull | (3ull << 10) | ((uint64_t)pso->codeOffset << 16), 6);
    uint32_t regHalves = std::max<uint32_t>(8, 2 * k.gprs);
    put(0x8du | ((((regHalves + 7) / 8) & 31) << 8) | (1u << 24), 4);
    put(0x88, 2);
    uint64_t uscVa;
    uint8_t* uw = usc.alloc(w.size(), 64, uscVa);
    if (!uw) {
        error = "out of USC memory";
        return false;
    }
    memcpy(uw, w.data(), w.size());
    uint32_t uscOff = (uint32_t)(uscVa - dev->drm->uscBase());

    // CDM Launch words; the batcher puts barriers between launches and terminates the stream.
    std::vector<uint8_t>& c = p.launch;
    auto put32c = [&](uint32_t v) {
        for (int i = 0; i < 4; i++) c.push_back((uint8_t)(v >> (8 * i)));
    };
    put32c((ceilDiv(std::max<uint32_t>(halves, 1), 64) & 7) << 1);
    put32c(uscOff);
    if (dev->drm->info().g14x) {
        put32c(0);
        put32c(1u << 30);
    }
    for (int d = 0; d < 3; d++) put32c(dc.grid[d]);
    for (int d = 0; d < 3; d++) put32c(dc.local[d]);
    return true;
}

// ---------------------------------------------------------------- simulator

static uint64_t simThreadLimit()
{
    static uint64_t v = getenv("METAL_ASAHI_SIM_MAX_THREADS") ? strtoull(getenv("METAL_ASAHI_SIM_MAX_THREADS"), nullptr, 0) : (1ull << 22);
    return v;
}

struct SimState {
    SimDispatch sd;
    std::vector<std::pair<uint8_t*, size_t>> targets; // where each dumped region lives on the host
    std::vector<std::vector<uint8_t>> before;         // snapshots of the inputs
};

// What the simulator's run needs of memory, by mode: preflight only whether it faults or hangs (the
// buffers as they are, nothing read back); sim its effects, applied in place (only the 4 KiB chunks
// it changed); verify snapshots of the inputs and every output region whole, for the comparison.
static void buildSim(DispatchCmd& dc, Prepared& p, SimState& s, ExecMode mode)
{
    const bool full = mode == ExecMode::Verify, apply = mode == ExecMode::Simulator;
    const CompiledKernel& k = *dc.pso->fn->kernel;
    s.sd.code = &k.code;
    for (int d = 0; d < 3; d++) {
        s.sd.grid[d] = dc.grid[d];
        s.sd.wg[d] = dc.local[d];
    }
    s.sd.sharedBytes = p.tgBytes;
    for (size_t i = 0; i + 4 <= p.uniforms.size(); i += 4) {
        uint32_t v;
        memcpy(&v, &p.uniforms[i], 4);
        if (v) s.sd.uniforms.push_back({(uint32_t)(i / 4), v});
    }
    if (const char* ms = getenv("METAL_ASAHI_SIM_MAX_STEPS")) s.sd.maxSteps = strtoull(ms, nullptr, 0);
    std::vector<Bo*> seen;
    auto addBo = [&](Bo* bo, size_t size, bool dump) {
        if (!bo) return;
        for (Bo* x : seen)
            if (x == bo) return;
        seen.push_back(bo);
        SimRegion r;
        r.va = bo->va;
        r.size = size;
        r.data = bo->map;
        if (full) {
            s.before.emplace_back(bo->map, bo->map + size);
            r.dump = dump;
        } else if (apply && dump) {
            r.patch = bo->map;
        }
        s.sd.regions.push_back(std::move(r));
        s.targets.push_back({bo->map, size});
    };
    for (MBuffer* b : p.buffers) addBo(b->bo, (std::max<uint64_t>(b->length, 4) + 3) & ~3ull, true);
    if (p.globals) addBo(p.globals, p.globals->size, true);
    for (auto& u : p.uploads) {
        SimRegion r;
        r.va = u.va;
        r.size = u.size;
        r.data = u.ptr;
        if (full) {
            s.before.emplace_back(u.ptr, u.ptr + u.size);
            r.dump = u.writable;
        } else if (apply && u.writable) {
            r.patch = u.ptr;
        }
        s.sd.regions.push_back(std::move(r));
        s.targets.push_back({u.ptr, u.size});
    }
    if (full)
        for (size_t i = 0; i < s.sd.regions.size(); i++) s.sd.regions[i].data = s.before[i].data();
}

// ---------------------------------------------------------------- execution

static double hostNow() { return CACurrentMediaTime(); }

static void failBuffer(MCommandBuffer* cb, intptr_t code, const std::string& msg)
{
    if (!cb->error) {
        cb->error = Ref<Object>(makeError(kCbErrorDomain, code, msg));
    }
    warn("command buffer failed: %s", msg.c_str());
}

static constexpr uint32_t kFrameTimestamps = 64;

static void addTimes(Frame* f, DrmDevice* drm, size_t n)
{
    double hz = (double)drm->info().timestampHz;
    // Firmware timestamps are GPU timer ticks; place them on the host clock through one GET_TIME sample.
    double offset = f->hostBefore - (double)f->gpuBefore * 1e-9;
    for (size_t k = 0; k < n && f->tsBase; k++) {
        auto t = drm->timestamp(f->tsBase + (uint32_t)k);
        if (!t.first || !t.second) continue;
        double s = (double)t.first / hz + offset, e = (double)t.second / hz + offset;
        if (f->firstStart == 0 || s < f->firstStart) f->firstStart = s;
        if (e > f->lastEnd) f->lastEnd = e;
    }
}

// Submits cmds on the frame's syncobj. Waits for them unless async (the last
// submission of a command buffer, which the completer thread retires).
static bool runBatch(MQueue* q, MCommandBuffer* cb, std::vector<ComputeCommand>& cmds, bool async = false)
{
    if (cmds.empty()) return false;
    Frame* f = q->cur;
    DrmDevice* drm = q->dev->drm;
    std::string err;
    f->hostBefore = hostNow();
    f->gpuBefore = drm->gpuTimeNs();
    drm->resetSyncobj(f->syncobj);
    gStats[4]++;
    size_t n = cmds.size();
    if (!drm->submit(q->drmQueue, cmds, 0, f->syncobj, f->tsBase, !async, err)) {
        failBuffer(cb, err.find("timeout") != std::string::npos ? 2 : 1, err);
        cmds.clear();
        return false;
    }
    cmds.clear();
    if (async) {
        f->tsUsed = n;
        return true;
    }
    addTimes(f, drm, n);
    return false;
}

// Puts dispatches into compute commands: several launches per CDM stream and
// several commands per submission. Between launches of a serial encoder (or
// across a concurrent one's memory barrier) a barrier word orders them: Mesa's
// base set 0x160 (Apple-M2-Playground 2026-11-10-cdm-barrier-bits: bit 8 with
// 5 or 6 orders and makes the data visible on other cores; bit 11 invalidates
// the i-cache), 3.15 instead of 3.9 us per empty dispatch than 0xff7ff;
// concurrent launches have none (0.55 us). The stream ends with the full
// barrier. Commands of a submission wait for the one before them.
struct Batcher {
    MQueue* q;
    MCommandBuffer* cb;
    size_t maxCmds = 64, perCmd = 64;
    std::vector<ComputeCommand> cmds;
    std::vector<uint8_t> stream;
    unsigned inStream = 0;
    // METAL_ASAHI_BARRIER_BITS: the bits of the barrier between launches (ordering needs bit 8 with 5 or 6).
    static inline const uint32_t kBetween = (3u << 29) | (getenv("METAL_ASAHI_BARRIER_BITS")
                                                              ? (uint32_t)strtoul(getenv("METAL_ASAHI_BARRIER_BITS"), nullptr, 16) & 0xfffffu
                                                              : 0x160u);
    static constexpr uint32_t kEnd = (3u << 29) | 0xfffffu;

    void word(uint32_t v)
    {
        for (int i = 0; i < 4; i++) stream.push_back((uint8_t)(v >> (8 * i)));
    }
    // barrier: the launch waits for the ones before it (a serial encoder, or a memory
    // barrier of a concurrent one); without, it may overlap them.
    void add(const std::vector<uint8_t>& launch, bool barrier = true)
    {
        if (inStream && barrier) word(kBetween);
        stream.insert(stream.end(), launch.begin(), launch.end());
        if (++inStream >= perCmd) close();
    }
    void close()
    {
        if (!inStream) return;
        word(kEnd);
        word(2u << 29); // stream terminate
        word(0);
        uint64_t va;
        uint8_t* dst = q->data->alloc(stream.size(), 64, va);
        if (!dst) {
            failBuffer(cb, 1, "out of upload memory for the CDM stream");
        } else {
            memcpy(dst, stream.data(), stream.size());
            ComputeCommand c;
            c.cdmBase = va;
            c.cdmEnd = va + stream.size();
            c.barrierOnPrevious = true;
            cmds.push_back(c);
        }
        stream.clear();
        inStream = 0;
        if (cmds.size() >= maxCmds) runBatch(q, cb, cmds);
    }
    void flush()
    {
        close();
        runBatch(q, cb, cmds);
    }
    // The command buffer's last submission, left in flight: true if there was one.
    bool flushAsync()
    {
        close();
        return runBatch(q, cb, cmds, true);
    }
};

static bool compareVerify(DispatchCmd& dc, SimState& s)
{
    bool ok = true;
    for (size_t i = 0; i < s.sd.regions.size(); i++) {
        auto& r = s.sd.regions[i];
        if (!r.dump) continue;
        const uint8_t* hw = s.targets[i].first;
        if (memcmp(hw, r.out.data(), r.size) == 0) continue;
        size_t first = 0;
        while (first < r.size && hw[first] == r.out[first]) first++;
        size_t diffs = 0;
        for (size_t j = 0; j < r.size; j++) diffs += hw[j] != r.out[j];
        uint32_t hv = 0, sv = 0;
        memcpy(&hv, hw + (first & ~3ull), std::min<size_t>(4, r.size - (first & ~3ull)));
        memcpy(&sv, r.out.data() + (first & ~3ull), std::min<size_t>(4, r.size - (first & ~3ull)));
        warn("verify: %s: GPU and simulator differ in region %zu (va %#llx): %zu bytes, first at +%zu (GPU %08x, sim %08x)",
             dc.pso->fn->kernel->name.c_str(), i, (unsigned long long)r.va, diffs, first, hv, sv);
        ok = false;
    }
    return ok;
}

// Runs a command buffer's commands. Returns true when its last GPU submission
// is still in flight (the completer finishes it), false when it is done.
static bool execute(MQueue* q, MCommandBuffer* cb)
{
    ExecMode mode = execMode();
    Batcher batch{q, cb};
    batch.maxCmds = q->dev->drm ? std::min<size_t>(q->dev->drm->info().maxCommandsPerSubmission, 64) : 1;
    if (const char* mb = getenv("METAL_ASAHI_MAX_BATCH")) batch.maxCmds = std::max<size_t>(1, std::min<size_t>(batch.maxCmds, atoi(mb)));
    if (const char* pc = getenv("METAL_ASAHI_DISPATCHES_PER_COMMAND")) batch.perCmd = std::max(1, atoi(pc));
    bool barrierNext = true;
    double kStart = hostNow();
    for (Command& c : cb->commands) {
        if (cb->error) break;
        switch (c.kind) {
        case Command::Barrier: barrierNext = true; break;
        case Command::Dispatch: {
            DispatchCmd& dc = *c.dispatch;
            gStats[0]++;
            Prepared p;
            std::string err;
            if (!prepare(q, dc, p, err)) {
                failBuffer(cb, 1, err);
                break;
            }
            if (p.launch.empty()) break; // empty grid
            uint64_t threads = (uint64_t)dc.grid[0] * dc.grid[1] * dc.grid[2];
            bool simulate = mode != ExecMode::Hardware && (mode == ExecMode::Simulator || threads <= simThreadLimit());
            if (mode != ExecMode::Hardware && !simulate) {
                gStats[5]++;
                if (debugLevel() > 0)
                    warn("%s: %llu threads is above METAL_ASAHI_SIM_MAX_THREADS; running on the GPU without simulation",
                         dc.pso->fn->kernel->name.c_str(), (unsigned long long)threads);
            }
            if (!simulate) {
                batch.add(p.launch, barrierNext);
                barrierNext = false;
                break;
            }
            // Simulated dispatches run alone, so the simulator sees the memory the GPU would.
            batch.flush();
            q->drain();
            if (cb->error) break;
            SimState s;
            buildSim(dc, p, s, mode == ExecMode::Simulator || !q->dev->drm ? ExecMode::Simulator : mode);
            gStats[1]++;
            SimResult r = masahi::simulate(s.sd);
            if (!r.ok) {
                gStats[2]++;
                failBuffer(cb, r.hang ? 2 : 1,
                           std::string(r.hang ? "simulator: kernel hangs" : "simulator: kernel faults") + " (" +
                               dc.pso->fn->kernel->name + "); not sent to the GPU: " + r.error);
                break;
            }
            if (debugLevel() > 1)
                warn("sim %s: %llu instructions, %.2f s", dc.pso->fn->kernel->name.c_str(), (unsigned long long)r.instructions, r.seconds);
            if (mode == ExecMode::Simulator || !q->dev->drm) {
                for (size_t i = 0; i < s.sd.regions.size(); i++)
                    if (s.sd.regions[i].dump) memcpy(s.targets[i].first, s.sd.regions[i].out.data(), s.sd.regions[i].size);
                break;
            }
            batch.add(p.launch);
            batch.flush();
            // METAL_ASAHI_VERIFY_SKIP=1: results that depend on the order atomics happen in (still simulated first).
            const char* skip = getenv("METAL_ASAHI_VERIFY_SKIP");
            if (!cb->error && mode == ExecMode::Verify && !(skip && *skip == '1') && !compareVerify(dc, s)) gStats[3]++;
            barrierNext = false;
            break;
        }
        case Command::Blit: {
            batch.flush();
            q->drain();
            BlitCmd& b = *c.blit;
            auto* dst = static_cast<MBuffer*>(b.dst.get());
            if (b.kind == BlitCmd::Copy) {
                auto* src = static_cast<MBuffer*>(b.src.get());
                if (b.srcOff + b.size > src->length || b.dstOff + b.size > dst->length) {
                    failBuffer(cb, 1, "blit copy out of range");
                    break;
                }
                memmove(dst->bo->map + b.dstOff, src->bo->map + b.srcOff, b.size);
            } else {
                if (b.dstOff + b.size > dst->length) {
                    failBuffer(cb, 1, "blit fill out of range");
                    break;
                }
                memset(dst->bo->map + b.dstOff, b.value, b.size);
            }
            barrierNext = true;
            break;
        }
        case Command::Event: {
            batch.flush();
            q->drain();
            auto* e = static_cast<MEvent*>(c.event->event.get());
            if (c.event->wait) {
                std::unique_lock<std::mutex> l(e->m);
                e->cv.wait(l, [&] { return e->value >= c.event->value; });
            } else {
                {
                    std::lock_guard<std::mutex> l(e->m);
                    e->value = c.event->value;
                }
                e->cv.notify_all();
            }
            barrierNext = true;
            break;
        }
        }
    }
    q->cur->kernelStart = kStart;
    return !cb->error && batch.flushAsync();
}

// The command buffer's times and status, its completed handlers; frees its frame.
static void finish(MQueue* q, MCommandBuffer* cb, Frame* f)
{
    double kEnd = hostNow();
    cb->kernelStart = f->kernelStart;
    cb->kernelEnd = kEnd;
    cb->gpuStart = f->firstStart ? f->firstStart : f->kernelStart;
    cb->gpuEnd = f->lastEnd ? f->lastEnd : kEnd;
    f->data->reset();
    f->usc->reset();
    f->firstStart = f->lastEnd = 0;
    f->cb = nullptr;
    {
        std::lock_guard<std::mutex> l(cb->m);
        cb->status = cb->error ? StatusError : Completed;
    }
    for (void* h : cb->completedHandlers) callBlock(h, (Object*)cb);
    cb->cv.notify_all();
    {
        std::lock_guard<std::mutex> l(q->fm);
        q->freeFrames.push_back(f);
    }
    q->fcv.notify_all();
    release(cb);
}

void MQueue::run()
{
    for (;;) {
        MCommandBuffer* cb;
        {
            std::unique_lock<std::mutex> l(m);
            cv.wait(l, [&] { return stop || !pending.empty(); });
            if (pending.empty()) return;
            cb = pending.front();
            pending.pop_front();
        }
        Frame* f;
        {
            std::unique_lock<std::mutex> l(fm);
            fcv.wait(l, [&] { return !freeFrames.empty(); });
            f = freeFrames.front();
            freeFrames.pop_front();
        }
        cur = f;
        data = f->data.get();
        usc = f->usc.get();
        {
            std::lock_guard<std::mutex> l(cb->m);
            cb->status = Scheduled;
        }
        cb->cv.notify_all();
        for (void* h : cb->scheduledHandlers) callBlock(h, (Object*)cb);
        f->firstStart = f->lastEnd = 0;
        if (execute(this, cb)) {
            f->cb = cb;
            {
                std::lock_guard<std::mutex> l(fm);
                inFlight.push_back(f);
            }
            fcv.notify_all();
        } else {
            drain(); // command buffers complete in order
            finish(this, cb, f);
        }
        cur = nullptr;
    }
}

// The completer: retires in-flight frames in submission order.
void MQueue::complete()
{
    for (;;) {
        Frame* f;
        {
            std::unique_lock<std::mutex> l(fm);
            fcv.wait(l, [&] { return completerStop || !inFlight.empty(); });
            if (inFlight.empty()) return;
            f = inFlight.front();
        }
        DrmDevice* drm = dev->drm;
        MCommandBuffer* cb = f->cb;
        int r = drm->waitSyncobj(f->syncobj, 10ull * 1000000000ull);
        if (r != 0)
            failBuffer(cb, r == ETIME ? 2 : 1,
                       r == ETIME ? "GPU timeout (10 s): the command did not finish" : "syncobj wait failed: " + std::string(strerror(r)));
        else
            addTimes(f, drm, f->tsUsed);
        {
            std::lock_guard<std::mutex> l(fm);
            inFlight.pop_front();
        }
        finish(this, cb, f);
    }
}

void MQueue::drain()
{
    std::unique_lock<std::mutex> l(fm);
    fcv.wait(l, [&] { return inFlight.empty(); });
}

MQueue::~MQueue()
{
    {
        std::lock_guard<std::mutex> l(m);
        stop = true;
    }
    cv.notify_all();
    if (worker.joinable()) worker.join();
    {
        std::lock_guard<std::mutex> l(fm);
        completerStop = true;
    }
    fcv.notify_all();
    if (completer.joinable()) completer.join();
    for (auto& f : frames) {
        f->data.reset();
        f->usc.reset();
        if (dev && dev->drm) {
            if (f->syncobj) dev->drm->destroySyncobj(f->syncobj);
            dev->drm->freeTimestamps(f->tsBase, kFrameTimestamps);
        }
    }
    if (dev && dev->drm && drmQueue != ~0u) dev->drm->destroyQueue(drmQueue);
}

MCommandBuffer::~MCommandBuffer()
{
    for (void* h : completedHandlers) _Block_release(h);
    for (void* h : scheduledHandlers) _Block_release(h);
}

Object* newQueue(MDevice* d, uintptr_t)
{
    MQueue* q = make<MQueue>("MTLAsahiCommandQueue");
    q->dev = d;
    q->devRef = Ref<Object>(d);
    // METAL_ASAHI_MAX_IN_FLIGHT: command buffers the GPU may have at once (1: each waits for the one before).
    int depth = 8;
    if (const char* e = getenv("METAL_ASAHI_MAX_IN_FLIGHT")) depth = std::max(1, std::min(32, atoi(e)));
    if (d->drm) {
        q->drmQueue = d->drm->createQueue(1 /* medium */);
        if (q->drmQueue == ~0u) {
            warn("cannot create a GPU queue: %s", strerror(errno));
            q->destroy(q);
            return nullptr;
        }
    }
    for (int i = 0; i < depth; i++) {
        auto f = std::make_unique<Frame>();
        f->data = std::make_unique<Arena>(d, false);
        f->usc = std::make_unique<Arena>(d, true);
        if (d->drm) {
            f->syncobj = d->drm->createSyncobj();
            f->tsBase = d->drm->allocTimestamps(kFrameTimestamps);
            if (!f->syncobj) {
                warn("cannot create a syncobj: %s", strerror(errno));
                q->destroy(q);
                return nullptr;
            }
        }
        q->freeFrames.push_back(f.get());
        q->frames.push_back(std::move(f));
    }
    q->worker = std::thread([q] { q->run(); });
    q->completer = std::thread([q] { q->complete(); });
    return q;
}

// ---------------------------------------------------------------- registration

struct Size3 {
    uintptr_t w, h, d;
};

static MCommandBuffer* newCommandBuffer(MQueue* q, bool retained)
{
    MCommandBuffer* cb = make<MCommandBuffer>("MTLAsahiCommandBuffer");
    cb->queue = q;
    cb->queueRef = Ref<Object>(q);
    cb->retainedReferences = retained;
    return static_cast<MCommandBuffer*>(autorelease(cb));
}

static MEncoder* newEncoder(MCommandBuffer* cb, const char* cls, bool concurrent)
{
    if (cb->status != NotEnqueued && cb->status != Enqueued) fatal("encoder created on a committed command buffer");
    MEncoder* e = make<MEncoder>(cls);
    e->cb = cb;
    e->concurrent = concurrent;
    cb->currentEncoder = Ref<Object>(e);
    if (!cb->commands.empty()) {
        Command c{Command::Barrier, nullptr, nullptr, nullptr};
        cb->commands.push_back(std::move(c));
    }
    return static_cast<MEncoder*>(autorelease(e));
}

static void bindBuffer(MEncoder* e, Object* buf, uint64_t off, uint32_t idx)
{
    Binding& b = e->bindings[idx];
    b.buffer = Ref<Object>(buf);
    b.offset = off;
    b.bytes.clear();
    b.bound = buf != nullptr;
}

static void recordDispatch(MEncoder* e, const uint32_t grid[3], const uint32_t local[3])
{
    if (!e->pipeline) fatal("dispatch without a compute pipeline state");
    auto* pso = static_cast<MPipeline*>(e->pipeline.get());
    auto dc = std::make_unique<DispatchCmd>();
    dc->pipeline = e->pipeline;
    dc->pso = pso;
    for (const KernelArg& a : pso->fn->kernel->args) {
        auto it = e->bindings.find(a.location);
        if (it != e->bindings.end()) dc->bindings.push_back({a.location, it->second});
    }
    int tgl = pso->fn->kernel->threadgroupBufferLocation;
    if (tgl >= 0) {
        dc->tgLengths.assign((size_t)tgl + 1, 0);
        auto it = e->tgLengths.find((uint32_t)tgl);
        if (it != e->tgLengths.end()) dc->tgLengths[tgl] = it->second;
    }
    for (int d = 0; d < 3; d++) {
        dc->grid[d] = grid[d];
        dc->local[d] = local[d];
    }
    dc->used = e->used;
    if (!e->concurrent && !e->cb->commands.empty() && e->cb->commands.back().kind == Command::Dispatch) {
        Command b{Command::Barrier, nullptr, nullptr, nullptr};
        e->cb->commands.push_back(std::move(b));
    }
    Command c{Command::Dispatch, std::move(dc), nullptr, nullptr};
    e->cb->commands.push_back(std::move(c));
}

static uint32_t clamp32(uintptr_t v, const char* what)
{
    if (v > 0xffffffffull) fatal("%s %zu does not fit in 32 bits", what, (size_t)v);
    return (uint32_t)v;
}

void registerExec()
{
    Class obj = lookupClass("NSObject");
    Class q = defineClass("MTLAsahiCommandQueue", obj);
    def(q, "commandBuffer", +[](MQueue* q, SEL) -> Object* { return newCommandBuffer(q, true); });
    def(q, "commandBufferWithUnretainedReferences", +[](MQueue* q, SEL) -> Object* { return newCommandBuffer(q, false); });
    def(q, "commandBufferWithDescriptor:", +[](MQueue* q, SEL, Object*) -> Object* { return newCommandBuffer(q, true); });
    def(q, "device", +[](MQueue* q, SEL) -> Object* { return q->dev; });
    def(q, "label", +[](MQueue* q, SEL) -> Object* { return q->label.empty() ? nullptr : tempString(q->label); });
    def(q, "setLabel:", +[](MQueue* q, SEL, Object* s) { q->label = str(s); });
    def(q, "insertDebugCaptureBoundary", +[](MQueue*, SEL) {});
    def(q, "addResidencySet:", +[](MQueue*, SEL, Object*) {});
    def(q, "removeResidencySet:", +[](MQueue*, SEL, Object*) {});

    Class cb = defineClass("MTLAsahiCommandBuffer", obj);
    def(cb, "device", +[](MCommandBuffer* c, SEL) -> Object* { return c->queue->dev; });
    def(cb, "commandQueue", +[](MCommandBuffer* c, SEL) -> Object* { return c->queue; });
    def(cb, "label", +[](MCommandBuffer* c, SEL) -> Object* { return c->label.empty() ? nullptr : tempString(c->label); });
    def(cb, "setLabel:", +[](MCommandBuffer* c, SEL, Object* s) { c->label = str(s); });
    def(cb, "retainedReferences", +[](MCommandBuffer* c, SEL) -> bool { return c->retainedReferences; });
    def(cb, "computeCommandEncoder", +[](MCommandBuffer* c, SEL) -> Object* { return newEncoder(c, "MTLAsahiComputeCommandEncoder", false); });
    def(cb, "computeCommandEncoderWithDispatchType:", +[](MCommandBuffer* c, SEL, uintptr_t t) -> Object* {
        return newEncoder(c, "MTLAsahiComputeCommandEncoder", t == 1);
    });
    def(cb, "computeCommandEncoderWithDescriptor:", +[](MCommandBuffer* c, SEL, Object* d) -> Object* {
        auto get = reinterpret_cast<uintptr_t (*)(Object*, SEL)>(objc_msgSend);
        uintptr_t t = d ? get(d, sel_registerName("dispatchType")) : 0;
        return newEncoder(c, "MTLAsahiComputeCommandEncoder", t == 1);
    });
    def(cb, "blitCommandEncoder", +[](MCommandBuffer* c, SEL) -> Object* { return newEncoder(c, "MTLAsahiBlitCommandEncoder", false); });
    def(cb, "blitCommandEncoderWithDescriptor:", +[](MCommandBuffer* c, SEL, Object*) -> Object* {
        return newEncoder(c, "MTLAsahiBlitCommandEncoder", false);
    });
    def(cb, "encodeSignalEvent:value:", +[](MCommandBuffer* c, SEL, Object* ev, uint64_t v) {
        auto e = std::make_unique<EventCmd>();
        e->event = Ref<Object>(ev);
        e->value = v;
        Command cmd{Command::Event, nullptr, nullptr, std::move(e)};
        c->commands.push_back(std::move(cmd));
    });
    def(cb, "encodeWaitForEvent:value:", +[](MCommandBuffer* c, SEL, Object* ev, uint64_t v) {
        auto e = std::make_unique<EventCmd>();
        e->event = Ref<Object>(ev);
        e->value = v;
        e->wait = true;
        Command cmd{Command::Event, nullptr, nullptr, std::move(e)};
        c->commands.push_back(std::move(cmd));
    });
    def(cb, "addCompletedHandler:", +[](MCommandBuffer* c, SEL, void* block) { c->completedHandlers.push_back(_Block_copy(block)); });
    def(cb, "addScheduledHandler:", +[](MCommandBuffer* c, SEL, void* block) { c->scheduledHandlers.push_back(_Block_copy(block)); });
    def(cb, "enqueue", +[](MCommandBuffer* c, SEL) {
        if (c->status == NotEnqueued) c->status = Enqueued;
    });
    def(cb, "commit", +[](MCommandBuffer* c, SEL) {
        if (c->status >= Committed) fatal("commit called twice on a command buffer");
        MQueue* q = c->queue;
        {
            std::lock_guard<std::mutex> l(c->m);
            c->status = Committed;
        }
        retain(c);
        {
            std::lock_guard<std::mutex> l(q->m);
            q->pending.push_back(c);
        }
        q->cv.notify_all();
    });
    def(cb, "waitUntilCompleted", +[](MCommandBuffer* c, SEL) {
        std::unique_lock<std::mutex> l(c->m);
        if (c->status < Committed) {
            warn("waitUntilCompleted on a command buffer that was not committed");
            return;
        }
        c->cv.wait(l, [&] { return c->status >= Completed; });
    });
    def(cb, "waitUntilScheduled", +[](MCommandBuffer* c, SEL) {
        std::unique_lock<std::mutex> l(c->m);
        if (c->status < Committed) return;
        c->cv.wait(l, [&] { return c->status >= Scheduled; });
    });
    def(cb, "status", +[](MCommandBuffer* c, SEL) -> uintptr_t {
        std::lock_guard<std::mutex> l(c->m);
        return (uintptr_t)c->status;
    });
    def(cb, "error", +[](MCommandBuffer* c, SEL) -> Object* { return c->error.get(); });
    def(cb, "errorOptions", +[](MCommandBuffer*, SEL) -> uintptr_t { return 0; });
    def(cb, "GPUStartTime", +[](MCommandBuffer* c, SEL) -> double { return c->gpuStart; });
    def(cb, "GPUEndTime", +[](MCommandBuffer* c, SEL) -> double { return c->gpuEnd; });
    def(cb, "kernelStartTime", +[](MCommandBuffer* c, SEL) -> double { return c->kernelStart; });
    def(cb, "kernelEndTime", +[](MCommandBuffer* c, SEL) -> double { return c->kernelEnd; });
    def(cb, "pushDebugGroup:", +[](MCommandBuffer*, SEL, Object*) {});
    def(cb, "popDebugGroup", +[](MCommandBuffer*, SEL) {});
    def(cb, "logs", +[](MCommandBuffer*, SEL) -> Object* { return nullptr; });
    def(cb, "useResidencySet:", +[](MCommandBuffer*, SEL, Object*) {});

    // Encoders (shared methods).
    Class enc = defineClass("MTLAsahiCommandEncoder", obj);
    def(enc, "endEncoding", +[](MEncoder* e, SEL) {
        e->ended = true;
        if (e->cb->currentEncoder.get() == e) e->cb->currentEncoder = Ref<Object>();
    });
    def(enc, "device", +[](MEncoder* e, SEL) -> Object* { return e->cb->queue->dev; });
    def(enc, "label", +[](MEncoder* e, SEL) -> Object* { return e->label.empty() ? nullptr : tempString(e->label); });
    def(enc, "setLabel:", +[](MEncoder* e, SEL, Object* s) { e->label = str(s); });
    def(enc, "pushDebugGroup:", +[](MEncoder*, SEL, Object*) {});
    def(enc, "popDebugGroup", +[](MEncoder*, SEL) {});
    def(enc, "insertDebugSignpost:", +[](MEncoder*, SEL, Object*) {});
    def(enc, "updateFence:", +[](MEncoder*, SEL, Object*) {});
    def(enc, "waitForFence:", +[](MEncoder*, SEL, Object*) {});
    def(enc, "barrierAfterQueueStages:beforeStages:", +[](MEncoder* e, SEL, uintptr_t, uintptr_t) {
        Command c{Command::Barrier, nullptr, nullptr, nullptr};
        e->cb->commands.push_back(std::move(c));
    });

    Class ce = defineClass("MTLAsahiComputeCommandEncoder", enc);
    def(ce, "dispatchType", +[](MEncoder* e, SEL) -> uintptr_t { return e->concurrent ? 1 : 0; });
    def(ce, "setComputePipelineState:", +[](MEncoder* e, SEL, Object* p) { e->pipeline = Ref<Object>(p); });
    def(ce, "setBuffer:offset:atIndex:", +[](MEncoder* e, SEL, Object* b, uintptr_t off, uintptr_t idx) { bindBuffer(e, b, off, (uint32_t)idx); });
    def(ce, "setBuffer:offset:attributeStride:atIndex:",
        +[](MEncoder* e, SEL, Object* b, uintptr_t off, uintptr_t, uintptr_t idx) { bindBuffer(e, b, off, (uint32_t)idx); });
    def(ce, "setBufferOffset:atIndex:", +[](MEncoder* e, SEL, uintptr_t off, uintptr_t idx) { e->bindings[(uint32_t)idx].offset = off; });
    def(ce, "setBufferOffset:attributeStride:atIndex:",
        +[](MEncoder* e, SEL, uintptr_t off, uintptr_t, uintptr_t idx) { e->bindings[(uint32_t)idx].offset = off; });
    def(ce, "setBuffers:offsets:withRange:", +[](MEncoder* e, SEL, Object** bufs, const uintptr_t* offs, uintptr_t loc, uintptr_t len) {
        for (uintptr_t i = 0; i < len; i++) bindBuffer(e, bufs[i], offs ? offs[i] : 0, (uint32_t)(loc + i));
    });
    def(ce, "setBuffers:offsets:attributeStrides:withRange:",
        +[](MEncoder* e, SEL, Object** bufs, const uintptr_t* offs, const uintptr_t*, uintptr_t loc, uintptr_t len) {
            for (uintptr_t i = 0; i < len; i++) bindBuffer(e, bufs[i], offs ? offs[i] : 0, (uint32_t)(loc + i));
        });
    def(ce, "setBytes:length:atIndex:", +[](MEncoder* e, SEL, const void* bytes, uintptr_t len, uintptr_t idx) {
        Binding& b = e->bindings[(uint32_t)idx];
        b.buffer = Ref<Object>();
        b.offset = 0;
        b.bytes.assign((const uint8_t*)bytes, (const uint8_t*)bytes + len);
        b.bound = true;
    });
    def(ce, "setBytes:length:attributeStride:atIndex:", +[](MEncoder* e, SEL, const void* bytes, uintptr_t len, uintptr_t, uintptr_t idx) {
        Binding& b = e->bindings[(uint32_t)idx];
        b.buffer = Ref<Object>();
        b.offset = 0;
        b.bytes.assign((const uint8_t*)bytes, (const uint8_t*)bytes + len);
        b.bound = true;
    });
    def(ce, "setThreadgroupMemoryLength:atIndex:", +[](MEncoder* e, SEL, uintptr_t len, uintptr_t idx) {
        e->tgLengths[(uint32_t)idx] = (uint32_t)((len + 15) & ~15ul);
    });
    def(ce, "dispatchThreads:threadsPerThreadgroup:", +[](MEncoder* e, SEL, Size3 grid, Size3 local) {
        uint32_t g[3] = {clamp32(grid.w, "grid"), clamp32(grid.h, "grid"), clamp32(grid.d, "grid")};
        uint32_t l[3] = {clamp32(local.w, "threadgroup"), clamp32(local.h, "threadgroup"), clamp32(local.d, "threadgroup")};
        recordDispatch(e, g, l);
    });
    def(ce, "dispatchThreadgroups:threadsPerThreadgroup:", +[](MEncoder* e, SEL, Size3 groups, Size3 local) {
        uint32_t l[3] = {clamp32(local.w, "threadgroup"), clamp32(local.h, "threadgroup"), clamp32(local.d, "threadgroup")};
        uint32_t g[3] = {clamp32((uint64_t)groups.w * l[0], "grid"), clamp32((uint64_t)groups.h * l[1], "grid"),
                         clamp32((uint64_t)groups.d * l[2], "grid")};
        recordDispatch(e, g, l);
    });
    def(ce, "memoryBarrierWithScope:", +[](MEncoder* e, SEL, uintptr_t) {
        Command c{Command::Barrier, nullptr, nullptr, nullptr};
        e->cb->commands.push_back(std::move(c));
    });
    def(ce, "memoryBarrierWithResources:count:", +[](MEncoder* e, SEL, Object**, uintptr_t) {
        Command c{Command::Barrier, nullptr, nullptr, nullptr};
        e->cb->commands.push_back(std::move(c));
    });
    def(ce, "useResource:usage:", +[](MEncoder* e, SEL, Object* r, uintptr_t) {
        if (isKindOf(r, lookupClass("MTLAsahiBuffer"))) e->used.push_back(Ref<Object>(r));
    });
    def(ce, "useResources:count:usage:", +[](MEncoder* e, SEL, Object** r, uintptr_t n, uintptr_t) {
        for (uintptr_t i = 0; i < n; i++)
            if (isKindOf(r[i], lookupClass("MTLAsahiBuffer"))) e->used.push_back(Ref<Object>(r[i]));
    });
    def(ce, "useHeap:", +[](MEncoder*, SEL, Object*) {});
    def(ce, "useHeaps:count:", +[](MEncoder*, SEL, Object**, uintptr_t) {});
    def(ce, "setTexture:atIndex:", +[](MEncoder*, SEL, Object*, uintptr_t) { fatal("textures are not supported yet"); });
    def(ce, "setSamplerState:atIndex:", +[](MEncoder*, SEL, Object*, uintptr_t) { fatal("samplers are not supported yet"); });
    def(ce, "dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:",
        +[](MEncoder* e, SEL, Object* buf, uintptr_t off, Size3 local) {
            // The group counts are read when the command buffer runs (in order, after earlier work).
            (void)e;
            (void)buf;
            (void)off;
            (void)local;
            fatal("indirect dispatch is not supported yet");
        });

    Class be = defineClass("MTLAsahiBlitCommandEncoder", enc);
    def(be, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
        +[](MEncoder* e, SEL, Object* src, uintptr_t so, Object* dst, uintptr_t dof, uintptr_t n) {
            auto b = std::make_unique<BlitCmd>();
            b->kind = BlitCmd::Copy;
            b->src = Ref<Object>(src);
            b->dst = Ref<Object>(dst);
            b->srcOff = so;
            b->dstOff = dof;
            b->size = n;
            Command c{Command::Blit, nullptr, std::move(b), nullptr};
            e->cb->commands.push_back(std::move(c));
        });
    def(be, "fillBuffer:range:value:", +[](MEncoder* e, SEL, Object* dst, uintptr_t loc, uintptr_t len, uint8_t v) {
        auto b = std::make_unique<BlitCmd>();
        b->kind = BlitCmd::Fill;
        b->dst = Ref<Object>(dst);
        b->dstOff = loc;
        b->size = len;
        b->value = v;
        Command c{Command::Blit, nullptr, std::move(b), nullptr};
        e->cb->commands.push_back(std::move(c));
    });
    def(be, "synchronizeResource:", +[](MEncoder*, SEL, Object*) {});
    def(be, "optimizeContentsForGPUAccess:", +[](MEncoder*, SEL, Object*) {});
    def(be, "optimizeContentsForCPUAccess:", +[](MEncoder*, SEL, Object*) {});
}

} // namespace masahi

extern "C" __attribute__((visibility("default"))) uint64_t metal_asahi_stat(int i)
{
    return i >= 0 && i < 8 ? masahi::gStats[i].load() : 0;
}

void masahi::registerMetal()
{
    extern void registerDevice();
    registerDevice();
    registerExec();
}
