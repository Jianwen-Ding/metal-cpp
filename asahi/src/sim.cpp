// libmetal_asahi: agx_run driver.
#include "sim.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

#ifndef METAL_ASAHI_DEFAULT_AGX_RUN
#define METAL_ASAHI_DEFAULT_AGX_RUN "agx_run"
#endif

namespace masahi {

const std::string& simulatorPath()
{
    static std::string p = [] {
        const char* v = getenv("METAL_ASAHI_AGX_RUN");
        return std::string(v && *v ? v : METAL_ASAHI_DEFAULT_AGX_RUN);
    }();
    return p;
}

bool simulatorAvailable() { return access(simulatorPath().c_str(), X_OK) == 0; }

static bool writeFile(const std::string& path, const void* data, size_t n)
{
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) return false;
    const uint8_t* p = (const uint8_t*)data;
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w <= 0) {
            close(fd);
            return false;
        }
        p += w;
        n -= (size_t)w;
    }
    return close(fd) == 0;
}

// Region files live in memory (memfd): the simulator opens them as /proc/self/fd/N, which
// it inherits. Large buffers never touch the disk.
struct MemFile {
    int fd = -1;
    std::string path;
    bool create(const char* name)
    {
        fd = memfd_create(name, 0); // inherited by the simulator
        if (fd < 0) return false;
        path = "/proc/self/fd/" + std::to_string(fd);
        return true;
    }
    bool write(const void* data, size_t n)
    {
        const uint8_t* p = (const uint8_t*)data;
        off_t off = 0;
        while (n) {
            ssize_t w = pwrite(fd, p, n, off);
            if (w <= 0) return false;
            p += w;
            off += w;
            n -= (size_t)w;
        }
        return true;
    }
    bool read(std::vector<uint8_t>& out)
    {
        struct stat st;
        if (fstat(fd, &st) != 0) return false;
        out.resize((size_t)st.st_size);
        size_t got = 0;
        while (got < out.size()) {
            ssize_t r = pread(fd, out.data() + got, out.size() - got, (off_t)got);
            if (r <= 0) return false;
            got += (size_t)r;
        }
        return true;
    }
    ~MemFile()
    {
        if (fd >= 0) close(fd);
    }
};

// Applies agx_run --dump-changed records (u64 offset, u64 length, bytes) to dst.
static bool applyChanged(const std::vector<uint8_t>& rec, uint8_t* dst, size_t size)
{
    size_t i = 0;
    while (i + 16 <= rec.size()) {
        uint64_t off, len;
        memcpy(&off, &rec[i], 8);
        memcpy(&len, &rec[i + 8], 8);
        i += 16;
        if (off > size || len > size - off || len > rec.size() - i) return false;
        memcpy(dst + off, &rec[i], len);
        i += len;
    }
    return i == rec.size();
}

static bool readFile(const std::string& path, std::vector<uint8_t>& out)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    out.resize(n > 0 ? (size_t)n : 0);
    bool ok = n <= 0 || fread(out.data(), 1, (size_t)n, f) == (size_t)n;
    fclose(f);
    return ok;
}

static std::string hex(uint64_t v)
{
    char b[32];
    snprintf(b, sizeof b, "%#llx", (unsigned long long)v);
    return b;
}

static void removeTree(const std::string& dir, const std::vector<std::string>& files)
{
    if (getenv("METAL_ASAHI_KEEP_SIM_FILES")) return;
    for (auto& f : files) unlink(f.c_str());
    rmdir(dir.c_str());
}

SimResult simulate(SimDispatch& d)
{
    SimResult r;
    auto t0 = std::chrono::steady_clock::now();
    const char* tmp = getenv("METAL_ASAHI_TMP");
    std::string tmpl = std::string(tmp && *tmp ? tmp : "/tmp") + "/metal-asahi-sim-XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back(0);
    if (!mkdtemp(buf.data())) {
        r.error = "mkdtemp: " + std::string(strerror(errno));
        return r;
    }
    std::string dir = buf.data();
    std::vector<std::string> files;
    std::vector<std::string> args = {simulatorPath()};
    std::string code = dir + "/code.bin";
    files.push_back(code);
    if (!writeFile(code, d.code->data(), d.code->size())) {
        r.error = "cannot write " + code;
        removeTree(dir, files);
        return r;
    }
    args.push_back(code);
    auto dims = [](const uint32_t* v) { return std::to_string(v[0]) + "," + std::to_string(v[1]) + "," + std::to_string(v[2]); };
    args.insert(args.end(), {"--grid", dims(d.grid), "--wg", dims(d.wg)});
    if (d.sharedBytes) args.insert(args.end(), {"--shared", std::to_string(d.sharedBytes)});
    if (d.maxSteps) args.insert(args.end(), {"--max-steps", std::to_string(d.maxSteps)});
    for (auto& u : d.uniforms) args.insert(args.end(), {"--uniform", std::to_string(u.first) + "=" + std::to_string(u.second)});
    if (d.object && !d.object->empty() && getenv("METAL_ASAHI_SIM_LINES")) {
        std::string obj = dir + "/kernel.o";
        files.push_back(obj);
        writeFile(obj, d.object->data(), d.object->size());
        args.insert(args.end(), {"--lines", obj, "--lines-offset", std::to_string(d.linesOffset)});
    }
    std::vector<MemFile> inFiles(d.regions.size()), outFiles(d.regions.size());
    for (size_t i = 0; i < d.regions.size(); i++) {
        auto& reg = d.regions[i];
        if (!inFiles[i].create("metal-asahi-sim-in") || !inFiles[i].write(reg.data, reg.size)) {
            r.error = "cannot write the simulator's memory: " + std::string(strerror(errno));
            removeTree(dir, files);
            return r;
        }
        args.insert(args.end(), {"--map", hex(reg.va) + "=" + inFiles[i].path});
        if (reg.dump || reg.patch) {
            if (!outFiles[i].create("metal-asahi-sim-out")) {
                r.error = "memfd_create: " + std::string(strerror(errno));
                removeTree(dir, files);
                return r;
            }
            args.insert(args.end(), {reg.dump ? "--dump" : "--dump-changed",
                                     hex(reg.va) + "=" + std::to_string(reg.size) + ":" + outFiles[i].path});
        }
    }
    std::string outPath = dir + "/stdout.txt";
    files.push_back(outPath);
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 1, outPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    posix_spawn_file_actions_adddup2(&fa, 1, 2);
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);
    if (getenv("METAL_ASAHI_DEBUG") && atoi(getenv("METAL_ASAHI_DEBUG")) >= 2) {
        fprintf(stderr, "metal-asahi: sim:");
        for (auto& a : args) fprintf(stderr, " %s", a.c_str());
        fprintf(stderr, "\n");
    }
    pid_t pid;
    int e = posix_spawn(&pid, argv[0], &fa, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    if (e != 0) {
        r.error = "cannot run the simulator " + simulatorPath() + ": " + strerror(e);
        removeTree(dir, files);
        return r;
    }
    double limit = getenv("METAL_ASAHI_SIM_TIMEOUT") ? atof(getenv("METAL_ASAHI_SIM_TIMEOUT")) : 600.0;
    int status = 0;
    bool killed = false;
    for (;;) {
        pid_t w = waitpid(pid, &status, WNOHANG);
        if (w == pid) break;
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > limit) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            killed = true;
            break;
        }
        usleep(2000);
    }
    std::vector<uint8_t> outBytes;
    readFile(outPath, outBytes);
    std::string out(outBytes.begin(), outBytes.end());
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (killed) {
        r.hang = true;
        r.error = "simulator wall-clock timeout (" + std::to_string((int)limit) + " s): treating the kernel as a hang";
        removeTree(dir, files);
        return r;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 || out.find("\"error\"") != std::string::npos) {
        r.error = out.empty() ? "simulator failed" : out;
        while (!r.error.empty() && r.error.back() == '\n') r.error.pop_back();
        if (r.error.find("step") != std::string::npos || r.error.find("max_steps") != std::string::npos) r.hang = true;
        removeTree(dir, files);
        return r;
    }
    auto pos = out.find("\"instructions\":");
    if (pos != std::string::npos) r.instructions = strtoull(out.c_str() + pos + 15, nullptr, 10);
    for (size_t i = 0; i < d.regions.size(); i++) {
        auto& reg = d.regions[i];
        if (reg.dump) {
            if (!outFiles[i].read(reg.out) || reg.out.size() != reg.size) {
                r.error = "simulator dump missing for region " + std::to_string(i);
                removeTree(dir, files);
                return r;
            }
        } else if (reg.patch) {
            std::vector<uint8_t> rec;
            if (!outFiles[i].read(rec) || !applyChanged(rec, reg.patch, reg.size)) {
                r.error = "bad simulator change records for region " + std::to_string(i);
                removeTree(dir, files);
                return r;
            }
        }
    }
    r.ok = true;
    removeTree(dir, files);
    return r;
}

} // namespace masahi
