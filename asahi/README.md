# metal-cpp on Asahi Linux

`libmetal_asahi` lets unmodified [metal-cpp](../README.md) compute programs run on Apple silicon GPUs under
[Asahi Linux](https://asahilinux.org). It talks to the GPU through the Asahi DRM uAPI directly (`asahi_drm.h`
ioctls: VM, GEM buffers, compute commands), and compiles Metal kernels from **AIR** (`.air` / `.metallib`) to AGX
machine code with the in-tree LLVM AGX backend. No Metal Shading Language, no Apple tool, no Mesa or Vulkan, and
no proprietary code is involved.

```
metal-cpp program ──objc_msgSend──▶ libmetal_asahi (Objective-C runtime + Foundation + Metal subset)
   .air / .metallib ──LLVM bitcode reader──▶ agx-lower-air ──▶ LLVM O2 ──▶ AGX codegen ──▶ ELF (.text, .agx.kernels)
   dispatch ──▶ uniforms + USC pipeline words + CDM control stream ──DRM_IOCTL_ASAHI_SUBMIT──▶ GPU
            └─▶ (first) AGX Performance Simulator agx_run: faults and hangs never reach the GPU
```

## Building

Needs the LLVM of `../../llvm-project` built with the AGX and AIR targets and `libLLVM.so`, its clang, and (for the
simulator modes) `agx_run` from `AGX-Performance-Simulator/rust`:

```sh
cmake -G Ninja ../../llvm-project/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD=AArch64 -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="AGX;AIR" -DLLVM_ENABLE_PROJECTS=clang \
  -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON          # in /scratch/llvm-build
ninja llc opt clang llvm-dis LLVM
(cd AGX-Performance-Simulator/rust && cargo build --release --bin agx_run)
make LLVM_BUILD=/scratch/llvm-build AGX_RUN=.../release/agx_run           # build/libmetal_asahi.so
make check                                                                 # the test suite (see below)
```

A metal-cpp program is compiled with clang++ (metal-cpp needs blocks and constant NSStrings) against the
compatibility headers in `include/` and linked with the library:

```sh
clang++ -std=c++20 -fblocks -Iasahi/include -I. app.cpp -Lasahi/build -lmetal_asahi
```

## Kernels

Load kernels as AIR: `device->newLibrary(NS::URL::fileURLWithPath(...), &err)` with a `.air` file (one wrapped AIR
bitcode module) or a `.metallib` (one module per kernel), `newLibrary(dispatch_data_t)`, or `newDefaultLibrary()`
(`default.metallib` / `default.air` next to the executable). Metal source (`newLibraryWithSource:`) is not
supported and returns an error.

The kernels must use Metal's binding convention: `!air.kernel` describes every parameter, device and constant
buffers take `[[buffer(n)]]` indices 0, 1, 2 ... in parameter order, one `[[threadgroup(0)]]` buffer may be bound
with `setThreadgroupMemoryLength`. The test suite builds its kernels without MSL (`tools/air-build.sh`):

- OpenCL C with `kernels/air_kernel.h` (work-item functions, math as LLVM intrinsics, atomics) through clang and
  `llc -mtriple=air64-apple-macosx26.0.0 -filetype=obj -air-output=air -air-buffer-per-argument`;
- AIR written as LLVM IR (`kernels/simd.ll`: SIMD-group functions).

`tools/airc` compiles an AIR file the way the library does and prints each kernel's launch record.

## Execution modes

Every dispatch can be run on the AGX Performance Simulator (`agx_run`, a functional model of the AGX ISA) before
it reaches the GPU. `METAL_ASAHI_MODE` selects:

| Mode | What happens |
| --- | --- |
| `preflight` (default) | each dispatch runs on the simulator first; one that faults or exceeds its step budget (a hang) fails its command buffer (`MTLCommandBufferStatusError`) and is never submitted |
| `verify` | preflight, then the GPU, then every buffer the dispatch can reach is compared byte for byte with the simulator's result (mismatches are reported on stderr) |
| `sim` | the simulator only (no GPU needed); its results are written into the buffers |
| `hw` | the GPU only |

Dispatches above `METAL_ASAHI_SIM_MAX_THREADS` threads (default 4M) skip the simulator in `preflight` / `verify`.
Other variables: `METAL_ASAHI_AGX_RUN` (simulator binary), `METAL_ASAHI_SIM_MAX_STEPS`, `METAL_ASAHI_SIM_TIMEOUT`,
`METAL_ASAHI_DEBUG` (1 info, 2 simulator command lines, 3 every message send), `METAL_ASAHI_CPU` (AGX processor,
default from the GPU: `agx-g14c` on M2 Max), `METAL_ASAHI_IR_PIPELINE` (default `default<O2>`),
`METAL_ASAHI_DISPATCHES_PER_COMMAND` (default 64), `METAL_ASAHI_MAX_IN_FLIGHT` (command buffers a queue keeps on
the GPU at once, default 8), `METAL_ASAHI_LLVM_ARGS` (extra LLVM options for the kernel compiler, e.g.
`-agx-tile-full-unroll-threshold=0`), `METAL_ASAHI_DUMP_IR` (file to write each module's IR to after the IR
pipeline), `METAL_ASAHI_CACHE_DIR` / `METAL_ASAHI_CACHE=0` (compiled-kernel cache, default
`$XDG_CACHE_HOME/metal-asahi` or `~/.cache/metal-asahi`), `METAL_ASAHI_COMPILE_THREADS` (threads compiling a
`.metallib`'s modules, default up to 8), `METAL_ASAHI_DRM_NODE`.

Compiled kernels are cached on disk, keyed by the AIR bitcode, the compile options, `METAL_ASAHI_LLVM_ARGS` and
the identity (path, size, mtime) of `libLLVM` and `libmetal_asahi`. An entry carries a 128-bit hash of its contents;
a damaged one is compiled again. A `.metallib`'s modules compile in parallel.

Command buffers are pipelined: the last GPU submission of a command buffer stays in flight while the queue
prepares and submits the next ones, and a completion thread retires them in commit order (status, GPU times,
completed handlers). Work the CPU does for a command buffer (blits, events, simulator runs, verify comparisons)
first waits for the command buffers before it.

## What is implemented

- Objective-C runtime subset: classes, interned selectors, `objc_msgSend` (an AAPCS64 trampoline), retain /
  release / autorelease pools, the Blocks runtime (`_Block_copy`, byref variables) for completion handlers.
- Foundation: `NSString` (including clang constant strings), `NSNumber`, `NSArray`, `NSDictionary`, `NSError`,
  `NSURL`, `NSBundle`, `NSProcessInfo`, `NSAutoreleasePool`.
- Metal compute: `MTLDevice` (properties, families, buffers, libraries, pipelines, queues, events),
  `MTLBuffer` (shared and private storage, `newBufferWithBytesNoCopy` sharing the client's pages),
  `MTLLibrary` / `MTLFunction`, `MTLComputePipelineState` (from functions or descriptors, sync or async),
  `MTLCommandQueue`, `MTLCommandBuffer` (commit, wait, scheduled/completed handlers, status, errors, GPU times,
  events), `MTLComputeCommandEncoder` (buffers, bytes, offsets, threadgroup memory, `dispatchThreads`,
  `dispatchThreadgroups`, serial and concurrent dispatch, memory barriers, `useResource`), `MTLBlitCommandEncoder`
  (copy, fill), `MTLSharedEvent`.
- Kernels: every AIR feature the AGX backend lowers: system values, device / constant / threadgroup memory,
  program-scope constants, atomics, barriers, SIMD-group functions, math, 64-bit integers, half.

Not yet: textures and samplers, kernels that need a stack (spills, dynamically indexed private arrays), indirect
dispatch, function constants, argument buffers, render pipelines.

## Benchmarks

`make bench` runs `bench/bench` scaled down in `verify` mode (simulator, then GPU, results compared), then at full
size on the GPU: compile time and memory, dispatch overhead, bandwidth, and SGEMM kernels (`kernels/bench.cl`). On
the M2 Max (38 cores, 1398 MHz, AC power), 2026-11-26:

| Benchmark | Result |
| --- | --- |
| compile `workloads.air` (14 kernels) | 25 ms (0.2 ms from the cache), 57 MB max RSS for the whole run (25 cached) |
| compile `workloads.metallib` (14 modules) | 44 ms on 1 thread, 14 ms on 8 |
| 1 dispatch, commit + wait | ~120 µs median |
| command buffers in flight, 1 dispatch each | 45 µs per command buffer (188 µs before pipelining) |
| dispatches in one command buffer | 3.5-4 µs GPU, 5.5-6 µs wall each |
| `copy4` (1 GiB each way) | 360-368 GB/s (400 GB/s peak) |
| `sgemm_simd` (SIMD-group matrices) 2048³ | 9.4 TFLOP/s (13.6 peak) |
| `sgemm_reg4x4` (4x4 per thread, threadgroup tiles) 2048³ | 2.6 TFLOP/s |

## Tests

`make check` builds the kernels and the test programs in `tests/` and runs them (`tools/run-tests.py`) in `verify`
mode; `make check-sim` runs them on the simulator alone and `make check-hw` on the GPU alone. The suite covers the
Objective-C runtime and Foundation, dispatch geometry (1-3D grids, every threadgroup size, ragged grids), argument
passing, threadgroup memory, atomics, math accuracy, SIMD-group functions, command buffer semantics, library loading
and errors, the simulator's refusal of hanging and faulting kernels, and realistic workloads (GEMM, scans, sorts,
stencils, softmax, GEMV, n-body, Black-Scholes).
