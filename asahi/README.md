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
`METAL_ASAHI_DISPATCHES_PER_COMMAND` (default 64), `METAL_ASAHI_DRM_NODE`.

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

## Tests

`make check` builds the kernels and the test programs in `tests/` and runs them (`tools/run-tests.py`) in `verify`
mode; `make check-sim` runs them on the simulator alone and `make check-hw` on the GPU alone. The suite covers the
Objective-C runtime and Foundation, dispatch geometry (1-3D grids, every threadgroup size, ragged grids), argument
passing, threadgroup memory, atomics, math accuracy, SIMD-group functions, command buffer semantics, library loading
and errors, the simulator's refusal of hanging and faulting kernels, and realistic workloads (GEMM, scans, sorts,
stencils, softmax, GEMV, n-body, Black-Scholes).
