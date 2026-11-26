// A small kernel-language header for writing Metal compute kernels in OpenCL C
// that the LLVM AIR target lowers to AIR (tools/air-build.sh). It replaces
// OpenCL's library header: work-item functions keep OpenCL's mangled names
// (air-from-generic-gpu maps them to AIR's system values), math becomes LLVM
// intrinsics (mapped to air.* functions), atomics become atomicrmw/cmpxchg.
#pragma once

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef __SIZE_TYPE__ size_t;

#define VEC(T, N) typedef T T##N __attribute__((ext_vector_type(N)))
VEC(float, 2); VEC(float, 3); VEC(float, 4);
VEC(int, 2); VEC(int, 3); VEC(int, 4);
VEC(uint, 2); VEC(uint, 3); VEC(uint, 4);
VEC(half, 2); VEC(half, 4);
VEC(uchar, 4); VEC(ushort, 2); VEC(short, 2);
#undef VEC

#define OVL __attribute__((overloadable))
OVL size_t get_global_id(uint d);
OVL size_t get_local_id(uint d);
OVL size_t get_group_id(uint d);
OVL size_t get_local_size(uint d);
OVL size_t get_num_groups(uint d);
OVL uint get_sub_group_local_id(void);
OVL uint get_sub_group_id(void);
OVL uint get_sub_group_size(void);
OVL uint get_num_sub_groups(void);
OVL void barrier(uint flags);
OVL size_t get_global_size(uint d); // the grid size in threads (Metal threads_per_grid)

#ifndef CLK_LOCAL_MEM_FENCE
#define CLK_LOCAL_MEM_FENCE 1
#define CLK_GLOBAL_MEM_FENCE 2
#endif

// Math: clang's element-wise builtins are LLVM intrinsics (scalars and vectors).
#define sqrt(x) __builtin_elementwise_sqrt(x)
#define sin(x) __builtin_elementwise_sin(x)
#define cos(x) __builtin_elementwise_cos(x)
#define tan(x) __builtin_elementwise_tan(x)
#define exp(x) __builtin_elementwise_exp(x)
#define exp2(x) __builtin_elementwise_exp2(x)
#define log(x) __builtin_elementwise_log(x)
#define log2(x) __builtin_elementwise_log2(x)
#define log10(x) __builtin_elementwise_log10(x)
#define pow(x, y) __builtin_elementwise_pow(x, y)
#define fabs(x) __builtin_elementwise_abs(x)
#define floor(x) __builtin_elementwise_floor(x)
#define ceil(x) __builtin_elementwise_ceil(x)
#define trunc(x) __builtin_elementwise_trunc(x)
#define rint(x) __builtin_elementwise_rint(x)
#define round(x) __builtin_elementwise_round(x)
#define fma(a, b, c) __builtin_elementwise_fma(a, b, c)
#define fmin(a, b) __builtin_elementwise_minnum(a, b)
#define fmax(a, b) __builtin_elementwise_maxnum(a, b)
#define min(a, b) __builtin_elementwise_min(a, b)
#define max(a, b) __builtin_elementwise_max(a, b)
#define copysign(a, b) __builtin_elementwise_copysign(a, b)
#define popcount(x) __builtin_elementwise_popcount(x)
#define clz(x) __builtin_elementwise_clzg(x, (__typeof__(x))(8 * sizeof(x)))
#define ctz(x) __builtin_elementwise_ctzg(x, (__typeof__(x))(8 * sizeof(x)))
#define reverse_bits(x) __builtin_elementwise_bitreverse(x)
#define tanh(x) __builtin_elementwise_tanh(x)
#define asin(x) __builtin_elementwise_asin(x)
#define acos(x) __builtin_elementwise_acos(x)
#define atan(x) __builtin_elementwise_atan(x)
#define atan2(y, x) __builtin_elementwise_atan2(y, x)
#define sinh(x) __builtin_elementwise_sinh(x)
#define cosh(x) __builtin_elementwise_cosh(x)
static inline float rsqrt(float x) { return 1.0f / sqrt(x); }
static inline float clampf(float x, float lo, float hi) { return fmin(fmax(x, lo), hi); }
static inline float mixf(float a, float b, float t) { return a + (b - a) * t; }
static inline uint mul_hi(uint a, uint b) { return (uint)(((ulong)a * b) >> 32); }
static inline int mul_hi_i(int a, int b) { return (int)(((long)a * b) >> 32); }
static inline uint rotate(uint x, uint n) { return __builtin_rotateleft32(x, n); }
static inline float dot4(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

// Atomics (relaxed, as Metal's memory_order_relaxed).
#define atomic_add(p, v) __atomic_fetch_add((p), (v), __ATOMIC_RELAXED)
#define atomic_sub(p, v) __atomic_fetch_sub((p), (v), __ATOMIC_RELAXED)
#define atomic_and(p, v) __atomic_fetch_and((p), (v), __ATOMIC_RELAXED)
#define atomic_or(p, v) __atomic_fetch_or((p), (v), __ATOMIC_RELAXED)
#define atomic_xor(p, v) __atomic_fetch_xor((p), (v), __ATOMIC_RELAXED)
#define atomic_min(p, v) __atomic_fetch_min((p), (v), __ATOMIC_RELAXED)
#define atomic_max(p, v) __atomic_fetch_max((p), (v), __ATOMIC_RELAXED)
#define atomic_xchg(p, v) __atomic_exchange_n((p), (v), __ATOMIC_RELAXED)
#define atomic_load(p) __atomic_load_n((p), __ATOMIC_RELAXED)
#define atomic_store(p, v) __atomic_store_n((p), (v), __ATOMIC_RELAXED)
static inline uint atomic_cmpxchg_u(volatile global uint* p, uint cmp, uint val)
{
    __atomic_compare_exchange_n(p, &cmp, val, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    return cmp;
}
static inline uint atomic_cmpxchg_local(volatile local uint* p, uint cmp, uint val)
{
    __atomic_compare_exchange_n(p, &cmp, val, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    return cmp;
}

// Reinterpretation.
#define as_uint(x) __builtin_astype((x), uint)
#define as_int(x) __builtin_astype((x), int)
#define as_float(x) __builtin_astype((x), float)
#define as_ulong(x) __builtin_astype((x), ulong)
#define clamp(x, lo, hi) __builtin_elementwise_min(__builtin_elementwise_max((x), (lo)), (hi))
#define FLT_MAX_ 3.402823466e+38f
