// Threadgroup memory (static arrays and the dynamic [[threadgroup(0)]] buffer),
// barriers and atomics.

// Reverses each threadgroup's slice through a static threadgroup array.
kernel void tg_reverse_static(global const uint* in, global uint* out) {
  local uint tile[256];
  uint l = get_local_id(0), n = get_local_size(0);
  tile[l] = in[get_global_id(0)];
  barrier(CLK_LOCAL_MEM_FENCE);
  out[get_global_id(0)] = tile[n - 1 - l];
}

// The same through the dynamic threadgroup buffer (setThreadgroupMemoryLength).
kernel void tg_reverse_dynamic(global const uint* in, global uint* out, local uint* tile) {
  uint l = get_local_id(0), n = get_local_size(0);
  tile[l] = in[get_global_id(0)];
  barrier(CLK_LOCAL_MEM_FENCE);
  out[get_global_id(0)] = tile[n - 1 - l];
}

// Tree reduction in threadgroup memory: one partial sum per group.
kernel void reduce_sum(global const float* in, global float* partial, local float* s, uint n) {
  uint l = get_local_id(0), g = get_global_id(0), size = get_local_size(0);
  s[l] = g < n ? in[g] : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = size / 2; stride > 0; stride >>= 1) {
    if (l < stride) s[l] += s[l + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (l == 0) partial[get_group_id(0)] = s[0];
}

kernel void atomic_ops(global uint* counters, global int* signed_counters, global const uint* values, uint n) {
  uint i = get_global_id(0);
  if (i >= n) return;
  uint v = values[i];
  atomic_add(&counters[0], 1u);
  atomic_add(&counters[1], v);
  atomic_max(&counters[2], v);
  atomic_min(&counters[3], v);
  atomic_or(&counters[4], 1u << (v & 31));
  atomic_xor(&counters[5], v);
  atomic_and(&counters[6], ~(1u << (i & 31)));
  atomic_sub(&signed_counters[0], (int)(v & 7));
  atomic_max(&signed_counters[1], (int)v - 500);
  atomic_min(&signed_counters[2], (int)v - 500);
}

// Every thread claims one distinct slot with an atomic counter.
kernel void atomic_slots(global uint* counter, global uint* slots, uint n) {
  uint i = get_global_id(0);
  if (i < n) {
    uint s = atomic_add(counter, 1u);
    slots[s] = i;
  }
}

// Lock-free increments through compare-and-swap.
kernel void cas_increment(global uint* word, uint n) {
  if (get_global_id(0) >= n) return;
  uint old = atomic_load(word);
  for (;;) {
    uint seen = atomic_cmpxchg_u((volatile global uint*)word, old, old + 1);
    if (seen == old) break;
    old = seen;
  }
}

// Histogram with threadgroup atomics, merged into the global one.
kernel void histogram(global const uchar* data, global uint* hist, uint n) {
  local uint h[256];
  uint l = get_local_id(0);
  for (uint b = l; b < 256; b += get_local_size(0)) h[b] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = get_global_id(0); i < n; i += get_global_size(0)) atomic_add(&h[data[i]], 1u);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint b = l; b < 256; b += get_local_size(0))
    if (h[b]) atomic_add(&hist[b], h[b]);
}

// Exchange through atomic_xchg: each thread swaps its id into one cell.
kernel void xchg(global uint* cell, global uint* got, uint n) {
  uint i = get_global_id(0);
  if (i < n) got[i] = atomic_xchg(cell, i + 1);
}

// Two arrays in threadgroup memory, with a barrier between phases.
kernel void tg_two_arrays(global const float* in, global float* out) {
  local float a[128];
  local float b[128];
  uint l = get_local_id(0);
  a[l] = in[get_global_id(0)];
  barrier(CLK_LOCAL_MEM_FENCE);
  b[l] = a[(l + 1) % get_local_size(0)] + a[l];
  barrier(CLK_LOCAL_MEM_FENCE);
  out[get_global_id(0)] = b[l] * 0.5f;
}

// Float atomic add (CAS loop over the bits).
kernel void float_atomic_sum(global const float* in, global uint* sum_bits, uint n) {
  uint i = get_global_id(0);
  if (i >= n) return;
  float v = in[i];
  uint old = atomic_load(sum_bits);
  for (;;) {
    uint want = as_uint(as_float(old) + v);
    uint seen = atomic_cmpxchg_u((volatile global uint*)sum_bits, old, want);
    if (seen == old) break;
    old = seen;
  }
}

// out[i] = in[(i + shift) % n] + 1: each thread reads another threadgroup's (another core's) element.
kernel void rotate_add(global const uint* in, global uint* out, uint n, uint shift) {
  uint i = get_global_id(0);
  if (i < n) out[i] = in[(i + shift) % n] + 1;
}
