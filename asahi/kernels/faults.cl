// Kernels that must never reach the GPU: the simulator preflight catches them.

// Never terminates (the trip count comes from memory and is huge, and the
// loop has a side effect the compiler cannot drop).
kernel void spin_forever(global volatile uint* flag, global uint* out) {
  uint n = 0;
  while (flag[0] == 0) {
    n++;
    out[get_global_id(0) & 7] = n;
  }
}

// Writes far outside any buffer.
kernel void wild_store(global uint* out, uint stride) {
  out[get_global_id(0) * stride + 0x10000000u] = 1;
}

// A well-behaved kernel from the same library, to show the library still works.
kernel void fine(global uint* out) {
  out[get_global_id(0)] = 42;
}
