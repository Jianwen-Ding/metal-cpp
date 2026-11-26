// Basic kernels for the metal-cpp on Asahi Linux test suite (OpenCL C -> AIR).
kernel void vadd(global const float* a, global const float* b, global float* c, uint n) {
  uint i = get_global_id(0);
  if (i < n) c[i] = a[i] + b[i];
}

kernel void fill_index(global uint* out) {
  out[get_global_id(0)] = get_global_id(0);
}

kernel void saxpy(global const float* x, global float* y, float a, uint n) {
  uint i = get_global_id(0);
  if (i < n) y[i] = a * x[i] + y[i];
}
