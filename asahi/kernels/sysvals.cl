// System values: Metal's thread_position_in_grid & co. through OpenCL's work-item functions.

// One record of 12 uints per thread, at its linear global index.
kernel void sysvals3d(global uint* out) {
  uint gx = get_global_id(0), gy = get_global_id(1), gz = get_global_id(2);
  uint sx = get_global_size(0), sy = get_global_size(1);
  uint idx = (gz * sy + gy) * sx + gx;
  global uint* r = out + idx * 12;
  r[0] = gx; r[1] = gy; r[2] = gz;
  r[3] = get_local_id(0); r[4] = get_local_id(1); r[5] = get_local_id(2);
  r[6] = get_group_id(0); r[7] = get_group_id(1); r[8] = get_group_id(2);
  r[9] = get_local_size(0) * 10000 + get_local_size(1) * 100 + get_local_size(2);
  r[10] = get_num_groups(0) * 10000 + get_num_groups(1) * 100 + get_num_groups(2);
  r[11] = 0xC0FFEE;
}

// SIMD-group facts per thread.
kernel void simd_ids(global uint* out) {
  uint i = get_global_id(0);
  out[4 * i + 0] = get_sub_group_local_id();
  out[4 * i + 1] = get_sub_group_id();
  out[4 * i + 2] = get_sub_group_size();
  out[4 * i + 3] = get_num_sub_groups();
}

// Writes 1 for each thread that runs: partial threadgroups must run exactly the grid.
kernel void count_threads(global uint* out, global uint* counter) {
  uint i = get_global_id(0) + get_global_id(1) * get_global_size(0);
  out[i] += 1;
  atomic_add(counter, 1u);
}

// 2D: out[y * w + x] = x * 1000 + y, bounded by (w, h).
kernel void index2d(global uint* out, uint w, uint h) {
  uint x = get_global_id(0), y = get_global_id(1);
  if (x < w && y < h) out[y * w + x] = x * 1000 + y;
}
