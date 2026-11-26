// Kernels for bench/bench.cpp.

kernel void empty(global uint* out) {}

kernel void copy4(global const float4* in, global float4* out) {
  uint i = get_global_id(0);
  out[i] = in[i];
}

// 64x64 C tile per 16x16 threadgroup, 4x4 outputs per thread, K tiles of 16 in threadgroup memory.
// M, N multiples of 64, K multiple of 16.
kernel void sgemm_reg4x4(global const float* A, global const float* B, global float* C, uint M, uint N, uint K) {
  local float As[16][64 + 4];   // As[k][m]
  local float Bs[16][64];       // Bs[k][n]
  uint tx = get_local_id(0), ty = get_local_id(1), lid = ty * 16 + tx;
  uint row0 = get_group_id(1) * 64, col0 = get_group_id(0) * 64;
  float acc[4][4];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
  for (uint t = 0; t < K; t += 16) {
    // 1024 elements of each tile, 4 per thread.
    for (uint e = lid; e < 1024; e += 256) {
      uint am = e / 16, ak = e % 16;          // A tile 64 x 16
      As[ak][am] = A[(row0 + am) * K + t + ak];
      uint bk = e / 64, bn = e % 64;          // B tile 16 x 64
      Bs[bk][bn] = B[(t + bk) * N + col0 + bn];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint k = 0; k < 16; k++) {
      float a[4], b[4];
      for (int i = 0; i < 4; i++) a[i] = As[k][ty + 16 * i];
      for (int j = 0; j < 4; j++) b[j] = Bs[k][tx + 16 * j];
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = fma(a[i], b[j], acc[i][j]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) C[(row0 + ty + 16 * i) * N + col0 + tx + 16 * j] = acc[i][j];
}

// As sgemm_reg4x4 with each thread's 4 rows and 4 columns contiguous (rows ty*4.., columns tx*4..):
// the tile reads of a k step are two 16-byte threadgroup loads.
kernel void sgemm_reg4x4v(global const float* A, global const float* B, global float* C, uint M, uint N, uint K) {
  local float4 As[16][16];      // As[k][m/4]
  local float4 Bs[16][16];      // Bs[k][n/4]
  uint tx = get_local_id(0), ty = get_local_id(1), lid = ty * 16 + tx;
  uint row0 = get_group_id(1) * 64, col0 = get_group_id(0) * 64;
  float4 acc[4];
  for (int i = 0; i < 4; i++) acc[i] = (float4)(0.0f);
  local float* as = (local float*)As;
  for (uint t = 0; t < K; t += 16) {
    for (uint e = lid; e < 1024; e += 256) {
      uint am = e / 16, ak = e % 16;
      as[ak * 64 + am] = A[(row0 + am) * K + t + ak];
    }
    {
      uint bk = lid / 16, bn = lid % 16;      // one float4 of B per thread
      Bs[bk][bn] = *(global const float4*)(B + (t + bk) * N + col0 + bn * 4);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint k = 0; k < 16; k++) {
      float4 a = As[k][ty], b = Bs[k][tx];
      acc[0] = fma((float4)(a.x), b, acc[0]);
      acc[1] = fma((float4)(a.y), b, acc[1]);
      acc[2] = fma((float4)(a.z), b, acc[2]);
      acc[3] = fma((float4)(a.w), b, acc[3]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int i = 0; i < 4; i++) *(global float4*)(C + (row0 + ty * 4 + i) * N + col0 + tx * 4) = acc[i];
}

// 8x8 outputs per thread (two 4-blocks each way, 32 apart), 128x128 C tile per 16x16 threadgroup.
kernel void sgemm_reg8x8(global const float* A, global const float* B, global float* C, uint M, uint N, uint K) {
  local float4 As[8][32];       // As[k][m/4], 128 rows
  local float4 Bs[8][32];       // Bs[k][n/4], 128 columns
  uint tx = get_local_id(0), ty = get_local_id(1), lid = ty * 16 + tx;
  uint row0 = get_group_id(1) * 128, col0 = get_group_id(0) * 128;
  float4 acc[8][2];
  for (int i = 0; i < 8; i++) acc[i][0] = acc[i][1] = (float4)(0.0f);
  local float* as = (local float*)As;
  for (uint t = 0; t < K; t += 8) {
    for (uint e = lid; e < 1024; e += 256) {   // A tile 128 x 8
      uint am = e / 8, ak = e % 8;
      as[ak * 128 + am] = A[(row0 + am) * K + t + ak];
    }
    {
      uint bk = lid / 32, bn = lid % 32;      // B tile 8 x 128: one float4 per thread
      Bs[bk][bn] = *(global const float4*)(B + (t + bk) * N + col0 + bn * 4);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint k = 0; k < 8; k++) {
      float4 a0 = As[k][ty], a1 = As[k][ty + 16], b0 = Bs[k][tx], b1 = Bs[k][tx + 16];
      float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      for (int i = 0; i < 8; i++) {
        acc[i][0] = fma((float4)(a[i]), b0, acc[i][0]);
        acc[i][1] = fma((float4)(a[i]), b1, acc[i][1]);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int i = 0; i < 8; i++) {
    uint r = row0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4);
    *(global float4*)(C + r * N + col0 + tx * 4) = acc[i][0];
    *(global float4*)(C + r * N + col0 + 64 + tx * 4) = acc[i][1];
  }
}

// SIMD-group matrices (Metal's simdgroup_float8x8), called by their AIR names.
typedef float __attribute__((ext_vector_type(64))) mat8f;
typedef long __attribute__((ext_vector_type(2))) dim2;
mat8f air_mload(global const float* p, dim2 dims, dim2 stride, dim2 origin) __asm("air.simdgroup_matrix_8x8_load.v64f32.p1f32");
mat8f air_mload_tg(local const float* p, dim2 dims, dim2 stride, dim2 origin) __asm("air.simdgroup_matrix_8x8_load.v64f32.p3f32");
void air_mstore(mat8f m, global float* p, dim2 dims, dim2 stride, dim2 origin) __asm("air.simdgroup_matrix_8x8_store.v64f32.p1f32");
mat8f air_mma(mat8f a, mat8f b, mat8f c) __asm("air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32");
mat8f air_mfill(float v) __asm("air.simdgroup_matrix_8x8_init_filled.v64f32.f32");

// 128 threads = 4 SIMD groups (2 x 2), each 32x32 of C as 4x4 8x8 matrices, A and B read
// from device memory. M, N multiples of 64, K of 8.
kernel void sgemm_simd(global const float* A, global const float* B, global float* C, uint M, uint N, uint K) {
  uint sg = get_local_id(0) / 32;
  uint row0 = get_group_id(1) * 64 + (sg / 2) * 32, col0 = get_group_id(0) * 64 + (sg % 2) * 32;
  dim2 d8 = (dim2)(8, 8), z = (dim2)(0, 0);
  mat8f acc[4][4];
  _Pragma("unroll") for (int i = 0; i < 4; i++)
    _Pragma("unroll") for (int j = 0; j < 4; j++) acc[i][j] = air_mfill(0.0f);
  for (uint k = 0; k < K; k += 8) {
    mat8f a[4], b[4];
    _Pragma("unroll") for (int i = 0; i < 4; i++) a[i] = air_mload(A + (row0 + 8 * i) * K + k, d8, (dim2)(1, K), z);
    _Pragma("unroll") for (int j = 0; j < 4; j++) b[j] = air_mload(B + k * N + col0 + 8 * j, d8, (dim2)(1, N), z);
    _Pragma("unroll") for (int i = 0; i < 4; i++)
      _Pragma("unroll") for (int j = 0; j < 4; j++) acc[i][j] = air_mma(a[i], b[j], acc[i][j]);
  }
  _Pragma("unroll") for (int i = 0; i < 4; i++)
    _Pragma("unroll") for (int j = 0; j < 4; j++) air_mstore(acc[i][j], C + (row0 + 8 * i) * N + col0 + 8 * j, d8, (dim2)(1, N), z);
}
