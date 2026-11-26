// Realistic compute workloads (the shapes llama.cpp, Blender and Geekbench run).

kernel void sgemm_naive(global const float* A, global const float* B, global float* C, uint M, uint N, uint K) {
  uint col = get_global_id(0), row = get_global_id(1);
  if (row >= M || col >= N) return;
  float acc = 0.0f;
  for (uint k = 0; k < K; k++) acc = fma(A[row * K + k], B[k * N + col], acc);
  C[row * N + col] = acc;
}

#define TILE 16
kernel void sgemm_tiled(global const float* A, global const float* B, global float* C, uint M, uint N, uint K) {
  local float As[TILE][TILE];
  local float Bs[TILE][TILE];
  uint tx = get_local_id(0), ty = get_local_id(1);
  uint col = get_group_id(0) * TILE + tx, row = get_group_id(1) * TILE + ty;
  float acc = 0.0f;
  for (uint t = 0; t < K; t += TILE) {
    As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.0f;
    Bs[ty][tx] = (col < N && t + ty < K) ? B[(t + ty) * N + col] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint k = 0; k < TILE; k++) acc = fma(As[ty][k], Bs[k][tx], acc);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

kernel void transpose(global const float* in, global float* out, uint w, uint h) {
  local float tile[TILE][TILE + 1];
  uint x = get_group_id(0) * TILE + get_local_id(0), y = get_group_id(1) * TILE + get_local_id(1);
  if (x < w && y < h) tile[get_local_id(1)][get_local_id(0)] = in[y * w + x];
  barrier(CLK_LOCAL_MEM_FENCE);
  x = get_group_id(1) * TILE + get_local_id(0);
  y = get_group_id(0) * TILE + get_local_id(1);
  if (x < h && y < w) out[y * h + x] = tile[get_local_id(0)][get_local_id(1)];
}

// Inclusive scan of each threadgroup's slice (Hillis-Steele, double buffered).
kernel void scan_group(global const uint* in, global uint* out, global uint* sums, local uint* buf) {
  uint l = get_local_id(0), n = get_local_size(0), g = get_global_id(0);
  local uint* a = buf;
  local uint* b = buf + n;
  a[l] = in[g];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint off = 1; off < n; off <<= 1) {
    b[l] = l >= off ? a[l] + a[l - off] : a[l];
    barrier(CLK_LOCAL_MEM_FENCE);
    local uint* t = a; a = b; b = t;
  }
  out[g] = a[l];
  if (l == n - 1) sums[get_group_id(0)] = a[l];
}

kernel void add_offsets(global uint* data, global const uint* scanned_sums) {
  uint grp = get_group_id(0);
  if (grp > 0) data[get_global_id(0)] += scanned_sums[grp - 1];
}

// One bitonic merge step over the whole array.
kernel void bitonic_step(global uint* data, uint j, uint k) {
  uint i = get_global_id(0), ixj = i ^ j;
  if (ixj > i) {
    uint a = data[i], b = data[ixj];
    bool up = (i & k) == 0;
    if ((a > b) == up) {
      data[i] = b;
      data[ixj] = a;
    }
  }
}

kernel void blur3x3(global const float* in, global float* out, uint w, uint h) {
  int x = get_global_id(0), y = get_global_id(1);
  if (x >= (int)w || y >= (int)h) return;
  float s = 0.0f;
  for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
      int xx = clamp(x + dx, 0, (int)w - 1), yy = clamp(y + dy, 0, (int)h - 1);
      s += in[yy * w + xx];
    }
  out[y * w + x] = s / 9.0f;
}

kernel void mandelbrot(global uint* out, uint w, uint h, uint max_iter) {
  uint px = get_global_id(0), py = get_global_id(1);
  if (px >= w || py >= h) return;
  float cx = -2.0f + 3.0f * px / w, cy = -1.25f + 2.5f * py / h;
  float x = 0.0f, y = 0.0f;
  uint it = 0;
  while (x * x + y * y <= 4.0f && it < max_iter) {
    float t = x * x - y * y + cx;
    y = 2.0f * x * y + cy;
    x = t;
    it++;
  }
  out[py * w + px] = it;
}

kernel void nbody_step(global const float4* pos, global float4* vel, global float4* newpos, uint n, float dt) {
  uint i = get_global_id(0);
  if (i >= n) return;
  float4 p = pos[i];
  float3 acc = (float3)(0.0f);
  for (uint j = 0; j < n; j++) {
    float4 q = pos[j];
    float3 d = q.xyz - p.xyz;
    float r2 = d.x * d.x + d.y * d.y + d.z * d.z + 0.01f;
    float inv = rsqrt(r2);
    acc += d * (q.w * inv * inv * inv);
  }
  float4 v = vel[i];
  v.xyz += acc * dt;
  vel[i] = v;
  newpos[i] = (float4)(p.xyz + v.xyz * dt, p.w);
}

// Row softmax: one threadgroup per row, reductions in threadgroup memory.
kernel void softmax_rows(global const float* in, global float* out, uint cols, local float* red) {
  uint row = get_group_id(0), l = get_local_id(0), n = get_local_size(0);
  global const float* x = in + row * cols;
  float m = -FLT_MAX_;
  for (uint c = l; c < cols; c += n) m = fmax(m, x[c]);
  red[l] = m;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = n / 2; s > 0; s >>= 1) {
    if (l < s) red[l] = fmax(red[l], red[l + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  m = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  float sum = 0.0f;
  for (uint c = l; c < cols; c += n) sum += exp(x[c] - m);
  red[l] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = n / 2; s > 0; s >>= 1) {
    if (l < s) red[l] += red[l + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float inv = 1.0f / red[0];
  for (uint c = l; c < cols; c += n) out[row * cols + c] = exp(x[c] - m) * inv;
}

// Matrix-vector product, one SIMD group's worth of threads per row (llama.cpp's mul_mv shape).
kernel void gemv(global const float* A, global const float* x, global float* y, uint rows, uint cols, local float* red) {
  uint row = get_group_id(0), l = get_local_id(0), n = get_local_size(0);
  if (row >= rows) return;
  float acc = 0.0f;
  for (uint c = l; c < cols; c += n) acc = fma(A[row * cols + c], x[c], acc);
  red[l] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = n / 2; s > 0; s >>= 1) {
    if (l < s) red[l] += red[l + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (l == 0) y[row] = red[0];
}

// Black-Scholes call/put prices.
static inline float cnd(float d) {
  const float A1 = 0.31938153f, A2 = -0.356563782f, A3 = 1.781477937f, A4 = -1.821255978f, A5 = 1.330274429f;
  float K = 1.0f / (1.0f + 0.2316419f * fabs(d));
  float c = 0.3989422804f * exp(-0.5f * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
  return d > 0 ? 1.0f - c : c;
}
kernel void black_scholes(global const float* S, global const float* X, global const float* T, global float* call,
                          global float* put, float r, float v, uint n) {
  uint i = get_global_id(0);
  if (i >= n) return;
  float s = S[i], x = X[i], t = T[i];
  float sq = sqrt(t);
  float d1 = (log(s / x) + (r + 0.5f * v * v) * t) / (v * sq), d2 = d1 - v * sq;
  float e = exp(-r * t);
  call[i] = s * cnd(d1) - x * e * cnd(d2);
  put[i] = x * e * (1.0f - cnd(d2)) - s * (1.0f - cnd(d1));
}

// Grid-stride copy (bandwidth).
kernel void stream_copy(global const float4* in, global float4* out, uint n) {
  for (uint i = get_global_id(0); i < n; i += get_global_size(0)) out[i] = in[i];
}

kernel void stream_triad(global const float* a, global const float* b, global float* c, float s, uint n) {
  for (uint i = get_global_id(0); i < n; i += get_global_size(0)) c[i] = a[i] + s * b[i];
}
