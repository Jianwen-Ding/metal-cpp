// Floating point and integer arithmetic.

// One output row of 24 floats per input.
kernel void float_funcs(global const float* x, global float* out, uint n) {
  uint i = get_global_id(0);
  if (i >= n) return;
  float v = x[i];
  float a = fabs(v) + 0.001f;
  global float* o = out + 24 * i;
  o[0] = sin(v);
  o[1] = cos(v);
  o[2] = exp(v * 0.1f);
  o[3] = log(a);
  o[4] = sqrt(a);
  o[5] = rsqrt(a);
  o[6] = pow(a, 1.7f);
  o[7] = floor(v);
  o[8] = ceil(v);
  o[9] = trunc(v);
  o[10] = rint(v);
  o[11] = fma(v, 1.5f, -2.0f);
  o[12] = fmin(v, 0.25f);
  o[13] = fmax(v, 0.25f);
  o[14] = exp2(v * 0.1f);
  o[15] = log2(a);
  o[16] = tanh(v * 0.3f);
  o[17] = atan(v);
  o[18] = copysign(1.0f, v);
  o[19] = v / 3.0f;
  o[20] = 1.0f / (v + 10.0f);
  o[21] = round(v);
  o[22] = tan(v * 0.2f);
  o[23] = log10(a);
}

// One output row of 16 uints per input pair.
kernel void int_ops(global const uint* a, global const uint* b, global uint* out, uint n) {
  uint i = get_global_id(0);
  if (i >= n) return;
  uint x = a[i], y = b[i] | 1;
  int sx = (int)x, sy = (int)(y >> 1) + 1;
  global uint* o = out + 16 * i;
  o[0] = x * y;
  o[1] = x / y;
  o[2] = x % y;
  o[3] = (uint)(sx / sy);
  o[4] = (uint)(sx % sy);
  o[5] = mul_hi(x, y);
  o[6] = (uint)mul_hi_i(sx, (int)y);
  o[7] = popcount(x);
  o[8] = clz(x | 1);
  o[9] = ctz(x | 0x80000000u);
  o[10] = rotate(x, y & 31);
  o[11] = x >> (y & 31);
  o[12] = (uint)(sx >> (y & 31));
  o[13] = reverse_bits(x);
  o[14] = min(x, y);
  o[15] = (uint)max(sx, -sy);
}

// 64-bit integers.
kernel void int64_ops(global const ulong* a, global const ulong* b, global ulong* out, uint n) {
  uint i = get_global_id(0);
  if (i >= n) return;
  ulong x = a[i], y = b[i] | 1;
  global ulong* o = out + 8 * i;
  o[0] = x + y;
  o[1] = x * y;
  o[2] = x / y;
  o[3] = x % y;
  o[4] = x << (y & 63);
  o[5] = x >> (y & 63);
  o[6] = (ulong)((long)x >> (y & 63));
  o[7] = x ^ (y * 0x9E3779B97F4A7C15ul);
}

// Conversions between floats and integers.
kernel void conversions(global const float* f, global int* out_i, global uint* out_u, global float* out_f, uint n) {
  uint i = get_global_id(0);
  if (i >= n) return;
  float v = f[i];
  out_i[i] = (int)v;
  out_u[i] = (uint)fmax(v, 0.0f);
  out_f[i] = (float)(int)(v * 3.0f) + (float)(uint)i;
}

// Half precision.
kernel void half_math(global const half* a, global const half* b, global half* out, uint n) {
  uint i = get_global_id(0);
  if (i < n) out[i] = a[i] * b[i] + (half)1.0f;
}

// Vector arithmetic.
kernel void vec_math(global const float4* a, global const float4* b, global float4* out, global float* dots, uint n) {
  uint i = get_global_id(0);
  if (i >= n) return;
  float4 x = a[i], y = b[i];
  out[i] = x * y + (float4)(1.0f, 2.0f, 3.0f, 4.0f) - y.wzyx;
  dots[i] = dot4(x, y);
}

// Control flow: data-dependent loops and branches.
kernel void collatz(global const uint* start, global uint* steps, uint n) {
  uint i = get_global_id(0);
  if (i >= n) return;
  uint x = start[i], s = 0;
  while (x != 1 && s < 10000) {
    x = (x & 1) ? 3 * x + 1 : x / 2;
    s++;
  }
  steps[i] = s;
}

kernel void switchy(global const uint* in, global uint* out, uint n) {
  uint i = get_global_id(0);
  if (i >= n) return;
  uint v = in[i];
  uint r;
  switch (v % 7) {
  case 0: r = v * 2; break;
  case 1: r = v + 100; break;
  case 2: r = v ^ 0xffff; break;
  case 3: r = v >> 3; break;
  case 4: r = v * v; break;
  case 5: r = 7; break;
  default: r = v - 1; break;
  }
  out[i] = r;
}
