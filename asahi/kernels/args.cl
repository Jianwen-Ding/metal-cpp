// Kernel arguments: scalars (setBytes), structures in constant buffers, constant
// tables read at variable indices (stay pointers), many buffers, offsets.

typedef struct {
  float scale;
  float bias;
  uint count;
  int shift;
  float4 color;
} Params;

kernel void apply_params(global const float* in, global float* out, constant Params* p) {
  uint i = get_global_id(0);
  if (i < p->count) out[i] = in[i] * p->scale + p->bias + p->color.y * (float)(p->shift);
}

// A constant table indexed by data: the buffer stays a pointer.
kernel void lookup(global const uint* idx, constant float* table, global float* out, uint n) {
  uint i = get_global_id(0);
  if (i < n) out[i] = table[idx[i] & 255];
}

kernel void many_buffers(global const uint* a, global const uint* b, global const uint* c, global const uint* d,
                         global const uint* e, global const uint* f, global const uint* g, global uint* out) {
  uint i = get_global_id(0);
  out[i] = a[i] + 2 * b[i] + 3 * c[i] + 4 * d[i] + 5 * e[i] + 6 * f[i] + 7 * g[i];
}

kernel void scalar_types(global uint* out, uchar a, ushort b, uint c, ulong d, float e, int f, short g) {
  if (get_global_id(0) == 0) {
    out[0] = a;
    out[1] = b;
    out[2] = c;
    out[3] = (uint)d;
    out[4] = (uint)(d >> 32);
    out[5] = as_uint(e);
    out[6] = (uint)f;
    out[7] = (uint)(int)g;
  }
}

// A large parameter block (more uniforms than one 64-half USC word).
typedef struct {
  float w[48];
} Weights;

kernel void big_params(global float* out, constant Weights* p) {
  uint i = get_global_id(0);
  if (i < 48) out[i] = p->w[i] * 2.0f + p->w[47 - i];
}

kernel void copy_u32(global const uint* in, global uint* out) {
  uint i = get_global_id(0);
  out[i] = in[i];
}

typedef struct {
  float x, y;
  uint id;
  float pad;
} Particle;

kernel void particles(global Particle* p, float dt, uint n) {
  uint i = get_global_id(0);
  if (i < n) {
    p[i].x += p[i].y * dt;
    p[i].id = i * 3;
  }
}

kernel void vec4_scale(global float4* v, float4 s) {
  uint i = get_global_id(0);
  v[i] = v[i] * s;
}
