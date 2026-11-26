// Program-scope constants: the global data buffer (.agx.globals).

constant float kCoeffs[8] = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f};
constant uint kPrimes[16] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};
constant float kWeights[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};

kernel void use_constants(global const uint* idx, global float* out, uint n) {
  uint i = get_global_id(0);
  if (i < n) {
    uint k = idx[i];
    out[i] = kCoeffs[k & 7] * (float)kPrimes[k & 15];
  }
}

kernel void filter5(global const float* in, global float* out, uint n) {
  int i = get_global_id(0);
  if (i >= (int)n) return;
  float s = 0.0f;
  for (int t = -2; t <= 2; t++) s += kWeights[t + 2] * in[clamp(i + t, 0, (int)n - 1)];
  out[i] = s;
}
