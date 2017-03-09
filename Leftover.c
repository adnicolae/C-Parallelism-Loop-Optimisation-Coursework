for (int i = 0; i < N; i++) {
  x[i] += dt * vx[i];
  if (x[i] >= 1.0f || x[i] <= -1.0f) vx[i] *= -1.0f;
}

for (int i = 0; i < N; i++) {
  y[i] += dt * vy[i];
  if (y[i] >= 1.0f || y[i] <= -1.0f) vy[i] *= -1.0f;
}

for (int i = 0; i < N; i++) {
  z[i] += dt * vz[i];
  if (z[i] >= 1.0f || z[i] <= -1.0f) vz[i] *= -1.0f;
}


#include <immintrin.h>
__m128 rsqrt_float4_single(__m128 x) {
 __m128 three = _mm_set1_ps(3.0f), half = _mm_set1_ps(0.5f);
 __m128 res = _mm_rsqrt_ps(x);
 __m128 muls = _mm_mul_ps(_mm_mul_ps(x, res), res);
 return res = _mm_mul_ps(_mm_mul_ps(half, res), _mm_sub_ps(three, muls));
}

__m128 invsqrt_compiler(__m128 x) {
 // surprisingly, gcc/clang just use rcpps + newton, not rsqrt + newton.
 // gcc fails to use FMA for the Newton-Raphson iteration, though: clang is better
 return _mm_set1_ps(1.0f) / _mm_sqrt_ps(x);
}

__m128 inv_compiler(__m128 x) {
 return _mm_set1_ps(1.0f) / x;
}
void compute() {

 double t0, t1;

 // Loop 0.
 t0 = wtime();
 for (int i = 0; i < N; i++) {
   ax[i] = 0.0f;
 }

 for (int i = 0; i < N; i++) {
   ay[i] = 0.0f;
 }

 for (int i = 0; i < N; i++) {
   az[i] = 0.0f;
 }

 t1 = wtime();
 l0 += (t1 - t0);

   // Loop 1.
 t0 = wtime();
 int unroll_n = (N/4) * 4;

 for (int i = 0; i < N; i+=4) {
   __m128 xi_v = _mm_load_ps(&x[i]);
   __m128 yi_v = _mm_load_ps(&y[i]);
   __m128 zi_v = _mm_load_ps(&z[i]);

   // vector accumulators for ax[i + 0..3] etc.
   __m128 axi_v = _mm_setzero_ps();
   __m128 ayi_v = _mm_setzero_ps();
   __m128 azi_v = _mm_setzero_ps();

   // AVX broadcast-loads are as cheap as normal loads,
   // and data-reuse meant that stand-alone load instructions were used anyway.
   // so we're not even losing out on folding loads into other insns
   // An inner-loop stride of only 4B is a huge win if memory / cache bandwidth is a bottleneck
   // even without AVX, the shufps instructions are cheap,
   // and don't compete with add/mul for execution units on Intel

   for (int j = 0; j < N; j++) {
     __m128 xj_v = _mm_set1_ps(x[j]);
     __m128 rx_v = _mm_sub_ps(xj_v, xi_v);

     __m128 yj_v = _mm_set1_ps(y[j]);
     __m128 ry_v = _mm_sub_ps(yj_v, yi_v);

     __m128 zj_v = _mm_set1_ps(z[j]);
     __m128 rz_v = _mm_sub_ps(zj_v, zi_v);

     __m128 mj_v = _mm_set1_ps(m[j]);
     // sum of squared differences
     __m128 r2_v = _mm_set1_ps(eps) + rx_v*rx_v + ry_v*ry_v + rz_v*rz_v;   // GNU extension
     /* __m128 r2_v = _mm_add_ps(_mm_set1_ps(eps), _mm_mul_ps(rx_v, rx_v));
     r2_v = _mm_add_ps(r2_v, _mm_mul_ps(ry_v, ry_v));
     r2_v = _mm_add_ps(r2_v, _mm_mul_ps(rz_v, rz_v));
     */

     // rsqrt and a Newton-Raphson iteration might have lower latency
     // but there's enough surrounding instructions and cross-iteration parallelism
     // that the single-uop sqrtps and divps instructions prob. aren't be a bottleneck
#define USE_RSQRT
#ifndef USE_RSQRT
     // even with -mrecip=vec-sqrt after -ffast-math, this still does sqrt(v)*v, then rcpps
     __m128 r2sqrt = _mm_sqrt_ps(r2_v);
     __m128 r6sqrt = _mm_mul_ps(r2_v, r2sqrt);  // v^(3/2) = sqrt(v)^3 = sqrt(v)*v
     __m128 s_v = _mm_div_ps(mj_v, r6sqrt);
#else
     __m128 r2isqrt = rsqrt_float4_single(r2_v);
     // can't use the sqrt(v)*v trick, unless we either do normal sqrt first then rcpps
     // or rsqrtps and rcpps.  Maybe it's possible to do a Netwon Raphson iteration on that product
     // instead of refining them both separately?
     __m128 r6isqrt = r2isqrt * r2isqrt * r2isqrt;
     __m128 s_v = _mm_mul_ps(mj_v, r6isqrt);
#endif
     __m128 srx_v = _mm_mul_ps(s_v, rx_v);
     __m128 sry_v = _mm_mul_ps(s_v, ry_v);
     __m128 srz_v = _mm_mul_ps(s_v, rz_v);

     axi_v = _mm_add_ps(axi_v, srx_v);
     ayi_v = _mm_add_ps(ayi_v, sry_v);
     azi_v = _mm_add_ps(azi_v, srz_v);
   }
   _mm_store_ps(&ax[i], axi_v);
   _mm_store_ps(&ay[i], ayi_v);
   _mm_store_ps(&az[i], azi_v);
 }
 t1 = wtime();
 l1 += (t1 - t0);

 // Loop 2.
 t0 = wtime();
 for (int i = 0; i < N; i++) {
   vx[i] += dmp * (dt * ax[i]);
 }

 for (int i = 0; i < N; i++) {
   vy[i] += dmp * (dt * ay[i]);
 }

 for (int i = 0; i < N; i++) {
   vz[i] += dmp * (dt * az[i]);
 }
 t1 = wtime();
 l2 += (t1 - t0);

 // Loop 3.
 t0 = wtime();
 for (int i = 0; i < N; i++) {
   x[i] += dt * vx[i];
   y[i] += dt * vy[i];
   z[i] += dt * vz[i];
   if (x[i] >= 1.0f || x[i] <= -1.0f) vx[i] *= -1.0f;
   if (y[i] >= 1.0f || y[i] <= -1.0f) vy[i] *= -1.0f;
   if (z[i] >= 1.0f || z[i] <= -1.0f) vz[i] *= -1.0f;

 }

 t1 = wtime();
 l3 += (t1 - t0);

}
