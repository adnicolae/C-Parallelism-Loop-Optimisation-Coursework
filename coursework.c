/**
 * The function to optimise as part of the coursework.
 *
 * l0, l1, l2 and l3 record the amount of time spent in each loop
 * and should not be optimised out. :)
 */
#include <immintrin.h>
#include <omp.h>
// __m128 rsqrt_NR(__m128 v) {
// 	__m128 three = _mm_set1_ps(3.0f), half = _mm_set1_ps(0.5f);
// 	__m128 result = _mm_rsqrt_ps(x);
// 	__m128 multiply = _mm_mul_ps(_mm_mul_ps(x, res), res);
// 	return res = _mm_mul_ps(_mm_mul_ps(half, res), _mm_sub_ps(three, muls));
// }
//
// __m128 hsum(__m128 v) {
// 	__m128 shuf = _mm_movehdup_ps(v);
// 	__m128 sums = _mm_add_ps(v, shuf);
//     shuf        = _mm_movehl_ps(shuf, sums);
//     return sums        = _mm_add_ss(sums, shuf);
// }

#include <immintrin.h>
#include <omp.h>
__m128 rsqrt_float4_single(__m128 x) {
 __m128 three = _mm_set1_ps(3.0f), half = _mm_set1_ps(0.5f);
 __m128 res = _mm_rsqrt_ps(x);
 __m128 muls = _mm_mul_ps(_mm_mul_ps(x, res), res);
 return res = _mm_mul_ps(_mm_mul_ps(half, res), _mm_sub_ps(three, muls));
}
__m128 hsum_ps_sse3(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums        = _mm_add_ss(sums, shuf);
    return        sums;
}
void compute() {
	double t0, t1;
	int i=N, j;
	int unroll_n = (N/4) * 4;
	omp_set_num_threads(4);

	// Loop 0
	l0 = 0;

	// Loop 1.
	t0 = wtime();
	__m128 _mm_eps = _mm_load1_ps(&eps);
	__m128 rx_v, ry_v, rz_v, r2_v, r2inv_v, r6inv_v, s_v, xi_v, yi_v, zi_v;
	#pragma omp parallel for num_threads(4) schedule(dynamic, 64) default(shared) private(i, rx_v, rz_v, r2_v, r2inv_v, r6inv_v, s_v)
	for (i = 0; i < N; i++) {
		xi_v = _mm_set1_ps(x[i]);
		yi_v = _mm_set1_ps(y[i]);
		zi_v = _mm_set1_ps(z[i]);

		__m128 axi_v = _mm_setzero_ps();
		__m128 ayi_v = _mm_setzero_ps();
		__m128 azi_v = _mm_setzero_ps();
		for (j = 0; j < unroll_n; j+=4) {
			rx_v = _mm_sub_ps(_mm_load_ps(&x[j]),xi_v);
			ry_v = _mm_sub_ps(_mm_load_ps(&y[j]),yi_v);
			rz_v = _mm_sub_ps(_mm_load_ps(&z[j]),zi_v);

			r2_v = _mm_add_ps(_mm_add_ps(_mm_mul_ps(rx_v,rx_v),_mm_mul_ps(ry_v,ry_v)),_mm_add_ps(_mm_mul_ps(rz_v,rz_v),_mm_eps));
			r2inv_v = _mm_rsqrt_ps(r2_v);
			r6inv_v = _mm_mul_ps(_mm_mul_ps(r2inv_v,r2inv_v),r2inv_v);

			s_v = _mm_mul_ps(_mm_load_ps(m+j),r6inv_v);

			axi_v = _mm_add_ps(axi_v,_mm_mul_ps(s_v,rx_v));
			ayi_v = _mm_add_ps(ayi_v,_mm_mul_ps(s_v,ry_v));
			azi_v = _mm_add_ps(azi_v,_mm_mul_ps(s_v,rz_v));
		}

		for (; j < N; j++) {
			float  rx = x[j] - x[i];
			float  ry = y[j] - y[i];
			float  rz = z[j] - z[i];
			float  r2 = rx*rx + ry*ry + rz*rz + eps;
			float  r2inv = 1.0f / sqrt(r2);
			float  r6inv = r2inv * r2inv * r2inv;
			float  s = m[j] * r6inv;
			ax[i] += s * rx;
			ay[i] += s * ry;
			az[i] += s * rz;
	  }

	  _mm_store_ss(&ax[i], _mm_hadd_ps(_mm_hadd_ps(axi_v,axi_v),_mm_hadd_ps(axi_v,axi_v)));
	  _mm_store_ss(&ay[i], _mm_hadd_ps(_mm_hadd_ps(ayi_v,ayi_v),_mm_hadd_ps(ayi_v,ayi_v)));
	  _mm_store_ss(&az[i], _mm_hadd_ps(_mm_hadd_ps(azi_v,azi_v),_mm_hadd_ps(azi_v,azi_v)));
	}

	t1 = wtime();
	l1 += (t1 - t0);


	// Loop 2.
	t0 = wtime();
	__m128 dt_v = _mm_set1_ps(dt);
	__m128 dmp_v = _mm_set1_ps(dmp);
	__m128 ax_v, ay_v, az_v, vx_v, vy_v, vz_v, dt_ax, dt_ay, dt_az, dmp_dt_ax, dmp_dt_ay, dmp_dt_az;

	for (i = 0; i < unroll_n; i+=4) {
		ax_v = _mm_load_ps(&ax[i]);
		ay_v = _mm_load_ps(&ay[i]);
		az_v = _mm_load_ps(&az[i]);

		vx_v = _mm_load_ps(&vx[i]);
		vy_v = _mm_load_ps(&vy[i]);
		vz_v = _mm_load_ps(&vz[i]);

		dt_ax = _mm_mul_ps(dt_v, ax_v);
		dt_ay = _mm_mul_ps(dt_v, ay_v);
		dt_az = _mm_mul_ps(dt_v, az_v);

		dmp_dt_ax = _mm_mul_ps(dmp_v, dt_ax);
		dmp_dt_ay = _mm_mul_ps(dmp_v, dt_ay);
		dmp_dt_az = _mm_mul_ps(dmp_v, dt_az);

		_mm_store_ps(&vx[i], _mm_add_ps(vx_v, dmp_dt_ax));
		_mm_store_ps(&vy[i], _mm_add_ps(vy_v, dmp_dt_ay));
		_mm_store_ps(&vz[i], _mm_add_ps(vz_v, dmp_dt_az));
	}

	for (; i < N; i++) {
		vx[i] += dmp * (dt * ax[i]);
		vy[i] += dmp * (dt * ay[i]);
		vz[i] += dmp * (dt * az[i]);
	}

	t1 = wtime();
	l2 += (t1 - t0);

	// Loop 3.
	t0 = wtime();
	__m128 one = _mm_set1_ps(1.0f);
	__m128 minus_one = _mm_set1_ps(-1.0f);
	__m128 two = _mm_set1_ps(2.0f);
	__m128 vxi_v, vyi_v, vzi_v, dt_vx_v, dt_vy_v, dt_vz_v;
	for (i = 0; i < unroll_n; i+=4) {
		xi_v = _mm_load_ps(&x[i]);
		vxi_v = _mm_load_ps(&vx[i]);
		dt_vx_v = _mm_mul_ps(dt_v, vxi_v);

		_mm_store_ps(&x[i], _mm_add_ps(xi_v, dt_vx_v));
		xi_v = _mm_load_ps(&x[i]);

		_mm_store_ps(&vx[i], _mm_mul_ps(vxi_v, _mm_sub_ps(_mm_min_ps(_mm_and_ps(_mm_cmplt_ps(xi_v, one), _mm_cmpgt_ps(xi_v, minus_one)), two), one)));

	}

	for (; i < N; i++) {
		x[i] += dt * vx[i];
		if (x[i] >= 1.0f || x[i] <= -1.0f) vx[i] *= -1.0f;
	}

	for (i = 0; i < unroll_n; i+=4) {
		yi_v = _mm_load_ps(&y[i]);
		vyi_v = _mm_load_ps(&vy[i]);
		dt_vy_v = _mm_mul_ps(dt_v, vyi_v);

		_mm_store_ps(&y[i], _mm_add_ps(yi_v, dt_vy_v));
		yi_v = _mm_load_ps(&y[i]);

		_mm_store_ps(&vy[i], _mm_mul_ps(vyi_v, _mm_sub_ps(_mm_min_ps(_mm_and_ps(_mm_cmplt_ps(yi_v, one), _mm_cmpgt_ps(yi_v, minus_one)), two), one)));
	}

	for (; i < N; i++) {
		y[i] += dt * vy[i];
		if (y[i] >= 1.0f || y[i] <= -1.0f) vy[i] *= -1.0f;
	}

	for (i = 0; i < unroll_n; i+=4) {
		zi_v = _mm_load_ps(&z[i]);
		vzi_v = _mm_load_ps(&vz[i]);
		dt_vz_v = _mm_mul_ps(dt_v, vzi_v);

		_mm_store_ps(&z[i], _mm_add_ps(zi_v, dt_vz_v));

		zi_v = _mm_load_ps(&z[i]);

		_mm_store_ps(&vz[i], _mm_mul_ps(vzi_v, _mm_sub_ps(_mm_min_ps(_mm_and_ps(_mm_cmplt_ps(zi_v, one), _mm_cmpgt_ps(zi_v, minus_one)), two), one)));
	}

	for (; i < N; i++) {
		z[i] += dt * vz[i];
		if (z[i] >= 1.0f || z[i] <= -1.0f) vz[i] *= -1.0f;
	}
	t1 = wtime();
	l3 += (t1 - t0);

}
