/**
 * The function to optimise as part of the coursework.
 *
 * l0, l1, l2 and l3 record the amount of time spent in each loop
 * and should not be optimised out. :)
 */
 #include <immintrin.h>
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

	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i+=4) {
			__m128 xj_v = _mm_set1_ps(x[j]);
			__m128 xi_v = _mm_load_ps(&x[i]);
			__m128 rx_v = _mm_sub_ps(xj_v, xi_v);

			__m128 yj_v = _mm_set1_ps(y[j]);
			__m128 yi_v = _mm_load_ps(&y[i]);
			__m128 ry_v = _mm_sub_ps(yj_v, yi_v);

			__m128 zj_v = _mm_set1_ps(z[j]);
			__m128 zi_v = _mm_load_ps(&z[i]);
			__m128 rz_v = _mm_sub_ps(zj_v, zi_v);

			__m128 r2_v = _mm_mul_ps(rx_v, rx_v) + _mm_mul_ps(ry_v, ry_v) + _mm_mul_ps(rz_v, rz_v) + _mm_set1_ps(eps);
			__m128 r2inv_v = _mm_div_ps(_mm_set1_ps(1.0f),_mm_sqrt_ps(r2_v));
			__m128 r6inv_1v = _mm_mul_ps(r2inv_v, r2inv_v);
			__m128 r6inv_v = _mm_mul_ps(r6inv_1v, r2inv_v);

			__m128 mj_v = _mm_set1_ps(m[j]);
			__m128 s_v = _mm_mul_ps(mj_v, r6inv_v);

			__m128 axi_v = _mm_load_ps(&ax[i]);
			__m128 ayi_v = _mm_load_ps(&ay[i]);
			__m128 azi_v = _mm_load_ps(&az[i]);

			__m128 srx_v = _mm_mul_ps(s_v, rx_v);
			__m128 sry_v = _mm_mul_ps(s_v, ry_v);
			__m128 srz_v = _mm_mul_ps(s_v, rz_v);

			axi_v = _mm_add_ps(axi_v, srx_v);
			ayi_v = _mm_add_ps(ayi_v, sry_v);
			azi_v = _mm_add_ps(azi_v, srz_v);

			_mm_store_ps(&ax[i], axi_v);
			_mm_store_ps(&ay[i], ayi_v);
			_mm_store_ps(&az[i], azi_v);
	    }
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
