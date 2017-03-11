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
  int i, j;
  int unroll_n = (N/4) * 4;
	for (i = 0; i < unroll_n; i+=4) {
    _mm_store_ps(&ax[i], _mm_set1_ps(0.0f));
	}

  for (; i < N; i++) {
    		ax[i] = 0.0f;
  }

	for (i = 0; i < unroll_n; i+=4) {
    _mm_store_ps(&ay[i], _mm_set1_ps(0.0f));
	}

  for (; i < N; i++) {
    	ay[i] = 0.0f;
  }

	for (i = 0; i < unroll_n; i+=4) {
    _mm_store_ps(&az[i], _mm_set1_ps(0.0f));
	}

  for (; i < N; i++) {
    		az[i] = 0.0f;
  }

	t1 = wtime();
	l0 += (t1 - t0);

    // Loop 1.
	t0 = wtime();

    for (j = 0; j < N; j ++) {
        __m128 xj_v = _mm_set1_ps(x[j]);
        __m128 yj_v = _mm_set1_ps(y[j]);
        __m128 zj_v = _mm_set1_ps(z[j]);
        __m128 mj_v = _mm_set1_ps(m[j]);
        for (i = 0; i < unroll_n; i+=4) {

			__m128 xi_v = _mm_load_ps(&x[i]);
			__m128 rx_v = _mm_sub_ps(xj_v, xi_v);


			__m128 yi_v = _mm_load_ps(&y[i]);
			__m128 ry_v = _mm_sub_ps(yj_v, yi_v);


			__m128 zi_v = _mm_load_ps(&z[i]);
			__m128 rz_v = _mm_sub_ps(zj_v, zi_v);




			// __m128 r2_v = _mm_mul_ps(rx_v, rx_v) + _mm_mul_ps(ry_v, ry_v) + _mm_mul_ps(rz_v, rz_v) + _mm_set1_ps(eps);
			__m128 r2_v = _mm_set1_ps(eps) + rx_v*rx_v + ry_v*ry_v + rz_v*rz_v;   // GNU extension
			__m128 r2inv_v = _mm_rsqrt_ps(r2_v);
            // _mm_div_ps(_mm_set1_ps(1.0f),_mm_sqrt_ps(r2_v));
			__m128 r6inv_1v = _mm_mul_ps(r2inv_v, r2inv_v);
			__m128 r6inv_v = _mm_mul_ps(r6inv_1v, r2inv_v);

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
        for (; i < N; i++) {
			float rx = x[j] - x[i];
			float ry = y[j] - y[i];
			float rz = z[j] - z[i];
			float r2 = rx*rx + ry*ry + rz*rz + eps;
			float r2inv = 1.0f / sqrt(r2);
			float r6inv = r2inv * r2inv * r2inv;
			float s = m[j] * r6inv;
			ax[i] += s * rx;
			ay[i] += s * ry;
			az[i] += s * rz;
		}
	}
	t1 = wtime();
	l1 += (t1 - t0);

	// Loop 2.
	t0 = wtime();

  for (i = 0; i < unroll_n; i+=4) {
    __m128 dt_v = _mm_set1_ps(dt);
    __m128 dmp_v = _mm_set1_ps(dmp);

    __m128 ax_v = _mm_load_ps(&ax[i]);
    __m128 ay_v = _mm_load_ps(&ay[i]);
    __m128 az_v = _mm_load_ps(&az[i]);

    __m128 vx_v = _mm_load_ps(&vx[i]);
    __m128 vy_v = _mm_load_ps(&vy[i]);
    __m128 vz_v = _mm_load_ps(&vz[i]);

    __m128 dt_ax = _mm_mul_ps(dt_v, ax_v);
    __m128 dt_ay = _mm_mul_ps(dt_v, ay_v);
    __m128 dt_az = _mm_mul_ps(dt_v, az_v);

    __m128 dmp_dt_ax = _mm_mul_ps(dmp_v, dt_ax);
    __m128 dmp_dt_ay = _mm_mul_ps(dmp_v, dt_ay);
    __m128 dmp_dt_az = _mm_mul_ps(dmp_v, dt_az);

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
