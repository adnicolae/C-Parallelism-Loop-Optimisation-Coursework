/**
 * The function to optimise as part of the coursework.
 *
 * l0, l1, l2 and l3 record the amount of time spent in each loop
 * and should not be optimised out. :)
 */
 #include <immintrin.h>
  #include <omp.h>
  float hsum_ps_sse(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
  }

  inline void horizontal_add_float(__m128 vec, float* dest)
{
  //Invert first 2 bits of vec
  __m128 temp = _mm_movehl_ps(vec, vec);
  temp = _mm_add_ps(vec, temp);
  temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, _MM_SHUFFLE(0,0,0,1)));
  _mm_store_ss(dest, temp);
}
void compute() {

	double t0, t1;
  omp_set_num_threads(4);

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
  // __m128 xj_v, zj_v, yj_v, rx_v, xi_v, ry_v, yi_v, zi_v, rz_v, r2_v, r2inv_v, r6inv_v, r6inv_1v, s_v, axi_v, ayi_v, azi_v, srx_v, sry_v, srz_v, mj_v;
  __m128 xi_v, yi_v, zi_v, rx_v, ry_v, rz_v, r2_v, r2inv_v, r6inv_v, s_v, xj_v, yj_v, zj_v;
      float X, Y, Z;

      #pragma omp parallel for schedule(static,64) shared(x,y,z,ax,ay,az,m,N,eps) private(i,xi_v,yi_v,zi_v,j,rx_v,ry_v,rz_v,xj_v, yj_v, zj_v,r2_v,r2inv_v,r6inv_v,s_v,X,Y,Z)
      	for (i = 0; i < N; i++) {
      	  xi_v = _mm_load_ps1(&x[i]);
      	  yi_v = _mm_load_ps1(&y[i]);
      	  zi_v = _mm_load_ps1(&z[i]);
      	  for (j = 0; j < unroll_n; j+=4) {
            xj_v =  _mm_load_ps(&x[j]);
            yj_v =  _mm_load_ps(&y[j]);
            zj_v =  _mm_load_ps(&z[j]);

      	    rx_v = _mm_sub_ps(xj_v, xi_v);
      	    ry_v = _mm_sub_ps(yj_v, yi_v);
      	    rz_v = _mm_sub_ps(zj_v, zi_v);

      	    r2_v = _mm_set1_ps(eps) + rx_v*rx_v + ry_v*ry_v + rz_v*rz_v;

            r2inv_v = _mm_rsqrt_ps(r2_v);

            r6inv_v = _mm_mul_ps(_mm_mul_ps(r2inv_v,r2inv_v),r2inv_v);

            s_v = _mm_mul_ps(_mm_load_ps(&m[j]),r6inv_v);

            horizontal_add_float(_mm_mul_ps(s_v,rx_v),&X); ax[i] += X;
      			horizontal_add_float(_mm_mul_ps(s_v,ry_v),&Y); ay[i] += Y;
      			horizontal_add_float(_mm_mul_ps(s_v,rz_v),&Z); az[i] += Z;
      	  }
      	  for (;j < N; j++) {
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
  __m128 one = _mm_set1_ps(1.0f);
  __m128 minus_one = _mm_set1_ps(-1.0f);
  __m128 two = _mm_set1_ps(2.0f);
  for (i = 0; i < unroll_n; i+=4) {
    __m128 x_v = _mm_load_ps(&x[i]);
    __m128 vx_v = _mm_load_ps(&vx[i]);
    __m128 dt_v = _mm_set1_ps(dt);
    __m128 dt_vx_v = _mm_mul_ps(dt_v, vx_v);

    _mm_store_ps(&x[i], _mm_add_ps(x_v, dt_vx_v));

	  // if (x[i] >= 1.0f || x[i] <= -1.0f) vx[i] *= -1.0f;
    _mm_store_ps(&vx[i], _mm_mul_ps(vx_v, _mm_sub_ps(_mm_min_ps(_mm_and_ps(_mm_cmplt_ps(_mm_load_ps(&x[i]), one), _mm_cmpgt_ps(_mm_load_ps(&x[i]), minus_one)), two), one)));

	}

	for (; i < N; i++) {
		x[i] += dt * vx[i];
    if (x[i] >= 1.0f || x[i] <= -1.0f) vx[i] *= -1.0f;
	}

	for (i = 0; i < unroll_n; i+=4) {
    __m128 y_v = _mm_load_ps(&y[i]);
    __m128 vy_v = _mm_load_ps(&vy[i]);
    __m128 dt_v = _mm_set1_ps(dt);
    __m128 dt_vy_v = _mm_mul_ps(dt_v, vy_v);

    _mm_store_ps(&y[i], _mm_add_ps(y_v, dt_vy_v));

	  // if (y[i] >= 1.0f || y[i] <= -1.0f) vy[i] *= -1.0f;
    // have to load y again because it h
    _mm_store_ps(&vy[i], _mm_mul_ps(vy_v, _mm_sub_ps(_mm_min_ps(_mm_and_ps(_mm_cmplt_ps(_mm_load_ps(&y[i]), one), _mm_cmpgt_ps(_mm_load_ps(&y[i]), minus_one)), two), one)));
	}

	for (; i < N; i++) {
		y[i] += dt * vy[i];
	  if (y[i] >= 1.0f || y[i] <= -1.0f) vy[i] *= -1.0f;
	}

	for (i = 0; i < unroll_n; i+=4) {
    __m128 z_v = _mm_load_ps(&z[i]);
    __m128 vz_v = _mm_load_ps(&vz[i]);
    __m128 dt_v = _mm_set1_ps(dt);
    __m128 dt_vz_v = _mm_mul_ps(dt_v, vz_v);

    _mm_store_ps(&z[i], _mm_add_ps(z_v, dt_vz_v));

	  // if (z[i] >= 1.0f || z[i] <= -1.0f) vz[i] *= -1.0f;
    _mm_store_ps(&vz[i], _mm_mul_ps(vz_v, _mm_sub_ps(_mm_min_ps(_mm_and_ps(_mm_cmplt_ps(_mm_load_ps(&z[i]), one), _mm_cmpgt_ps(_mm_load_ps(&z[i]), minus_one)), two), one)));
	}

	for (; i < N; i++) {
		z[i] += dt * vz[i];
	  if (z[i] >= 1.0f || z[i] <= -1.0f) vz[i] *= -1.0f;
	}
	t1 = wtime();
	l3 += (t1 - t0);

}
