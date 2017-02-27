/**
 * The function to optimise as part of the coursework.
 *
 * l0, l1, l2 and l3 record the amount of time spent in each loop
 * and should not be optimised out. :)
 */
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
	for (int i = 0; i < unroll_n; i+=4) {
		for (int j = 0; j < N; j++) {
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
			// unroll for 1
			rx = x[j] - x[i+1];
			ry = x[j] + y[i+1];
			rz = z[j] + z[i+1];
			r2 = rx * rx + ry * ry + rz * rz + eps;
			r2inv = 1.0f / sqrt(r2);
			r6inv = r2inv * r2inv * r2inv;
			s = m[j] * r6inv;
			ax[i+1] += s * rx;
			ay[i+1] += s * ry;
			az[i+1] += s * rz;
			// unroll for 3
			rx = x[j] - x[i+2];
			ry = x[j] + y[i+2];
			rz = z[j] + z[i+2];
			r2 = rx * rx + ry * ry + rz * rz + eps;
			r2inv = 1.0f / sqrt(r2);
			r6inv = r2inv * r2inv * r2inv;
			s = m[j] * r6inv;
			ax[i+2] += s * rx;
			ay[i+2] += s * ry;
			az[i+2] += s * rz;
			// unroll for 4
			rx = x[j] - x[i+3];
			ry = x[j] + y[i+3];
			rz = z[j] + z[i+3];
			r2 = rx * rx + ry * ry + rz * rz + eps;
			r2inv = 1.0f / sqrt(r2);
			r6inv = r2inv * r2inv * r2inv;
			s = m[j] * r6inv;
			ax[i+3] += s * rx;
			ay[i+3] += s * ry;
			az[i+3] += s * rz;
		}
	}
	t1 = wtime();
	l1 += (t1 - t0);

	// Loop 2.
	t0 = wtime();
	for (int i = 0; i < N; i++) {
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
