# CS257-coursework
Advanced Computer Architecture Coursework - C Code Optimisation

# Initial values on Fedora i5-2430m

CS257 Optimisation Coursework
-----------------------------------------
 Starting simulation of 1000 stars for 1000 timesteps.


 Loop 0 = 0.004863 seconds.
 
 Loop 1 = 16.031893 seconds.
 
 Loop 2 = 0.003379 seconds.
 
 Loop 3 = 0.005438 seconds.
 
 Total  = 16.045573 seconds.

 GFLOP/s = 1.245203
 
 GB/s = 0.002493

 Answer = 86672752.000000
 
 
 Loop 1 Unrolling by a factor 4 results (1000 1000 0)
 ----------------------------------------------------
 Loop 0 = 0.009664 seconds.
  
 Loop 1 = 16.067531 seconds.
 
 Loop 2 = 0.003277 seconds.
 
 Loop 3 = 0.008118 seconds.
 
 Total  = 16.088589 seconds.

 GFLOP/s = 1.241874
 
 GB/s = 0.002486

 Answer = 13772848.000000 (answer is way off, need to take a look at this)
 
 Observation: The timings for the loop did not improve at all, the answer has dramatically changed, GFLOP stays the same.
 
 Using SSE first attempt
 -----------------------------------------------------------------------
 __m128 r2inv_v = _mm_div_ps(_mm_set1_ps(1.0f),_mm_sqrt_ps(r2_v));
 loop interchanged
 everything inside the inner loop
 
 Loop 0 = 0.004085 seconds. (loop fission)
 
 Loop 1 = 3.475009 seconds.
 
 Loop 2 = 0.004975 seconds.
 
 Loop 3 = 0.007287 seconds.
 
 Total  = 3.491356 seconds.

 GFLOP/s = 5.722705
 GB/s = 0.011457

 Answer = 90729344.000000 (answer is off by 4.57% increase)
 
 ----------------------------------------------------
 Attempt at improving locality by using loads in the outer loop,
 but the speed decreased by at least 0.6s
 ----------------------------------------------------
