repeated_median_filter
===================

The function repeated_median_filter(x) is our implementation of the
method described in

   "High breakdown methods of time series analysis.
   Tatum, L.G., and Hurvich, C. M.
   Journal of the Royal Statistical Society. Series B (Methodological),
   pp. 881-896, 1993.

The code is based on an implementation by Falco Strasser, Signal Processing
Group, TU Darmstadt, October 2010.

Input
^^^^

* x	: 1d-array, signal

Output
^^^^

* xFRM	: repeated median filtered (outlier cleaned) signal
* ARM	: Fourier coefficients for cosine
* BRM	: Fourier coefficients for sine