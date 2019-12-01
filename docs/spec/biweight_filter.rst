biweight_filter
===================

The biweight_filter(x) is our implementation of the
method described in 

   "High breakdown methods of time series analysis.
   Tatum, L.G., and Hurvich, C. M.  
   Journal of the Royal Statistical Society. Series B (Methodological),
   pp. 881-896, 1993.
   
The code is based on an implementation by Falco Strasser, Signal Processing
Group, TU Darmstadt, October 2010.

Input
^^^^

* x	: 1d-array, the signal

Output
^^^^

* xFBi	: Biweight filtered (outlier cleaned) signa
* ABi	: Fourier coefficients for cosine
* BBi	: Fourier coefficients for sine

Note
^^^^

Produces slightly different results than matlab, due to
different curve fitting methods