madn
==========

computes the normalized median absolute deviation estimate of scale

Input
^^^^

* y	: numeric 1d-array of size N

Output
^^^^

* sig	: normalized median absolute deviations scale estimate

Example
^^^^

.. code-block:: python
  
import numpy as np
import robustsp as rsp

y = [0,2,3434,2]

rsp.madn(y)