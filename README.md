# Neighborhood Algorithm Optimization and Ensemble Appraisal 

Python 3 implementation of "neighborhood algorithm" direct-search optimization
and Bayesian ensemble appraisal. In short, a nearest-neighbor interpolant based
on Voronoi polygons is used to interpolate the misfit (search) and posterior
probability (appraisal) to allow efficient sampling and integration for
high-dimensional problems. Details on theory and implementation are supplied in
the references.

| ![Example search population for 2D Rosenbrock objective function](example_rosenbrock_2d.png?raw=true) |
| :----: |
| Example search population for 2D Rosenbrock objective function. Image include 18,000 samples collected in 20 iterations of the neighborhood algorithm direct search, with `num_samp=900` and `num_resamp=450`. The true minimum is `0` at `(1, 1)`, while best sample is `2e-10` at `(1.00000012, yy=1.0000016)`. Not bad!|

## Status

Optimization is implemented, ensemble appraisal is in progress.

## References

1. Sambridge, M. (1999). Geophysical inversion with a neighbourhood algorithm -
I. Searching a parameter space. Geophysical Journal International, 138(2),
479–494. http://doi.org/10.1046/j.1365-246X.1999.00876.x 

1. Sambridge, M. (1999). Geophysical inversion with a neighborhood algorithm -
II. Appraising the ensemble. Geophys, J. Int., 138, 727–746.
http://doi.org/10.1046/j.1365-246x.1999.00900.x
