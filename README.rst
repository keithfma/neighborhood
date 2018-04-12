Neighborhood Algorithm Optimization and Ensemble Appraisal
==========================================================

Travis CI: |image0|

Python 3 implementation of "neighborhood algorithm" direct-search
optimization and Bayesian ensemble appraisal. In short, a
nearest-neighbor interpolant based on Voronoi polygons is used to
interpolate the misfit (search) and posterior probability (appraisal) to
allow efficient sampling and integration for high-dimensional problems.
Details on theory and implementation are supplied in the references.

.. figure:: https://raw.githubusercontent.com/keithfma/neighborhood/master/example_rosenbrock_4d.png 
    :align: center
    :figclass: align-center

    Example search population for 4D Rosenbrock objective function. Image
    include 10,000 samples collected in 1,000 iterations of the neighborhood
    algorithm direct search, with ``num_samp=10`` and ``num_resamp=5``. The
    true minimum is ``0`` at ``(1, 1, 1, 1)``, while the best sample is
    ``0.0113`` at ``((0.976, 0.953, 0.908, 0.824)``. This result continues
    to converge for larger sample size (but the plot is less interesting since
    the density converges to a point!)

To generate the example figure above, you can run the internal demo,
like so:

.. code:: python

    import neighborhood as nbr

    nbr.demo_search(ndim=4, nsamp=10, nresamp=5, niter=500)

Equivalently, you can do the following:

.. code:: python

    import neighborhood as nbr

    num_dim = 4
    srch = nbr.Searcher(
        objective=nbr.rosenbrock,
        limits=[(-1.5, 1.5) for _ in range(num_dim)],
        num_samp=10,
        num_resamp=5,
        maximize=False,
        verbose=True
        )
    srch.update(500)
    srch.plot()

Status
------

Optimization is implemented, ensemble appraisal is in progress.

Testing
-------

This project uses `pytest <https://docs.pytest.org/en/latest/>`__ for
unit testing. The aim is not to be exhuastive, but to provide reasonable
assurances that everything works as advertised. To run, simply call
``pytest --verbose`` from somewhere in this package.

Release
-------

Release versions are tagged in the repository, built as distributions,
and uploaded to PyPI. The minimal commands to do this are:

.. code:: bash

    # update PyPI-readable README
    pandoc --from=markdown --to=rst --output=README.rst README.md
    # build with setuptools
    python3 setup.py sdist bdist_wheel
    # upload to PyPI test server (then check it out)
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    # upload to PyPI
    twine upload dist/*
    # tag release in git repo
    git tag -a X.X.X -m "vX.X.X"
    git push origin --tags

References
----------

1. Sambridge, M. (1999). Geophysical inversion with a neighbourhood
   algorithm - I. Searching a parameter space. Geophysical Journal
   International, 138(2), 479–494.
   http://doi.org/10.1046/j.1365-246X.1999.00876.x

2. Sambridge, M. (1999). Geophysical inversion with a neighborhood
   algorithm -

II. Appraising the ensemble. Geophys, J. Int., 138, 727–746.
    http://doi.org/10.1046/j.1365-246x.1999.00900.x

.. |image0| image:: https://travis-ci.org/keithfma/neighborhood.svg?branch=master
   :target: https://travis-ci.org/keithfma/neighborhood/branches
