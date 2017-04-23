.. fenics_mpm documentation master file, created by
   sphinx-quickstart on Sun Apr 23 09:09:06 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

fenics_mpm 1.0
*******************************

This first release of an implementation of the `material point method` as initial designed by [sulsky_1994]_, [sulsky_1995]_.  Currently, particles are represented with ``NumPy`` arrays and are integrated with a finite-element grid and basis functions provided by ``FEniCS``.  

Work remaining:

1. Parallelize the code to use ``MPI``.  This could be accomplished with ``PETSc`` or ``NumPy`` alone.  It would also be intersting to offload all the particle calculations to the graphics process using ``PyCUDA`` or something similar, and use the processor CPUs for grid calculations alone.
2. Design an efficient way to communicate between the particles and grid in parallel.  This is difficult because particles move from one grid domain to the next, and as such change processors due to the grid partitioning used by ``FEniCS``, ``ParMETIS``.
3. Create new finite-element basis functions with ``FIAT`` such as cubic splines.
4. Perform `method of manufactured solutions` verification tests in 1D.
5. Improve documentation.
6. All those other ideas that I don't want to discuss yet.

Documentation will be updated at each design iteration in order to remain relavent with the code.

.. _first:

.. toctree::
   :maxdepth: 2
   :caption: Preliminaries

   install
   hello_world

.. _module_overview:

.. toctree::
   :maxdepth: 2
   :caption: Module overview

   material
   gridmodel
   model
   helper


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


References:
------------

.. [sulsky_1994] https://doi.org/10.1016/0045-7825(94)90112-0

.. [sulsky_1995] https://doi.org/10.1016/0010-4655(94)00170-7

