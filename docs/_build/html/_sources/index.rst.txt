.. fenics_mpm documentation master file, created by
   sphinx-quickstart on Sun Apr 23 09:09:06 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The material-point method
*******************************

This is a parallel implementation of written in C++ with OpenMP using a Python front end of the `material-point method` as originally designed by [sulsky_1994]_, [sulsky_1995]_.  Particles are represented with ``NumPy`` arrays and are integrated with a finite-element grid and basis functions provided by ``FEniCS``.

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

