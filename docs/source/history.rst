*********
Changelog
*********

.. _release-0.2.1:

0.2.1
=====

:Date: October 14, 2021

Features
--------

* Add support for calculating probability density of multivariate Gaussian at a given Mahalanobis distance ``gaussianFunc.compute_density_r``

Incompatible changes
--------------------

* Rename the method ``gaussianFunc.multivariateGaussian`` to ``gaussianFunc.compute_density_r``

.. _release-0.2.0:

0.2.0
=====

:Date: October 14, 2021

Features
--------

* Add support for multi-dimensional arrays or floats for ``triangularFunc.interp``, ``fuzzyRule.apply``, ``fuzzySet.interp``, and ``fuzzySystem.compute``
* Update example ``TRS_example.py`` and documentation example to use these functionalities
* Add support for directly plotting ``triangularFunc`` using ``triangularFunc.view()``

Incompatible changes
--------------------

* Simplify API to directly import ``triangularFunc``, ``fuzzyRule``, ``fuzzySet``, ``fuzzySystem``, ``Design``, and ``gaussianFunc``

.. _release-0.1.0:

0.1.0
=====

:Date: October 9, 2021

Features
--------

* Introduce  ``fuzzyLib``, ``DOELib``, and ``uncertaintyLib``, and ``fuzzySystem.compute``
* Introduce fuzzy inference using ``dmLib.fuzzyLib.fuzzySystem.fuzzySystem.compute()`` for a ``dict`` of floats
* Add example ``TRS_example.py`` and documentation example to use these functionalities
