*********
Changelog
*********

.. _release-0.4.0:

0.4.0
=====

:Date: October 26, 2021

Features
--------

* Add building block for a Margin Analysis Network (MAN) as a class object ``MarginNode``
* Add ability to call ``MarginNode()`` using a set of requirement observations and design parameters in a Monte Carlo setting
* Add ability to view ``MarginNode`` excess pdf and cdf using ``MarginNode.view()`` and ``MarginNode.view_cdf()`` methods

Fixes
-----

* Transfer class object labels to plot axes for ``fuzzySystem.view()``, ``Distribution.view()``, and ``gaussianFunc.view()``

.. _release-0.3.0:

0.3.0
=====

:Date: October 23, 2021

Features
--------

* Add support for defining arbitrary probability densities using raw density values ``Distribution(p)``
* Add support for random sampling from instance of ``Distribution`` by calling it
* Add support for sampling from Gaussian distribution ``gaussianFunc`` by calling it directly
* Add support for viewing samples from defined distribution using the ``.view()`` method for ``Distribution`` and ``gaussianFunc`` instances
* Add support for viewing aggregate function after computing using ``.view()`` method for ``fuzzySystem`` after using ``.compute()`` method

Incompatible changes
--------------------

* Must manually reset ``fuzzySystem`` instance after ``.compute()`` to clear aggregate function

Fixes
-----

* Fixed problem with ``fuzzySystem.output_activation``` not being calculated properly using element-wise operations
* Add ``PDF_examples.py`` script
* Improve existing tests ``test_fuzzyInference_N``
* Add new tests ``test_gaussian_pdf_rvs`` and ``test_arbitrary_pdf_rvs``
* Update documentation ``conf.py`` to include class docstring from ``__init__``

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
