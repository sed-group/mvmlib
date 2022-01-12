*********
Changelog
*********

.. _release-0.4.6:

0.4.6
=====

:Date: January 14, 2022

Features
--------

* Add utilization calculation as part of the ``compute_absorption`` method
* Add utilization storage to ``AbsorptionMatrix`` class
* Add ``compute_MVP`` method to ``MarginNetwork`` class to show margin value map
* Add ``nearest`` method to ``dmLib`` to allow calculation of the distance metric for the MVP

Fixes
-----

* Adapt ``train_performance_surrogate``, ``view_perf``, and ``compute_impact`` to include scaling functionality when training Kriging model
* Add input specifications samples as input to performance surrogate in ``train_performance_surrogate`` to accommodate variability input specifications

Incompatible changes
--------------------

* ``InputSpec`` now requires the argument ``universe`` upon initialization

.. _release-0.4.5:

0.4.5
=====

:Date: January 06, 2022

Features
--------

* Add distribution type ``uniformFunc`` for multivariate uniform distributions

Fixes
-----

* Fix ``MarginNode.value`` property to retrieve the last available sample after calling the ``reset(N)`` method 
* Fix ``value_dist`` property of ``Performance`` and ``MarginNode`` classes to construct histogram of samples and then initialize a ``Distribution`` class from them
* Force absorption computing to ignore 0 deteriorations by outputting a ``np.nan``
* Make absorption computation sign independent
* Add relevant tests for absorption and deterioration computation
* Simplified length calculation procedure in ``strut_design.py`` example by using analytical expression instead of ``fsolve`` in ``B1`` model

.. _release-0.4.4:

0.4.4
=====

:Date: December 20, 2021

Features
--------

* Add ability to selectively choose how to randomize the MAN by redefining the ``randomize`` method of ``MarginNetwork``
* Selectively choose when to reset the outputs of the MAN using the ``reset_outputs`` method
* Can retrieve design parameters, input specs, excess, target thresholds, decided values, and performances using the properties ``design_vector``, ``spec_vector``, ``excess_vector``, ``dv_vector``, ``tt_vector``, and ``perf_vector``, respectively
* Add output storage class for a Margin Analysis Network (MAN) ``AbsorptionMatrix`` which stores absorption and deterioration
* Add method ``compute_absorption`` to compute an observation of the change absorption capability matrix and deterioration vector
* Add ``view()``, ``view_cdf()``, ``view_det()``, ``view_det_cdf()`` methods to ``AbsorptionMatrix`` class by inheritance from ``VisualizeDist``
  
Incompatible changes
--------------------

* instances of ``InputSpec`` should be called using the ``.value`` property just like ``DesignParam`` and ``FixedParam``
* ``train_performance_surrogate`` argument ``ext_samples`` now takes training points of (``excess``, ``performance``) instead of (``decided_value``, ``performance``)

.. _release-0.4.3:

0.4.3
=====

:Date: December 18, 2021

Features
--------

* Add ``VisualizeDist`` class to ``uncertaintyLib.py`` module
* Add output storage class for a Margin Analysis Network (MAN) ``Performance``
* Add output storage class for a Margin Analysis Network (MAN) ``ImpactMatrix``
* Add method ``train_performance_surrogate`` which uses the library `SMT <https://smt.readthedocs.io/en/latest/index.html>`_ to estimate threshold performances
* Add method ``compute_impact`` to compute an observation of the Impact on Performance matrix
* Add method ``view_perf`` to ``MarginNetwork`` class to visualize 2D projections of performance surrogate models
* Add ``view()`` and ``view_cdf()`` methods to ``Performance`` and ``ImpactMatrix`` classes by inheritance from ``VisualizeDist``
* ``Design`` class can now take array_like values for argument ``nsamples`` if using ``doe_type='full_fact'``

Incompatible changes
--------------------

* move ``compute_cdf()`` method from class ``MarginNode`` to module level method in ``uncertaintyLib.py`` module
* use property ``.values`` instead of ``excess`` to retrieve observations of excess from ``MarginNode`` object
* Added dependency on `SMT <https://smt.readthedocs.io/en/latest/index.html>`_
* ``view()`` and ``view_cdf()`` methods now take optional arguments ``folder``, ``file``, ``img_format``, instead of just ``savefile``
* Argument ``type`` of ``Design`` initialization changed to ``doe_type`` to avoid overloading python object ``type``

.. _release-0.4.2:

0.4.2
=====

:Date: December 17, 2021

Features
--------

* Add building block for a Margin Analysis Network (MAN) as a class object ``InputSpec``
* Add building block for a Margin Analysis Network (MAN) as a class object ``FixedParam``
* Add building block for a Margin Analysis Network (MAN) as a class object ``DesignParam``
* Add building block for a Margin Analysis Network (MAN) as a class object ``Behaviour``
* ``Behaviour`` ``__call__`` method must be redefined by the user
* Add ``MarginNetwork`` class object that must be inherited and redefined by user
* Add ability to call ``MarginNetwork.forward()`` in a Monte Carlo setting

.. _release-0.4.1:

0.4.1
=====

:Date: December 15, 2021

Incompatible changes
--------------------

* ``MarginNode`` class object is now called using ``MarginNode(decided_value,threshold)``, where ``decided_value`` and ``threshold`` are vectors of equal length sampled from their respective functions


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
