.. toctree::
   :titlesonly:

*********
Changelog
*********

.. _release-0.5.8:

0.5.8
=====

:Date: April 18, 2023

Fixes
-----
* Fix the ``compute_volume`` method to avoid float object TypeError on call to ``math.factorial``

.. _release-0.5.7:

0.5.7
=====

:Date: March 04, 2023

Features
--------
* ``compute_absorption()`` method of  ``MarginNetwork`` now has an optional ``method`` argument for specifying different root finding algorithms
* Margins can now be allocated using different margin allocation strategies for each margin node using the ``allocate_margins`` method
* Added technical documentation explaining the theory of the Margin Value Method

Fixes
-----
* Removed the update of ``Distribution`` histograms ``self.value_dist = self.values`` on every ``__call__`` of scalar parameters to improve performance

Incompatible changes
--------------------

* ``forward()`` method of ``MarginNetwork`` now requires the argument ``strategy`` to be a list of strategies one for each ``Decision`` instance

.. _release-0.5.6:

0.5.6
=====

:Date: June 29, 2022

Features
--------
* The method ``allocate_margins()`` allows the user to specify a margin allocation strategy such as ``min_excess`` or ``manual``
* The ``Decision`` class now stores ``selection_values`` in order to allow the use to probe the frequency of each decision made

Fixes
-----
* Fixed case when errors occur in ``MarginNode`` objects due to ``value_dist`` property, when attempting to generate histogram from samples
* ``random()`` method of  ``Specification`` now calls the ``random()`` method of ``Distribution`` implicitly
* ``MarginNetwork`` instances now store ``Specification`` instances when using the ``save`` and ``load`` methods
* ``MarginNetwork`` instances now store ``deterioration_vector``and ``utilization_matrix``
* Fixed the ``train_surrogate``, ``train_inverse``, and ``train_performance_surrogate`` methods to cap data at 100 * dim for ``KRG`` models

Incompatible changes
--------------------

* ``forward()`` method of ``MarginNetwork`` now requires two optional arguments ``allocate_margin``, and ``strategy`` for all ``Decision`` instances
* ``__call__()`` method of ``Decision`` now requires two arguments ``allocate_margin``, and ``strategy``
* The method ``allocate_margins()`` must now be called before every ``forward()`` call of a ``MarginNetwork`` instance
* It is recommended to use ``self.input_spec`` instead of those defined globally outside ``forward()`` and ``randomize()`` implementations

.. _release-0.5.5:

0.5.5
=====

:Date: April 15, 2022

Fixes
-----
* ``Behaviour`` objects now support the ``train_inverse`` method (works only if the ``train_surrogate`` method has completed successfully) for training a surrogate model that inverts decided variables to any variable of choice (given by the ``key`` argument) and intermediate parameters (given by the ``intermediate`` attribute)
* ``Behaviour`` objects now support the ``inv_call`` which can be user defined (must define the ``intermediate`` and ``inverted`` attributes) or from the ``train_inverse`` method
* if ``use_estimate`` is False, the ``MarginNetwork`` must have equal number of decision and margin nodes associated with each other (1:1 ratio)
* The user-implemented ``forward`` method of the ``MarginNetwork`` object must include the ``outputs`` optional argument if using ``inv_call`` anywhere in the method
* Added error handling for ``train_surrogate`` function by rejecting NaN values during sampling of the blackbox

Incompatible changes
--------------------

* ``use_estimate`` argument of ``compute_impact`` method now relates to whether a surrogate is used or not for impact calculation
* ``train_surrogate`` argument of ``Behaviour`` class now needs the arguments ``n_i``, ``n_p``, ``n_dv``, and ``n_tt`` for each of the output parameters
* output of a decision node is given by the attribute ``output_value`` instead of ``decided_value``

.. _release-0.5.4:

0.5.4
=====

:Date: April 10, 2022

Fixes
-----

* Removed example scripts since they are more for research than for utility
* Modified impact on performance calculation ``compute_impact`` to cap performance parameters by their maximum and minimum values
* Fixed impact on performance calculation ``compute_impact`` to utilized target thresholds and decided values instead of excess values from surrogate
* Added visualization tests

.. _release-0.5.3:

0.5.3
=====

:Date: March 29, 2022

Fixes
-----

* removed dependency on ``py``

.. _release-0.5.2:

0.5.2
=====

:Date: March 29, 2022

Features
--------

* First public release of ``mvmlib``

.. _release-0.5.1:

0.5.1
=====

:Date: March 27, 2022

Incompatible changes
--------------------

* Rename the library to ``mvmlib`` and the main module to ``mvm``

.. _release-0.5.0:

0.5.0
=====

:Date: March 14, 2022

Incompatible changes
--------------------

* The ``init_decisions()`` must be called before the first ``forward()`` and after every ``randomize()`` or changing the input specifications to store the universe of decided values
* if the argument ``num_threads`` of ``train_performance_surrogate()``, ``compute_absorption()``, or ``compute_decisions()`` is greater then 1 then Jupyter notebook cannot be used with ur script
* The argument ``value`` of ``InputSpec`` initializer must be a float or an integer only. A ``Distribution`` object must be passed to the optional argument ``distribution``

Features
--------

* The ``Decision`` class can now support multiple decided values using the ``n_nodes`` optional argument. User must supple equal length of target thresholds
* Can pass additional arguments as ``kwargs`` to behaviour model for ``Decision`` class during the ``__call__`` method
* The method ``train_performance_surrogate()`` of the class ``MarginNetwork``now supports different surrogate models specified using the ``sm_type`` argument
* The method ``train_performance_surrogate()`` of the class ``MarginNetwork`` now supports parallel processing specified using the ``num_threads`` argument
* The method ``compute_absorption()`` of the class ``MarginNetwork`` now supports parallel processing of the decision universe (which changes every time the input specifications ``self.spec_vector`` are iterated during absorption computation) specified using the ``num_threads`` argument
* The method ``compute_decisions()`` of the class ``Decision`` now supports parallel processing specified using the ``num_threads`` argument


Fixes
-----

* More efficient ``compute_absorption()`` method by lumping threshold limit and specification limit calculations
* Fixed ``train_performance_surrogate()`` method to properly handle ordinal type variables such as ``INT``
* Fixed ``compute_absorption()`` method to properly handle input specifications that have negative nominal values
* Fixed ``compute_absorption()`` method to change nan values to zero
* Added ``__deepcopy__`` directives to all classes

.. _release-0.4.8:

0.4.8
=====

:Date: February 20, 2022

Incompatible changes
--------------------

* Added mandatory argument ``variable_type`` to ``InputSpec`` and ``DesignParam`` classes during initialization

Features
--------

* Add the ``Decision`` class for defining decision nodes and off-the-shelf components
* Add the integer and continuous type variables for ``InputSpec`` and ``DesignParam`` classes


.. _release-0.4.7:

0.4.7
=====

:Date: February 17, 2022

Incompatible changes
--------------------

* ``AbsorptionMatrix`` class is removed, instead call ``MarginNetwork.absoprtion_matrix.value`` to retrieve absorption values
* ``ImpactMatrix`` class is removed, instead call ``MarginNetwork.impact_matrix.value`` to retrieve impact values
* ``AbsorptionMatrix.deteriorations`` attribute is removed, instead call ``MarginNetwork.deterioration_vector.value`` to retrieve deterioration values
* All random sampling functions use the ``.random()`` method to draw samples. Cannot use the ``__call__`` operator anymore
* ``MarginNetwork`` method ``reset_outputs()`` not takes a single optional argument to reset by ``N`` samples
* ``dist`` method ``reset_outputs()`` not takes a single optional argument to reset by ``N`` samples
* ``Distribution`` object method ``.random()`` (previously ``__call__``) now returns a 1D ``np.ndarray`` for one dimensional pdfs
* Rename methods to comply with PEP 582 standard ``compute_mvp``, ``get_array``, ``set_func``
* Rename classes to comply with PEP 582 standard ``GuassianFunc``, ``UniformFunc``, ``TriangularFunc``, ``FuzzySet``, ``FuzzyFunc``, ``FuzzySystem``, ``FuzzyRule``

Features
--------

* Separate absorption, deterioration, and impact matrics into separate ``MarginNetwork`` attributes
* Use a Factory design parameter for defining matrix and vector caches used during stochastic simulation of ``MarginNetwork``

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
