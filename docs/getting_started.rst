.. _Getting started:

Getting started
===============

Create a python virtual environment
-----------------------------------

Create a virtual environment for your project

**MacOS/Linux**

::

    $ python -m venv .venv
    $ source .venv/bin/activate


**Windows**

::

    python -m venv .env
    .env\Scripts\activate


Install prerequisites
---------------------

``mvmlib`` requires ``numpy`` make sure you have it installed

::

	$ pip install numpy

Then install ``mvmlib`` using the command 

::

	$ pip install mvmlib

- This will install the required dependencies:
    + ``matplotlib`` for visualization tools.
    + ``pyDOE`` for design space exploration.
    + ``scikit-fuzzy`` for fuzzy logic modeling .
    + ``smt`` for surrogate modeling of expensive models.
    + ``multiprocess`` for parallel processing of certain MVM computations.


Testing the installation
------------------------

Start a python console and try the following command

::

    >>> from mvm import *

This will make sure that all modules can be loaded properly and no dependencies are missing.


Notebooks
---------

Several notebooks are available to get up to speed with SMT:

* `Quick tutorial on probability density functions <https://github.com/sed-group/mvmlib/blob/master/docs/notebooks/PDF_examples.ipynb>`_
* `An example that involves fuzzy-logic modeling <https://github.com/sed-group/mvmlib/blob/master/docs/notebooks/TRS_example.ipynb>`_
* `A full-blown engineering example <https://github.com/sed-group/mvmlib/blob/master/docs/notebooks/strut_example.ipynb>`_

Uninstalling mvmlib
-------------------

Simply use the following command

::
    
    $ pip uninstall mvmlib
