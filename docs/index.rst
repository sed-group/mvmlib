.. mvmlib documentation master file, created by
   sphinx-quickstart on Fri Oct  8 18:02:05 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The margin value method library
-------------------------------

The margin value method (:term:`MVM`) library is an open-source Python package consisting of libraries for defining complex engineering systems using a margin analysis network (MAN). The library provides class objects that represent the various components of the MAN and can be interfaced together to form a network. The library provides a number of computational tools to assess the ability of the system to absorb a change in its inputs (when for example a design requirement changes) by using excess margins.

.. figure:: ./images/efm_example.png
  :scale: 80 %
  :align: center

Currently, ``mvmlib`` is designed for engineers with a python programming background to use since they would need to implement the behaviour models inline. A function-block modeling approach is being developed as a front-end for this library. The library includes a lot of modeling tools that are commonly used in engineering design such as probabilistic and fuzzy logic modeling tools to model uncertainty where it may exist.

The library also provides an intuitive and efficient way to store computations and results for later use and postprocessing as well as a number of visualization tools.

License & copyright
-------------------

© Khalil Al Handawi

Cite us
-------

To cite ``mvmlib``: A. Brahma and D. C. Wynn.
`Margin value method for engineering design improvement <https://doi.org/10.1007/s00163-020-00335-8>`_. 

.. code-block:: none

   @article{Brahma2020,
      author   = {Brahma, A. and Wynn, D. C.},
      doi      = {10.1007/s00163-020-00335-8},
      isbn     = {0123456789},
      issn     = {14356066},
      journal  = {Research in Engineering Design},
      number   = {3},
      pages    = {353--381},
      title    = {{Margin value method for engineering design improvement}},
      url      = {https://doi.org/10.1007/s00163-020-00335-8},
      volume   = {31},
      year     = {2020}}

References
----------
.. bibliography:: refs.bib
   :style: unsrt

Documentation contents
----------------------

.. Provides a brief background on the margin value method (:term:`MVM`) from an engineering design standpoint an outlines the algorithmic implementation of the method used in this library.

.. toctree::
   :caption: Technical Documentation

   background
   glossary

.. Contains several examples demonstrating the usage of the library. The examples range from simple analytical toy problems to more complex engineering-inspired ones.

.. toctree::
   :titlesonly:
   :maxdepth: 2
   :caption: Demo Documentation

   getting_started
   examples

.. Outlines the library's application programming interface (:term:`API`) and history of revisions.

.. toctree::
   :maxdepth: 2
   :caption: Code Documentation

   api
   history

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`