"""
Design Margins Library
----------------------

A library for computing design margins using 
fuzzy logic and probabilistic methods for 
uncertainty modelling

"""

__version__ = '0.5.0'
__all__ = ['TriangularFunc', 'FuzzySet', 'FuzzyRule', 'FuzzySystem', 'GaussianFunc', 'UniformFunc', 'Distribution',
           'VisualizeDist', 'compute_cdf', 'Design', 'MarginNode', 'InputSpec', 'FixedParam', 'DesignParam',
           'Behaviour', 'MatrixParam', 'ScalarParam', 'VectorParam', 'Performance', 'MarginNetwork', 'Decision', 'nearest']

from .fuzzyLib import TriangularFunc, FuzzySet, FuzzyRule, FuzzySystem
from .uncertaintyLib import GaussianFunc, UniformFunc, Distribution, VisualizeDist, compute_cdf
from .DOELib import Design
from .designMarginsLib import Cache, InputSpec, FixedParam, DesignParam, Behaviour, MarginNode, MatrixParam, \
    Performance, MarginNetwork, ScalarParam, VectorParam, Decision, nearest
