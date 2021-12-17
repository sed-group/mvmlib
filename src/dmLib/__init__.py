"""
Design Margins Library
----------------------

A library for computing design margins using 
fuzzy logic and probabilistic methods for 
uncertainty modelling

"""

__version__ = '0.4.3'
__all__=['triangularFunc', 'fuzzySet', 'fuzzyRule','fuzzySystem', 'gaussianFunc', 'Distribution',
    'VisualizeDist', 'compute_cdf', 'Design', 'MarginNode','InputSpec', 'FixedParam', 'DesignParam', 'Behaviour', 
    'Performance', 'ImpactMatrix', 'MarginNetwork']

from .fuzzyLib import triangularFunc, fuzzySet, fuzzyRule, fuzzySystem
from .uncertaintyLib import gaussianFunc, Distribution, VisualizeDist, compute_cdf
from .DOELib import Design
from .designMarginsLib import InputSpec, FixedParam, DesignParam, Behaviour, MarginNode, Performance, ImpactMatrix, MarginNetwork