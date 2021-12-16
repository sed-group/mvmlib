"""
Design Margins Library
----------------------

A library for computing design margins using 
fuzzy logic and probabilistic methods for 
uncertainty modelling

"""

__version__ = '0.4.2'
__all__=['triangularFunc', 'fuzzySet', 'fuzzyRule','fuzzySystem', 'gaussianFunc', 'Distribution'
    'Design', 'MarginNode','InputSpec', 'FixedParam', 'DesignParam', 'Behaviour']

from .fuzzyLib import triangularFunc, fuzzySet, fuzzyRule, fuzzySystem
from .uncertaintyLib import gaussianFunc, Distribution
from .DOELib import Design
from .designMarginsLib import InputSpec, FixedParam, DesignParam, Behaviour, MarginNode, MarginNetwork