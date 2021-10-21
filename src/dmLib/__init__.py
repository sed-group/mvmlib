"""
Desin Margins Library
---------------------

A library for computing design margins using 
fuzzy logic and probabilistic methods for 
uncertainty modelling

"""

__version__ = '0.3.0'
__all__=["triangularFunc", "fuzzySet", "fuzzyRule", 
    "fuzzySystem", "gaussianFunc", "Distribution"
    "Design"]

from .fuzzyLib import triangularFunc, fuzzySet, fuzzyRule, fuzzySystem
from .uncertaintyLib import gaussianFunc, Distribution
from .DOELib import Design