"""
Desin Margins Library
---------------------

A library for computing design margins using 
fuzzy logic and probabilistic methods for 
uncertainty modelling

"""

__version__ = '0.2.0'
__all__=["triangularFunc", "fuzzySet", "fuzzyRule", 
    "fuzzySystem", "gaussianFunc", "probFunction",
    "Design"]

from .fuzzyLib import triangularFunc, fuzzySet, fuzzyRule, fuzzySystem
from .uncertaintyLib import gaussianFunc, probFunction
from .DOELib import Design