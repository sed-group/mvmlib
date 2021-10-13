import pytest
import numpy as np

from dmLib import Design

def test_fullfactorial():
    """testing full factorial design"""

    lb = np.array([0.0, 0.0,])
    ub = np.array([6.0, 4.0,])

    n_levels = 3

    output = np.array([
        [0.0, 0.0,],
        [0.0, 2.0,],
        [0.0, 4.0,],
        [3.0, 0.0,],
        [3.0, 2.0,],
        [3.0, 4.0,],
        [6.0, 0.0,],
        [6.0, 2.0,],
        [6.0, 4.0,]
        ])

    test = Design(lb,ub,n_levels,'fullfact').unscale()
    
    check = test == output
    assert check.all()