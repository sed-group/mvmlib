import pytest
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import List, Tuple

from dmLib import Design, gaussianFunc, Distribution, MarginNode

# Input set 1
@pytest.fixture
def stochastic_inputs() -> Tuple[gaussianFunc,gaussianFunc]:
    # Target threshold
    mu = np.array([4.0,])
    Sigma = np.array([[0.3**2,],])
    threshold = gaussianFunc(mu, Sigma, 'T1')

    # Behaviour and capability
    mu = np.array([4.6,])
    Sigma = np.array([[0.3**2,],])
    behaviour = gaussianFunc(mu, Sigma, 'B1')

    return threshold, behaviour

@pytest.fixture
def deterministic_inputs():
    # Target threshold
    threshold = 4.0

    # Behaviour and capability
    behaviour = 5.8

    return threshold, behaviour

@pytest.fixture
def requirement():
    # Requirement specifications
    mu = np.array([370,580])
    Sigma = np.array([
        [50, 25],
        [75, 100],
        ])
    requirement = gaussianFunc(mu, Sigma, 'R1')

    return requirement

def test_deterministic(deterministic_inputs):
    """
    Tests the MarginNode excess calculation method for deterministic threshold and behaviour
    """

    # # DEBUG:
    # # Target threshold
    # threshold = 4.0

    # # Behaviour and capability
    # behaviour = 5.8

    # example_mean_std = (threshold, behaviour)

    ######################################################
    # Defining a MarginNode object

    threshold, behaviour = deterministic_inputs

    class ThermalNode(MarginNode):

        def behaviour(self,T,D):
            # Compute for given inputs
            return behaviour

        def threshold(self,T,D):
            # some specific model-dependent behaviour
            return threshold

    ######################################################
    # Check excess calculation for one sample
    ThermalNode_1 = ThermalNode('EM1')
    ThermalNode_1(np.ones(1),None)
    assert ThermalNode_1.excess == np.array([behaviour-threshold])

    ######################################################
    # Check return for multiple inputs
    ThermalNode_1.reset()
    ThermalNode_1(np.ones(10),None)
    assert (ThermalNode_1.excess == np.ones(10) * (behaviour-threshold)).all()

def test_stochastic(stochastic_inputs):
    """
    Tests the MarginNode excess calculation method for stochastic threshold and behaviour
    """

    # DEBUG:
    # Target threshold
    mu = np.array([4.0,])
    Sigma = np.array([[0.3**2,],])
    threshold = gaussianFunc(mu, Sigma, 'T1')

    # Behaviour and capability
    mu = np.array([4.6,])
    Sigma = np.array([[0.3**2,],])
    behaviour = gaussianFunc(mu, Sigma, 'B1')

    stochastic_inputs = (threshold, behaviour)

    ######################################################

    # Defining a MarginNode object
    threshold, behaviour = stochastic_inputs
    behaviour.samples
    class ThermalNode(MarginNode):

        def behaviour(self,T,D):
            # Compute for given inputs
            return behaviour()

        def threshold(self,T,D):
            # some specific model-dependent behaviour
            return threshold()
    
    ######################################################
    # Check excess calculation for one sample
    ThermalNode_1 = ThermalNode('EM1')
    ThermalNode_1(np.ones(1),None)
    assert ThermalNode_1.excess == behaviour.samples-threshold.samples

    ######################################################
    # Check sampling accuracy of mean and standard deviaction of excess

    ThermalNode_1.reset()
    behaviour.reset()
    threshold.reset()

    mu_excess = behaviour.mu - threshold.mu # calculate composite random variable mean
    Sigma_excess = behaviour.Sigma + (((-1)**2) * threshold.Sigma) # calculate composite random variable variance

    ThermalNode_1(np.ones(10000),None)
    
    # Check that means and variances of excess
    assert np.math.isclose(np.mean(ThermalNode_1.excess), mu_excess.squeeze(), rel_tol=1e-1)
    assert np.math.isclose(np.var(ThermalNode_1.excess), Sigma_excess.squeeze(), rel_tol=1e-1)

    ######################################################
    # Check that CDF computation is correct
    bin_centers, cdf, excess_limit, reliability = ThermalNode_1.compute_cdf(bins=500)

    test_excess_pdf = norm(loc=mu_excess,scale=np.sqrt(Sigma_excess))
    test_reliability = 1 - test_excess_pdf.cdf(0).squeeze() 
    test_excess_limit = test_excess_pdf.ppf(0.9).squeeze()
    test_excess_cdf = test_excess_pdf.cdf(bin_centers).squeeze()

    # ThermalNode_1.view_cdf(xlabel='Excess')
    # ThermalNode_1.axC.plot(bin_centers,test_excess_cdf,'--r')
    # ThermalNode_1.figC.show()

    assert np.math.isclose(reliability, test_reliability, rel_tol=1e-1)
    assert np.math.isclose(excess_limit, test_excess_limit, rel_tol=1e-1)
    assert np.allclose(cdf, test_excess_cdf, atol=1e-1)

if __name__ == "__main__":
    test_stochastic(stochastic_inputs)