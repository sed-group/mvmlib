import pytest
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import List, Tuple

from dmLib import Design, gaussianFunc, Distribution, MarginNode, Performance, MarginNetwork, compute_cdf

# Input set 1
@pytest.fixture
def stochastic_inputs() -> Tuple[gaussianFunc,gaussianFunc]:
    # Target threshold
    mu = np.array([4.0,])
    Sigma = np.array([[0.3**2,],])
    threshold = gaussianFunc(mu, Sigma, 'T1')

    # decided value (capability)
    mu = np.array([4.6,])
    Sigma = np.array([[0.3**2,],])
    decided_value = gaussianFunc(mu, Sigma, 'B1')

    return threshold, decided_value

@pytest.fixture
def deterministic_inputs():
    # Target threshold
    threshold = 4.0

    # decided value (capability)
    decided_value = 5.8

    return threshold, decided_value

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

@pytest.fixture
def perf_model():
    # Analytical test performance models in terms of decided values
    p1_model = lambda dv : (dv[0]**2) + (dv[1]**2) + (2*dv[2]**2)
    p2_model = lambda dv : dv[0] + 2*dv[1] + dv[2]

    return p1_model,p2_model

@pytest.fixture
def impact_calc_inputs():
    # test decided values and target thresholds for checking impact matrix calculation
    dv_vector = np.array([0.5,0.5,0.25])
    tt_vector = np.array([0.25,0.25,0.75])

    return dv_vector,tt_vector

@pytest.fixture
def noise():
    # Gaussian noise for adding stochasticity
    return gaussianFunc(0.0,0.00125)

def test_deterministic_MarginNode(deterministic_inputs):
    """
    Tests the MarginNode excess calculation method for deterministic threshold and behaviour
    """

    # # DEBUG:
    # # Target threshold
    # threshold = 4.0

    # # decided value (capability)
    # decided_value = 5.8

    # example_mean_std = (threshold, decided_value)

    ######################################################
    # Defining a MarginNode object

    threshold, decided_value = deterministic_inputs

    ######################################################
    # Check excess calculation for one sample
    ThermalNode = MarginNode('EM1')
    ThermalNode(decided_value,threshold)
    assert ThermalNode.values == np.array([decided_value-threshold])

    ######################################################
    # Check return for multiple inputs
    ThermalNode.reset()
    ThermalNode(np.ones(10)*decided_value,np.ones(10)*threshold)
    assert (ThermalNode.values == np.ones(10) * (decided_value-threshold)).all()

def test_stochastic_MarginNode(stochastic_inputs):
    """
    Tests the MarginNode excess calculation method for stochastic threshold and behaviour
    """

    # # DEBUG:
    # # Target threshold
    # mu = np.array([4.0,])
    # Sigma = np.array([[0.3**2,],])
    # threshold = gaussianFunc(mu, Sigma, 'T1')

    # # decided value (capability)
    # mu = np.array([4.6,])
    # Sigma = np.array([[0.3**2,],])
    # decided_value = gaussianFunc(mu, Sigma, 'B1')

    # stochastic_inputs = (threshold, decided_value)

    ######################################################

    # Defining a MarginNode object
    threshold, decided_value = stochastic_inputs
    decided_value.samples
    
    ######################################################
    # Check excess calculation for one sample
    ThermalNode = MarginNode('EM1')
    ThermalNode(decided_value(),threshold())
    assert ThermalNode.values == decided_value.samples-threshold.samples

    ######################################################
    # Check sampling accuracy of mean and standard deviaction of excess

    ThermalNode.reset()
    decided_value.reset()
    threshold.reset()

    mu_excess = decided_value.mu - threshold.mu # calculate composite random variable mean
    Sigma_excess = decided_value.Sigma + (((-1)**2) * threshold.Sigma) # calculate composite random variable variance

    ThermalNode(decided_value(10000),threshold(10000))
    
    # Check that means and variances of excess
    assert np.math.isclose(np.mean(ThermalNode.values), mu_excess.squeeze(), rel_tol=1e-1)
    assert np.math.isclose(np.var(ThermalNode.values), Sigma_excess.squeeze(), rel_tol=1e-1)

    ######################################################
    # Check that CDF computation is correct
    ThermalNode(decided_value(10000),threshold(10000))
    bin_centers, cdf, excess_limit, reliability = compute_cdf(ThermalNode.values,bins=500,
        cutoff=ThermalNode.cutoff,buffer_limit=ThermalNode.buffer_limit)

    test_excess_pdf = norm(loc=mu_excess,scale=np.sqrt(Sigma_excess))
    test_reliability = 1 - test_excess_pdf.cdf(0).squeeze() 
    test_excess_limit = test_excess_pdf.ppf(0.9).squeeze()
    test_excess_cdf = test_excess_pdf.cdf(bin_centers).squeeze()

    # ThermalNode.view_cdf(xlabel='Excess')
    # ThermalNode.axC.plot(bin_centers,test_excess_cdf,'--r')
    # ThermalNode.figC.show()

    assert np.math.isclose(reliability, test_reliability, rel_tol=1e-1)
    assert np.math.isclose(excess_limit, test_excess_limit, rel_tol=1e-1)
    assert np.allclose(cdf, test_excess_cdf, atol=1e-1)

def test_deterministic_ImpactMatrix(perf_model,impact_calc_inputs):
    """
    Tests the ImpactMatrix calculation method for deterministic threshold and decided values
    """

    # # DEBUG:
    # # Analytical test performance models in terms of decided values
    # p1_model = lambda dv : (dv[0]**2) + (dv[1]**2) + (2*dv[2]**2)
    # p2_model = lambda dv : dv[0] + 2*dv[1] + dv[2]

    # perf_model = (p1_model, p2_model)

    # # test decided values and target thresholds for checking impact matrix calculation
    # dv_vector = np.array([0.5,0.5,0.25])
    # tt_vector = np.array([0.25,0.25,0.75])

    # impact_calc_inputs = (dv_vector, tt_vector)

    ######################################################
    # Construct MAN

    p1_model, p2_model      = perf_model
    dv_vector, tt_vector    = impact_calc_inputs

    # Define margin nodes
    e1 = MarginNode('E1',type='must_exceed')
    e2 = MarginNode('E2',type='must_exceed')
    e3 = MarginNode('E3',type='must_exceed')
    margin_nodes = [e1,e2,e3]

    # Define performances
    p1 = Performance('P1')
    p2 = Performance('P1')
    performances = [p1,p2]

    # Define the MAN
    class MAN(MarginNetwork):
        def forward(self):

            # retrieve MAN components
            e1 = self.margin_nodes[0]
            e2 = self.margin_nodes[1]
            e3 = self.margin_nodes[2]

            p1 = self.performances[0]

            # Compute excesses
            e1(tt_vector[0],dv_vector[0])
            e2(tt_vector[1],dv_vector[1])
            e3(tt_vector[2],dv_vector[2])

            # Compute performances
            p1(p1_model(tt_vector))
            p2(p2_model(tt_vector))

    man = MAN([],[],[],[],margin_nodes,performances,'MAN_test')

    ######################################################
    # Create training data and train response surface
    n_samples = 100
    excess_space = Design(np.zeros(len(margin_nodes)),np.ones(len(margin_nodes)),n_samples,'LHS').unscale()

    p_space = np.empty((n_samples,len(performances)))
    p_space[:,0] = np.apply_along_axis(p1_model,axis=1,arr=excess_space+dv_vector) # mat + vec is automatically broadcasted
    p_space[:,1] = np.apply_along_axis(p2_model,axis=1,arr=excess_space+dv_vector) # mat + vec is automatically broadcasted

    man.train_performance_surrogate(n_samples=100,ext_samples=(excess_space,p_space))
    man.forward()
    man.compute_impact()

    # Check outputs
    input = np.tile(tt_vector,(len(margin_nodes),1))

    p = np.empty((len(margin_nodes),len(performances)))
    p[:,0] = np.apply_along_axis(p1_model,axis=1,arr=input)
    p[:,1] = np.apply_along_axis(p2_model,axis=1,arr=input)

    np.fill_diagonal(input,dv_vector)

    p_t = np.empty((len(margin_nodes),len(performances)))
    p_t[:,0] = np.apply_along_axis(p1_model,axis=1,arr=input)
    p_t[:,1] = np.apply_along_axis(p2_model,axis=1,arr=input)

    test_impact = (p - p_t) / p_t

    # test_impact = np.array([
    #     [ 0.42857143,  0.16666667],
    #     [ 0.42857143,  0.4       ],
    #     [-0.61538462, -0.22222222]
    #     ])

    assert np.allclose(man.impact_matrix.impact, test_impact, rtol=1e-3)

def test_stochastic_ImpactMatrix(perf_model,impact_calc_inputs,noise):
    """
    Tests the ImpactMatrix calculation method for stochastic threshold and decided values
    """

    # # DEBUG:
    # # Analytical test performance models in terms of decided values
    # p1_model = lambda dv : (dv[0]**2) + (dv[1]**2) + (2*dv[2]**2)
    # p2_model = lambda dv : dv[0] + 2*dv[1] + dv[2]

    # perf_model = (p1_model, p2_model)

    # # test decided values and target thresholds for checking impact matrix calculation
    # dv_vector = np.array([0.5,0.5,0.25])
    # tt_vector = np.array([0.25,0.25,0.75])

    # impact_calc_inputs = (dv_vector, tt_vector)

    # noise = gaussianFunc(0.0,0.00125)

    ######################################################
    # Construct MAN

    p1_model, p2_model      = perf_model
    dv_vector, tt_vector    = impact_calc_inputs

    # Define margin nodes
    e1 = MarginNode('E1',type='must_exceed')
    e2 = MarginNode('E2',type='must_exceed')
    e3 = MarginNode('E3',type='must_exceed')
    margin_nodes = [e1,e2,e3]

    # Define performances
    p1 = Performance('P1')
    p2 = Performance('P1')
    performances = [p1,p2]

    # Define the MAN
    class MAN(MarginNetwork):
        def forward(self):
            
            # retrieve MAN components
            e1 = self.margin_nodes[0]
            e2 = self.margin_nodes[1]
            e3 = self.margin_nodes[2]

            p1 = self.performances[0]

            # Compute excesses
            e1(tt_vector[0]+noise(),dv_vector[0]+noise())
            e2(tt_vector[1]+noise(),dv_vector[1]+noise())
            e3(tt_vector[2]+noise(),dv_vector[2]+noise())

            # Compute performances
            p1(p1_model(tt_vector))
            p2(p2_model(tt_vector))

    man = MAN([],[],[],[],margin_nodes,performances,'MAN_test')

    ######################################################
    # Create training data and train response surface
    n_samples = 100
    excess_space = Design(-np.ones(len(margin_nodes)),np.ones(len(margin_nodes)),n_samples,'LHS').unscale()

    p_space = np.empty((n_samples,len(performances)))
    p_space[:,0] = np.apply_along_axis(p1_model,axis=1,arr=excess_space+dv_vector) # mat + vec is automatically broadcasted
    p_space[:,1] = np.apply_along_axis(p2_model,axis=1,arr=excess_space+dv_vector) # mat + vec is automatically broadcasted

    man.train_performance_surrogate(n_samples=100,ext_samples=(excess_space,p_space))

    n_runs = 1000
    for n in range(n_runs):
        man.forward()
        man.compute_impact()

    mean_impact = np.mean(man.impact_matrix.impacts,axis=2)

    # Check outputs
    input = np.tile(tt_vector,(len(margin_nodes),1))

    p = np.empty((len(margin_nodes),len(performances)))
    p[:,0] = np.apply_along_axis(p1_model,axis=1,arr=input)
    p[:,1] = np.apply_along_axis(p2_model,axis=1,arr=input)

    np.fill_diagonal(input,dv_vector)

    p_t = np.empty((len(margin_nodes),len(performances)))
    p_t[:,0] = np.apply_along_axis(p1_model,axis=1,arr=input)
    p_t[:,1] = np.apply_along_axis(p2_model,axis=1,arr=input)

    test_impact = (p - p_t) / p_t

    # test_impact = np.array([
    #     [ 0.42857143,  0.16666667],
    #     [ 0.42857143,  0.4       ],
    #     [-0.61538462, -0.22222222]
    #     ])

    assert np.allclose(mean_impact, test_impact, rtol=1e-1)

    ######################################################
    # Check reset functionality

    for performance in man.performances:
        assert len(performance.values) == n_runs

    for node in man.margin_nodes:
        assert len(node.values) == n_runs

    man.reset(5)

    for performance in man.performances:
        assert len(performance.values) == n_runs - 5

    for node in man.margin_nodes:
        assert len(node.values) == n_runs - 5

    ######################################################
    # Check visualization
    # man.view_perf(e_indices=[1,2],p_index=1)
    # man.impact_matrix.view(2,1)