import pytest
import numpy as np
import os
from scipy.stats import norm
import matplotlib
from typing import List, Tuple
from scipy.optimize import fsolve, minimize

matplotlib.use('Agg')

from mvm import Design, GaussianFunc, UniformFunc, MarginNode, Performance, MarginNetwork, \
    InputSpec, DesignParam, Behaviour, Decision, compute_cdf, nearest
from mvm.utilities import check_folder

# Input set 1
@pytest.fixture
def stochastic_inputs() -> Tuple[GaussianFunc, GaussianFunc]:
    # Target threshold
    mu = np.array([4.0,])
    Sigma = np.array([[0.3**2,],])
    threshold = GaussianFunc(mu, Sigma, 'T1')

    # decided value (capability)
    mu = np.array([4.6,])
    Sigma = np.array([[0.3**2,],])
    decided_value = GaussianFunc(mu, Sigma, 'B1')

    return threshold, decided_value

@pytest.fixture
def deterministic_inputs() -> Tuple[float,float]:
    # Target threshold
    threshold = 4.0

    # decided value (capability)
    decided_value = 5.8

    return threshold, decided_value

@pytest.fixture
def Impact_test_inputs() -> Tuple[np.ndarray,np.ndarray]:
    # test decided values and target thresholds for checking impact matrix calculation
    tt_vector = np.array([0.5,0.5,0.25])
    dv_vector = np.array([0.25,0.25,0.75])

    return tt_vector,dv_vector

@pytest.fixture
def design_parameters() -> List[DesignParam]:
    # test decided values for checking absorption matrix calculation
    d1 = DesignParam(4.0, 'D1', universe=[1.0, 4.0], variable_type='FLOAT')
    d2 = DesignParam(2.0, 'D2', universe=[0.5, 2.0], variable_type='FLOAT')
    d3 = DesignParam(1.5, 'D3', universe=[0.25, 1.5], variable_type='FLOAT')
    design_params = [d1, d2, d3]

    return design_params

@pytest.fixture
def deterministic_specs() -> Tuple[np.ndarray,List[InputSpec]]:
    # Define input specs
    centers = np.array([1.2,1.0])
    s1 = InputSpec(centers[0],'S1',universe=(0.0,2.0),variable_type='FLOAT',symbol='T1',inc = -1e-0,inc_type='rel')
    s2 = InputSpec(centers[1],'S2',universe=(0.0,2.0),variable_type='FLOAT',symbol='T2',inc = -1e-0,inc_type='rel')
    input_specs = [s1,s2]

    return centers,input_specs

@pytest.fixture
def stochastic_specs() -> Tuple[UniformFunc, np.ndarray, np.ndarray, List[InputSpec]]:
    # Define input specs
    centers = np.array([1.2,1.0])
    ranges = np.ones(2) * 0.1
    dist = UniformFunc(centers, ranges)
    s1 = InputSpec(centers[0],'S1',universe=(0.0,2.0),variable_type='FLOAT',symbol='T1',inc = -1e-0,inc_type='rel',distribution=dist,cov_index=0)
    s2 = InputSpec(centers[1],'S2',universe=(0.0,2.0),variable_type='FLOAT',symbol='T2',inc = -1e-0,inc_type='rel',distribution=dist,cov_index=1)
    input_specs = [s1,s2]

    return dist,centers,ranges,input_specs

@pytest.fixture
def man_components() -> Tuple[List[Behaviour],List[Performance],List[MarginNode]]:
    
    # Define behaviour models
    class B1(Behaviour):
        def __call__(self,s1,s2):
            # Analytical behaviour models in terms of input specs
            tt1_model = lambda spec : (spec[0]**2) + (2*spec[0]) + (spec[1])
            tt2_model = lambda spec : spec[0] + 2*spec[1]
            tt3_model = lambda spec : spec[0] + spec[1]

            self.threshold = [tt1_model((s1,s2)),tt2_model((s1,s2)),tt3_model((s1,s2))]

            return self.threshold

    class B2(Behaviour):

        def __init__(self, key: str = ''):
            super().__init__(key=key)
            self.p1_model = lambda dv : (dv[0]**2) + (dv[1]**2) + (2*dv[2]**2)
            self.p2_model = lambda dv : dv[0] + 2*dv[1] + dv[2]

        def __call__(self,dv1,dv2,dv3):
            # Analytical test performance models in terms of decided values
            self.performance = [self.p1_model((dv1,dv2,dv3)),self.p2_model((dv1,dv2,dv3))]

            return self.performance


    class B3(Behaviour):

        def __call__(self,value):
            # Translate decision variable to decided value
            if value == '0':
                self.decided_value = 5.0
            elif value == '1':
                self.decided_value = 4.0
            elif value == '2':
                self.decided_value = 3.0
            elif value == '3':
                self.decided_value = 2.0
            elif value == '4':
                self.decided_value = 1.0

            return self.decided_value


    class B4(Behaviour):

        def __call__(self,value):
            # Translate decision variable to decided value
            if value == '0':
                self.decided_value = 4.0
            elif value == '1':
                self.decided_value = 2.0
            elif value == '2':
                self.decided_value = 1.0
            elif value == '3':
                self.decided_value = 0.5

            return self.decided_value


    class B5(Behaviour):

        def __call__(self,value):
            # Translate decision variable to decided value
            if value == '1':
                self.decided_value = [4.0,2.0]
            elif value == '2':
                self.decided_value = [3.0,3.0]
            elif value == '3':
                self.decided_value = [2.0,0.5]
            elif value == '4':
                self.decided_value = [1.0,0.25]

            return self.decided_value


    b1 = B1('B1')
    b2 = B2('B2')
    b3 = B3('B3')
    b4 = B4('B4')
    b5 = B5('B5')
    behaviours = [b1,b2,b3,b4,b5]

    # Define decision nodes
    decision_1 = Decision(universe=['0','1', '2', '3', '4'], variable_type='ENUM', key='decision_1',
                          direction='must_exceed', decided_value_model=b3)
    decision_2 = Decision(universe=['0','1', '2', '3'], variable_type='ENUM', key='decision_2',
                          direction='must_exceed', decided_value_model=b4)
    decision_3 = Decision(universe=['1', '2', '3', '4'], variable_type='ENUM', key='decision_2',
                          direction=['must_exceed','must_exceed'], decided_value_model=b5, n_nodes=2)
    decisions = [decision_1, decision_2, decision_3]

    # Define performances
    p1 = Performance('P1', direction='less_is_better')
    p2 = Performance('P2', direction='less_is_better')
    performances = [p1,p2]

    # Define margin nodes
    e1 = MarginNode('E1', direction='must_exceed')
    e2 = MarginNode('E2', direction='must_exceed')
    e3 = MarginNode('E3', direction='must_exceed')
    margin_nodes = [e1,e2,e3]

    return behaviours, decisions, performances, margin_nodes

@pytest.fixture
def deterministic_man(design_parameters: List[DesignParam],
                      man_components: Tuple[List[Behaviour],List[Performance],List[MarginNode]],
                      deterministic_specs: Tuple[np.ndarray,List[InputSpec]]) -> MarginNetwork:
    ######################################################
    # Construct MAN
    behaviours, decisions, performances, margin_nodes = man_components
    centers,input_specs = deterministic_specs

    # Define the MAN
    class MAN(MarginNetwork):
        def randomize(self):
            pass
        
        def forward(self,**kwargs):
            
            # retrieve design parameters
            d1 = self.design_params[0]
            d2 = self.design_params[1]
            d3 = self.design_params[2]

            # retrieve input specs
            s1 = self.input_specs[0]
            s2 = self.input_specs[1]

            # get behaviour
            b1 = self.behaviours[0]
            b2 = self.behaviours[1]

            # retrieve margin nodes
            e1 = self.margin_nodes[0]
            e2 = self.margin_nodes[1]
            e3 = self.margin_nodes[2]

            # get performance
            p1 = self.performances[0]
            p2 = self.performances[1]

            # Execute behaviour models
            b1(s1.value,s2.value)
            b2(d1.value,d2.value,d3.value)

            # Compute excesses
            e1(b1.threshold[0],d1.value)
            e2(b1.threshold[1],d2.value)
            e3(b1.threshold[2],d3.value)

            # Compute performances
            p1(b2.performance[0])
            p2(b2.performance[1])

    man = MAN(design_parameters,input_specs,[],behaviours,[],margin_nodes,performances,'MAN_test')

    return man

@pytest.fixture
def stochastic_man(design_parameters: List[DesignParam],
                   man_components: Tuple[List[Behaviour],List[Performance],List[MarginNode]],
                   stochastic_specs: Tuple[np.ndarray,List[InputSpec]]) -> MarginNetwork:
    ######################################################
    # Construct MAN
    behaviours, decisions, performances, margin_nodes = man_components
    dist,centers,ranges,input_specs = stochastic_specs

    s1,s2 = input_specs

    class MAN(MarginNetwork):
        def randomize(self):
            dist.random()
            s1.random()
            s2.random()

        def forward(self,**kwargs):

            # retrieve design parameters
            d1 = self.design_params[0]
            d2 = self.design_params[1]
            d3 = self.design_params[2]

            # retrieve input specs
            s1 = self.input_specs[0]
            s2 = self.input_specs[1]

            # get behaviour
            b1 = self.behaviours[0]
            b2 = self.behaviours[1]

            # retrieve margin nodes
            e1 = self.margin_nodes[0]
            e2 = self.margin_nodes[1]
            e3 = self.margin_nodes[2]

            # get performance
            p1 = self.performances[0]
            p2 = self.performances[1]

            # Execute behaviour models
            b1(s1.value,s2.value)
            b2(d1.value,d2.value,d3.value)

            # Compute excesses
            e1(b1.threshold[0],d1.value)
            e2(b1.threshold[1],d2.value)
            e3(b1.threshold[2],d3.value)

            # Compute performances
            p1(b2.performance[0])
            p2(b2.performance[1])

    man = MAN(design_parameters,input_specs,[],behaviours,[],margin_nodes,performances,'MAN_test')

    return man

def decision_man(design_parameters: List[DesignParam],
                 man_components: Tuple[List[Behaviour],List[Performance],List[MarginNode]],
                 deterministic_specs: Tuple[np.ndarray,List[InputSpec]],
                 tt_factor: List[float]=3*[1.0,]) -> MarginNetwork:
    ######################################################
    # Construct MAN
    behaviours, decisions, performances, margin_nodes = man_components
    centers,input_specs = deterministic_specs

    s1,s2 = input_specs

    class MAN(MarginNetwork):
        def randomize(self):
            pass

        def forward(self,recalculate_decisions=False,override_decisions=False,**kwargs):
            
            # retrieve design parameters
            d1 = self.design_params[0]
            d2 = self.design_params[1]
            d3 = self.design_params[2]

            # retrieve input specs
            s1 = self.input_specs[0]
            s2 = self.input_specs[1]

            # get behaviour
            b1 = self.behaviours[0]
            b2 = self.behaviours[1]
            b3 = self.behaviours[2]
            b4 = self.behaviours[3]

            # get decision nodes
            decision_1 = self.decisions[0]
            decision_2 = self.decisions[1]

            # retrieve margin nodes
            e1 = self.margin_nodes[0]
            e2 = self.margin_nodes[1]
            e3 = self.margin_nodes[2]

            # get performance
            p1 = self.performances[0]
            p2 = self.performances[1]

            # Execute behaviour models
            b1(s1.value,s2.value)

            # Execute decision node and translation model
            tt_vector = [f*t for f,t in zip(tt_factor,b1.threshold)]
            decision_1(tt_vector[0], override_decisions, recalculate_decisions)
            b3(decision_1.selection_value)

            decision_2(tt_vector[1], override_decisions, recalculate_decisions)
            b4(decision_2.selection_value)

            # Compute excesses
            e1(tt_vector[0],decision_1.decided_value)
            e2(tt_vector[1],decision_2.decided_value)
            e3(tt_vector[2],d3.value)

            # Compute performances
            b2(decision_1.decided_value,decision_2.decided_value,d3.value)
            p1(b2.performance[0])
            p2(b2.performance[1])

    man = MAN(design_parameters,input_specs,[],behaviours,decisions,margin_nodes,performances,'MAN_test')

    return man

def multidecision_man(design_parameters: List[DesignParam],
                      man_components: Tuple[List[Behaviour],List[Performance],List[MarginNode]],
                      deterministic_specs: Tuple[np.ndarray,List[InputSpec]],
                      tt_factor: List[float]=3*[1.0,]) -> MarginNetwork:
    ######################################################
    # Construct MAN
    behaviours, decisions, performances, margin_nodes = man_components
    centers,input_specs = deterministic_specs

    s1,s2 = input_specs

    class MAN(MarginNetwork):
        def randomize(self):
            pass

        def forward(self,recalculate_decisions=False,override_decisions=False,**kwargs):

            # retrieve design parameters
            d1 = self.design_params[0]
            d2 = self.design_params[1]
            d3 = self.design_params[2]

            # retrieve input specs
            s1 = self.input_specs[0]
            s2 = self.input_specs[1]

            # get behaviour
            b1 = self.behaviours[0]
            b2 = self.behaviours[1]
            b3 = self.behaviours[2]
            b4 = self.behaviours[3]
            b5 = self.behaviours[4]

            # get decision nodes
            decision_1 = self.decisions[0]

            # retrieve margin nodes
            e1 = self.margin_nodes[0]
            e2 = self.margin_nodes[1]
            e3 = self.margin_nodes[2]

            # get performance
            p1 = self.performances[0]
            p2 = self.performances[1]

            # Execute behaviour models
            b1(s1.value,s2.value)

            # Execute decision node and translation model
            tt_vector = [f*t for f,t in zip(tt_factor,b1.threshold)]
            decision_1(tt_vector[:2], override_decisions, recalculate_decisions)
            b5(decision_1.selection_value)

            # Compute excesses
            e1(tt_vector[0],decision_1.decided_value[0])
            e2(tt_vector[1],decision_1.decided_value[1])
            e3(tt_vector[2],d3.value)

            # Compute performances
            b2(decision_1.decided_value[0],decision_1.decided_value[1],d3.value)
            p1(b2.performance[0])
            p2(b2.performance[1])

    man = MAN(design_parameters,input_specs,[],behaviours,[decisions[2],],margin_nodes,performances,'MAN_test')

    return man

@pytest.fixture
def noise() -> GaussianFunc:
    # Gaussian noise for adding stochasticity
    return GaussianFunc(0.0, 0.00125)

def test_deterministic_MarginNode(deterministic_inputs:Tuple[float,float]):
    """
    Tests the MarginNode excess calculation method for deterministic threshold and behaviour
    """

    ######################################################
    # Defining a MarginNode object

    threshold, decided_value = deterministic_inputs

    ######################################################
    # Check excess calculation for one sample
    ThermalNode = MarginNode('EM1')
    ThermalNode(decided_value,threshold)
    assert ThermalNode.excess.values == np.array([decided_value-threshold])

    ######################################################
    # Check return for multiple inputs
    ThermalNode.reset()
    ThermalNode(np.ones(10)*decided_value,np.ones(10)*threshold)
    assert np.allclose(ThermalNode.excess.values, np.ones(10) * (decided_value-threshold), atol=1e-3)

@pytest.mark.dependency()
def test_stochastic_MarginNode(stochastic_inputs:Tuple[GaussianFunc, GaussianFunc]):
    """
    Tests the MarginNode excess calculation method for stochastic threshold and behaviour
    """

    ######################################################

    # Defining a MarginNode object
    threshold, decided_value = stochastic_inputs
    decided_value.samples
    
    ######################################################
    # Check excess calculation for one sample
    ThermalNode = MarginNode('EM1')
    ThermalNode(decided_value.random(),threshold.random())
    assert all(ThermalNode.excess.values == decided_value.samples-threshold.samples)

    ######################################################
    # Check sampling accuracy of mean and standard deviaction of excess

    ThermalNode.reset()
    decided_value.reset()
    threshold.reset()

    mu_excess = decided_value.mu - threshold.mu # calculate composite random variable mean
    Sigma_excess = decided_value.Sigma + (((-1)**2) * threshold.Sigma) # calculate composite random variable variance

    ThermalNode(decided_value.random(10000),threshold.random(10000))
    
    # Check that means and variances of excess
    assert np.math.isclose(np.mean(ThermalNode.excess.values), mu_excess.squeeze(), rel_tol=1e-1)
    assert np.math.isclose(np.var(ThermalNode.excess.values), Sigma_excess.squeeze(), rel_tol=1e-1)

    ######################################################
    # Check that CDF computation is correct
    ThermalNode(decided_value.random(10000),threshold.random(10000))
    bin_centers, cdf, excess_limit, reliability = compute_cdf(ThermalNode.excess.values,bins=500,
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

    test_dir = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(test_dir,'..','test_dumps')
    check_folder(folder)

    ThermalNode.save(os.path.join(folder,'test_margin_node'))

@pytest.mark.dependency()
@pytest.mark.parametrize(
    'deviation', [0.0, 0.5],
)
def test_deterministic_ImpactMatrix(man_components:Tuple[List[Behaviour],List[Performance],List[MarginNode]],
    Impact_test_inputs:Tuple[np.ndarray,np.ndarray],deviation:float):
    """
    Tests the ImpactMatrix calculation method for deterministic threshold and decided values
    """

    ######################################################
    # Construct MAN

    behaviours, decisions, performances, margin_nodes = man_components
    tt_vector, dv_vector = Impact_test_inputs

    # Define the MAN
    class MAN(MarginNetwork):
        def randomize(self):
            pass

        def forward(self):

            # retrieve MAN components
            e1 = self.margin_nodes[0]
            e2 = self.margin_nodes[1]
            e3 = self.margin_nodes[2]

            # get performances
            p1 = self.performances[0]
            p2 = self.performances[1]

            # get behaviour
            b1 = self.behaviours[0]
            b2 = self.behaviours[1]

            # Execute behaviour models
            b2(dv_vector[0]+deviation,dv_vector[1]+deviation,dv_vector[2]+deviation)

            # Compute excesses
            e1(tt_vector[0],dv_vector[0]+deviation)
            e2(tt_vector[1],dv_vector[1]+deviation)
            e3(tt_vector[2],dv_vector[2]+deviation)

            # Compute performances
            p1(b2.performance[0])
            p2(b2.performance[1])

    man = MAN([],[],[],behaviours,[],margin_nodes,performances,'MAN_test')

    ######################################################
    # Create training data and train response surface
    n_samples = 1000
    doe_node = Design(-np.ones(len(margin_nodes)),np.ones(len(margin_nodes)),n_samples,'LHS').unscale()

    p_space = np.empty((n_samples,len(performances)))
    p_space[:,0] = np.apply_along_axis(man.behaviours[1].p1_model,axis=1,arr=doe_node+dv_vector) # mat + vec is automatically broadcasted
    p_space[:,1] = np.apply_along_axis(man.behaviours[1].p2_model,axis=1,arr=doe_node+dv_vector) # mat + vec is automatically broadcasted

    man.train_performance_surrogate(n_samples=100,sm_type='KRG',ext_samples=(doe_node+dv_vector,p_space))
    man.forward()
    man.view_perf(e_indices=[0,1],p_index=0)
    man.compute_impact(use_estimate=True)

    # Compute test impact
    input = np.tile(dv_vector+deviation,(len(margin_nodes),1))

    p = np.empty((len(margin_nodes),len(performances)))
    p[:,0] = np.apply_along_axis(man.behaviours[1].p1_model,axis=1,arr=input)
    p[:,1] = np.apply_along_axis(man.behaviours[1].p2_model,axis=1,arr=input)

    np.fill_diagonal(input,tt_vector)

    p_t = np.empty((len(margin_nodes),len(performances)))
    p_t[:,0] = np.apply_along_axis(man.behaviours[1].p1_model,axis=1,arr=input)
    p_t[:,1] = np.apply_along_axis(man.behaviours[1].p2_model,axis=1,arr=input)

    test_impact = (p - p_t) / p_t

    # test_impact = np.array([
    #     [ 0.42857143,  0.16666667],
    #     [ 0.42857143,  0.4       ],
    #     [-0.61538462, -0.22222222]
    #     ])

    assert np.allclose(man.impact_matrix.value, test_impact, rtol=1e-1)

@pytest.mark.dependency()
def test_stochastic_ImpactMatrix(man_components:Tuple[List[Behaviour],List[Performance],List[MarginNode]],
                                 Impact_test_inputs:Tuple[np.ndarray,np.ndarray], noise:GaussianFunc):
    """
    Tests the ImpactMatrix calculation method for stochastic threshold and decided values
    """

    ######################################################
    # Construct MAN

    behaviours, decisions, performances, margin_nodes = man_components
    tt_vector, dv_vector = Impact_test_inputs

    # Define the MAN
    class MAN(MarginNetwork):
        def randomize(self):
            pass

        def forward(self):

            # retrieve MAN components
            e1 = self.margin_nodes[0]
            e2 = self.margin_nodes[1]
            e3 = self.margin_nodes[2]

            # get performances
            p1 = self.performances[0]
            p2 = self.performances[1]

            # get behaviour
            b1 = self.behaviours[0]
            b2 = self.behaviours[1]

            # Execute behaviour models
            b2(dv_vector[0],dv_vector[1],dv_vector[2])

            # Compute excesses
            e1(tt_vector[0]+noise.random(),dv_vector[0]+noise.random())
            e2(tt_vector[1]+noise.random(),dv_vector[1]+noise.random())
            e3(tt_vector[2]+noise.random(),dv_vector[2]+noise.random())

            # Compute performances
            p1(b2.performance[0])
            p2(b2.performance[1])

    man = MAN([],[],[],behaviours,[],margin_nodes,performances,'MAN_test')

    ######################################################
    # Create training data and train response surface
    n_samples = 100
    doe_node = Design(-np.ones(len(margin_nodes)),np.ones(len(margin_nodes)),n_samples,'LHS').unscale()

    p_space = np.empty((n_samples,len(performances)))
    p_space[:,0] = np.apply_along_axis(man.behaviours[1].p1_model,axis=1,arr=doe_node+dv_vector) # mat + vec is automatically broadcasted
    p_space[:,1] = np.apply_along_axis(man.behaviours[1].p2_model,axis=1,arr=doe_node+dv_vector) # mat + vec is automatically broadcasted

    man.train_performance_surrogate(n_samples=100,ext_samples=(doe_node+dv_vector,p_space))

    n_runs = 1000
    for n in range(n_runs):
        man.forward()
        man.compute_impact(use_estimate=True)

    mean_impact = np.mean(man.impact_matrix.values,axis=2)

    # Compute test impact
    input = np.tile(dv_vector,(len(margin_nodes),1))

    p = np.empty((len(margin_nodes),len(performances)))
    p[:,0] = np.apply_along_axis(man.behaviours[1].p1_model,axis=1,arr=input)
    p[:,1] = np.apply_along_axis(man.behaviours[1].p2_model,axis=1,arr=input)

    np.fill_diagonal(input,tt_vector)

    p_t = np.empty((len(margin_nodes),len(performances)))
    p_t[:,0] = np.apply_along_axis(man.behaviours[1].p1_model,axis=1,arr=input)
    p_t[:,1] = np.apply_along_axis(man.behaviours[1].p2_model,axis=1,arr=input)

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
        assert len(node.excess.values) == n_runs
        assert len(node.decided_value.values) == n_runs
        assert len(node.target.values) == n_runs

    man.reset(5)

    for performance in man.performances:
        assert len(performance.values) == n_runs - 5

    for node in man.margin_nodes:
        assert len(node.excess.values) == n_runs - 5
        assert len(node.decided_value.values) == n_runs - 5
        assert len(node.target.values) == n_runs - 5

def test_deterministic_Absorption(deterministic_man: MarginNetwork,
    man_components: Tuple[List[Behaviour],List[Performance],List[MarginNode]],
    deterministic_specs: Tuple[np.ndarray,List[InputSpec]],
    design_parameters: List[DesignParam]):
    """
    Tests the Absorption calculation method for deterministic specifications
    """

    ######################################################
    # Construct MAN
    behaviours, decisions, performances, margin_nodes = man_components
    centers,input_specs = deterministic_specs
    man = deterministic_man

    d1 = design_parameters[0]
    d2 = design_parameters[1]
    d3 = design_parameters[2]
    design_parameters_vector = [d.value for d in design_parameters]
    dv_vector = design_parameters_vector
    ######################################################
    # Create training data and train response surface
    b1, b2, _, _, _ = behaviours

    man.forward()
    man.compute_absorption()

    mean_absorption = np.mean(man.absorption_matrix.values,axis=2)
    mean_utilization = np.mean(man.utilization_matrix.values,axis=2)

    # Check outputs

    s1_limit = np.array([
        -1 + np.sqrt(1-centers[1]+dv_vector[0]),
        (dv_vector[1]-2*centers[1]),
        (dv_vector[2]-centers[1]),
    ])
    
    s2_limit = np.array([
        dv_vector[0]-(centers[0]**2)-(2*centers[0]),
        (dv_vector[1]-centers[0]) / 2,
        dv_vector[2]-centers[0],
    ])

    s1_limit = np.max(s1_limit)
    s2_limit = np.max(s2_limit)
    spec_limit = np.array([s1_limit,s2_limit])

    # deterioration matrix
    signs = np.array([-1,-1])
    nominal_specs = np.array([centers[0],centers[1]])
    deterioration = signs*(spec_limit - nominal_specs) / nominal_specs
    deterioration_matrix = np.tile(deterioration,(len(margin_nodes),1))

    #deterioration_matrix = [len(margin_nodes), len(input_specs)]

    # threshold matrix
    nominal_tt = np.array(b1(centers[0],centers[1]))
    nominal_tt = np.reshape(nominal_tt,(len(margin_nodes),-1))
    target_thresholds = np.tile(nominal_tt,(1,len(input_specs)))

    #target_thresholds = [len(margin_nodes), len(input_specs)]

    # Compute performances at the spec limit for each margin node
    new_tt_1 = np.array(b1(s1_limit,centers[1]))
    new_tt_2 = np.array(b1(centers[0],s2_limit))

    new_tt_1 = np.reshape(new_tt_1,(len(margin_nodes),-1))
    new_tt_2 = np.reshape(new_tt_2,(len(margin_nodes),-1))
    new_thresholds = np.empty((len(margin_nodes),0))
    new_thresholds = np.hstack((new_thresholds,new_tt_1))
    new_thresholds = np.hstack((new_thresholds,new_tt_2))

    #new_thresholds = [len(margin_nodes), len(input_specs)]

    test_absorption = abs(new_thresholds - target_thresholds) / (target_thresholds * deterioration_matrix)
    # test_absorption = np.array([
    #     [ 1.04132231  , 0.20661157],
    #     [ 0.375       , 0.625     ],
    #     [0.54545455   , 0.45454545]
    #     ])

    #test_absorption = [len(margin_nodes), len(input_specs)]

    # compute utilization
    decided_value = np.reshape(dv_vector,(len(margin_nodes),-1))
    decided_values = np.tile(decided_value,(1,len(input_specs)))

    #decided_values = [len(margin_nodes), len(input_specs)]

    test_utilization = 1 - ((new_thresholds - decided_values) / (target_thresholds - decided_values))

    #test_utilization = [len(margin_nodes), len(input_specs)]

    assert np.allclose(mean_absorption, test_absorption, rtol=1e-1)
    assert np.allclose(mean_utilization, test_utilization, rtol=1e-1)

@pytest.mark.dependency()
def test_stochastic_Absorption(stochastic_man: MarginNetwork, 
                               man_components: Tuple[List[Behaviour],List[Performance],List[MarginNode]],
                               stochastic_specs: Tuple[np.ndarray,List[InputSpec]],
                               design_parameters: List[DesignParam]):
    """
    Tests the Absorption calculation method for stochastic specifications
    """

    ######################################################
    # Construct MAN
    behaviours, decisions, performances, margin_nodes = man_components
    dist,centers,ranges,input_specs = stochastic_specs
    man = stochastic_man

    d1 = design_parameters[0]
    d2 = design_parameters[1]
    d3 = design_parameters[2]
    design_parameters_vector = [d.value for d in design_parameters]

    ######################################################
    # Create training data and train response surface

    b1, b2, _, _, _ = behaviours

    n_runs = 100
    for n in range(n_runs):
        man.randomize()
        man.forward()
        man.compute_absorption()

    mean_absorption = np.mean(man.absorption_matrix.values,axis=2)
    mean_utilization= np.mean(man.utilization_matrix.values,axis=2)

    # Check outputs
    
    s1_limit = np.array([
        -1 + np.sqrt(1-centers[1]+d1.value),
        (d2.value-2*centers[1]),
        (d3.value-centers[1]),
    ])
    
    s2_limit = np.array([
        d1.value-(centers[0]**2)-(2*centers[0]),
        (d2.value-centers[0]) / 2,
        d3.value-centers[0],
    ])

    s1_limit = np.max(s1_limit)
    s2_limit = np.max(s2_limit)
    spec_limit = np.array([s1_limit,s2_limit])

    # deterioration matrix
    signs = np.array([-1,-1])
    nominal_specs = np.array([centers[0],centers[1]])
    deterioration = signs*(spec_limit - nominal_specs) / nominal_specs
    deterioration_matrix = np.tile(deterioration,(len(margin_nodes),1))

    #deterioration_matrix = [len(margin_nodes), len(input_specs)]

    # threshold matrix
    nominal_tt = np.array(b1(centers[0],centers[1]))
    nominal_tt = np.reshape(nominal_tt,(len(margin_nodes),-1))
    target_thresholds = np.tile(nominal_tt,(1,len(input_specs)))

    #target_thresholds = [len(margin_nodes), len(input_specs)]

    # Compute performances at the spec limit for each margin node
    new_tt_1 = np.array(b1(s1_limit,centers[1]))
    new_tt_2 = np.array(b1(centers[0],s2_limit))

    new_tt_1 = np.reshape(new_tt_1,(len(margin_nodes),-1))
    new_tt_2 = np.reshape(new_tt_2,(len(margin_nodes),-1))
    new_thresholds = np.empty((len(margin_nodes),0))
    new_thresholds = np.hstack((new_thresholds,new_tt_1))
    new_thresholds = np.hstack((new_thresholds,new_tt_2))

    #new_thresholds = [len(margin_nodes), len(input_specs)]

    test_absorption =abs(new_thresholds - target_thresholds) / (target_thresholds * deterioration_matrix)
    # test_absorption = np.array([
    #     [ 1.04132231  , 0.20661157],
    #     [ 0.375       , 0.625     ],
    #     [0.54545455   , 0.45454545]
    #     ])

    #test_absorption = [len(margin_nodes), len(input_specs)]

    # compute utilization
    decided_value = np.reshape(design_parameters_vector,(len(margin_nodes),-1))
    decided_values = np.tile(decided_value,(1,len(input_specs)))

    #decided_values = [len(margin_nodes), len(input_specs)]

    test_utilization = 1 - ((new_thresholds - decided_values) / (target_thresholds - decided_values))

    #test_utilization = [len(margin_nodes), len(input_specs)]

    assert np.allclose(mean_absorption, test_absorption, rtol=0.2)
    assert np.allclose(mean_utilization, test_utilization, rtol=0.2)

    test_dir = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(test_dir,'..','test_dumps')

    man.save('absorption_man',folder=folder)

@pytest.mark.parametrize(
    'test_dict', 
    [
        {
            'true_decisions' : ['1','1'],
            'tt_factor' : [1.0,1.0,1.0],
            'tt_vector' : [4.84,3.2,2.2],
        },
        {
            'true_decisions' : ['3','2'],
            'tt_factor' : [0.5,0.5,1.0],
            'tt_vector' : [2.42,1.6,1.1],
        },
        {
            'true_decisions' : ['4','3'],
            'tt_factor' : [0.25,0.25,1.0],
            'tt_vector' : ['1','1'],
        },
    ]
)
def test_decision(man_components: Tuple[List[Behaviour],List[Performance],List[MarginNode]],
                  deterministic_specs: Tuple[np.ndarray,List[InputSpec]],
                  design_parameters: List[DesignParam], test_dict):
    """
    Tests the Decision node capability of the MAN
    """

    ######################################################
    # Construct MAN
    behaviours, decisions, performances, margin_nodes = man_components
    centers,input_specs = deterministic_specs

    d1 = design_parameters[0]
    d2 = design_parameters[1]
    d3 = design_parameters[2]
    design_parameters_vector = [d.value for d in design_parameters]
    dv_vector = np.empty(3)
    dv_vector[2] = d3.value
    ######################################################
    # Loop over different decision possibilities
    b1, b2, b3, b4, b5 = behaviours

    # Check outputs
    man = decision_man(design_parameters,man_components,deterministic_specs,test_dict['tt_factor'])
    man.init_decisions()
    man.forward()
    man.compute_absorption()

    mean_absorption = np.mean(man.absorption_matrix.values,axis=2)
    mean_utilization = np.mean(man.utilization_matrix.values,axis=2)

    ######################################################
    # Calculate the test values
    # check that correct decisions are made
    assert all([d == d_t for d,d_t in zip(man.decision_vector, test_dict['true_decisions'])])

    dv_vector[0] = b3(decisions[0].selection_value)
    dv_vector[1] = b4(decisions[1].selection_value)

    s1_limit = np.array([
        -1 + np.sqrt(1-centers[1]+(dv_vector[0]/test_dict['tt_factor'][0])),
        ((dv_vector[1]/test_dict['tt_factor'][1])-2*centers[1]),
        ((dv_vector[2]/test_dict['tt_factor'][2])-centers[1]),
    ])
    
    s2_limit = np.array([
        (dv_vector[0]/test_dict['tt_factor'][0])-(centers[0]**2)-(2*centers[0]),
        ((dv_vector[1]/test_dict['tt_factor'][1])-centers[0]) / 2,
        (dv_vector[2]/test_dict['tt_factor'][2])-centers[0],
    ])

    s1_limit = np.max(s1_limit)
    s2_limit = np.max(s2_limit)
    spec_limit = np.array([s1_limit,s2_limit])

    # deterioration matrix
    signs = np.array([-1,-1])
    nominal_specs = np.array([centers[0],centers[1]])
    deterioration = signs*(spec_limit - nominal_specs) / nominal_specs
    deterioration_matrix = np.tile(deterioration,(len(margin_nodes),1))

    #deterioration_matrix = [len(margin_nodes), len(input_specs)]

    # threshold matrix
    nominal_tt = np.array(b1(centers[0],centers[1])) * np.array(test_dict['tt_factor'])
    nominal_tt = np.reshape(nominal_tt,(len(margin_nodes),-1))
    target_thresholds = np.tile(nominal_tt,(1,len(input_specs)))

    #target_thresholds = [len(margin_nodes), len(input_specs)]

    # Compute performances at the spec limit for each margin node
    new_tt_1 = np.array(b1(s1_limit,centers[1])) * np.array(test_dict['tt_factor'])
    new_tt_2 = np.array(b1(centers[0],s2_limit)) * np.array(test_dict['tt_factor'])

    new_tt_1 = np.reshape(new_tt_1,(len(margin_nodes),-1))
    new_tt_2 = np.reshape(new_tt_2,(len(margin_nodes),-1))
    new_thresholds = np.empty((len(margin_nodes),0))
    new_thresholds = np.hstack((new_thresholds,new_tt_1))
    new_thresholds = np.hstack((new_thresholds,new_tt_2))

    #new_thresholds = [len(margin_nodes), len(input_specs)]

    test_absorption = abs(new_thresholds - target_thresholds) / (target_thresholds * deterioration_matrix)

    #test_absorption = [len(margin_nodes), len(input_specs)]

    # compute utilization
    decided_value = np.reshape(dv_vector,(len(margin_nodes),-1))
    decided_values = np.tile(decided_value,(1,len(input_specs)))

    #decided_values = [len(margin_nodes), len(input_specs)]

    test_utilization = 1 - ((new_thresholds - decided_values) / (target_thresholds - decided_values))

    #test_utilization = [len(margin_nodes), len(input_specs)]

    assert np.allclose(mean_absorption, test_absorption, rtol=1e-1)
    assert np.allclose(mean_utilization, test_utilization, rtol=1e-1)

@pytest.mark.parametrize(
    'test_dict', 
    [
        {
            'true_decisions' : ['2',],
            'tt_factor' : [1.0,1.0,1.0],
            'tt_vector' : [4.84,3.2,2.2],
        },
        {
            'true_decisions' : ['3',],
            'tt_factor' : [0.5,0.5,1.0],
            'tt_vector' : [2.42,1.6,1.1],
        },
        {
            'true_decisions' : ['4',],
            'tt_factor' : [0.25,0.25,1.0],
            'tt_vector' : ['1','1'],
        },
    ]
)
def test_decision_multinode(man_components: Tuple[List[Behaviour],List[Performance],List[MarginNode]],
                            deterministic_specs: Tuple[np.ndarray,List[InputSpec]],
                            design_parameters: List[DesignParam], test_dict):
    """
    Tests the Decision node capability of the MAN
    """

    ######################################################
    # Construct MAN
    behaviours, decisions, performances, margin_nodes = man_components
    centers,input_specs = deterministic_specs

    d1 = design_parameters[0]
    d2 = design_parameters[1]
    d3 = design_parameters[2]
    design_parameters_vector = [d.value for d in design_parameters]
    dv_vector = np.empty(3)
    dv_vector[2] = d3.value

    ######################################################
    # Loop over different decision possibilities
    b1, b2, b3, b4, b5 = behaviours
    
    # Check outputs
    man = multidecision_man(design_parameters,man_components,deterministic_specs,test_dict['tt_factor'])
    man.init_decisions()
    man.forward(override_decisions=False)
    man.compute_absorption()

    mean_absorption = np.mean(man.absorption_matrix.values,axis=2)
    mean_utilization = np.mean(man.utilization_matrix.values,axis=2)

    ######################################################
    # Calculate the test values
    # check that correct decisions are made
    assert all([d == d_t for d,d_t in zip(man.decision_vector, test_dict['true_decisions'])])

    dv_vector[:2] = b5(decisions[2].selection_value)

    s1_limit = np.array([
        -1 + np.sqrt(1-centers[1]+(dv_vector[0]/test_dict['tt_factor'][0])),
        ((dv_vector[1]/test_dict['tt_factor'][1])-2*centers[1]),
        ((dv_vector[2]/test_dict['tt_factor'][2])-centers[1]),
    ])
    
    s2_limit = np.array([
        (dv_vector[0]/test_dict['tt_factor'][0])-(centers[0]**2)-(2*centers[0]),
        ((dv_vector[1]/test_dict['tt_factor'][1])-centers[0]) / 2,
        (dv_vector[2]/test_dict['tt_factor'][2])-centers[0],
    ])

    s1_limit = np.max(s1_limit)
    s2_limit = np.max(s2_limit)
    spec_limit = np.array([s1_limit,s2_limit])

    # deterioration matrix
    signs = np.array([-1,-1])
    nominal_specs = np.array([centers[0],centers[1]])
    deterioration = signs*(spec_limit - nominal_specs) / nominal_specs
    deterioration_matrix = np.tile(deterioration,(len(margin_nodes),1))

    #deterioration_matrix = [len(margin_nodes), len(input_specs)]

    # threshold matrix
    nominal_tt = np.array(b1(centers[0],centers[1])) * np.array(test_dict['tt_factor'])
    nominal_tt = np.reshape(nominal_tt,(len(margin_nodes),-1))
    target_thresholds = np.tile(nominal_tt,(1,len(input_specs)))

    #target_thresholds = [len(margin_nodes), len(input_specs)]

    # Compute performances at the spec limit for each margin node
    new_tt_1 = np.array(b1(s1_limit,centers[1])) * np.array(test_dict['tt_factor'])
    new_tt_2 = np.array(b1(centers[0],s2_limit)) * np.array(test_dict['tt_factor'])

    new_tt_1 = np.reshape(new_tt_1,(len(margin_nodes),-1))
    new_tt_2 = np.reshape(new_tt_2,(len(margin_nodes),-1))
    new_thresholds = np.empty((len(margin_nodes),0))
    new_thresholds = np.hstack((new_thresholds,new_tt_1))
    new_thresholds = np.hstack((new_thresholds,new_tt_2))

    #new_thresholds = [len(margin_nodes), len(input_specs)]

    test_absorption = abs(new_thresholds - target_thresholds) / (target_thresholds * deterioration_matrix)

    #test_absorption = [len(margin_nodes), len(input_specs)]

    # compute utilization
    decided_value = np.reshape(dv_vector,(len(margin_nodes),-1))
    decided_values = np.tile(decided_value,(1,len(input_specs)))

    #decided_values = [len(margin_nodes), len(input_specs)]

    test_utilization = 1 - ((new_thresholds - decided_values) / (target_thresholds - decided_values))

    #test_utilization = [len(margin_nodes), len(input_specs)]

    assert np.allclose(mean_absorption, test_absorption, rtol=1e-1)
    assert np.allclose(mean_utilization, test_utilization, rtol=1e-1)

@pytest.mark.dependency()
def test_mixed_variables(man_components: Tuple[List[Behaviour],List[Performance],List[MarginNode]],
                         deterministic_specs: Tuple[np.ndarray,List[InputSpec]],
                         design_parameters: List[DesignParam]):

    """
    Tests the train performance surrogate functionality when decision nodes are present
    """
    n_threads = 1
    ######################################################
    # Construct MAN
    behaviours, decisions, performances, margin_nodes = man_components
    centers,input_specs = deterministic_specs

    d1 = design_parameters[0]
    d2 = design_parameters[1]
    d3 = design_parameters[2]
    design_parameters_vector = [d.value for d in design_parameters]
    dv_vector = np.empty(3)
    dv_vector[2] = d3.value
    ######################################################
    # Loop over different decision possibilities
    b1, b2, b3, b4, b5 = behaviours

    # Check outputs

    test_dict = {
            'true_decisions' : ['1','1'],
            'tt_factor' : [1.0,1.0,1.0],
            'tt_vector' : [4.84,3.2,2.2],
        }

    man = decision_man(design_parameters,man_components,deterministic_specs,test_dict['tt_factor'])

    # Create surrogate model for estimating threshold performance
    man.train_performance_surrogate(n_samples=500, sm_type='KRG', sampling_freq=1, num_threads=n_threads, bandwidth=[1e-3, ])
    man.init_decisions(num_threads=n_threads)
    man.forward()
    man.compute_impact()

    ######################################################
    # Calculate the test values
    # check that correct decisions are made

    input = np.tile(man.dv_vector,(len(margin_nodes),1))

    p = np.empty((len(margin_nodes),len(performances)))
    p[:,0] = np.apply_along_axis(man.behaviours[1].p1_model,axis=1,arr=input)
    p[:,1] = np.apply_along_axis(man.behaviours[1].p2_model,axis=1,arr=input)

    np.fill_diagonal(input,man.tt_vector)

    p_t = np.empty((len(margin_nodes),len(performances)))
    p_t[:,0] = np.apply_along_axis(man.behaviours[1].p1_model,axis=1,arr=input)
    p_t[:,1] = np.apply_along_axis(man.behaviours[1].p2_model,axis=1,arr=input)

    test_impact = (p - p_t) / p_t

    assert np.allclose(man.impact_matrix.value, test_impact, rtol=1e-1)

    test_dir = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(test_dir,'..','test_dumps')

    man.save('impact_man',folder=folder)

@pytest.mark.parametrize('n_threads', [1,])
def test_behaviour(n_threads):

    def fun(x,value):
        if value == '1':
            p = -0.1 * x
        elif value == '2':
            p = 0.5 * x
        elif value == '3':
            p = 50

        return p

    # Define behaviour models
    class B1(Behaviour):
        def __call__(self,x,value):
            p = fun(x,value)
            return p
    b = B1()

    # Train surrogate
    variable_dict = {
        'x' : {'type' : 'FLOAT', 'limits' : [-100,100]},
        'value' : {'type' : 'ENUM', 'limits' : ['1','2','3']},
    }
    b.train_surrogate(variable_dict,n_outputs=1,n_samples=100,num_threads=n_threads)

    # test outputs
    assert fun(1.0,'1') == b(1.0,'1')
    assert fun(1.0,'2') == b(1.0,'2')
    assert fun(1.0,'3') == b(1.0,'3')

def test_nearest():
    """
    Check calculation to the nearest point along a line to an arbitrary point
    """

    p1 = np.array([1,1])
    p2 = np.array([5,2])
    s = np.array([3,3])
    ######################################################
    
    pn, d = nearest(p1,p2,s)
    
    ######################################################
    
    r_t = lambda t : p1 + t*(p2-p1) # parameteric equation of the line
    f = lambda  t: np.dot(s-r_t(t),p2-p1)

    t_sol=fsolve(f,0.5)

    pn_test = r_t(t_sol)
    d_test = np.linalg.norm(pn_test - s)

    assert all(pn == pn_test)
    assert np.isclose(d, d_test, rtol=1e-1)

@pytest.mark.dependency(depends=["test_stochastic_MarginNode"])
def test_margin_visualization(stochastic_inputs:Tuple[GaussianFunc, GaussianFunc]):

    # Defining a MarginNode object
    threshold, decided_value = stochastic_inputs
    ThermalNode = MarginNode('EM1')

    test_dir = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(test_dir,'..','test_dumps')

    ThermalNode.load(os.path.join(folder,'test_margin_node'))

    mu_excess = decided_value.mu - threshold.mu # calculate composite random variable mean
    Sigma_excess = decided_value.Sigma + (((-1)**2) * threshold.Sigma) # calculate composite random variable variance
    ######################################################
    # Check CDF visualization
    ThermalNode.excess.view_cdf(xlabel='Excess')
    ######################################################
    # Check visualization
    ThermalNode.excess.view()

@pytest.mark.dependency(depends=["test_deterministic_ImpactMatrix", "test_stochastic_ImpactMatrix", "test_mixed_variables"])
def test_perf_visualization(man_components: Tuple[List[Behaviour],List[Performance],List[MarginNode]],
                            deterministic_specs: Tuple[np.ndarray,List[InputSpec]],
                            design_parameters: List[DesignParam]):

    test_dict = {
            'true_decisions' : ['1','1'],
            'tt_factor' : [1.0,1.0,1.0],
            'tt_vector' : [4.84,3.2,2.2],
        }
    man = decision_man(design_parameters,man_components,deterministic_specs,test_dict['tt_factor'])

    test_dir = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(test_dir,'..','test_dumps')

    man.load('impact_man',folder=folder)
    man.init_decisions()
    man.forward()
    ######################################################
    # Check visualization
    man.view_perf(e_indices=[0,1],p_index=0)
    man.impact_matrix.view(2,1)

@pytest.mark.dependency(depends=["test_stochastic_Absorption"])
def test_absorption_visualization(stochastic_man: MarginNetwork):
    man = stochastic_man
    test_dir = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(test_dir,'..','test_dumps')

    man.load('absorption_man',folder=folder)
    man.init_decisions()
    man.forward()
    ######################################################
    # Check visualization
    man.deterioration_vector.view(0)
    man.deterioration_vector.view(1)

    man.absorption_matrix.view(0,0)
    man.absorption_matrix.view(1,0)
    man.absorption_matrix.view(2,0)

    man.absorption_matrix.view(0,1)
    man.absorption_matrix.view(1,1)
    man.absorption_matrix.view(2,1)