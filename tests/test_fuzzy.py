import pytest
import numpy as np
import matplotlib.pyplot as plt

from dmLib import TriangularFunc, FuzzySet, FuzzyRule, FuzzySystem

def create_set(lb,ub):
    """" Create a fuzzy set from a given range """

    universe = np.linspace(lb, ub, 11) # grid for all variables

    # Define shape of triangular membership functions
    shape_lo = np.array([lb,                lb,             lb + (ub-lb)/4])
    shape_md = np.array([lb + (ub-lb)/4,    lb + (ub-lb)/2, ub - (ub-lb)/4])
    shape_hi = np.array([lb + 1*(ub-lb)/4,  ub,             ub])

    # Generate fuzzy membership functions
    lo = TriangularFunc(universe)
    lo.set_func(shape_lo[0], shape_lo[1], shape_lo[2])

    md = TriangularFunc(universe)
    md.set_func(shape_md[0], shape_md[1], shape_md[2])

    hi = TriangularFunc(universe)
    hi.set_func(shape_hi[0], shape_hi[1], shape_hi[2])

    fuzzy_set = FuzzySet(lo, md, hi)

    return fuzzy_set

# Input set 1
@pytest.fixture
def input_set_1():

    lb = 0.0
    ub = 10.0

    fuzzyset_in_1 = create_set(lb,ub)
    fuzzyset_in_1.set_label('Input_1')

    return fuzzyset_in_1

# Input set 2
@pytest.fixture
def input_set_2():

    lb = 0.0
    ub = 10.0

    fuzzyset_in_2 = create_set(lb,ub)
    fuzzyset_in_2.set_label('Input_2')

    return fuzzyset_in_2

# Output set
@pytest.fixture
def output_set():

    lb = 0.0
    ub = 100.0

    fuzzyset_out = create_set(lb,ub)
    fuzzyset_out.set_label('output')

    return fuzzyset_out

# 1D array for testing fuzzy inference
@pytest.fixture
def input_1D():
    return np.array([2.0, 0.5,])

# 2D array with one sample for testing fuzzy inference
@pytest.fixture
def input_2D_1():
    return np.array([[2.0, 0.5,],])

# 2D array with five samples for testing fuzzy inference
@pytest.fixture
def input_2D_5():
    return np.array([
                    [1.0, 0.5,],
                    [2.0, 1.5,],
                    [3.0, 2.5,],
                    [4.0, 3.5,],
                    [5.0, 4.5,],
                    ])


def test_traingular():
    """ test triangular fuzzy function object """

    # Generate universe variables
    lb = 0.0
    ub = 10.0

    universe = np.linspace(lb, ub, 11) # grid for all variables

    # Define shape of triangular membership functions
    shape = np.array([lb + (ub-lb)/4, lb + (ub-lb)/2, ub - (ub-lb)/4])

    # Generate fuzzy membership functions
    function = TriangularFunc(universe, 'test')
    function.set_func(shape[0], shape[1], shape[2])

    # test that a triangular function is returned
    test_array = np.array([0. , 0. , 0. , 0.2, 0.6, 1. , 0.6, 0.2, 0. , 0. , 0. ])
    assert (function.get_array() == test_array).all()

    # test that correct membership is inferred
    assert function.interp(5) == 1.0

def test_traingular_N():
    """ test triangular fuzzy function object when 
    presented with an array of inputs for interpreting 
    their membership 
    """

    # Generate universe variables
    lb = 0.0
    ub = 10.0

    universe = np.linspace(lb, ub, 11) # grid for all variables

    # Define shape of triangular membership functions
    shape = np.array([lb + (ub-lb)/4, lb + (ub-lb)/2, ub - (ub-lb)/4])

    # Generate fuzzy membership functions
    function = TriangularFunc(universe, 'test')
    function.set_func(shape[0], shape[1], shape[2])

    # test that a triangular function is returned
    test_array = np.array([0. , 0. , 0. , 0.2, 0.6, 1. , 0.6, 0.2, 0. , 0. , 0. ])
    assert (function.get_array() == test_array).all()

    # test that correct membership is inferred for array of inputs
    output = function.interp(np.array([4,5,6]))
    test = np.array([0.6, 1. , 0.6,])
    assert (output == test).all()

def test_fuzzyRule(input_set_1,input_set_2,output_set,input_1D):
    """ test implementation of a fuzzy rule """

    # # DEBUG:

    # lb = 0.0
    # ub = 10.0
    # input_set_1 = create_set(lb,ub)
    # input_set_1.setLabel('Input_1')

    # lb = 0.0
    # ub = 10.0
    # input_set_2 = create_set(lb,ub)
    # input_set_2.setLabel('Input_2')

    # lb = 0.0
    # ub = 100.0
    # output_set = create_set(lb,ub)
    # output_set.setLabel('output')

    # input_1D = np.array([2.0, 0.5,])

    ######################################################

    # Define fuzzy rules
    rule = FuzzyRule([{'fun1': input_set_1.lo, 'fun2': input_set_2.lo, 'operator': 'OR'}, ], output_set.lo, label='rule')

    input_set_1.lo.interp(input_1D[0])
    output = rule.apply(input_1D)

    # Test result by manual calculation
    rule_test = np.fmax(input_set_1.lo.interp(input_1D[0]), input_set_2.lo.interp(input_1D[1]))
    activation_test = np.fmin(rule_test, output_set.lo.get_array())  # removed entirely to 0

    assert (output == activation_test).all()

def test_fuzzyRule_N(input_set_1,input_set_2,output_set,input_2D_1,input_2D_5):
    """ test implementation of a fuzzy 
    rule for multiple inputs or array of inputs
    """

    # # DEBUG:

    # lb = 0.0
    # ub = 10.0
    # input_set_1 = create_set(lb,ub)
    # input_set_1.setLabel('Input_1')

    # lb = 0.0
    # ub = 10.0
    # input_set_2 = create_set(lb,ub)
    # input_set_2.setLabel('Input_2')

    # lb = 0.0
    # ub = 100.0
    # output_set = create_set(lb,ub)
    # output_set.setLabel('output')

    # input_2D_1 = np.array([[2.0, 0.5,],])

    # input_2D_5 = np.array([
    #     [1.0, 0.5,],
    #     [2.0, 1.5,],
    #     [3.0, 2.5,],
    #     [4.0, 3.5,],
    #     [5.0, 4.5,],
    #     ])

    ################ 1 input in a 2D array ###############

    # Define fuzzy rules
    rule = FuzzyRule([{'fun1': input_set_1.lo, 'fun2': input_set_2.lo, 'operator': 'OR'}, ], output_set.lo, label='rule')

    input_set_1.lo.interp(input_2D_1[:,0])
    output = rule.apply(input_2D_1)

    # Test result by manual calculation
    rule_test = np.fmax(input_set_1.lo.interp(input_2D_1[:,0]), input_set_2.lo.interp(input_2D_1[:,1]))
    
    # rule_test [n]

    rule_test = rule_test.reshape((len(rule_test),1))

    # rule_test [n, 1]
    
    rule_test = np.tile(rule_test,(1,len(output_set.universe))) # broadcasting

    # rule_test [n, len output_set.universe]
    
    activation_test = np.fmin(rule_test, output_set.lo.get_array())  # removed entirely to 0

    # activation_test [n, len output_set.universe]

    assert (output == activation_test.squeeze()).all()

    ################ 5 inputs in a 2D array ###############

    # Define fuzzy rules
    rule = FuzzyRule([{'fun1': input_set_1.lo, 'fun2': input_set_2.lo, 'operator': 'OR'}, ], output_set.lo, label='rule')
    test_input = np.array([
        [1.0, 0.5,],
        [2.0, 1.5,],
        [3.0, 2.5,],
        [4.0, 3.5,],
        [5.0, 4.5,],
        ])

    input_set_1.lo.interp(input_2D_5[:,0])
    output = rule.apply(input_2D_5)

    # Test result by manual calculation
    rule_test = np.fmax(input_set_1.lo.interp(input_2D_5[:,0]), input_set_2.lo.interp(input_2D_5[:,1]))
    
    # rule_test [n]

    rule_test = rule_test.reshape((len(rule_test),1))

    # rule_test [n, 1]
    
    rule_test = np.tile(rule_test,(1,len(output_set.universe))) # broadcasting

    # rule_test [n, len output_set.universe]
    
    activation_test = np.fmin(rule_test, output_set.lo.get_array())  # removed entirely to 0

    # activation_test [n, len output_set.universe]

    assert (output == activation_test.squeeze()).all()

def test_fuzzyInference(input_set_1,input_set_2,output_set,input_1D):
    """ test implementation of a fuzzy rule """

    # # DEBUG:

    # lb = 0.0
    # ub = 10.0
    # input_set_1 = create_set(lb,ub)
    # input_set_1.setLabel('Input_1')

    # lb = 0.0
    # ub = 10.0
    # input_set_2 = create_set(lb,ub)
    # input_set_2.setLabel('Input_2')

    # lb = 0.0
    # ub = 100.0
    # output_set = create_set(lb,ub)
    # output_set.setLabel('output')

    # input_1D = np.array([2.0, 0.5,])

    ######################################################

    # Define fuzzy rules
    rule_1 = FuzzyRule([{'fun1': input_set_1.lo, 'fun2': input_set_2.lo, 'operator': 'OR'}, ], output_set.lo, label='rule_1')
    rule_2 = FuzzyRule([{'fun1': input_set_1.hi, 'fun2': input_set_2.hi, 'operator': 'OR'}, ], output_set.hi, label='rule_2')

    # Define fuzzy control system
    rules = [rule_1, rule_2,]
    sim = FuzzySystem([input_set_1, input_set_2], output_set, rules)

    # Compute for given inputs
    test_input = np.array([2.0, 0.5,])
    value, aggregate, activation = sim.compute(input_1D, normalize=False)

    # Test result by manual calculation
    rule_test_1 = np.fmax(input_set_1.lo.interp(input_1D[0]), input_set_2.lo.interp(input_1D[1]))
    activation_test_1 = np.fmin(rule_test_1, output_set.lo.get_array())

    rule_test_2 = np.fmax(input_set_1.hi.interp(input_1D[0]), input_set_2.hi.interp(input_1D[1]))
    activation_test_2 = np.fmin(rule_test_2, output_set.hi.get_array())

    aggregate_test = np.fmax(activation_test_1,activation_test_2)

    # # Visualize this
    # fig, ax0 = plt.subplots(figsize=(8, 3))

    # n_0 = np.zeros_like(input_set_1.universe)

    # ax0.fill_between(output_set.universe, n_0, aggregate, facecolor='Orange', alpha=0.7)
    # ax0.plot([value, value], [0, activation], 'k', linewidth=1.5, alpha=0.9)
    # ax0.set_title('Aggregated membership and result (line)')

    assert (aggregate == aggregate_test).all()

def test_fuzzyInference_N(input_set_1,input_set_2,output_set,input_2D_1,input_2D_5):
    """ test implementation of a fuzzy rule """

    # # DEBUG:

    # lb = 0.0
    # ub = 10.0
    # input_set_1 = create_set(lb,ub)
    # input_set_1.setLabel('Input_1')

    # lb = 0.0
    # ub = 10.0
    # input_set_2 = create_set(lb,ub)
    # input_set_2.setLabel('Input_2')

    # lb = 0.0
    # ub = 100.0
    # output_set = create_set(lb,ub)
    # output_set.setLabel('output')

    # input_2D_1 = np.array([[2.0, 0.5,],])

    # input_2D_5 = np.array([
    #     [1.0, 0.5,],
    #     [2.0, 1.5,],
    #     [3.0, 2.5,],
    #     [4.0, 3.5,],
    #     [5.0, 4.5,],
    #     ])

    ######################################################

    # Define fuzzy rules
    rule_1 = FuzzyRule([{'fun1': input_set_1.lo, 'fun2': input_set_2.lo, 'operator': 'OR'}, ], output_set.lo, label='rule_1')
    rule_2 = FuzzyRule([{'fun1': input_set_1.hi, 'fun2': input_set_2.hi, 'operator': 'OR'}, ], output_set.hi, label='rule_2')

    # Define fuzzy control system
    rules = [rule_1, rule_2,]
    sim = FuzzySystem([input_set_1, input_set_2], output_set, rules)

    ################ 1 input in a 2D array ###############
    # Compute for given inputs
    value, aggregate, activation = sim.compute(input_2D_1, normalize=False)

    print(aggregate)
    assert value.ndim == 0
    assert activation.ndim == 0
    assert (aggregate.ndim == 1) & (aggregate.shape == output_set.universe.shape)

    # Test result by manual calculation
    rule_test_1 = np.fmax(input_set_1.lo.interp(input_2D_1[:,0]), input_set_2.lo.interp(input_2D_1[:,1]))
    
    # rule_test_1 [n]

    rule_test = rule_test_1.reshape((len(rule_test_1),1))

    # rule_test_1 [n, 1]
    
    rule_test_1 = np.tile(rule_test_1,(1,len(output_set.universe))) # broadcasting

    # rule_test [n, len output_set.universe]
    
    activation_test_1 = np.fmin(rule_test_1, output_set.lo.get_array())  # removed entirely to 0

    # activation_test_1 [n, len output_set.universe]

    ######################################################
    rule_test_2 = np.fmax(input_set_1.hi.interp(input_2D_1[:,0]), input_set_2.hi.interp(input_2D_1[:,1]))
    
    # rule_test_2 [n]

    rule_test_2 = rule_test_2.reshape((len(rule_test_2),1))

    # rule_test_2 [n, 1]
    
    rule_test_2 = np.tile(rule_test_2,(1,len(output_set.universe))) # broadcasting

    # rule_test_2 [n, len output_set.universe]
    
    activation_test_2 = np.fmin(rule_test_2, output_set.hi.get_array())  # removed entirely to 0

    # activation_test_2 [n, len output_set.universe]

    ######################################################
    aggregate_test = np.fmax(activation_test_1,activation_test_2)

    # aggregate_test [n, len output_set.universe]

    assert (aggregate == aggregate_test).all()
    
    ############### 5 inputs in a 2D array ###############
    # Compute for given inputs
    sim.reset() # reset storred aggregation
    value, aggregate, activation = sim.compute(input_2D_5, normalize=False)

    assert (value.ndim == 1) & (len(value) == 5)
    assert (activation.ndim == 1) & (len(activation) == 5)
    assert (aggregate.ndim == 2) & (aggregate.shape == (5,len(output_set.universe)))

    # Test result by manual calculation
    rule_test_1 = np.fmax(input_set_1.lo.interp(input_2D_5[:,0]), input_set_2.lo.interp(input_2D_5[:,1]))
    
    # rule_test_1 [n]

    rule_test_1 = rule_test_1.reshape((len(rule_test_1),1))

    # rule_test_1 [n, 1]
    
    rule_test_1 = np.tile(rule_test_1,(1,len(output_set.universe))) # broadcasting

    # rule_test [n, len output_set.universe]
    
    activation_test_1 = np.fmin(rule_test_1, output_set.lo.get_array())  # removed entirely to 0

    # activation_test_1 [n, len output_set.universe]

    ######################################################
    rule_test_2 = np.fmax(input_set_1.hi.interp(input_2D_5[:,0]), input_set_2.hi.interp(input_2D_5[:,1]))
    
    # rule_test_2 [n]

    rule_test_2 = rule_test_2.reshape((len(rule_test_1),1))

    # rule_test_2 [n, 1]
    
    rule_test_2 = np.tile(rule_test_2,(1,len(output_set.universe))) # broadcasting

    # rule_test_2 [n, len output_set.universe]
    
    activation_test_2 = np.fmin(rule_test_2, output_set.hi.get_array())  # removed entirely to 0

    # activation_test_2 [n, len output_set.universe]

    ######################################################
    aggregate_test = np.fmax(activation_test_1,activation_test_2)

    # aggregate_test [n, len output_set.universe]

    assert (aggregate == aggregate_test).all()

if __name__ == "__main__":

    # DEBUG:

    lb = 0.0
    ub = 10.0
    input_set_1 = create_set(lb,ub)
    input_set_1.set_label('Input_1')

    lb = 0.0
    ub = 10.0
    input_set_2 = create_set(lb,ub)
    input_set_2.set_label('Input_2')

    lb = 0.0
    ub = 100.0
    output_set = create_set(lb,ub)
    output_set.set_label('output')

    input_2D_1 = np.array([[2.0, 0.5,],])

    input_2D_5 = np.array([
        [1.0, 0.5,],
        [2.0, 1.5,],
        [3.0, 2.5,],
        [4.0, 3.5,],
        [5.0, 4.5,],
        ])

    test_fuzzyInference_N(input_set_1,input_set_2,output_set,input_2D_1,input_2D_5)
    