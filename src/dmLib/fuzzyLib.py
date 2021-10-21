import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, AnyStr, List

"""
Fuzzy Library for computing aggregate membership functions for fuzzy variables
"""

class fuzzyFunction():

    def __init__(self,universe,label=''):
        """
        Contrains description and implementation of
        different fuzzy membership functions
        
        Parameters
        ----------
        universe : np.1darray
            1d array of length n
            n=number of samples
        label : str, optional
            string to tag instance with
        """

        self.universe = universe
        self.label = label

class triangularFunc(fuzzyFunction):

    def __init__(self, universe,label=''):
        super().__init__(universe,label)

    def setFunc(self,low,medium,high):
        """
        Create an instance of the triangular membership function

        Parameters
        ----------
        low : int
            left foot of triangle
        medium : int
            peak of triangle
        high : int
            right foot of triangle
        """

        self.low = low; self.medium = medium; self.high = high

    def setLabel(self,label):
        """
        Changes label property of function

        Parameters
        ----------
        label : float
            new label name
        """

        self.label=label

    def getArray(self):
        """
        Sample an array of values from membership function
        
        Returns
        -------
        array : np.1darray
            1d array of length universe
        """

        self.array = fuzz.trimf(self.universe, [self.low, self.medium, self.high])
        return self.array

    def interp(self,input=None):
        """
        Interpret membership of input for fuzzy function

        Parameters
        ----------
        input : float OR np.1darray
            value(s) at which membership is to be interpreted

        Returns
        -------
        level : float OR np.1darray
            interpreted membership level(s) of the input(s)
        """

        if input is not None:
            level = fuzz.interp_membership(self.universe, self.getArray(), input)
        else:
            level = fuzz.interp_membership(self.universe, self.getArray(), self.universe[0])

        return level
    
class fuzzyRule():

    def __init__(
        self,
        input_statements : Dict[str, fuzzyFunction],
        output : fuzzyFunction,
        label=''
        ):
        """
        Defines a fuzzy rules by connecting fuzzy inputs
        to outputs

        Parameters
        ----------
        input_statements : list 
            list of dicts, structure of each dict
            {
            
            'fun1': fuzzyFunction object

            'fun2': fuzzyFunction object

            operator: 'AND', 'OR'

            }
        label : str, optional
            string to tag instance with
        """
        
        self.input_statements = input_statements
        self.output = output
        self.label = label

        # TODO : add case for only one set

    def apply(self,inputs):
        """
        Apply rule on fuzzy sets

        Parameters
        ----------
        inputs : np.ndarray
            pair(s) of values to be evaluated by rule

        Returns
        -------
        activation : np.ndarray
            2d array with dimensions = len inputs * length universe
            holding activation function values
        """

        if inputs.ndim == 1:
            inputs = inputs.reshape((1,2)) # reshape 1D arrays to 2D

        # inputs = [n, 2]

        rules = np.ones(inputs.shape[0])

        # rules = [n]

        for statement in self.input_statements:

            fun1 = statement['fun1']
            fun2 = statement['fun2']

            # The OR operator means we take the maximum of these two.
            # The AND operator means we take the minimum of these two.

            if statement['operator'] == 'AND':
                rule = np.fmin(fun1.interp(inputs[:,0]), fun2.interp(inputs[:,1]))
            elif statement['operator'] == 'OR':
                rule = np.fmax(fun1.interp(inputs[:,0]), fun2.interp(inputs[:,1]))

            # TODO: add case for no operator

            rules = np.fmin(rules,rule) # AND statement between each line of statements

        # Now we apply rule by clipping the top off the corresponding output
        # membership function with `np.fmin`

        rules = rules.reshape((len(rules),1))

        # rules [n, 1]
        
        rules = np.tile(rules,(1,len(self.output.universe))) # broadcasting

        # rules [n, len universe]

        activation = np.fmin(rules, self.output.getArray()) # removed entirely to 0

        # activation [n, len universe]

        return activation

class fuzzySet():

    def __init__(self,
        lo : fuzzyFunction,
        md : fuzzyFunction,
        hi : fuzzyFunction,
        label=''
        ):
        """
        Contains all fuzzy funtions describing the 
        low, medium, and high level membership functions

        Parameters
        ----------
        lo : fuzzyFunction
            low instance of fuzzyFunction class
        md : fuzzyFunction
            medium instance of fuzzyFunction class
        hi : fuzzyFunction
            high instance of fuzzyFunction class
        label : str, optional
            string to tag instance with
        """

        self.lo = lo
        self.md = md
        self.hi = hi
        
        self.universe = lo.universe
        self.label = label

        self.lo.setLabel(label)
        self.md.setLabel(label)
        self.hi.setLabel(label)

        # TODO : if different universes raise an error


    def setLabel(self,label):
        """
        Changes label property of fuzzy set and all its membership functions

        Parameters
        ----------
        label : float
            new label name
        """

        self.label=label
        self.lo.setLabel(label)
        self.md.setLabel(label)
        self.hi.setLabel(label)

    def interp(self,inputs):

        """
        Interpret membership of input
        
        Parameters
        ----------
        inputs : float OR np.1darray
            The input(s) at which membership is to be interpreted

        Returns
        -------
        level_lo : float OR np.1darray
            The interpreted membership to the low 
            membership function of the input(s)
        level_md : float OR np.1darray
            The interpreted membership to the medium 
            membership function of the input(s)
        level_hi : float OR np.1darray
            The interpreted membership to the high 
            membership function of the input(s)
        """

        level_lo = self.lo.interp(inputs) 
        level_md = self.md.interp(inputs)
        level_hi = self.hi.interp(inputs)

        return level_lo,level_md,level_hi

    def view(self):
        """
        Used to view the distribution of all associated membership functions
        """

        # Visualize these universes and membership functions
        fig, ax = plt.subplots(nrows=1, figsize=(8, 3))

        ax.plot(self.universe, self.lo.getArray(), 'b', linewidth=1.5, label='Low')
        ax.plot(self.universe, self.md.getArray(), 'g', linewidth=1.5, label='Medium')
        ax.plot(self.universe, self.hi.getArray(), 'r', linewidth=1.5, label='High')
        ax.set_title(self.label)
        ax.legend()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        plt.tight_layout()

class fuzzySystem():

    def __init__(self,
        antecedents : List[fuzzySet],
        consequent : fuzzySet,
        rules : List[fuzzyRule],
        label=''
        ):
        """
        Contains all fuzzy inputs, outputs, rules,
        and fuzzy logic interpreter
        
        Parameters
        ----------
        antecedents : list
            list of fuzzySet objects defining the inputs
        consequent : fuzzySet
            fuzzySet objects defining the output
        rules : list
            fuzzyRule objects
        label: str, optional
            string to tag instance with
        """

        self.antecedents = antecedents
        self.consequent = consequent

        self.rules = rules
        self.label = label

        # Empty aggregate function
        self.aggregate = np.zeros((1,self.consequent.universe.shape[0]))

        # aggregate = [1, len consequent.universe]

        self.output = np.zeros(1)
        self.output_activation = np.zeros(1)

    def compute(self,
        inputs : np.ndarray, 
        normalize=False
        ): 
        """
        Compute fuzzy output of system

        Parameters
        ----------
        inputs : np.ndarray
            2d Array of inputs of shape [n, n_inputs]
            n = number of data points to be computed and 
            n_inputs = len(antecedents)
        normalize: bool, optional
            if True normalize aggregate output by area
        
        Returns
        -------

        output : float OR np.1darray
            defuzzified crisp value(s)
        aggregate : np.ndarray
            array of shape [n, len universe] 
            n = number of data points in the input
            len universe = len(self.universe)
        output_activation : np.1darray
            activation values(s) for defuzzified output
        """

        if inputs.ndim == 1:
            inputs = inputs.reshape((1,len(inputs))) # reshape 1D arrays to 2D

        # inputs = [n, len antecedents]

        aggregate = np.tile(self.aggregate,(inputs.shape[0],1))

        # aggregate = [n, len consequent.universe]

        for rule in self.rules:

            activation = rule.apply(inputs)

            # activation = [n, len consequent.universe]

            aggregate = np.fmax(activation,aggregate)

            # aggregate = [n, len consequent.universe]

        # normalize aggregate membership function
        if normalize:
            area = np.trapz(aggregate, x=self.consequent.universe)

            # area = [n]

            aggregate = np.divide(aggregate.T,area).T

            # aggregate = [n, len consequent.universe]

        self.aggregate = aggregate

        # Calculate defuzzified result
        def defuzzify(a):
            """Defuzzify an aggregate membership function"""
            return fuzz.defuzz(self.consequent.universe, a, 'centroid')

        self.output = np.apply_along_axis(defuzzify, 1, self.aggregate)

        # output = [n]

        # Calculate defuzzified result
        self.output_activation = np.zeros(self.output.shape)

        i = 0
        for a,b in zip(self.aggregate,self.output):
            # Defuzzify an aggregate membership function
            self.output_activation[i] = fuzz.interp_membership(self.consequent.universe, a, b)
            i += 0
        
        # output_activation = [n]

        if inputs.ndim == 1:
            inputs = inputs.reshape((1,len(inputs))) # reshape 1D arrays to 2D

        return self.output.squeeze(), self.aggregate.squeeze(), self.output_activation.squeeze()

    def reset(self):
        """
        Resets intermediate aggregate membership function to zeros
        """
        self.output = np.zeros(1)
        self.output_activation = np.zeros(1)
        self.aggregate = np.zeros((1,self.consequent.universe.shape[0]))

    def view(self):
        """
        View the aggregate membership function on the universe
        """

        fig, ax = plt.subplots(figsize=(8, 3))

        n_0 = np.zeros_like(self.consequent.universe)

        ax.fill_between(self.consequent.universe, n_0, self.aggregate[0,:], facecolor='Orange', alpha=0.7)
        ax.plot([self.output[0], self.output[0]], [0, self.output_activation[0]], 'k', linewidth=1.5, alpha=0.9)
        ax.set_title('Aggregated membership and result (line)')

        plt.show()