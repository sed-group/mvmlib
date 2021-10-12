import skfuzzy as fuzz
import numpy as np
from scipy.interpolate import interp1d
from typing import Dict, Any, AnyStr, List

"""Fuzzy Library for computing aggregate membership functions for fuzzy variables"""

class fuzzyFunction():

    def __init__(self,universe,label=''):
        """
        Contrains description and implementation of
        different fuzzy membership functions
        
        Parameters
        ----------
        universe : np.array
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
        array : np.array
            1d array of length universe
        """

        self.array = fuzz.trimf(self.universe, [self.low, self.medium, self.high])
        return self.array

    def interp(self,input=None):
        """
        Interpret membership of input ot fuzzy function

        Parameters
        ----------
        input : float

        Returns
        -------
        level : float
            interpreted membership level of the input
        """

        if input:
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
            list of dicts 
            structure of each dict
            {
                'fun1': fuzzyFunction object
                'fun2': fuzzyFunction object
                operator: 'AND', 'OR'
            }
        
        # TODO : add case for only one set
        label : str, optional
            string to tag instance with
        """
        
        self.input_statements = input_statements
        self.output = output
        self.label = label

    def apply(self,input):
        """
        Apply rule on fuzzy sets

        Parameters
        ----------
        input : dict
            dict of structure
            {
            'label': float,
            'label': float,
            }
        
        Returns
        -------
        activation : np.array
            1d array of length universe
            holding activation function 
            values
        """

        rules = 1.0

        for statement in self.input_statements:

            fun1 = statement['fun1']
            fun2 = statement['fun2']

            # The OR operator means we take the maximum of these two.
            # The AND operator means we take the minimum of these two.

            if statement['operator'] == 'AND':
                rule = np.fmin(fun1.interp(input[fun1.label]), fun2.interp(input[fun2.label]))
            elif statement['operator'] == 'OR':
                rule = np.fmax(fun1.interp(input[fun1.label]), fun2.interp(input[fun2.label]))

            # TODO: add case for no operator

            rules = np.fmin(rules,rule) # AND statement between each line of statements

        # Now we apply rule by clipping the top off the corresponding output
        # membership function with `np.fmin`
        activation = np.fmin(rules, self.output.getArray())  # removed entirely to 0

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

    def interp(self,input):

        """
        Interpret membership of input
        
        Parameters
        ----------
        input : float
        """

        level_lo = self.lo.interp(input) 
        level_md = self.md.interp(input)
        level_hi = self.hi.interp(input)

        return level_lo,level_md,level_hi

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
        self.aggregate = np.zeros_like(self.consequent.universe)


    def compute(self,
        input : Dict[str,float], 
        normalize=False
        ): 
        """
        Compute fuzzy output of system

        Parameters
        ----------
        input : dict
            dict of structure
            {
            'label': float,
            'label': float,
            }
        normalize: bool, optional
            if True normalize aggregate output by area
        
        Returns
        -------

        output : float
            defuzzified value
        aggregate : np.array
            array of length universe
        output_activation : float
            activation for defuzzified output
        """

        self.reset()

        aggregate = self.aggregate

        for rule in self.rules:

            activation = rule.apply(input)
            aggregate = np.fmax(activation,aggregate)

        # normalize aggregate membership function
        if normalize:
            dx = self.consequent.universe[1] - self.consequent.universe[0]
            area = np.trapz(aggregate, dx=dx)
            aggregate /= area

        self.aggregate = aggregate

        # Calculate defuzzified result
        output = fuzz.defuzz(self.consequent.universe, aggregate, 'centroid')
        output_activation = fuzz.interp_membership(self.consequent.universe, aggregate, output) # for plot

        return output, aggregate, output_activation

    def interpolate_activation(self,value):
        """interpolate the activation of any value in the
        consequent universe

        Parameters
        ----------
        value : float or np.array
            value(s) inside consequent universe 
            to interpolate activation at.

        Returns
        -------
        interp_activation : float or np.array
            interpolated activation value(s)
        """

        # Linear interpolation of aggregate membership
        f_a = interp1d(self.consequent.universe, self.aggregate)

        interp_activation = f_a(value)

        return interp_activation

    def reset(self):
        """
        Resets intermediate aggregate membership function to zeros
        """
        self.aggregate = np.zeros_like(self.consequent.universe)