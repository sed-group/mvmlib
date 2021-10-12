import skfuzzy as fuzz
import numpy as np

class fuzzySystem():

    def __init__(self,antecedents,consequent,rules,label=''):
        """
        Contains all fuzzy inputs, outputs, rules,
        and fuzzy logic interpreter
        
        Parameters
        ----------
        antecedents : list
            fuzzySet objects
        consequent : list
            fuzzySet objects
        rules : list
            fuzzyRule objects
        label: str, optional
            string to tag instance with
        """

        self.antecedents = antecedents
        self.consequent = consequent

        self.rules = rules
        self.label = label

    def compute(self,input, normalize=False): 
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

        aggregate = np.zeros_like(self.antecedents[0].universe)

        for rule in self.rules:

            activation = rule.apply(input)
            aggregate = np.fmax(activation,aggregate)

        # normalize aggregate membership function
        if normalize:
            dx = self.consequent.universe[1] - self.consequent.universe[0]
            area = np.trapz(aggregate, dx=dx)
            aggregate /= area

        # Calculate defuzzified result
        output = fuzz.defuzz(self.consequent.universe, aggregate, 'centroid')
        output_activation = fuzz.interp_membership(self.consequent.universe, aggregate, output) # for plot

        return output, aggregate, output_activation

