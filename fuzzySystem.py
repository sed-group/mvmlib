import skfuzzy as fuzz
import numpy as np

class fuzzySystem():

    def __init__(self,antecedents,consequent,rules,label=''):

        """
        Constructor

        input: dict of structure
        {
            'label': float,
            'label': float,
        }

        antecedents: list of fuzzySet objects
        consequent: list of fuzzySet objects
        rules: list of fuzzyRule objects
        """

        self.antecedents = antecedents
        self.consequent = consequent

        self.rules = rules
        self.label = label

    def compute(self,input):
        
        aggregate = np.zeros_like(self.antecedents[0].universe)

        for rule in self.rules:

            activation = rule.apply(input)
            aggregate = np.fmax(activation,aggregate)

        # Calculate defuzzified result
        output = fuzz.defuzz(self.consequent.universe, aggregate, 'centroid')
        output_activation = fuzz.interp_membership(self.consequent.universe, aggregate, output) # for plot

        return output, aggregate, output_activation

