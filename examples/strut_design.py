import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.optimize import minimize_scalar,fsolve

from dmLib import triangularFunc, fuzzySet, fuzzyRule, fuzzySystem
from dmLib import Design
from dmLib import MarginNode, FixedParam, DesignParam, InputSpec, Behaviour, MarginNetwork
from dmLib import Distribution, gaussianFunc

# define fixed parameters
p1 = FixedParam(1.17E-04    ,'P1',description='Coefficient of thermal expansion' ,symbol='alpha' )
p2 = FixedParam(156.3E3     ,'P2',description='Youngs modulus'                   ,symbol='E'     )
p3 = FixedParam(8.19e-06    ,'P3',description='Material density'                 ,symbol='rho'   )
p4 = FixedParam(346.5       ,'P4',description='Radius of the hub'                ,symbol='r1'    )
p5 = FixedParam(536.5       ,'P5',description='Radius of the shroud'             ,symbol='r2'    )
fixed_params = [p1,p2,p3,p4,p5]

# define design parameters
d1 = DesignParam(100.0  ,'D1'   ,universe=(70.0,130.0) ,description='vane length'  ,symbol='w'     )
d2 = DesignParam(10.0   ,'D2'   ,universe=(0.5,20.0)   ,description='vane height'  ,symbol='h'     )
d3 = DesignParam(85.0   ,'D1'   ,universe=(0.0,90.0)    ,description='lean angle'   ,symbol='theta' )
design_params = [d1,d2,d3]

# T1,T2 distribution
mu = np.array([370,580])
Sigma = np.array([
    [50, 25],
    [75, 100],
    ])
Requirement = gaussianFunc(mu, Sigma, 'temp')

# define input specifications
s1 = InputSpec(Requirement  ,'S1'   ,cov_index=0    ,description='nacelle temperature'      ,symbol='T1'        )
s2 = InputSpec(Requirement  ,'S2'   ,cov_index=1    ,description='gas surface temperature'  ,symbol='T2'        )
s3 = InputSpec(30.0e5       ,'S3'                   ,description='allowable force'          ,symbol='f_all'     )
s4 = InputSpec(460.0        ,'S4'                   ,description='yield stress'             ,symbol='sigma_y'   )
input_specs = [s1,s2,s3,s4]

# define the behaviour models

# this is the length model
class B1(Behaviour):
    def __call__(self,theta,r1,r2):
        def f(L):
            return (L**2) + 2*r1*L*np.cos(np.deg2rad(theta)) - ((r2**2) - (r1**2))
        
        def f_prime(L):
            return 2*L + 2*r1*np.cos(np.deg2rad(theta))
        
        lb = r2-r1
        ub = np.sqrt(r2**2 - r1**2)
        # length=minimize_scalar(f,bounds=(lb,ub),method='bounded')
        length=fsolve(f,lb + (ub-lb)*0.5)[0]
        self.intermediate = length

# this is the weight model
class B2(Behaviour):
    def __call__(self,rho,w,h,L):
        weight = rho*w*h*L
        self.performance = weight

# this is the force model
class B3(Behaviour):
    def __call__(self,alpha,E,T1,T2,w,h,theta):
        force = (E*w*h*alpha)*(T2-T1)*np.cos(np.deg2rad(theta))
        self.decided_value = force

# this is the stress model
class B4(Behaviour):
    def __call__(self,alpha,E,T1,T2,h,theta,L):
        sigma_m = ((3*E*h)/(2*L))*(alpha*(T2-T1)*np.sin(np.deg2rad(theta)))
        self.decided_value = sigma_m

b1 = B1('B1')
b2 = B2('B2')
b3 = B3('B3')
b4 = B4('B4')
behaviours = [b1,b2,b3,b4]

e1 = MarginNode('E1',type='must_not_exceed')
e2 = MarginNode('E2',type='must_not_exceed')
margin_nodes = [e1,e2]

# Define the MAN
class MAN(MarginNetwork):
    def forward(self):

        Requirement()

        # retrieve MAN components
        d1 = self.design_params[0]
        d2 = self.design_params[1]
        d3 = self.design_params[2]

        s1 = self.input_specs[0] # stochastic
        s2 = self.input_specs[1] # stochastic
        s3 = self.input_specs[2]
        s4 = self.input_specs[3]

        p1 = self.fixed_params[0]
        p2 = self.fixed_params[1]
        p3 = self.fixed_params[2]
        p4 = self.fixed_params[3]
        p5 = self.fixed_params[4]

        b1 = self.behaviours[0]
        b2 = self.behaviours[1]
        b3 = self.behaviours[2]
        b4 = self.behaviours[3]

        e1 = self.margin_nodes[0]
        e2 = self.margin_nodes[1]

        # Execute behaviour models
        b1(d3.value,p4.value,p5.value)
        b2(p3.value,d1.value,d2.value,b1.intermediate)
        b3(p1.value,p2.value,s1(),s2(),d1.value,d2.value,d3.value)
        b4(p1.value,p2.value,s1(),s2(),d2.value,d3.value,b1.intermediate)

        e1(b3.decided_value,s3())
        e2(b4.decided_value,s4())

man = MAN(design_params,input_specs,fixed_params,behaviours,margin_nodes,'MAN_1')

for n in range(1000):
    man.forward()

man.margin_nodes[0].view()
man.margin_nodes[1].view()