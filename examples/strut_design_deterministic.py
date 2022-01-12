import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.optimize import minimize_scalar,fsolve
import sys

from dmLib import triangularFunc, fuzzySet, fuzzyRule, fuzzySystem
from dmLib import Design
from dmLib import FixedParam, DesignParam, InputSpec, Behaviour, Performance, MarginNode, MarginNetwork
from dmLib import Distribution, gaussianFunc, uniformFunc
from dmLib import nearest

# define fixed parameters
i1 = FixedParam(7.17E-06    ,'I1',description='Coefficient of thermal expansion'    ,symbol='alpha'     )
i2 = FixedParam(156.3E3     ,'I2',description='Youngs modulus'                      ,symbol='E'         )
i3 = FixedParam(8.19e-06    ,'I3',description='Material density'                    ,symbol='rho'       )
i4 = FixedParam(346.5       ,'I4',description='Radius of the hub'                   ,symbol='r1'        )
i5 = FixedParam(536.5       ,'I5',description='Radius of the shroud'                ,symbol='r2'        )
i6 = FixedParam(1.0         ,'I6',description='Column effective length factor'      ,symbol='K'         )
i7 = FixedParam(25.0        ,'I7',description='ambient temperature'                 ,symbol='Ts'        )
i8 = FixedParam(460.0       ,'I8',description='yield stress'                        ,symbol='sigma_y'   )

fixed_params = [i1,i2,i3,i4,i5,i6,i7,i8]

# define design parameters
d1 = DesignParam(100.0  ,'D1'   ,universe=(70.0,130.0)  ,description='vane length'  ,symbol='w'     )
d2 = DesignParam(15.0   ,'D2'   ,universe=(5.0,20.0)    ,description='vane height'  ,symbol='h'     )
d3 = DesignParam(10.0   ,'D3'   ,universe=(0.0,30.0)    ,description='lean angle'   ,symbol='theta' )
design_params = [d1,d2,d3]

# T1,T2 distribution (Uniform)
center = np.array([450,425])
Range = np.array([100, 100])/20
Requirement = uniformFunc(center, Range, 'temp')

# define input specifications
s1 = InputSpec(450,'S1',universe=(325,550),description='nacelle temperature',
    symbol='T1',inc = -1e-0,inc_type='rel')
s2 = InputSpec(425,'S2',universe=(325,550),description='gas surface temperature',
    symbol='T2',inc=+1e-0,inc_type='rel')
input_specs = [s1,s2]

# define the behaviour models

# this is the length model
class B1(Behaviour):
    def __call__(self,theta,r1,r2):
        length=-r1*np.cos(np.deg2rad(theta)) + np.sqrt(r2**2 - (r1*np.sin(np.deg2rad(theta)))**2)
        self.intermediate = length

# this is the weight model
class B2(Behaviour):
    def __call__(self,rho,w,h,L):
        weight = rho*w*h*L
        self.performance = weight

# this is the axial stress model
class B3(Behaviour):
    def __call__(self,alpha,E,r1,r2,Ts,T1,T2,w,h,theta,L):
        force = (E*w*h*alpha)*((T2*0.95*r2)-(T1*r1)-(Ts*(r2-r1)))*np.cos(np.deg2rad(theta)) / L
        sigma_a = (E*alpha)*((T2*1.05*r2)-(T1*r1)-(Ts*(r2-r1)))*np.cos(np.deg2rad(theta)) / L
        self.threshold = [force/1000,sigma_a]

# this is the bending stress model
class B4(Behaviour):
    def __call__(self,alpha,E,r1,r2,Ts,T1,T2,h,theta,L):
        sigma_m = (3/2)*((E*h)/(L**2))*(alpha*((T2*0.97*r2)-(T1*r1)-(Ts*(r2-r1)))*np.sin(np.deg2rad(theta)))
        self.threshold = sigma_m

# this is the buckling model
class B5(Behaviour):
    def __call__(self,E,K,w,h,L):
        f_buckling = ((np.pi**2)*E*w*(h**3)) / (12*((K*L)**2))
        self.decided_value = f_buckling/1000

b1 = B1('B1')
b2 = B2('B2')
b3 = B3('B3')
b4 = B4('B4')
b5 = B5('B4')
behaviours = [b1,b2,b3,b4,b5]

# Define margin nodes
e1 = MarginNode('E1',type='must_not_exceed')
e2 = MarginNode('E2',type='must_not_exceed')
e3 = MarginNode('E3',type='must_not_exceed')
margin_nodes = [e1,e2,e3]

# Define performances
p1 = Performance('P1',type='less_is_better')
performances = [p1,]

# Define the MAN
class MAN(MarginNetwork):

    def forward(self):

        # retrieve MAN components
        d1 = self.design_params[0] # w
        d2 = self.design_params[1] # h
        d3 = self.design_params[2] # theta

        s1 = self.input_specs[0] # T1 (stochastic)
        s2 = self.input_specs[1] # T2 (stochastic)

        i1 = self.fixed_params[0] # alpha
        i2 = self.fixed_params[1] # E
        i3 = self.fixed_params[2] # rho
        i4 = self.fixed_params[3] # r1
        i5 = self.fixed_params[4] # r2
        i6 = self.fixed_params[5] # K
        i7 = self.fixed_params[6] # Ts
        i8 = self.fixed_params[7] # sigma_y

        b1 = self.behaviours[0] # calculates length
        b2 = self.behaviours[1] # calculates weight
        b3 = self.behaviours[2] # calculates axial force and stress
        b4 = self.behaviours[3] # calculates bending stress
        b5 = self.behaviours[4] # calculates buckling load

        e1 = self.margin_nodes[0] # margin against buckling (F,F_buckling)
        e2 = self.margin_nodes[1] # margin against axial failure (sigma_a,sigma_y)
        e3 = self.margin_nodes[2] # margin against bending failure (sigma_m,sigma_y)

        p1 = self.performances[0] # weight

        # Execute behaviour models
        b1(d3.value,i4.value,i5.value)
        b2(i3.value,d1.value,d2.value,b1.intermediate)
        b3(i1.value,i2.value,i4.value,i5.value,i7.value,s1.value,s2.value,d1.value,d2.value,d3.value,b1.intermediate)
        b4(i1.value,i2.value,i4.value,i5.value,i7.value,s1.value,s2.value,d2.value,d3.value,b1.intermediate)
        b5(i2.value,i6.value,d1.value,d2.value,b1.intermediate)

        # Compute excesses
        e1(b3.threshold[0],b5.decided_value)
        e2(b3.threshold[1],i8.value)
        e3(b4.threshold,i8.value)

        # Compute performances
        p1(b2.performance)

man = MAN(design_params,input_specs,fixed_params,
    behaviours,margin_nodes,performances,'MAN_1')

# Create surrogate model for estimating threshold performance
man.train_performance_surrogate(n_samples=700,bandwidth=[1e-3,],sampling_freq=1)
man.forward()
man.view_perf(e_indices=[0,1],p_index=0)
man.view_perf(e_indices=[0,2],p_index=0)
man.view_perf(e_indices=[1,2],p_index=0)

# Run a forward pass of the MAN
man.forward()
man.compute_impact()
man.compute_absorption()

# View value of excess
print(man.excess_vector)

# View Impact on Performance
print(man.impact_matrix.impact)

# View Deterioration
print(man.absorption_matrix.deterioration)

# View Absorption capability
print(man.absorption_matrix.absorption)

# display the margin value plot
d = man.compute_MVP('scatter',show_neutral=True)

# Effect of alternative designs
n_designs = 100
design_doe = Design(man.lb_d,man.ub_d,n_designs,'LHS')

# create empty figure
fig, ax = plt.subplots(figsize=(7, 8))
ax.set_xlabel('Impact on performance')
ax.set_ylabel('Change absoption capability')

X = np.empty((1,len(man.margin_nodes)))
Y = np.empty((1,len(man.margin_nodes)))
for i,design in enumerate(design_doe.unscale()):
    man.nominal_design_vector = design
    man.reset()
    man.reset_outputs()

    # Display progress bar
    sys.stdout.write("Progress: %d%%   \r" %((i/n_designs)* 100))
    sys.stdout.flush()

    # Perform MAN computations
    man.forward()
    man.compute_impact()
    man.compute_absorption()

    # Extract x and y
    x = np.mean(man.impact_matrix.impacts,axis=(1,2)).ravel() # average along performance parameters (assumes equal weighting)
    y = np.mean(man.absorption_matrix.absorptions,axis=(1,2)).ravel() # average along input specs (assumes equal weighting)

    if not all(np.isnan(y)):
        X = np.vstack((X,x))
        Y = np.vstack((Y,y))

    # plot the results
    color = np.random.random((1,3))
    ax.scatter(x,y,c=color)

plt.show()

# Calculate distance metric
p1 = np.array([X.min(),Y.min()])
p2 = np.array([X.max(),Y.max()])
p1 = p1 - 0.1*abs(p2 - p1)
p2 = p2 + 0.1*abs(p2 - p1)

distances = np.empty(0)
for i,(x,y) in enumerate(zip(X,Y)):

    dist = 0
    for node in range(len(x)):
        s = np.array([x[node],y[node]])
        pn,d = nearest(p1,p2,s)
        dist += d

    distances = np.append(distances,dist)

# display distances only for three designs
X_plot = X[0:3,:]
Y_plot = Y[0:3,:]

# create empty figure
colors = ['#FF0000','#27BE1E','#0000FF']
fig, ax = plt.subplots(figsize=(7, 8))
ax.set_xlabel('Impact on performance')
ax.set_ylabel('Change absoption capability')
ax.set_xlim(p1[0],p2[0])
ax.set_ylim(p1[1],p2[1])

p = ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='k',linestyle=(5,(10,5)))

distances = np.empty(0)
for i,(x,y) in enumerate(zip(X_plot,Y_plot)):

    ax.scatter(x,y,c=colors[i])

    dist = 0
    for node in range(len(x)):
        s = np.array([x[node],y[node]])
        pn,d = nearest(p1,p2,s)
        dist += d

        x_d = [s[0],pn[0]]
        y_d = [s[1],pn[1]]
        ax.plot(x_d,y_d,marker='.',linestyle='--',color=colors[i])

    distances = np.append(distances,dist)

plt.show()