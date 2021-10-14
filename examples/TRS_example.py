import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from dmLib import triangularFunc, fuzzySet, fuzzyRule, fuzzySystem
from dmLib import Design

# Generate universe variables
lb = np.array([250, 480, 1.0])
ub = np.array([450, 680, 6.0])
labels = ['T1','T2','n_safety']

universe = np.linspace(lb, ub, 100) # grid for all variables

# Define shape of triangular membership functions
shapes_lo = np.array([lb,               lb,             lb + (ub-lb)/2  ])
shapes_md = np.array([lb + (ub-lb)/4,   lb + (ub-lb)/2, ub - (ub-lb)/4  ])
shapes_hi = np.array([lb + 1*(ub-lb)/2, ub,             ub              ])

# Generate fuzzy membership functions
fuzzy_sets = []
for i in range(len(lb)):
    lo = triangularFunc(universe[:,i],labels[i])
    lo.setFunc(shapes_lo[0,i],shapes_lo[1,i],shapes_lo[2,i])

    md = triangularFunc(universe[:,i],labels[i])
    md.setFunc(shapes_md[0,i],shapes_md[1,i],shapes_md[2,i])

    hi = triangularFunc(universe[:,i],labels[i])
    hi.setFunc(shapes_hi[0,i],shapes_hi[1,i],shapes_hi[2,i])

    fuzzyset_i = fuzzySet(lo,md,hi,labels[i])

    fuzzy_sets += [fuzzyset_i]

# label each fuzzy set
temp_T1 = fuzzy_sets[0]
temp_T2 = fuzzy_sets[1]
n_safety = fuzzy_sets[2]

# Visualize these universes and membership functions
temp_T1.view()
temp_T2.view()
n_safety.view()

# Define fuzzy rules
rule1 = fuzzyRule([{'fun1': temp_T1.lo, 'fun2': temp_T2.hi, 'operator': 'AND'},],n_safety.lo,label='R1')
rule2 = fuzzyRule([{'fun1': temp_T1.hi, 'fun2': temp_T2.lo, 'operator': 'AND'},],n_safety.hi,label='R2')
rule3 = fuzzyRule([{'fun1': temp_T1.lo, 'fun2': temp_T2.lo, 'operator': 'AND'},],n_safety.md,label='R3')
rule4 = fuzzyRule([{'fun1': temp_T1.hi, 'fun2': temp_T2.hi, 'operator': 'AND'},],n_safety.md,label='R4')

rule5 = fuzzyRule([{'fun1': temp_T1.md, 'fun2': temp_T2.hi, 'operator': 'AND'},],n_safety.lo,label='R5')
rule6 = fuzzyRule([{'fun1': temp_T1.md, 'fun2': temp_T2.lo, 'operator': 'AND'},],n_safety.hi,label='R6')
rule7 = fuzzyRule([{'fun1': temp_T1.lo, 'fun2': temp_T2.md, 'operator': 'AND'},],n_safety.lo,label='R7')
rule8 = fuzzyRule([{'fun1': temp_T1.hi, 'fun2': temp_T2.md, 'operator': 'AND'},],n_safety.hi,label='R8')

rule9 = fuzzyRule([{'fun1': temp_T1.md, 'fun2': temp_T2.md, 'operator': 'AND'},],n_safety.md,label='R9')

# Define fuzzy control system
rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]
sim = fuzzySystem([temp_T1,temp_T2],n_safety,rules)

# Compute for given inputs
inputs = np.array([
    370.0, # T1
    580.0, # T2
    ])
n_safety_value, aggregate, n_safety_activation = sim.compute(inputs, normalize=True)

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

n_0 = np.zeros_like(n_safety.universe)

ax0.fill_between(n_safety.universe, n_0, aggregate, facecolor='Orange', alpha=0.7)
ax0.plot([n_safety_value, n_safety_value], [0, n_safety_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated membership and result (line)')

# Simulate at higher resolution the control space in 2D
n_levels = 100
grid_in = Design(lb[:2],ub[:2],n_levels,"fullfact").unscale()

# Loop through the system to collect the control surface
z,a,_ = sim.compute(grid_in,normalize=True)

x = grid_in[:,0].reshape((n_levels,n_levels))
y = grid_in[:,1].reshape((n_levels,n_levels))
z = z.reshape((n_levels,n_levels))

# %% Figure 1

# Plot the result in 2D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()

surf = ax.contourf(x, y, z, cmap=plt.cm.jet,)
ax.set_xlabel('T1')
ax.set_ylabel('T2')

cbar = plt.cm.ScalarMappable(cmap=plt.cm.jet)
cbar.set_array(z)

boundaries = np.linspace(1, 6, 51)
cbar_h = fig.colorbar(cbar, boundaries=boundaries)
cbar_h.set_label('safety factor', rotation=90, labelpad=3)

plt.show()

# Requirement definition

from dmLib import gaussianFunc

mu = np.array([370,580])
Sigma = np.array([
    [50, 25],
    [75, 100],
    ])

Requirement = gaussianFunc(mu, Sigma, 'thermal_requirement')
r = 3

# %% Figure 2

p = Requirement.compute_density(grid_in) # get probability density
p = p.reshape((n_levels,n_levels))

# Compute capability of the design
threshold = 3.0 # threshold in universe
feasible_condition = n_safety.universe > threshold
area = np.trapz(a[:, feasible_condition], x=n_safety.universe[feasible_condition])
area = area.reshape((n_levels,n_levels))

# find boundary of requirements
requirement = np.zeros_like(x)
requirement[p < Requirement.compute_density_r(r=3)] = 1

# Plot the result in 2D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()

surf = ax.contourf(x, y, area, cmap=plt.cm.jet,)
ax.set_xlabel('T1')
ax.set_ylabel('T2')

cbar = plt.cm.ScalarMappable(cmap=plt.cm.jet)
cbar.set_array(area)

cbar_h = fig.colorbar(cbar)
cbar_h.set_label('Probability of no failure', rotation=90, labelpad=3)

c1 = ax.contourf( x, y, requirement, alpha=0.0, levels=[-20, 0, 20], colors=['#FF0000','#FF0000'], 
							hatches=['//', None])
ax.contour(c1, colors='#FF0000', linewidths = 2.0, zorder=2)

label='requirement 3-$\sigma$'
actor = patches.Rectangle((20,20), 20, 20, linewidth=2, edgecolor='#FF0000', facecolor='none', fill='None', hatch='///')

ax.legend([actor,],[label,])

plt.show()
# %% Margin computation
n_levels = 200
grid_margins = Design(lb[:2],ub[:2],n_levels,"fullfact").unscale()

total_V = np.prod(ub[:2] - lb[:2])
n = grid_margins.shape[0]
p = Requirement.compute_density(grid_margins) # get probability density of requirements

# Compute capability
# Loop through the system to collect the control surface
z,a,_ = sim.compute(grid_margins,normalize=True)
# Compute capability of the design
feasible_condition = n_safety.universe > threshold
area = np.trapz(a[:, feasible_condition], x=n_safety.universe[feasible_condition])

# calculate capability
capability = (1/n) * np.sum(area)

# calculate reliability
p_reliability = area * p.copy() # np.array is mutable!!

reliability = (total_V/n) * np.sum(p_reliability)

req_cond = p >= Requirement.compute_density_r(r=3)

# calculate buffer
p_buffer = area.copy() # np.array is mutable!!
p_buffer[~req_cond] = 0 # apply indicator function I_C

buffer = (1/n) * np.sum(p_buffer)

# calculate excess
p_excess = area.copy() # np.array is mutable!!
p_excess[req_cond] = 0 # apply indicator function I_C

excess = (1/n) * np.sum(p_excess)