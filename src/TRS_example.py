import numpy as np
import matplotlib.pyplot as plt
# import skfuzzy as fuzz
# from skfuzzy import control as ctrl
from dmLib.fuzzyLib.fuzzyFunctions import triangularFunc
from dmLib.fuzzyLib.fuzzySet import fuzzySet
from dmLib.fuzzyLib.fuzzyRule import fuzzyRule
from dmLib.fuzzyLib.fuzzySystem import fuzzySystem

from dmLib.DOELib.Design import Design

# Generate universe variables
nominal = np.array([350, 425, 410, 580, 3.5])
lb = np.array([250, 325, 310, 480, 1.0])
ub = np.array([450, 525, 510, 680, 6.0])
labels = ['T1','T2','T3','T4','n_safety']

universe = np.linspace(lb, ub, 100) # grid for all variables

# Define shape of triangular membership functions
shapes_lo = np.array([lb,lb,lb + (ub-lb)/2])
shapes_md = np.array([lb + (ub-lb)/4,lb + (ub-lb)/2,ub - (ub-lb)/4])
shapes_hi = np.array([lb + 1*(ub-lb)/2,ub,ub])

# Generate fuzzy membership functions
fuzzy_sets = []
for i in range(len(nominal)):
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
temp_T3 = fuzzy_sets[2]
temp_T4 = fuzzy_sets[3]
n_safety = fuzzy_sets[4]

# Visualize these universes and membership functions
fig, axis = plt.subplots(nrows=5, figsize=(15, 9))

i = 1
for ax,var,label in zip(axis,fuzzy_sets,labels):

    ax.plot(var.universe, var.lo.getArray(), 'b', linewidth=1.5, label='Low')
    ax.plot(var.universe, var.md.getArray(), 'g', linewidth=1.5, label='Medium')
    ax.plot(var.universe, var.hi.getArray(), 'r', linewidth=1.5, label='High')
    ax.set_title(label)
    ax.legend()

    i += 1

# Turn off top/right axes
for ax in axis:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

# Define fuzzy rules
rule1 = fuzzyRule([{'fun1': temp_T1.lo, 'fun2': temp_T4.hi, 'operator': 'AND'},],n_safety.lo,label='R1')
rule2 = fuzzyRule([{'fun1': temp_T1.hi, 'fun2': temp_T4.lo, 'operator': 'AND'},],n_safety.hi,label='R2')
rule3 = fuzzyRule([{'fun1': temp_T1.lo, 'fun2': temp_T4.lo, 'operator': 'AND'},],n_safety.md,label='R3')
rule4 = fuzzyRule([{'fun1': temp_T1.hi, 'fun2': temp_T4.hi, 'operator': 'AND'},],n_safety.md,label='R4')

rule5 = fuzzyRule([{'fun1': temp_T1.md, 'fun2': temp_T4.hi, 'operator': 'AND'},],n_safety.lo,label='R5')
rule6 = fuzzyRule([{'fun1': temp_T1.md, 'fun2': temp_T4.lo, 'operator': 'AND'},],n_safety.hi,label='R6')
rule7 = fuzzyRule([{'fun1': temp_T1.lo, 'fun2': temp_T4.md, 'operator': 'AND'},],n_safety.lo,label='R7')
rule8 = fuzzyRule([{'fun1': temp_T1.hi, 'fun2': temp_T4.md, 'operator': 'AND'},],n_safety.hi,label='R8')

rule9 = fuzzyRule([{'fun1': temp_T1.md, 'fun2': temp_T4.md, 'operator': 'AND'},],n_safety.md,label='R9')

# Define fuzzy control system
rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]
sim = fuzzySystem([temp_T1,temp_T2],n_safety,rules)

# Compute for given inputs
inputs = {'T1' : 370, 'T4' : 580,}
n_safety_value, aggregate, n_safety_activation = sim.compute(inputs, normalize=True)

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

n_0 = np.zeros_like(n_safety.universe)

ax0.fill_between(n_safety.universe, n_0, aggregate, facecolor='Orange', alpha=0.7)
ax0.plot([n_safety_value, n_safety_value], [0, n_safety_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated membership and result (line)')

# Simulate at higher resolution the control space in 2D
x, y = np.meshgrid(universe[:,0], universe[:,3])
z = np.zeros_like(x)

# Loop through the system to collect the control surface
for i in range(len(universe)):
    for j in range(len(universe)):
        inputs = {'T1' : x[i, j], 'T4' : y[i, j],}
        z[i, j],_,_ = sim.compute(inputs)

# Plot the result in 2D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()

surf = ax.contourf(x, y, z, cmap=plt.cm.jet,)
ax.set_xlabel('T1')
ax.set_ylabel('T4')

cbar = plt.cm.ScalarMappable(cmap=plt.cm.jet)
cbar.set_array(z)

boundaries = np.linspace(1, 6, 51)
cbar_h = fig.colorbar(cbar, boundaries=boundaries)
cbar_h.set_label('safety factor', rotation=90, labelpad=3)

plt.show()

# Define threshold safety factor
threshold = 2.8

# Latin hypercube for n_safety
lb_nsafety = np.array([lb[-1],])
ub_nsafety = np.array([ub[-1],])
nsamples = 100

lh_nsafety = Design(lb_nsafety,ub_nsafety,nsamples,'LHS')


