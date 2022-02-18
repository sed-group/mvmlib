"""
Examples
--------
Example showing the usage if discrete density 
function values to define a multivariate Guassian
"""

# 1D example
import numpy as np
from dmLib import GaussianFunc, Distribution

mean = 10.0
sd = 5.0
x = np.linspace(-100, 100, 100)  # 2D grid

function = GaussianFunc(np.array([mean]), np.array([[sd ** 2, ]]))
p = function.compute_density(x[:, None])  # get density values
dist = Distribution(p)

print(dist.random(10000).mean(axis=1))  # should be close to 10.0
print(dist.random(10000).std(axis=1))  # should be close to 5.0

# View distribution
import matplotlib.pyplot as plt

plt.hist(dist.random(10000).squeeze(), bins=100, density=True)
# plt.plot(np.linspace(-6,6),pdf)
plt.show()

# 2D example
import numpy as np
from dmLib import Design, GaussianFunc, Distribution

mean = 10.0
sd = 5.0

x = Design(np.array([-100, -100]), np.array([100, 100]), 512, "fullfact").unscale()  # 2D grid
function = GaussianFunc(np.array([mean, mean]), np.array([[sd ** 2, 0], [0, sd ** 2]]))
p = function.compute_density(x)  # get density values

dist = Distribution(p.reshape((512, 512)))
print(dist.random(1000000).mean(axis=1))  # should be close to 10.0
print(dist.random(1000000).std(axis=1))  # should be close to 5.0

# View distribution
import matplotlib.pyplot as plt

plt.scatter(*dist.random(1000))
plt.show()
