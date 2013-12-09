
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

# hypothesis testing regarding single cell errors

# plzf single cell data
D = np.zeros((2,2,2)) # three cell options
D[1,0,0] = 185.
D[0,1,0] = 99.
D[0,0,1] = 49.
D[1,1,0] = 9.
D[0,1,1] = 11.
D[1,0,1] = 19.
D[1,1,1] = 6.

# calculate probability of t or more triplet wells out of Nw wells
# given probability a non-single well becomes a triplet
# and the probability of having a single well
def triplet_prob(Nw,t,ptriplet,psingle):
    weights = np.zeros(Nw+1)
    # number of non-singles must be greater than or equal to t
    for ns in xrange(t,Nw+1):
        weights[ns] = 1.0 - binom.cdf(t-1,ns,ptriplet)
        
    return sum(weights * binom.pmf(np.linspace(0,Nw,Nw+1),Nw,(1.0-psingle)))

# test against completely pre-committed singletons
test_p = np.linspace(0.,1.,200)[1:-1]
single_test = [triplet_prob(378,6,1.0/27.0,p) for p in test_p]

plt.scatter(test_p,single_test)
plt.show()

# test against pre-committed duals
dtest_p = np.linspace(0.7,1.,300)[1:-1]
dual_test = [triplet_prob(378,6,2.0/9.0,p) for p in dtest_p]

plt.scatter(dtest_p,dual_test)
plt.show()

# test against even chance dual combo
ctest_p = np.linspace(0.7,1.,300)[1:-1]
combo_test = [triplet_prob(378,6,0.25,p) for p in ctest_p]

plt.scatter(ctest_p,combo_test)
plt.show()

