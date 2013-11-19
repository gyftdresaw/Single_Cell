
# simple binary decision making
# from a fixed pool of progenitors
#
# each data sample only indicates
# the categories fulfilled: 1, 2, or 1&2

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import comb
from scipy.stats import beta as beta_distr

# data set will be contained in the multi-dimensional array
# D[0][1] = # trials with only category 2 fulfilled

# simulate binary decision process with fixed progenitor pool
# N: number of progenitors per trial
# T: number of trials
# p: probability a progenitor differentiates into state 1
def binary_fixed_sim(N,T,p):
    D = np.zeros((2,2))
    for i in xrange(T):
        t = np.random.binomial(N,p)
        if t == N:
            D[1][0] += 1
        elif t == 0:
            D[0][1] += 1
        else:
            D[1][1] += 1
    return D

N = 3 # number of progenitors per trials

# test data set with N = 3, T = 100, p = 0.5
D = np.array([[0,8],[13,79]])

# get number of trials
T = D.sum() # sum over all elements in D
Ntotal = T * N

# preallocate array for N1 possibilities
N1s = np.zeros(Ntotal+1)
N1s[0] = 1.0

# perform convolution step
def convolute(N1s,updater):
    # use the built in numpy convolve function
    cv = np.convolve(N1s,updater)
    return cv[:len(N1s)].copy() # don't change length of N1s

# iterate over data matrix
it = np.nditer(D, flags=['multi_index'])
while not it.finished:
    dval = it[0]
    dindex = it.multi_index

    w = 2 ** N - 2 # to weight combination trials

    # need to perform convolution on N1s for each recorded trial
    for i in xrange(dval):
        updater = np.zeros(N+1) # hold prob weights to convolute by
        if dindex == (1,0):
            updater[N] += 1
        elif dindex == (0,1):
            updater[0] += 1
        elif dindex == (1,1):
            for j in xrange(1,N):
                updater[j] += comb(N,j) / w
        else: # ignore (0,0)
            pass
        N1s = convolute(N1s,updater)
        N1s = N1s / sum(N1s) # just to maintain normalization
    it.iternext()

# now we have the N1 weights
# for bayesian inference provide alpha,beta for p prior
alpha = 1.5
beta = 1.5

# calculate P(heads|D)
# using means of beta distributions
exp_p = 0.0
for i in xrange(len(N1s)):
    exp_p += N1s[i] * (i + alpha) / (Ntotal + alpha + beta)

# plot distribution
npoints = 300
x = np.linspace(0,1,npoints)
y = np.zeros(npoints)
for i in xrange(len(N1s)):
    y += N1s[i] * beta_distr.pdf(x,i+alpha,Ntotal-i+beta)

plt.plot(x,y)
plt.axvline(x=exp_p)
plt.show()
