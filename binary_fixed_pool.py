
# simple binary decision making
# from a fixed pool of progenitors
#
# each data sample only indicates
# the categories fulfilled: 1, 2, or 1&2

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import comb
from scipy.stats import beta as beta_distr
import matplotlib.cm as cm
from scipy.special import beta as beta_func

# data set will be contained in the multi-dimensional array
# D[0][1] = # trials with only category 2 fulfilled

# simulate binary decision process with fixed progenitor pool
# N: number of progenitors per trial
# T: number of trials
# p: probability a progenitor differentiates into state 1
def binary_fixed_sim(N,T,p):
    D = np.zeros((2,2))
    total = 0
    for i in xrange(T):
        t = np.random.binomial(N,p)
        if t == N:
            D[1][0] += 1
        elif t == 0:
            D[0][1] += 1
        else:
            D[1][1] += 1
        total += t
    return D,total

class BinaryFixedModel:
    # in init perform fit
    def __init__(self,D,N,alpha=1.5,beta=1.5):
        self.D = D
        self.N = N
        # for bayesian inference provide alpha,beta for p prior
        self.alpha = alpha
        self.beta = beta
        self.fit(D,N)

    # perform convolution step
    def convolute(self,N1s,updater):
        # use the built in numpy convolve function
        cv = np.convolve(N1s,updater)
        return cv[:len(N1s)].copy() # don't change length of N1s

    def fit(self,D,N):
        # get number of trials
        T = D.sum() # sum over all elements in D
        self.T = T # save this
        Ntotal = T * N
        self.Ntotal = Ntotal # save this 

        # preallocate array for N1 possibilities
        N1s = np.zeros(Ntotal+1)
        N1s[0] = 1.0

        w = 2 ** N - 2 # to weight combination trials

        # iterate over data matrix
        it = np.nditer(D, flags=['multi_index'])
        while not it.finished:
            dval = it[0]
            dindex = it.multi_index

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
                N1s = self.convolute(N1s,updater)
                N1s = N1s / sum(N1s) # just to maintain normalization
            it.iternext()
        
        # save result
        self.N1s = N1s
        # produce proper weights for beta distributions
        bweights = np.zeros(Ntotal+1)
        for i in xrange(len(N1s)):
            if N1s[i] > 0:
                bweights[i] = N1s[i] * beta_func(i + self.alpha,Ntotal-i + self.beta)

        # sometimes beta weights are so ridiculously small everything just ends up 0
        # especially I suspect if N is off
        if sum(bweights) > 0:
            bweights = bweights / sum(bweights)
        else:
            bweights = N1s
        self.bweights = bweights
        return N1s
    
    def estimate_p(self):
        if hasattr(self,'exp_p'):
            return self.exp_p
        # calculate P(heads|D)
        # using means of beta distributions
        exp_p = 0.0
        for i in xrange(len(self.bweights)):
            exp_p += self.bweights[i] * (i + self.alpha) / (self.Ntotal + self.alpha + self.beta)
        self.exp_p = exp_p
        return exp_p

    def estimate_ll(self):
        if not hasattr(self,'exp_p'):
            self.estimate_p()
        exp_p = self.exp_p
        # calculate log-likelihood using this estimate
        # not exactly the max-likelihood estimate, but probably reasonable

        # iterate over data matrix
        loglik = 0.0
        it = np.nditer(self.D, flags=['multi_index'])
        while not it.finished:
            dval = it[0]
            dindex = it.multi_index

            if dindex == (1,0):
                loglik += dval*self.N*np.log(exp_p)
            elif dindex == (0,1):
                loglik += dval*self.N*np.log(1-exp_p)
            elif dindex == (1,1):
                loglik += dval*np.log(1 - exp_p ** self.N - (1 - exp_p) ** self.N)
            else: # ignore (0,0)
                pass
            it.iternext()
        self.loglik = loglik
        return loglik

# N is number of progenitors
# test data set with N = 3, T = 100, p = 0.5
# D = np.array([[0,8],[13,79]])
p = np.random.uniform()

N = 3
samples = [10,20,70]
colors = cm.Dark2(np.linspace(0,1,len(samples)))
f = plt.figure()
print p
D = np.zeros((2,2))
N1 = 0
ps = [None for i in xrange(len(samples))]
for i in xrange(len(samples)):
    Dnext,N1next = binary_fixed_sim(N,samples[i],p)
    D += Dnext
    N1 += N1next
    print D,N1
    # fit and plot
    B = BinaryFixedModel(D,N,alpha=1.5,beta=1.5)
    print B.estimate_p()
    npoints = 300
    x = np.linspace(0,1,npoints)
    y = np.zeros(npoints)
    for j in xrange(len(B.bweights)):
        if B.bweights[j] > 0:
            y += B.bweights[j] * beta_distr.pdf(x,j+B.alpha,B.Ntotal-j+B.beta)
        
    yn = beta_distr.pdf(x,N1+B.alpha,B.Ntotal-N1+B.beta)
    
    ps[i] = plt.plot(x,y,color=colors[i])[0] # this is a horrible hack for the legend
    plt.plot(x,yn,'--',color=colors[i])
    # plt.axvline(x=B.estimate_p(),color=colors[i])

plt.legend(ps,np.cumsum(samples))
plt.axvline(x=p,color='red')
plt.xlabel('p')
f.savefig('converge.png')
plt.show()

# plot the log likelihoods
logliks = [BinaryFixedModel(D,i).estimate_ll() for i in xrange(2,16)]
for i in xrange(2,16):
    B = BinaryFixedModel(D,i)
    print i,B.estimate_p(),B.estimate_ll()
f = plt.figure()
plt.plot(range(2,16),logliks)
plt.title('log-likelihood')
plt.xlabel('N')
f.savefig('loglik.png')
plt.show()

'''
# now plot distribution of N's predicted
trials = 5
ncounts = np.zeros(16)
for i in xrange(trials):
    D,N1 = binary_fixed_sim(3,100,np.random.uniform())
    logliks = np.array([BinaryFixedModel(D,j).estimate_ll() for j in xrange(2,16)])
    ncounts[logliks.argmax()+2] += 1

f = plt.figure()
plt.bar(range(16),ncounts/trials,align='center')
f.savefig('Ndistr.png')
plt.show()
'''

'''
# plot distribution
# this isn't going to work for something with N1s > 250
# since the beta_distr can't handle such large parameters
# if really wanted to can just use normal approx probably
npoints = 300
x = np.linspace(0,1,npoints)
y = np.zeros(npoints)
for i in xrange(len(N1s)):
    y += N1s[i] * beta_distr.pdf(x,i+alpha,Ntotal-i+beta)

plt.plot(x,y)
plt.axvline(x=exp_p)
plt.show()
'''

