
# simple binary decision making
# from a variable pool of progenitors
# (distribution of N progenitors per well)
#
# each data sample only indicates
# the categories fulfilled: 1, 2, or 1&2

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import comb
from scipy.stats import beta as beta_distr
import matplotlib.cm as cm
from scipy.special import beta as beta_func
from scipy.signal import convolve2d
from scipy.signal import fftconvolve

# data set will be contained in the multi-dimensional array
# D[0][1] = # trials with only category 2 fulfilled

# simulate binary decision process with variable progenitor pool
# T: number of trials
# p: probability a progenitor differentiates into state 1
# pN: array of probabilities of obtaining N progenitors in a well
#     p(N = 0) should be equal to 0
def binary_variable_sim(T,p,pN):
    D = np.zeros((2,2))
    Ntotal = 0
    total = 0
    for i in xrange(T):
        u = np.random.uniform()
        # this is a hack to select N = i with prob pN[i]
        N = min(np.array(range(len(pN)))[u < np.cumsum(pN)])
        t = np.random.binomial(N,p)
        if t == N:
            D[1][0] += 1
        elif t == 0:
            D[0][1] += 1
        else:
            D[1][1] += 1
        Ntotal += N
        total += t
    return D,Ntotal,total

class BinaryVariableModel:
    # in init perform fit
    # going to assume pN is already reduced (large N combined in single category)
    # or approximated so that we're not looking over 
    # an extremely large number of N possibilities
    # 
    # pN[i] should be the probability of N = i
    def __init__(self,D,pN,alpha=1.5,beta=1.5):
        self.D = D
        self.pN = pN
        # for bayesian inference provide alpha,beta for p prior
        self.alpha = alpha
        self.beta = beta
        self.fit(D,pN)

    # perform convolution step
    def convolute(self,N1s,updater):
        # use the built in numpy convolve function
        cv = np.convolve(N1s,updater)
        return cv[:len(N1s)].copy() # don't change length of N1s

    def convolute2D(self,Ns,updater):
        r,c = Ns.shape
        # use the built in scipy convolve2D function
        cv2D = convolve2d(Ns,updater)
        return cv2D[:r,:c]
        # try fftconvolve for speed
        # 
        # with appropriate thresholding of Ns, it is no longer incorrect
        # 200 trials with 4 N options takes ~63.4 seconds to fit
        # ~57 seconds are spent doing fftn and ifftn back and forth
        # should be able to get a big speedup from a convolution power function
        # that doesn't transform back over and over.
        # cv2DF = fftconvolve(Ns,updater)
        # return cv2DF[:r,:c]

    def get_updater(self,config,pN):
        Nmax = len(pN) - 1
        updater = np.zeros((Nmax+1,Nmax+1)) # hold 2D prob weights to convolute
        if config == (1,0):
            for i in xrange(1,Nmax+1):
                updater[i,i] += pN[i] # all trials contribute to N1
        elif config == (0,1):
            for i in xrange(1,Nmax+1):
                updater[i,0] += pN[i] # all trials contribute to N but not N1
        elif config == (1,1):
            for i in xrange(2,Nmax+1):
                w = 2 ** i - 2 # to weight combination trials
                for j in xrange(1,i):
                    updater[i,j] += pN[i] * comb(i,j) / w

        return updater

    def fit(self,D,pN):
        # get number of trials
        T = int(D.sum()) # sum over all elements in D
        self.T = T # save this

        Nmax = len(pN) - 1 # maximum possible N value in a well
        Ntotalmax = T * Nmax # maximum total possible N's summed up
        self.Ntotalmax = Ntotalmax # save this 

        # preallocate array for (N,N1) possibilities
        # 2D array with (Ntotalmax x Ntotalmax) possibilities
        Ns = np.zeros((Ntotalmax + 1,Ntotalmax + 1))
        Ns[0,0] = 1.0 # without any data full weight on (0,0)
        
        # precompute updaters
        updaters = np.zeros((2,2,Nmax+1,Nmax+1))
        for i in xrange(0,2):
            for j in xrange(0,2):
                updaters[i,j] = self.get_updater((i,j),pN)

        # iterate over data matrix
        it = np.nditer(D, flags=['multi_index'])
        while not it.finished:
            dval = it[0]
            dindex = it.multi_index

            # need to perform convolution on 2D Ns for each recorded trial
            for i in xrange(dval):
                Ns = self.convolute2D(Ns,updaters[dindex])
                Ns = Ns / Ns.sum() # just to maintain normalization
            it.iternext()
        
        # save result
        self.Ns = Ns
        
        # produce proper weights for beta distributions
        bweights = np.zeros((Ntotalmax+1,Ntotalmax+1))
        for i in xrange(Ntotalmax+1):
            for j in xrange(i+1): # can only have at most value N for N1
                # with the fftconvolve this was overweighting things with beta_func
                # values O(0.1), threshold 1e-15 works well for fft
                if Ns[i,j] > 0:
                    bweights[i,j] = Ns[i,j] * beta_func(j + self.alpha,i-j + self.beta)

        # sometimes beta weights are so ridiculously small everything just ends up 0
        # especially I suspect if N is off
        if bweights.sum() > 0:
            bweights = bweights / bweights.sum()
        else:
            bweights = Ns
        self.bweights = bweights
        return Ns
    
    def estimate_p(self):
        if hasattr(self,'exp_p'):
            return self.exp_p
        # calculate P(heads|D)
        # using means of beta distributions
        exp_p = 0.0
        for i in xrange(self.Ntotalmax + 1):
            for j in xrange(i+1):
                exp_p += self.bweights[i,j] * (j + self.alpha) / (i + self.alpha + self.beta)
        self.exp_p = exp_p
        return exp_p

    def estimate_ll(self):
        if not hasattr(self,'exp_p'):
            self.estimate_p()
        exp_p = self.exp_p
        # calculate log-likelihood using this estimate
        # not exactly the max-likelihood estimate, but probably reasonable

        # precalc psums
        psum = np.ones((2,2))
        psum[1,0] = sum([self.pN[i] * (exp_p ** i) for i in xrange(len(self.pN))])
        psum[0,1] = sum([self.pN[i] * ((1-exp_p) ** i) for i in xrange(len(self.pN))])
        psum[1,1] = sum([self.pN[i] * (1 - (exp_p ** i) - ((1 - exp_p) ** i)) for i in xrange(len(self.pN))])

        # iterate over data matrix
        loglik = 0.0
        it = np.nditer(self.D, flags=['multi_index'])
        while not it.finished:
            dval = it[0]
            dindex = it.multi_index
            loglik += dval*np.log(psum[dindex])
            it.iternext()

        self.loglik = loglik
        return loglik

# some basic tests
p = np.random.uniform()
pN = np.array([0.0,0.25,0.25,0.25,0.25])
D,Nt,N1 = binary_variable_sim(50,p,pN)

C = BinaryVariableModel(D,pN)
print p
# print D,Nt,N1
print C.estimate_p()
print np.unravel_index(C.bweights.argmax(),C.bweights.shape)
'''

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

