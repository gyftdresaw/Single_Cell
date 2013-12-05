
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
    
    # direct convolution power calculation
    # for comparison, this should be 'exact'
    def convpower2Ddirect(self,dist,M):
        return reduce(convolve2d,[dist for i in xrange(int(M))])

    # at the very least we can get a speed up to 
    # O(log(M)) convolutions for calculating 
    # a convolution power to M, by doing dividing and conquering
    #
    # return full convolution, i.e. R+(R-1)*(M-1) x C+(C-1)*(M-1) dist
    def convpower2D(self,dist,M):
        final_dists = []
        power_left = int(M)
        current_dist = dist
        while power_left > 0:
            if power_left % 2:
                final_dists.append(current_dist)
            current_dist = fftconvolve(current_dist,current_dist)
            # maintain normalization and positive values
            current_dist = (current_dist * (current_dist > 0))
            current_dist = current_dist / current_dist.sum()
            power_left = power_left >> 1
        # convolve final_dists to get result
        F = reduce(fftconvolve,final_dists)
        F = F * (F > 0)
        F = F / F.sum()
        return F

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
        
        # precompute updaters
        updaters = np.zeros((2,2,Nmax+1,Nmax+1))
        for i in xrange(0,2):
            for j in xrange(0,2):
                updaters[i,j] = self.get_updater((i,j),pN)

        tmp_Ns = []
        # iterate over data matrix
        it = np.nditer(D, flags=['multi_index'])
        while not it.finished:
            dval = it[0]
            dindex = it.multi_index

            # need to perform convolution on 2D Ns for each recorded trial
            if not dval == 0:
                tmp_Ns.append(self.convpower2D(updaters[dindex],dval))
            '''
            for i in xrange(dval):
                Ns = self.convolute2D(Ns,updaters[dindex])
                Ns = Ns / Ns.sum() # just to maintain normalization
            '''
            it.iternext()
        
        Ns = reduce(fftconvolve,tmp_Ns)
        Ns = Ns * (Ns > 0)
        Ns = Ns / Ns.sum()

        # save result
        self.Ns = Ns
        
        # produce proper weights for beta distributions
        bweights = np.zeros((Ntotalmax+1,Ntotalmax+1))
        for i in xrange(Ntotalmax+1):
            for j in xrange(i+1): # can only have at most value N for N1
                # with the fftconvolve this was overweighting things with beta_func
                # values O(0.1), threshold 1e-15 works well for fft
                if Ns[i,j] > 1e-15:
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

# terminating galton-watson process probabilities
# assume start with single cell generation 0
# return vector of probabilities of seeing n cells
# pN[nmax] = probability of seeing nmax or more cells
# pdiff is the probability of differentiating (terminating)
def gw_process(pdiff,nmax):
    pN = np.zeros(nmax+1)
    for i in xrange(1,nmax):
        pN[i] = pdiff + (1 - pdiff)*(pN[i-1] ** 2)
        # this is currently cumulative probability of having N progenitors
    for i in xrange(1,nmax):
        pN[nmax-i] = pN[nmax-i] - pN[nmax-i-1]
    # put rest of probs in last entry
    pN[nmax] = 1 - pN.sum()
    return pN

p = np.random.uniform()
pdiff = np.random.uniform(0.5,1.0)
pN = gw_process(pdiff,15)
D,Nt,N1 = binary_variable_sim(20,p,pN)

# print data
print p,pdiff
print D,Nt,N1

ptry = np.linspace(0.51,0.99,10)
fresults = []
for i in xrange(len(ptry)):
    C = BinaryVariableModel(D,gw_process(ptry[i],6))
    fresults.append((C.estimate_p(),ptry[i],C.estimate_ll()))
    print fresults[-1]
    # print np.unravel_index(C.bweights.argmax(),C.bweights.shape)

# convert log-likelihoods to probabilities
presults = np.zeros(len(fresults))
presults[0] = 1.0
for i in xrange(1,len(fresults)):
    presults[i] = presults[i-1]*np.exp(fresults[i][2] - fresults[i-1][2])

presults = presults / presults.sum()


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

