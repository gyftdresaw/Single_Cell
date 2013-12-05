
import numpy as np
from binary_fixed_pool import BinaryFixedModel
from binary_variable_pool import BinaryVariableModel
import matplotlib.pyplot as plt
from scipy.misc import comb
from scipy.stats import beta as beta_distr
from scipy.stats import norm
from scipy.stats import binom
import matplotlib.cm as cm
from scipy.special import beta as beta_func

# trying to do some fitting on the plzf single cell data
D = np.zeros((2,2,2)) # three cell options
D[1,0,0] = 185.
D[0,1,0] = 99.
D[0,0,1] = 49.
D[1,1,0] = 9.
D[0,1,1] = 11.
D[1,0,1] = 19.
D[1,1,1] = 6.

# first some binary fits on the aggregate data
# agg_D[i] is aggregate data of ith cell vs rest
agg_D = np.zeros((3,2,2))
# just write out the different possibilities
# for p1 vs rest
agg_D[0][0,1] = D[0,0:2,0:2].sum()
agg_D[0][1,0] = D[1,0,0]
agg_D[0][1,1] = D[1,0:2,0:2].sum() - D[1,0,0]
# for p2 vs rest
agg_D[1][0,1] = D[0:2,0,0:2].sum()
agg_D[1][1,0] = D[0,1,0]
agg_D[1][1,1] = D[0:2,1,0:2].sum() - D[0,1,0]
# for p3 vs rest
agg_D[2][0,1] = D[0:2,0:2,0].sum()
agg_D[2][1,0] = D[0,0,1]
agg_D[2][1,1] = D[0:2,0:2,1].sum() - D[0,0,1]

'''
## fixed N fitting to aggregate data
for i in xrange(3):
    print 'p%d' % i
    for j in xrange(2,16):
        B = BinaryFixedModel(agg_D[i],j)
        print j,B.estimate_p(),B.estimate_ll()
'''
# we're gonna go with N=3
N = 3
ps = np.zeros(3)
Bs = [None for i in xrange(3)]
for i in xrange(3):
    Bs[i] = BinaryFixedModel(agg_D[i],N)
    ps[i] = Bs[i].estimate_p()

# generate p distribution plots and CIs
# colors = cm.Dark2(np.linspace(0,1,len(samples)))
npoints = 10000
x = np.linspace(0,1,npoints)
ydist = np.zeros((3,npoints))
for i in xrange(3):
    f = plt.figure()
    for j in xrange(len(Bs[i].bweights)):
        if Bs[i].bweights[j] > 0:
            bd = beta_distr.pdf(x,j+Bs[i].alpha,Bs[i].Ntotal-j+Bs[i].beta)
            if np.isnan(bd).any():
                a = j + Bs[i].alpha
                b = Bs[i].Ntotal-j+Bs[i].beta
                bd = norm.pdf(x,a/(a+b),np.sqrt((a*b)/((a+b)*(a+b)*(a + b + 1))))
            ydist[i] += Bs[i].bweights[j] * bd
    
    plt.plot(x,ydist[i])
    plt.xlabel('p')
    f.savefig('p%d_dist.png' % i)
    plt.show()

# get an estimate 'CI'
CIs = np.zeros((3,2))
for i in xrange(3):
    lindex = min(np.array(range(len(ydist[i])))[0.05 < np.cumsum(ydist[i]/ydist[i].sum())])
    hindex = min(np.array(range(len(ydist[i])))[0.95 < np.cumsum(ydist[i]/ydist[i].sum())])
    CIs[i][0] = x[lindex]
    CIs[i][1] = x[hindex]

# rough p-value for aggregate mixed wells
for i in xrange(3):
    T = agg_D[i].sum()
    p = Bs[i].estimate_p()
    print binom.cdf(agg_D[i][1,1],T,(1 - p**3 - (1-p)**3))

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

plt.bar(range(100+1),gw_process(0.6,100),align='center')
plt.show()

# try some fitting to ilc data
ptry = np.linspace(0.51,0.99,49)

# for each aggregated data set
# try to fit galton watson process
fresults = []
for j in xrange(3):
    fresults.append([])
    for i in xrange(len(ptry)):
        C = BinaryVariableModel(np.round(agg_D[j]/2),gw_process(ptry[i],6))
        fresults[j].append((C.estimate_p(),ptry[i],C.estimate_ll()))
        print fresults[j][-1]
        # print np.unravel_index(C.bweights.argmax(),C.bweights.shape)

# convert log-likelihoods to probabilities
presults = np.zeros((len(fresults[0]),3))
presults[0,:] = 1.0
for j in xrange(3):
    for i in xrange(1,len(fresults[0])):
        presults[i,j] = presults[i-1,j]*np.exp(fresults[j][i][2] - fresults[j][i-1][2])

for j in xrange(3):
    presults[:,j] = presults[:,j] / presults[:,j].sum()

# plot these distributions for pdiff
for j in xrange(3):
    f = plt.figure()
    plt.plot(ptry,presults[:,j])
    plt.xlabel('prob differentiation')
    f.savefig('pdiff_aggregate_ilc%d.png' % (j+1))
    plt.show()
