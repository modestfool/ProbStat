# coding: utf-8

# ## Part I: Problem Solving

# ### 1 (b) 

# In[2]:

import scipy.stats as ss


# In[3]:

dice_toss = ss.binom(100,1.0/6.0)


# In[4]:

lower_bound,upper_bound = dice_toss.ppf(0.025),dice_toss.ppf(0.975)


# In[5]:

print lower_bound, upper_bound


# ### 1(c) . Given the following number of rolls until the dice become biased, can you conclude that it is 95% probable that aluminum performs better? (hint: may assume rolls until biased is well approximated as a Normal).
# aluminum rolls until biased = [136, 73, 118, 122, 114, 103, 149, 118, 113, 105]
# plastic rolls until biased = [129, 89, 97, 94, 124, 77, 85, 86, 86, 69]
# 

# In[6]:

aluminium_rolls = [136, 73, 118, 122, 114, 103, 149, 118, 113, 105]
plastic_rolls = [129, 89, 97, 94, 124, 77, 85, 86, 86, 69]


# In[7]:

import numpy as np


# In[8]:

mean1, mean2 = np.mean(aluminium_rolls), np.mean(plastic_rolls)


# In[9]:

print mean1, mean2


# In[10]:

s1, s2 = np.std(aluminium_rolls), np.std(plastic_rolls)
print s1, s2


# In[11]:

n1 = n2 = 10
df1,df2 = n1-1,n2-1


# In[12]:

t = (float)(mean1-mean2)/np.sqrt((s1**2/n1) + (s2**2)/n2)


# In[13]:

print t


# In[14]:

p = 1 - ss.t.cdf(np.abs(t), df1+df2)


# In[15]:

print p


# In[16]:

ss.ttest_ind(aluminium_rolls,plastic_rolls)


# ### 2 (e)

# In[17]:

import numpy

from numpy import matrix

X = matrix ([[-1,2,1],[4,12,1],[3,5,1],[4,6,1],[-3,9,1],[6,7,1]])

R = ((X.T) * X).I * (X.T)

Y = matrix ([[2.5],[10],[8.5],[4],[1.5],[14]])
out = R* Y

print out


# ### 2 (f)

# In[18]:

import math
from __future__ import division
beta_0 = 0.0
beta_p = 0.0
left_handed = [0, 0, 1, 0, 0, 1]
plus_minus = [-1,4,3,4,-3,6]

diag = [0]*6
z = [0]*6
I = ([1]*6)
X = np.matrix([I, plus_minus]).T
#print X
while(1):
    prev_beta_0 = beta_0
    prev_beta_p = beta_p
    for i in range(6):
        q = math.exp(beta_0 + plus_minus[i]*beta_p)
        p = q/(1+q)
        diag[i] = p*(1-p)
        z[i] = math.log(p/(1-p)) + ((left_handed[i] - p)/(p*(1-p)))
    W = np.matrix(np.diag(diag))
    #print W
    Z = np.matrix(z).T
    #print Z
    beta = (((X.T)*W*X).I)*(X.T)*W*(Z)
    beta_p = beta.item(1)
    beta_0 = beta.item(0)
    #print beta
    if (round(beta_p,2) - round(prev_beta_p,2) < 0.01):
        break
print beta


# In[19]:

X = np.matrix ([[-1, 1],[4,1],[3,1],[4,1],[-3,1],[6,1]])
X_t = X.T
W = np.matrix(np.diag([0.25, 0.25, 0.25,0.25,0.25,0.25]))
z = np.matrix([[-2],[-2],[2],[-2],[-2],[2]])
r = X_t*W
r = (r*X).I
q = r*X_t
s = q*W
result = s*z
print result

