'''
Author : Sindhuja Kancharla
ID : 110283450
Project : Expedia Hotel Recommendations
Course : Probability and Statistics - Final Project

Identify the correlation between all the attributes and
Discover some important patterns.
'''
# coding: utf-8

# In[1]:

#get_ipython().magic(u'matplotlib inline')

#python includes
import sys

#standard probability includes:
import numpy as np #matrices and data structures
import scipy.stats as ss #standard statistical operations
import pandas as pd #keeps data organized, works well with data
import matplotlib
import matplotlib.pyplot as plt #plot visualization
from sklearn.naive_bayes import GaussianNB
import math
from datetime import datetime
import operator


# In[2]:

filename = 'data/Top_10k_users.csv'
data = pd.read_csv(filename)
data = data.dropna()
data = data.reset_index(drop=True)


# In[3]:

features = ["user_location_country","user_location_city","is_mobile","hotel_cluster"
                           ,"is_package","posa_continent","site_name","user_location_region","channel",
                           "srch_destination_id","srch_destination_type_id","is_booking","hotel_continent",
                           "hotel_country","hotel_market","srch_adults_cnt","srch_children_cnt","srch_rm_cnt",
                          "log2_orig_destination_distance","log1p_num_days_to_checkin","log1p_num_days"]


# In[12]:

feature_data = data[features]
feature_data.head()


# ### Correlate each hotel cluster with features

# In[64]:

N = len(data)

alpha = 0.01/N
correlations = {}
for i in range(len(features)):
    feat1 = features[i]
    hotel_market_data = data[feat1]
    X = (hotel_market_data - np.mean(hotel_market_data)) / np.std(hotel_market_data)

    for j in range(i,len(features)):
        feat2 = features[j]
        if feat1==feat2:
            continue
        srch_children_count = data[feat2]

        Y = (srch_children_count - np.mean(srch_children_count)) / np.std(srch_children_count)
        
        corr = Y.corr(X)
        t = corr*np.sqrt((N-2)/(1-corr**2))
        pval = 1-ss.t.cdf(abs(t),N-2)
        
        # Consider only significant correlations for discovery
        if pval < alpha:
            correlations[feat1+","+feat2] = Y.corr(X)
        else:
            continue


print "Most Negative Correlated"

sorted_pos = sorted(correlations.items(), key=operator.itemgetter(1))[:30]
sorted_neg = sorted(correlations.items(), key=operator.itemgetter(1), reverse= True)[:30]

for i in sorted_pos:
    print i
    
print "Most Positive Correlated"
for i in sorted_neg:
    print i


# In[55]:

import seaborn as sns

sns.set(style="white")

# Compute the correlation matrix
corr = feature_data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(16, 16))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 5, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
hm = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
            square=True, xticklabels=corr.columns, yticklabels=corr.columns,
            linewidths=.3, cbar_kws={"shrink": .3}, ax=ax)

f.savefig("correlation.png")



