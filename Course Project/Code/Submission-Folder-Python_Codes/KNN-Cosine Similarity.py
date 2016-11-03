'''
Author : Sindhuja Kancharla
ID : 110283450
Project : Expedia Hotel Recommendations
Course : Probability and Statistics - Final Project

Computes the K Nearest Neighbors for all the users
based on similar attributes.
'''
# coding: utf-8

# In[1]:

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

## Grouping rows by user_id
grouped_users_data = data.groupby("user_id")


# In[4]:

grouped_users_data.head()


# In[5]:

#The set of features taken into consideration for computing cosine similarity between users
features_for_cosine_sim = ["user_location_country","user_location_city","is_mobile","hotel_cluster"
                           ,"is_package","posa_continent","site_name","user_location_region","channel",
                           "srch_destination_id","srch_destination_type_id","is_booking","hotel_continent",
                           "hotel_country","hotel_market","srch_adults_cnt","srch_children_cnt","srch_rm_cnt",
                          "log2_orig_destination_distance","log1p_num_days_to_checkin","log1p_num_days"]


# In[6]:

features_summary = pd.DataFrame(columns=features_for_cosine_sim)


# In[7]:

# For every user, compute a normalized score for every value for every attribute.
for user_id, group_user_id in grouped_users_data:
    feat_freq_dict = {}
    
    for feature in features_for_cosine_sim:
        value_counts_dict = dict(group_user_id[feature].value_counts(normalize= True).round(decimals=5))
        feat_freq_dict[feature] = value_counts_dict
        
    feat_freq_dict['user_id'] = user_id
    features_summary = features_summary.append(feat_freq_dict,ignore_index=True)
    #break


# In[126]:

features_summary.to_csv("features_summary.csv")


# In[81]:

knn_df_columns = ["user_id","list_of_neighbors"]
knn_df = pd.DataFrame(columns=knn_df_columns)


# In[82]:

# Computing the cosine similarity between any two users
for index1,row1 in features_summary.iterrows():
    users_neighbors_map = {}
    user1 = int(row1["user_id"])
    users_neighbors_map["user_id"] = user1

    user_score_map = {}
    for index2,row2 in features_summary.iterrows():
        user2 = int(row2["user_id"])
        score = 0.0
        if index1 > index2:
            user1_list = knn_df[knn_df["user_id"]==user2]
            user1_list = dict(user1_list["list_of_neighbors"])
            user_score_map[user2] = user1_list[index2][user1]
            continue
        else:

            for feature in features_for_cosine_sim:
                dict1 = row1[feature]
                dict2 = row2[feature]
               
                xy = 0.0
                x2 = 0.0
                y2 = 0.0
                for key1, val1 in dict1.iteritems():
                    x2 += val1*val1
                    if key1 in dict2.keys():
                        val2 = dict2[key1]
                        xy += val1*val2
                for key2,val2 in dict2.iteritems():
                    y2 += val2*val2

                score += xy/(x2*y2)
            user_score_map[user2] = score

    users_neighbors_map["list_of_neighbors"] = user_score_map
        
    knn_df = knn_df.append(users_neighbors_map,ignore_index=True)
    print len(knn_df) 


# In[86]:

knn_df.to_csv("knn.csv.gzip",compression="gzip")


# In[127]:

top_fifty_neighbors_df = pd.DataFrame(columns=["user_id","top_50_neighbors"])


# In[128]:

# Extracting only top 50 neighbors for every user and saving them in a file for Matrix Factorization method
for indexid, all_neighbors in knn_df.iterrows():
    top_dict = {}
    
    sorted_top_ten = sorted(all_neighbors[1].items(), key=operator.itemgetter(1), reverse=True) [:50]  
    
    top_dict["top_50_neighbors"] = sorted_top_ten
    top_dict["user_id"] = all_neighbors[0]
    top_fifty_neighbors_df = top_fifty_neighbors_df.append(top_dict, ignore_index=True)


# In[129]:

top_fifty_neighbors_df.head()


# In[130]:

top_fifty_neighbors_df.to_csv("knn_top_50.csv")
