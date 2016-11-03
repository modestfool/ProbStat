#1. Read and tokenize the corpus.

# In[53]:
from __future__ import division
import sys
import os
import pandas as pd
import numpy as np
import csv
import scipy.stats as ss
from happierfuntokenizing import Tokenizer
from BeautifulSoup import BeautifulSoup as Soup
import traceback
import json
import re
import operator
import math

def parse_blogs(path):
    tokenizer = Tokenizer()
    users = {}
    global_words_dict = {}
    industry_map = {}
    total_users = 0
    total_blog_posts = 0
    topics = pd.read_csv('wwbpFBtopics_condProb.csv')
    
    regex = r'<post>[\s\S]*?</post>'
    
    for filename in os.listdir(path):
        parts = filename.split(".")
        #print parts
        features_dict = {}
        words_dict ={}
        user_total_words = 0
        
        user_id = (int)(parts[0])
        gender = parts[1]
        age = (int)(parts[2])
        industry = parts[3]
        star_sign = parts[4]
        
        if user_id in users:
            features_dict = users[user_id]
        
        if industry in industry_map:
            tmp_count = industry_map[industry]
            tmp_count = tmp_count + 1
            industry_map[industry] = tmp_count
        else:
            industry_map[industry] = 1
                
        with open(path + filename, 'r') as user_blog:
            user_blogs = user_blog.read().replace('\n', '').replace('\r', '').replace('\t', '')
    
        #soup = Soup(user_blog)
        blog_posts = re.findall(regex, user_blogs, re.DOTALL)

        total_blog_posts = total_blog_posts + len(blog_posts)
        user_total_words = 0
        #for blog in soup.findAll('post'):
        for blog in blog_posts:  
            words = tokenizer.tokenize(blog.strip())
            user_total_words += len(words)
            
            if 'dict' in features_dict:
                words_dict = features_dict['dict']

            for word in words:
                if word in words_dict:
                    count = words_dict[word]
                    count = count + 1
                    words_dict[word] = count
                else:
                    words_dict[word] = 1
                    
                if word in global_words_dict:
                    count = global_words_dict[word]
                    count = count + 1
                    global_words_dict[word] = count
                else:
                    global_words_dict[word] = 1
                    
        topics_prob = {}

        for i in range(2000):
            prob = 0.0
            topic_dict = topics[topics['category'] == i]

            for row in topic_dict.itertuples():
                word = row[1]
                prob_topic_given_word = row[3]
                if word in words_dict:
                    count_user_word = words_dict[word]
                    prob_word_given_user = count_user_word/user_total_words

                    cur = prob_topic_given_word * prob_word_given_user

                    prob = prob + cur
            topics_prob[i] = prob
        
        features_dict['dict'] = words_dict
        features_dict['age'] = age
        features_dict['industry'] = industry
        features_dict['star_sign'] = star_sign
        features_dict['user_id'] = user_id
        features_dict['topics'] = topics_prob
        features_dict['total_count'] = user_total_words
        features_dict['gender'] = gender
        
        users[user_id] = features_dict
    return (users, global_words_dict, industry_map, total_blog_posts)


users, all_dict, industry_map, num_blogs  = parse_blogs('blogs/')
print "1. a) posts: " ,num_blogs
print "1. b) users: " , len(users)
print "1. c) words: " , len(all_dict)
print "1. d) ", industry_map



# In[142]:

topics = pd.read_csv('wwbpFBtopics_condProb.csv')
topic_map = {}

topic_map[463] = topics[topics["category"]== 463]
topic_map[963] = topics[topics["category"]== 963]
topic_map[981] = topics[topics["category"]== 981]

user_ids = sorted(k for k in users)
lowest_user_ids = sorted(user_ids)[0:3]

print "2.a)"
for userid in lowest_user_ids:
    print "%d" %userid
    
    word_counts_map = users[userid]['dict']

    all_words_wc = 0
    
    for w,c in word_counts_map.iteritems():
        all_words_wc = all_words_wc + word_counts_map[w]
        
    #print word_counts_map
    
    for topic in topic_map:
        print topic," : "
        
        prob = 0
        topic_pd = topic_map[topic]
        
        #print topic_pd
        for index, row in topic_pd.iterrows():
            word = row[0]
            prob_topic_given_word = row[2]

            if word in word_counts_map:
                count_user_word = word_counts_map[word]
                prob_word_given_user = count_user_word / all_words_wc

                #print prob_topic_given_word , prob_word_given_user
                cur = prob_topic_given_word * prob_word_given_user

                prob = prob + cur
            
        print prob


# ### 3 Correlate each topic usage with user age, adjusting for gender.

# In[316]:

beta_topics = {}
beta_gender = {}
beta_significant = {}
beta_p_value = {}

topics_user ={}

alpha = 0.05/2000

for topic in range(2000):
    topics = []
    gender = []
    Y_rows = []
    for user,features in users.iteritems():
        
        if topic in topics_user:
            user_topic = topics_user[topic]
        else:
            user_topic = {}
        
        user_topic[user] = features['topics'][topic]
        
        topics_user[topic] = user_topic
        
        topics.append(features['topics'][topic])
        gender.append(1 if features['gender'] == 'male' else 0)
        Y_rows.append([features['age']])
    try:
       
        topics = (topics - np.mean(topics))/np.std(topics)
        gender = (gender - np.mean(gender))/np.std(gender)
        Y_rows = (Y_rows - np.mean(Y_rows))/np.std(Y_rows)
        
        
        #break
        #X = np.matrix([topics,gender]).T
        X = []
        for i in range(len(topics)):
            X.append([topics[i],gender[i]])
        Y = []
        for i in range(len(topics)):
            Y.append([Y_rows[i]])
        X = np.matrix(X)
        Y = np.matrix(Y_rows)
        Y = (Y - np.mean(Y))/np.std(Y)
        '''print X.shape
        print Y.shape'''
        #break
        beta = (((X.T)*X).I)*(X.T)*Y
        #print beta
        beta_topics[topic] = beta.item(0)
        beta_gender[topic] = beta.item(1)
    except Exception as e:
        #print e, topic, user
        continue
    RSS = 0
    SE_topic = 0
    SE_gender = 0
    mean_topic = np.mean(topics)
    mean_gender = np.mean(gender)

    for i in range(len(users)):
        y_i = Y_rows[i][0]
        x_topic_i = topics[i]
        x_gender_i = gender[i]
        beta_topic_ = beta.item(0)
        beta_gender_ = beta.item(1)
        x_i = (beta_topic_*x_topic_i) + (beta_gender_*x_gender_i)
        
        RSS = RSS + ((y_i- x_i)**2)
        SE_topic =  SE_topic + ((x_topic_i - mean_topic)**2)
        #print y_i, x_i, RSS, SE_topic
    
    n = len(users)
    k = 2
    s_square = RSS/(n-k-1)
    t_topic = beta_topic_/(math.sqrt(s_square/SE_topic))
    p_value = ss.t.sf(np.abs(t_topic),len(users)-2)*2
    beta_p_value[topic] = p_value
    beta_significant[topic] = 'Yes' if p_value < alpha else 'No'

top_positive_topics = sorted(beta_topics.items(), key=operator.itemgetter(1),reverse = True)[:10]
top_negative_topics = sorted(beta_topics.items(), key=operator.itemgetter(1))[:10]
print '3.a)'
for topic, corr in top_positive_topics:
    print 'topic_id: %d, correlation: %0.3f, p-value: %f, signficant after correction? %s'     %(topic,corr,beta_p_value[topic], beta_significant[topic])
print '3. b)' 
for topic, corr in top_negative_topics:
    print 'topic_id: %d, correlation: %0.3f, p-value: %f, signficant after correction? %s'     %(topic,corr,beta_p_value[topic],beta_significant[topic])


# ### 4. Correlate each topic usage with user industry, adjusting for gender and age.

# In[304]:

industry_topic_beta = {}
for industry in industry_map:
    if industry_map[industry] < 30:
        continue
    topics_list = {}
    for topic in range(2000):
        beta_0 = 0.0
        beta_topic= 0.0
        beta_age = 0.0
        beta_gender = 0.0
        topics = []
        industry_row = []
        gender = []
        age = []
        constant = [1]*len(users)
        for user,features in users.iteritems():
            if 'topics' in features:
                topics.append(features['topics'][topic])
            else:
                topics.append(0)
            gender.append(1 if features['gender'] == 'male' else 0)
            age.append(features['age'])
            industry_row.append(0 if features['industry'] == industry else 1)
        
        age = (age - np.mean(age))/np.std(age)
        gender = (gender - np.mean(gender))/np.std(gender)
        topics = (topics - np.mean(topics))/np.std(topics)
        
        X = []
        for i in range(len(users)):
            X.append([topics[i],gender[i],age[i]])
        X = np.matrix(X)
       
        #X = np.matrix([topics,gender,age,constant]).T
        Y = np.matrix(industry_row).T
         
        try:
            while(1):
                prev_beta_0 = beta_0
                prev_beta_topic = beta_topic
                prev_beta_age = beta_age
                prev_beta_gender = beta_gender
                beta_matrix = np.matrix([beta_topic,beta_gender,beta_age,beta_0])
                diag = [0]*len(users)
                z = [0]*len(users)
            
                for i in range(len(users)):
                    q = math.exp(topics[i]*beta_topic + gender[i]*beta_gender + age[i]*beta_age)
                    p = q/(1+q)
                    diag[i] = p*(1-p)
                    z[i] = math.log(p/(1-p)) + ((industry_row[i] - p)/(p*(1-p)))
                    
                W = np.matrix(np.diag(diag))
                #print W
                Z = np.matrix(z).T
                #print Z
                beta = (((X.T)*W*X).I)*(X.T)*W*(Z)

                beta_topic = beta.item(0)
                beta_gender = beta.item(1)
                beta_age = beta.item(2)
                #print beta
                if (round(beta_topic,2) - round(prev_beta_topic,2) < 0.01):
                    break
        except Exception as e:
                print e
                break
                continue
        topics_list[topic] = beta_topic
    #print topics_list
    industry_topic_beta [industry] = topics_list

#print industry_topic_beta



filtered_topics = []
print '4 a).'
for industry,topics_list in industry_topic_beta.iteritems():
    for topic,beta_topic in sorted(topics_list.items(), key=operator.itemgetter(1),reverse = True)[:5]:
        if topic not in filtered_topics:
            filtered_topics.append(topic)
        print 'industry: %s, topic_id: %d, coefficient: %f' %(industry,topic, beta_topic)
print '4 b).'
for industry,topics_list in industry_topic_beta.iteritems():
    for topic,beta_topic in sorted(topics_list.items(), key=operator.itemgetter(1))[:5]:
        if topic not in filtered_topics:
            filtered_topics.append(topic)
        print 'industry: %s, topic_id: %d, coefficient: %f' %(industry,topic, beta_topic)


# ### 5 Plot topics by Age

# In[320]:

#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib

topics_terms_pd = pd.read_csv("2000topics.top20freqs.keys.csv")

Topic_ages_mean = {}

colors =  matplotlib.colors.cnames.keys()[:len(users)]

for topic in filtered_topics:

    topic_users = topics_user[topic]
    
    top_topic_users = sorted(topic_users.items(),key=operator.itemgetter(1),reverse=True)

    ages = []
    num_top_users = int (round(0.25*len(topic_users)))

    for user,beta in top_topic_users[:num_top_users]:
        #print user
        ages.append(users[user]['age'])

    #print ages
    ages_mean = np.mean(ages)
    Topic_ages_mean[topic] = round(ages_mean,3)

fig = plt.figure(figsize=(15,15))
sub_plot = 0
for industry,topics_list in industry_topic_beta.iteritems():

    top5pairs = sorted(topics_list.items(), key=operator.itemgetter(1),reverse = True)[:5]
    bottom5pairs = sorted(topics_list.items(), key=operator.itemgetter(1))[:5]

    top5pairs.extend(bottom5pairs)
    #print top5pairs
    X = []
    Y = []
    topic_terms_map = {}
    for tid in top5pairs:
        
        words_list = []
        topic = tid[0]
        
        X.append(round(tid[1],3))
        Y.append(Topic_ages_mean[topic])
        
        topic_terms = topics_terms_pd.iloc[topic-1]
    
        #print topic_terms.values
        for wd in topic_terms.values:
            if isinstance(wd, str):
                if len(words_list) > 3:
                    break
                else:
                    words_list.append(wd)
                
        topic_terms_map[topic] = words_list
    
    tpc_words_list = topic_terms_map.values()
    
    fig.suptitle(industry, fontsize=14, fontweight='bold')
    sub_plot += 1 
    ax = fig.add_subplot(len(industry_topic_beta),1,sub_plot)
    fig.subplots_adjust(top=20)
    ax.set_title('Topic Correlation with %s Vs Mean Age of top 25 percent users' % (industry))

    ax.set_xlabel('Topic Correlation with %s '% (industry))
    ax.set_ylabel('Mean Age of top 25% users')
    
    for i in range(10):
        #for j in range(len(tpc_words_list[i])):
        ax.text(X[i], Y[i],'\n'.join([','.join(tpc_words_list[i][0:2]),','.join(tpc_words_list[i][2:])]),                 fontsize=12, color = colors[i])
        
    
    ax.axis([np.min(X),np.max(X)*1.5, np.min(Y), np.max(Y)])

plt.savefig("out.png", transparent = True)
#plt.show()




