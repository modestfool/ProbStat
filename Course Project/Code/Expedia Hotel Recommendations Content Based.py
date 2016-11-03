

# coding: utf-8

# In[ ]:

#get_ipython().magic(u'matplotlib inline')
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.decomposition import PCA


# In[ ]:

train_set = pd.read_csv('../train.csv', parse_dates=['srch_ci', 'srch_co'])


# In[ ]:

train_set.describe()


# In[ ]:

users_group = train_set.groupby('user_id')
bookings = users_group.aggregate({'is_booking':np.sum})
print np.sum(bookings)


# In[ ]:

train_set['num_days'] = (train_set['srch_co'] - train_set['srch_ci'])                                                            .values.astype('timedelta64[D]').astype(int)


# ## Get the top 10000 users with most transactions

# In[ ]:

filtered_user_ids_10k = list(train_set['user_id'].value_counts()[0:10000].index)


# In[ ]:

train_set['user_id'].value_counts()[0:10000].tail(5)


# In[ ]:

filtered_train_set_10k = train_set[train_set['user_id'].isin(filtered_user_ids_10k)]


# In[ ]:

filtered_train_set_10k.describe()


# In[ ]:

filtered_train_set_10k['srch_ci'] = pd.to_datetime(filtered_train_set_10k['srch_ci'])
filtered_train_set_10k['srch_co'] = pd.to_datetime(filtered_train_set_10k['srch_co'])


# In[ ]:

temp = pd.to_datetime(filtered_train_set_10k['srch_ci'],errors = 'ignore')
filtered_train_set_10k['srch_ci'] = temp
temp = pd.to_datetime(filtered_train_set_10k['srch_co'],errors = 'ignore')
filtered_train_set_10k['srch_co'] = temp


# In[ ]:

filtered_train_set_10k['num_days'] = (filtered_train_set_10k['srch_co'] - filtered_train_set_10k['srch_ci'])                                                            .values.astype('timedelta64[D]').astype(int)


# In[ ]:

filtered_train_set_10k['num_days'] = filtered_train_set_10k['num_days'].apply(lambda x: abs(x))


# In[ ]:

filtered_train_set_1k ['num_days'] =  (filtered_train_set_1k['srch_co'] - filtered_train_set_1k['srch_ci']).values.astype('timedelta64[D]').astype(int)


# In[ ]:

erroneous = filtered_train_set_10k[filtered_train_set_10k['num_days'] < 0]
len(erroneous)


# In[ ]:

filtered_train_set_10k = filtered_train_set_10k[filtered_train_set_10k['num_days'] >= 0] 


# In[ ]:

temp = pd.to_datetime(train_set['srch_ci'],errors = 'coerce')
train_set['srch_ci'] = temp
temp = pd.to_datetime(train_set['srch_co'],errors = 'coerce')
train_set['srch_co'] = temp


# In[ ]:

temp = pd.to_datetime(test_data['srch_ci'],errors = 'coerce')
test_data['srch_ci'] = temp
temp = pd.to_datetime(test_data['srch_co'],errors = 'coerce')
test_data['srch_co'] = temp


# In[ ]:

test_data['num_days'] = (test_data['srch_co'] - test_data['srch_ci'])                                                            .values.astype('timedelta64[D]').astype(int)


# In[ ]:

filtered_train_set_10k = filtered_train_set_10k.drop('hours_to_checkin',axis=1)


# In[ ]:

filtered_train_set_10k['date_time'] = pd.to_datetime(filtered_train_set_10k["date_time"])


# In[ ]:

filtered_train_set_10k['num_days_to_checkin'] =             (filtered_train_set_10k['srch_ci'] - filtered_train_set_10k['date_time'])                            .values.astype('timedelta64[D]').astype(int)


# In[ ]:

(filtered_train_set_10k['num_days_to_checkin'] < 0).sum()


# In[ ]:

filtered_train_set_10k['num_days_to_checkin'].describe()


# In[ ]:

filtered_train_set_10k['num_days'] = filtered_train_set_10k['num_days'].apply(lambda x: abs(x))


# In[ ]:

filtered_train_set_10k['month'] = filtered_train_set_10k['date_time'].apply(lambda x: x.month)


# In[ ]:

len(filtered_train_set_10k)


# In[ ]:

filtered_train_set_10k = filtered_train_set_10k[filtered_train_set_10k['num_days_to_checkin']<366]
print len(filtered_train_set_10k)


# In[ ]:

filtered_train_set_10k = filtered_train_set_10k[filtered_train_set_10k['num_days'] < 365]


# In[ ]:

filtered_train_set_10k = filtered_train_set_10k.dropna()


# In[ ]:

filtered_train_set_10k['log2_orig_destination_distance'] =                 ((np.log2(filtered_train_set_10k['orig_destination_distance'])).round() + 7)


# In[ ]:

print filtered_train_set_10k['log2_orig_destination_distance'].hist()
print len(filtered_train_set_10k)


# In[ ]:

filtered_train_set_10k['log1p_num_days_to_checkin'] =             (np.log1p(filtered_train_set_10k['num_days_to_checkin'])*10).round()


# In[ ]:

filtered_train_set_10k['log1p_num_days'] =              ((np.log1p(filtered_train_set_10k['num_days']))*10).round()


# ## PCA Model - destinations.csv

# In[ ]:

destinations = pd.read_csv('../destinations.csv')
cols = list(destinations.columns)
cols.remove('srch_destination_id')
destinations[cols] = (destinations[cols] - destinations[cols].mean())/(destinations[cols].max() - destinations[cols].min())
destinations.describe()


# In[ ]:

pca = PCA(n_components=3)
dest_small = pca.fit_transform(destinations[["d{0}".format(i + 1) for i in range(149)]])
dest_small = pd.DataFrame(dest_small,columns=['dest1','dest2','dest3'])
dest_small["srch_destination_id"] = destinations["srch_destination_id"]


# In[ ]:

np.sum(pca.explained_variance_ratio_)


# In[ ]:

dest_small.head()


# In[ ]:

dat = filtered_train_set_10k[:100]
dat.info()


# In[ ]:

dummy = pd.merge(filtered_train_set_10k,dest_small,on = 'srch_destination_id')


# In[ ]:

dummy.info()


# In[ ]:

filtered_train_set_10k = pd.DataFrame(dummy)


# In[ ]:

filtered_train_set_10k.to_csv('../Top_10k_users.csv.gzip',compression='gzip',index=False)


# In[ ]:

shortlist_users =[]
for c in user_id_100:
    print c
    break
    if c == 1:
        shortlist_users.append()
print shortlist_users


# In[ ]:

test_set = pd.read_csv('test.csv')


# In[ ]:

test_set.info()


# # Top 10000 Users

# In[1]:

#get_ipython().magic(u'matplotlib inline')
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import naive_bayes
import operator
from pprint import pprint
from sklearn.decomposition import PCA
import time
from multiprocessing.dummy import Pool as ThreadPool 


# In[80]:

filtered_data = pd.read_csv('../Top_10k_users.csv')


# In[ ]:

#test_data = pd.read_csv('../test.csv')


# In[ ]:

#destinations_data = pd.read_csv('../destinations.csv')


# ## Randomly permute data before proceeding further and filter users with atleast 10 records

# In[81]:

s = filtered_data['user_id'].value_counts()
users_records_10 = s[s >= 10].index


# In[82]:

print users_records_10


# In[85]:

filtered_data = filtered_data[filtered_data['user_id'].isin(users_records_10)]
filtered_data.info()


# In[86]:

filtered_data = filtered_data.iloc[np.random.permutation(len(filtered_data))]


# In[87]:

X_columns = filtered_data.columns


# In[88]:

X_columns = X_columns.drop(['hotel_cluster','date_time','srch_ci','srch_co','dest1','dest2','dest3','is_booking']).astype(list)
print X_columns


# In[5]:

def content_based_nbc (X,Y,X_test,Y_test):
    gnb = naive_bayes.GaussianNB()
    y_predict = gnb.fit(X,Y).predict(X_test)
    y_predict_pb = gnb.predict_proba(X_test)
    accuracy_top = 0.0
    Y_predicted_top = pd.DataFrame(columns=['cb_recommendations'])
    for i in range(len(y_predict_pb)):
        ordered ={}
        ordered['cb_recommendations'] =            sorted(zip (gnb.classes_,y_predict_pb[i]), key= operator.itemgetter(1),reverse=True)[0:5]
        Y_predicted_top = Y_predicted_top.append(ordered,ignore_index = True)
        accuracy_top += (Y_test[i] in dict(ordered['cb_recommendations']).keys())
        #break
        #print "From Top %s,%s,%s" %(ordered[0][0],y_predict[i],Y[i])
    '''%xdel gnb
    %xdel y_predict_pb
    %xdel y_predict
    %xdel ordered'''
    accuracy = (Y_test == y_predict).sum()*1.0
    #accuracy_top = accuracy_top*1.0/len(X_test)
    #print accuracy,accuracy_top
    return (accuracy,accuracy_top,Y_predicted_top)


# In[6]:

print "Before dropping na", len(filtered_data)#, len(test_data)
#filtered_data['orig_destination_distance'] = filtered_data['orig_destination_distance'].fillna(-1)
filtered_data = filtered_data.dropna()
#test_data = test_data.dropna()
filtered_data = filtered_data.reset_index(drop=True)
#test_data = test_data.reset_index(drop=True)
print "After dropping na", len(filtered_data)#, len(test_data)
#filtered_data = filtered_data[filtered_data['num_days']>=0]


# In[ ]:

def get_recommendations(test_users_grouped):
    predictions = pd.DataFrame(columns=['id','hotel_cluster'])
    for user,filtered_test in test_users_grouped:
        try:
            filtered_train = train_grouped.get_group(user)
            
            filtered_train = filtered_train.fillna(0)
            filtered_train = filtered_train.reset_index(drop=True)

            filtered_test = filtered_test.fillna(0)
            filtered_test = filtered_test.reset_index(drop=True)

            X_train = filtered_train[columns]
            Y_train = filtered_train['hotel_cluster']
            X_test = filtered_test[columns]
            X_test_id = filtered_test['id'].astype(int)
            if len(X_train) < 1:
                print "No training data"
                continue
            if len(X_test) < 1:
                print "No test data"
                continue
            pred_dict = {}
            Y_predicted_top = content_based_nbc(X_train,Y_train,X_test)
            for i in range(len(X_test)):
                pred_dict['id'] = int(X_test_id[i])
                pred_dict['hotel_cluster'] = ' '.join([str(x) for x in Y_predicted_top[i]])
                predictions = predictions.append(pred_dict,ignore_index=True)
                #pprint(pred_dict)
            #pprint(Y_predicted_top)
            get_ipython().magic(u'xdel X_train')
            get_ipython().magic(u'xdel X_test')
            get_ipython().magic(u'xdel Y_train')
            get_ipython().magic(u'xdel Y_predicted_top')
        except ValueError as e:
            print e, user
    predictions.to_csv('predictions_group.csv',index=False)


# In[110]:

users = filtered_data['user_id'].unique()
print "Number of unique users: ",len(users)


# In[98]:

def k_fold(filtered_data, k=5):
    accuracy = {}
    avg_accuracy = 0.0
    avg_accuracy_top = 0.0
    
    #users = filtered_data['user_id'].unique()
    filtered_with_recommendations = pd.DataFrame(columns =list(X_columns) + ['hotel_cluster','cb_recommendations'])
    #print len(users)
    filtered_data_grouped = filtered_data.groupby('user_id')
    print len(filtered_data_grouped)
    #for user in users:
    for user,filtered in filtered_data_grouped:
        try:
            filtered_with_recommendations_user = pd.DataFrame(columns =list(X_columns) +                                                               ['hotel_cluster','cb_recommendations'])
            '''filtered = filtered_data[filtered_data['user_id'] == user]

            filtered = filtered.dropna()'''
            filtered = filtered.reset_index(drop=True)
            
            fold_size = len(filtered)/k
            
            avg_accuracy_user = 0.0
            avg_accuracy_top_user = 0.0
            
            skipped_folds = 0
            for i in range(k):
                filtered_train = filtered[:fold_size*(i)].append(filtered[fold_size*(i+1):])
                filtered_test =  filtered[fold_size*(i):fold_size*(i+1)]
                
                filtered_train = filtered_train.reset_index(drop=True)
                filtered_test = filtered_test.reset_index(drop=True)
               
                X_train = filtered_train[X_columns]
                Y_train = filtered_train['hotel_cluster']
                
                X_test = filtered_test[X_columns]
                Y_test = filtered_test['hotel_cluster']
                
                #print X_train.shape, Y_train.shape
                #print X_test.shape, Y_test.shape
                #break
                
                if len(X_train) < 1:
                    skipped_folds +=1
                    #print fold_size, len(filtered)
                    #print "No training data"
                    continue
                if len(X_test) < 1:
                    skipped_folds +=1
                    #print fold_size, len(filtered)
                    #print "No test data"
                    continue
                
                acc,acc_top,Y_predicted_top = content_based_nbc(X_train,Y_train,X_test,Y_test)
                accuracy[user] = acc
                avg_accuracy_user += acc
                avg_accuracy_top_user += acc_top
            if k != skipped_folds:
                avg_accuracy += avg_accuracy_user*1.0/(k-skipped_folds)
                avg_accuracy_top += avg_accuracy_top_user*1.0/(k-skipped_folds)

            #break
        except ValueError as e:
            print e, user
    print avg_accuracy*1.0/len(users),avg_accuracy_top*1.0/len(users)


# In[19]:

def get_cb_recommendations(train_set,test_set):
    accuracy = {}
    avg_accuracy = 0.0
    avg_accuracy_top = 0.0

    filtered_with_recommendations = pd.DataFrame(columns = list(X_columns) + ['hotel_cluster','cb_recommendations'])
    Y_predicted_top = pd.DataFrame(columns=['cb_recommendations'])
    #print len(users)
    train_set_grouped = train_set.groupby('user_id')
    test_set_grouped = test_set.groupby('user_id')
    
    for user,filtered_test in test_set_grouped:
        try:
            filtered_with_recommendations_user = pd.DataFrame(columns = list(X_columns) +                                                              ['hotel_cluster','cb_recommendations'])
            filtered_train = train_set_grouped.get_group(user)
            
            #filtered_train = filtered_train.dropna()
            filtered_train = filtered_train.reset_index(drop=True)

            #filtered_test = filtered_test.dropna()
            filtered_test = filtered_test.reset_index(drop=True)

            X_train = filtered_train[X_columns]
            Y_train = filtered_train['hotel_cluster']
            
            X_test = filtered_test[X_columns]
            Y_test = filtered_test['hotel_cluster']
            
            if len(X_train) < 1:
                print "No training data"
                continue
            if len(X_test) < 1:
                print "No test data"
                continue
            #return values from the NBC model
            acc,acc_top,Y_predicted_top = content_based_nbc(X_train,Y_train,X_test,Y_test)
            
            filtered_with_recommendations_user = pd.concat([X_test,Y_test,Y_predicted_top],axis=1)
            filtered_with_recommendations =                 pd.concat([filtered_with_recommendations,filtered_with_recommendations_user],axis=0,ignore_index=True)
            #print len(X_test),len(Y_test),len(Y_predicted_top),len(filtered_with_recommendations)
            
            accuracy[user] = acc
            avg_accuracy += acc
            avg_accuracy_top += acc_top
            #break
        except ValueError as e:
            print e, user
            
    print avg_accuracy*1.0/len(filtered_with_recommendations)
    print avg_accuracy_top*1.0/len(filtered_with_recommendations)
    return filtered_with_recommendations


# In[27]:

curr_time  = datetime.fromtimestamp(int(time.time()))
print curr_time


# In[28]:

filtered_data_with_recos = get_cb_recommendations(filtered_data,filtered_data)


# In[48]:

filtered_data_with_recos = get_cb_recommendations(filtered_data[:1000],filtered_data[:1000])


# In[108]:

for col in X_columns:
    try:
        k = filtered_data[col][:100000].plot(kind='kde',title = col,legend = True)
        print col
        plt.savefig(col +'.png')
        plt.show()
        #break
    except TypeError as e:
        continue



