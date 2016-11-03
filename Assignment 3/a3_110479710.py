
# coding: utf-8

# In[293]:

import pandas as pd
import numpy as np
import scipy.stats as ss
import sklearn
import csv
import string
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
import sys
#get_ipython().magic(u'matplotlib inline')


# ## 1. Read in the CSV - counties with at least 30,000 people, Only columns ending in “Value”,  Only with non-nan values for each value column

# In[294]:

if(len(sys.argv) != 2):
    input_csv = '2015_CHR_Analytic_Data.csv'
else:
    input_csv = sys.argv[1]


# In[295]:

data = pd.read_csv(input_csv,low_memory = False)


# In[296]:

data.head()


# In[297]:

columns = data.columns


# In[298]:

filtered_columns = list()
filtered_columns.append('COUNTYCODE')
for c in columns:
    if c.endswith("Value"):
           filtered_columns.append(c)


# In[299]:

data = data[filtered_columns]
#print data.head()


# In[300]:

for c in filtered_columns:
    if not (data[c].dtype == np.float64 or data[c].dtype == np.int64):
        data[c] = data[c].apply(lambda val: float(string.replace(str(val),',',''))) ##change to string then floats...
        data[c] = data[c].astype('float')


# In[301]:

data = data[data['COUNTYCODE'] != 0]


# In[302]:

data = data[data['2011 population estimate Value'] >= 30000]
data = data.dropna()


# In[303]:


print "1. TOTAL NUMBER OF COUNTIES: %d" %(len(data))


# ## 2. Create a new column, ‘log_paamv’ which is the log transform of 'Premature age-adjusted mortality Value'.

# In[304]:

data['log_paamv'] = np.log(data['Premature age-adjusted mortality Value'])


# In[305]:

loghist = data['log_paamv'].hist(bins = 25)
loghist.set_title('log_paamv HISTOGRAM')
loghist.get_figure().savefig('2. log_paamv HISTOGRAM: 2histogram.png')
print "2. log_paamv HISTOGRAM: 2histogram.png - Saved "
#plt.show()


# ## 3. Predict “log_paamv” (y) using all “Value” variables 

# In[306]:

removed_columns = ['COUNTYCODE', 'log_paamv', 'Premature age-adjusted mortality Value', 'Premature death Value',                'Uninsured adults Value', 'Teen births Value', 'Food insecurity Value', 'Physical inactivity Value',                   'Adult smoking Value', 'Injury deaths Value', 'Motor vehicle crash deaths Value',                    'Drug poisoning deaths Value',  'Child mortality Value', 'Uninsured Value']
#print len(removed_columns)


# In[307]:

prediction_columns = []
for f in filtered_columns:
    if f not in removed_columns:
        prediction_columns.append(f)
#print prediction_columns, len(prediction_columns)


# In[308]:

shuffled_data = data.reindex(np.random.permutation(data.index))
shuffled_data = shuffled_data.reset_index(drop=True)
data_stdzd = (shuffled_data - shuffled_data.mean())/shuffled_data.std()
#print data.info()
#print data_stdzd.info()


# In[309]:

X_ = data[prediction_columns]
Y_ = data['log_paamv']
X_ = ss.zscore(np.array(X_),ddof = 1)
Y_ = ss.zscore(np.array(Y_),ddof = 1)


# In[310]:

X_data = pd.DataFrame(data_stdzd[prediction_columns])
Y_data = pd.DataFrame(data_stdzd['log_paamv'])


# In[311]:

#print len(X_data)


# In[312]:

def k_cross_fold(len_data,X,y,method="linear",k=10,include_dev_set = False):
    fold_size = len_data/k
    mse = 0.0
    if method == "linear" or method == "pca":
        model = linear_model.LinearRegression()
    elif method == "ridge":
        model = linear_model.Ridge()
    elif method == "lasso":
        model = linear_model.Lasso()
    for i in range(0,k-1):
        if method == "pca":
            X_train = X[:fold_size*(i)].append(X[fold_size*(i+2):])
            Y_train= y[:fold_size*(i)].append(y[fold_size*(i+2):])
            X_dev = X[fold_size*i : fold_size*(i+1)]
            Y_dev = y[fold_size*i : fold_size*(i+1)]
            X_test = X[fold_size*(i+1) : fold_size*(i+2)]
            Y_test = y[fold_size*(i+1) : fold_size*(i+2)]
            n_components = 1
            least_err = 100
            best_n_components = 1
            while n_components < len(X.columns):
                pca = PCA(n_components = n_components)
                pca.fit(X)
                X_train_ = pd.DataFrame(pca.transform(X_train))
                X_dev_ = pd.DataFrame(pca.transform(X_dev))
                model.fit(X_train_,Y_train)
                err = np.mean((model.predict(X_dev_) - Y_dev)**2)
                if (least_err - err[0]) > 0.00001:
                    least_err = err[0]
                    best_n_components = n_components
                n_components += 1
            #print best_n_components
            pca = PCA()
            pca.set_params(n_components = best_n_components)
            pca.fit(X_train)
            X_train = pd.DataFrame(pca.transform(X_train))
            X_test = pd.DataFrame(pca.transform(X_test))
            model.fit(X_train,Y_train)
            err = np.mean((model.predict(X_test) - Y_test)**2)
            #print err , i
        elif method == "ridge" or method == "lasso":
            X_train = X[:fold_size*(i)].append(X[fold_size*(i+2):])
            Y_train= y[:fold_size*(i)].append(y[fold_size*(i+2):])
            X_dev = X[fold_size*i : fold_size*(i+1)]
            Y_dev = y[fold_size*i : fold_size*(i+1)]
            X_test = X[fold_size*(i+1) : fold_size*(i+2)]
            Y_test = y[fold_size*(i+1) : fold_size*(i+2)]
            least_err = 100
            alphas = [(10**i)/10000.0 for i in xrange(-6,6)]
            best_alpha = 0
            for alpha in alphas:
                #print "alpha %f"%alpha
                model.set_params(alpha=alpha)
                model.fit(X_train,Y_train)
                y_hat = model.predict(X_dev)
                y_hat = pd.DataFrame(y_hat)
                err = np.mean((y_hat.values - Y_dev.values)**2)
                #print err
                if (least_err - err) > 0:
                    least_err = err
                    best_alpha = alpha
            #print best_alpha
            model.set_params(alpha=best_alpha)
            model.fit(X_train,Y_train)
            y_hat = model.predict(X_test)
            y_hat = pd.DataFrame(y_hat)
            
            err = np.mean((y_hat.values - Y_test.values)**2)
            #print err
        else:
            X_train = X[:fold_size*(i)].append(X[fold_size*(i+1):])
            Y_train= y[:fold_size*(i)].append(y[fold_size*(i+1):])
            X_test = X[fold_size*i : fold_size*(i+1)]
            Y_test = y[fold_size*i : fold_size*(i+1)]
            model.fit(X_train,Y_train)
            err = np.mean((model.predict(X_test) - Y_test)**2)
        mse += err
    mean_squared_error = (mse/(k-1))
    #print "3. Non-regularized Linear Regression MSE: %0.3f" % mean_squared_error
    return mean_squared_error


# In[313]:

reg = linear_model.LinearRegression()
print "3. Non-regularized Linear Regression MSE: %0.3f" % k_cross_fold(len(X_data),X_data,Y_data,k=10)


# ## 4. Run PCA 
#     Print the percentage of variance explained for each of the first 3 components

# In[314]:

pca = PCA(n_components=3)
pca.fit(X_data)
print "4. Percentage variance explained of first three components: %s"         %str(np.round(pca.explained_variance_ratio_*1000)/1000)


# ## 5. Run regularized predictions 

# In[315]:

mse = k_cross_fold(len(X_data), X_data,Y_data,method="pca", k=10, include_dev_set=False)
print "5. a) principal components regression mse: %0.3f" %mse


# In[316]:

'''n_components = 1
least_mse = 100
best_n_components = 1
reg = linear_model.LinearRegression()
while n_components < len(X_data.columns):
    pca = PCA(n_components = n_components)
    pca.fit(X_data)
    X_pca = pd.DataFrame(pca.transform(X_data))
    mse = k_cross_fold(len(X_pca), X_pca,Y_data,k=10, include_dev_set=True)
    if (least_mse - mse[0]) > 0.001:
        least_mse = mse[0]
        best_n_components = n_components
    n_components *= 2
print "Best number of components %d" %best_n_components
pca = PCA(n_components= best_n_components)
pca.fit(X_data)
X_pca = pd.DataFrame(pca.transform(X_data))
mse = k_cross_fold(len(X_pca), X_pca,Y_data,k=10, include_dev_set=False)
print "5. a) principal components regression mse: %0.3f" %mse'''


# In[317]:

mse = k_cross_fold(len(X_data),X_data,Y_data,method="ridge",k=10,include_dev_set=False)
print "5. b) L2 regularized mse: %0.4f" %mse


# In[318]:

mse = k_cross_fold(len(X_data),X_data,Y_data,method="lasso",k=10,include_dev_set=False)
print "5. c) L1 regularized mse: %0.4f" %mse


# In[ ]:



