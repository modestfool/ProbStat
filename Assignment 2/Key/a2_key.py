##CSE594: Probability and Statistics for Data Science: Assignment 2
##
## KEY v.1 - H. Andrew Schwartz
###################################################################
#%matplotlib inline

#python includes
import sys
import os
import fnmatch
import re
import csv
import pprint

#standard probability includes:
import numpy as np #matrices and data structures
import scipy.stats as ss #standard statistical operations
import pandas as pd #keeps data organized, works well with data
import matplotlib
matplotlib.use('Agg') #if not plotting to screen
import matplotlib.pyplot as plt #plot visualization
#import statsmodels.api as sm #uncomment for testing

#project specific includes
from happierfuntokenizing import Tokenizer

(_, blogsDir, topic_cp_file, topic_count_file) = sys.argv

#first LOAD topic data:
print "[LOADING TOPIC DATA]"
num_topics = 2000
min_per_industry = 30
pTopicsGivenWord = dict()
with open(topic_cp_file, 'rb') as csvfile:
    topicreader = csv.reader(csvfile)
    topicreader.next()#throw out header
    for row in topicreader:
        word, topic, w = row
        try: 
            pTopicsGivenWord[word][int(topic)]= float(w)
        except KeyError:
            pTopicsGivenWord[word] = {int(topic):float(w)}
wordsForTopic = dict()
with open(topic_count_file, 'rb') as csvfile:
    topicreader = csv.reader(csvfile)
    topicreader.next()#throw out header
    for row in topicreader:
        topic = int(row[0])
        words = row[1:][::2][:4]
        wordsForTopic[topic] = words
print "[DONE]"


################################
#1. READ AND TOKENIZE THE CORPUS
#READ CORPUS
dirFiles = os.listdir(blogsDir)
print "[LOADING CORPUS (%d files) AND CALCULATING TOPIC USAGE]" % len(dirFiles)
tkzer = Tokenizer()
postsRe = re.compile(r'<post>(.*?)</post>', re.DOTALL + re.I) #.*? = non-geedy match
userData = dict() #dictionary of user_id => {age, gender, industry, topics}
filesRead = 0
(numPosts, numWords, industries) = (0, 0, dict()) #for answering question one.
for file in dirFiles:
    if fnmatch.fnmatch(file, '*.xml'):
        user_id, gender, age, industry, zodiac = file.split('.')[:5]
        industry = industry.lower()
        wordCounts = dict()
        totalWords = 0
        currentFile = open(blogsDir+'/'+file).read()
        posts = postsRe.findall(currentFile)
        for post in posts:
            words = tkzer.tokenize(post)
            for word in words:
                try:
                    wordCounts[word] += 1
                except KeyError:
                    wordCounts[word] = 1
            totalWords += len(words)
            numPosts+=1
        numWords+=totalWords
    
        #############################################
        #2. CALCULATE USERS PROBABILITY OF MENTIONING A TOPIC
        fTotalWords = float(totalWords)#for floating point division
        pTopicGivenUser = [0] * num_topics #initial probabilities
        for word, count in wordCounts.iteritems():
            pWordGivenUser = count / fTotalWords
            if word in pTopicsGivenWord:
                for topic, w in pTopicsGivenWord[word].iteritems():
                    pTopicGivenUser[topic] += pWordGivenUser*w
            
        userData[user_id] = {'age': int(age), 
                             'gender': 1 if gender=='female' else 0, 
                             'industry': industry,
                             'topics': pTopicGivenUser}
        filesRead += 1
        if filesRead % 100 == 0:
            print "    %d files read" %filesRead
        try:
            industries[industry] +=1
        except KeyError:
            industries[industry] = 1

print "[DONE]"

###############################################
## OUTPUT FOR 1:

print "1. a) posts: %d" % numPosts
print "1. b) users: %d" % len(userData)
print "1. c) words: %d" % numWords
print "1. d) : %s" % pprint.pformat(industries, indent=8).lstrip('{       ').rstrip('}')
#data = pd.read_csv(file,sep=',',low_memory=False)

###############################################
## OUTPUT FOR 2:
topicsToPrint = [463, 963, 981]
userTopics = dict()

for user_id in sorted(userData.keys())[:3]:
    userTopics.update({user_id: dict([(t, userData[user_id]['topics'][t]) for t in topicsToPrint])})
print "2. a) %s" % str(userTopics)


################################################
#3. Correlate each topic usage with user age, adjusting for gender.

def linReg(X, y, intercept = False):
    #reweighted least squares logistic regression
    #add intercept:
    if intercept:
        X = np.insert(X, X.shape[1], 1, axis=1)

    y = np.array([y]).T #make column

    #fit regression:
    betas = np.dot(np.dot(np.linalg.inv((np.dot(X.T,X))), X.T), y)

    #calculate p-values:
    error = y - (np.dot(X,betas))
    RSS = np.sum(error**2)
    betas = betas.flatten()
    df = float((X.shape[0] - (len(betas) - 1 if intercept else 0)) - 1)
    s2 = RSS / df
    #print s2
    beta_ses = np.sqrt(s2 / (np.sum( (X - np.mean(X,0))**2, 0)))
    #print beta_ses
    ts = [betas[j] / beta_ses[j] for j in range(len(betas))]
    pvalues = (1 - ss.t(df).cdf(np.abs(ts))) * 2 #two-tailed

    ##FOR TESTING:
    #print (betas, pvalues)#DEBUG
    #for comparison purposes:
    #results = sm.OLS(y, X).fit() #DEBUG
    #print (results.params, results.pvalues)
    return betas, pvalues


print "\n[Correlating (multiple linear regression) topics with user age]"

topicBPsWithAge = dict() #topic betas and p-values with age
for topic_id in xrange(num_topics):
    #create the standardized predictors X and dependent variable (y)
    Xy = ss.zscore(np.array([[user['topics'][topic_id], user['gender'], user['age']] for user in userData.itervalues()]))
    #zscore: standardize
    X, y = Xy[:, [0, 1]], Xy[:, 2]
    betas, pvalues = linReg(X, y)
    topicBPsWithAge[topic_id] = (betas[0], pvalues[0])


#################################################
## OUTPUT FOR 3.
def printTopicCorrelInfo (data, alpha=0.05):
    for topic_id, (beta, p) in data:
        p_corrected = p*num_topics #Bonferonni correction
        print "\t topic_id: %s, correlation: %.4f, p-value: %.5f, significant after correction? %s" \
            % (topic_id, beta, p, 'y' if (p_corrected < alpha) else 'n')

sortedTopicBPsWithAge = sorted(topicBPsWithAge.items(), key=lambda x: x[1][0])

print "3.a)"
printTopicCorrelInfo(reversed(sortedTopicBPsWithAge[-10:]))

print "3.b)"
printTopicCorrelInfo(sortedTopicBPsWithAge[:10])


################################################
#4. Correlate each topic usage with user industry, adjusting for gender and age.

def rwLogReg(X, y, precision = .005, intercept = True):
    #reweighted least squares logistic regression
    #add intercept:
    if intercept:
        X = np.insert(X, X.shape[1], 1, axis=1)

    betas = np.zeros((X.shape[1], 1))
    y = np.array([y]).T #make column

    #the convergence loop:
    while True:
        lastbetas = np.array(betas, copy=True)
        p = np.array([ (np.exp(np.dot(X[i],betas)))/(1 + np.exp(np.dot(X[i],betas))) for i in xrange(X.shape[0]) ]).flatten()
        w = [pi * (1 - pi) for pi in p]
        W = np.diag(w)
        z = np.array([ np.dot(X[i],betas) + ((y[i] - p[i])/(p[i]*(1 - p[i]))) for i in xrange(X.shape[0])])
        betas = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(X.T, W), X)), X.T), W),  z)
        #print betas
        if max(np.abs(betas - lastbetas)) < precision:
            break #converged

    betas = betas.flatten()

    ##FOR TESTING:
    #print betas#DEBUG
    ##for comparison purposes:
    #results = sm.Logit(y, X, missing='drop').fit(disp=False) #DEBUG
    #print results.params#DEBUG

    return betas, [0.0]*len(betas)

#find industries with at least 30 users:
industryNames = [k for k, count in industries.items() if count >= min_per_industry]

topicBPsWithIndustry = dict() #topic betas and p-values with age
print " "

for i in industryNames:
    print "[Running Logistic Regression for industry: %s]" % i
    topicBPsWithIndustry[i] = dict()
    for topic_id in xrange(num_topics):
        #create the standardized predictors X and dependent variable (y)
        Xy = np.array([[user['topics'][topic_id], user['gender'], user['age'],
                                  1 if (user['industry'] == i) else 0] 
                                 for user in userData.itervalues()])
        #zscore: standardize
        X, y = ss.zscore(Xy[:, [0, 1, 2]]), Xy[:, 3]
        betas, ps = rwLogReg(X, y)
        topicBPsWithIndustry[i][topic_id] = (betas[0], ps[0])


#################################################
## OUTPUT FOR 4.
def printIndustryTopicCorrelInfo (industry, data, alpha=0.05):
    for topic_id, (beta, p) in data:
        p_corrected = p*num_topics #Bonferonni correction
        print "\t industry: %s, topic_id: %s, correlation: %.4f, p-value: %.5f, significant after correction? %s" \
            % (industry, topic_id, beta, p, 'y' if (p_corrected < alpha) else 'n')


sortedTopicsByIndustry = dict()
outputLimit = 5

toPlot = dict()  #for question 5.
print "4.a)"
for i in industryNames:
    sortedTopicsByIndustry[i] = sorted(topicBPsWithIndustry[i].items(), key=lambda x: x[1][0])
    topicData = list(reversed(sortedTopicsByIndustry[i][-1*outputLimit:]))
    printIndustryTopicCorrelInfo(i, topicData)
    toPlot[i] = topicData

print "4.b)"
for i in industryNames:
    topicData = sortedTopicsByIndustry[i][:outputLimit]
    printIndustryTopicCorrelInfo(i, topicData)
    toPlot[i].extend(topicData)


#################################################
#5. Plot topics by industry x age

#first, figure out top 25% mean age by top quartile of age
print "\n[CALCULATING MEAN 25% AGE FOR NEEDED TOPICS]"
meanAge = dict()
for i, data in toPlot.iteritems():
    for topic_id, (beta, _) in data:
        topicAge = np.array([[user['topics'][topic_id], user['age']] for user in userData.itervalues()])
        sortedTopicAge = topicAge[topicAge[:,0].argsort()]
        meanAge[topic_id] = np.mean(sortedTopicAge[-1*int(.25 * len(topicAge)):][:,1])

print "\n[DRAWING PLOTS]"
#now plot each industry
rows = int((len(toPlot)+1)/2)*4
print rows
fig = plt.figure(figsize=(8, rows))
fig.suptitle('Topics by industry', fontsize=14, fontweight='bold')
n = 1
fig.subplots_adjust(top=0.85)

for i, data in toPlot.iteritems():
    #setup subplot:
    ax = plt.subplot(rows/4,2,n)
    n += 1
    ax.set_title(i)
    ax.set_xlabel('correlation with industry')
    ax.set_ylabel('mean age of top 25%')
    xs = []
    ys = []

    #plot each topic
    for topic_id, (beta, _) in data:
        (x, y) = (beta, meanAge[topic_id])
        words = ','.join(wordsForTopic[topic_id][:2])+"\n"+','.join(wordsForTopic[topic_id][2:])
        #print "(%.3f, %.3f) %s" % (x, y, word)
        c = tuple([r*.4 for r in np.random.rand(3).tolist()]) #set a random color to help distinguish each topic
        ax.text(x, y, words, fontsize=8, color=c, alpha=0.8, verticalalignment='center', horizontalalignment='center')
        xs.append(x)
        ys.append(y)

    ax.axis([min(xs)-.15, max(xs)+.15, min(ys)-.5, max(ys)+.5])
    fig.add_subplot(ax)
    plt.show()

fig.tight_layout()
pngfile = "5a_industry_plots.png"
print "5. a) %s" % pngfile
fig.savefig(pngfile)

