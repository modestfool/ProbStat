__author__ = 'sreepradhanathirumalaiswami'

import numpy as np
import pandas as pd
import os

import ast
import sys
#reads the data filtered to top10,000 users
df=pd.read_csv("Top_10k_users.csv")

userhoteldict={}
# this dict maps users to the llist of hotels along with frequencies

usercountrydict={}
#this dict maps user to the list of countries they have visited. This helps in filtering while finding similar users

#loop which populates userhotel and usercountrydict for every user
for i in df['user_id'].unique():
        df1=df[df['user_id']==i]
        userhoteldict[i]=[]
        usercountrydict[i]=[]
        userhoteldict[i].append(sorted(df1['hotel_cluster']))
        usercountrydict[i].append(sorted(df1['hotel_country']))

clusters=[]
#fucntion implementing matrix factorization
def matrix_fact(R, A, B):

        #R is the matrix to be factorized into two matrices A,B
        B=B.T
        e=0
        iter=0
        alp=0.0001#rate of approaching the minimum
        bet=0.01
        dim=3
        #the loop is controlled by error value and is terminated of it exceeds 3000 iterations wothout the error falling below 0.05
        while(e>0.05):
            for i in xrange(len(R)):
                for j in xrange(len(R[i])):
                    if R[i][j]is not 0:
                        #error computed by finding difference between the actual value and computed value
                        eij=R[i][j]-np.dot(A[i,:],B[:,j])
                        for k in xrange(dim):
                            # regularization introduced using bet parameter
                            A[i][k]=A[i][k]+alp*(2*eij*B[k][j]-bet*A[i][k])
                            B[k][j]=B[k][j]+alp*(2*eij*A[i][k]-bet*B[k][j])

            e=0
            for i in xrange(len(R)):
                for j in xrange(len(R[i])):
                    #finding error with respect to all values in the matrix
                    if R[i][j]is not 0:
                        e=e+pow(R[i][j]-np.dot(A[i,:],B[:,j]),2)
                        for k in xrange(dim):
                            e=e+(bet/2)*(pow(A[i][k],2)+pow(B[k][j],2))
            iter+=1

            if iter>3000:
                break
        #returns computed value for resultant matrix which had to be factorized
        return np.dot(A,B)
for l in xrange(1):

    #obtains the parameter, determining for which row the prediction has to be made
    rowselection=os.system(sys.argv[1])

    #obtining userid and country for the user instance prediction of which should be made
    uid= df[rowselection:rowselection+1]['user_id'].iloc[0]
    country=df[rowselection:rowselection+1]['hotel_country'].iloc[0]


    userid=uid

    hotelcountrydict={}
    #dictionary mapping country to the list of hotel_clusters in the country

    #loop for populating hotelcountrydict
    for j in df['hotel_country'].unique():
        df2=df[df['hotel_country']==j]
        hotelcountrydict[j]=[]
        hotelcountrydict[j].append(df2['hotel_cluster'])


    fmat = {}

    i=0
    targetuser=0

    #for every user, the frequency distribution of selecting several clusters is given
    for k in userhoteldict:



        if k==userid:
            targetuser=k

        fmat[k]=[]
        for j in xrange(100):

            fmat[k].append(userhoteldict[k][0].count(j))
        i=i+1





    matfact = []
    #reads the file containing the resukts of KNN implementation=> the userids associated with the top 50 similar users
    KNN=pd.read_csv("knn_top_50.csv",nrows=7810)


    #finding similar users to the target user for whom prediction has to be done
    id=KNN[KNN['user_id']==userid].index
    topind=[]

    selectedusers=[]
    for t in xrange(50):
       topind.append(list(ast.literal_eval (KNN['top_50_neighbors'][id[0]]))[t][0])
    #forms a list topind with the top 50 similar users
    for t in topind:
        #selects only the similar users who have visited the destination earlier for appropriate predictions
        if country in usercountrydict[t][0]:
            selectedusers.append(t)
    #the matrix to be factorized is finalized to contain the rows with values of similar users
    for t in selectedusers[:5]:
        matfact.append(fmat[t])
    #the matrix to be factorized has rows corresponding to rows and cols corresponding to hotel clusters
    R=np.array(matfact)
    N=len(R)
    M=len(R[0])
    # R is afctorized to two matrices A and B
    #Two matrices A,B are assigned random values
    A=np.random.rand(N,3)
    B=np.random.rand(M,3)
    Result = matrix_fact(R,A,B)

    hotelclusters=[]


    for r in Result:
        # End result of matrix factorization is normalized for easy comparision
        r= (r-min(r))/(max(r)-min(r))
        # The top recommendation from every similar user considered is appended to the final list
        hotelclusters.append(list(r).index(max(r)))

    clusters.append(hotelclusters)

# the result of collaborative filtering is written to a file to access for merging with content based
df3=pd.DataFrame()
df3['hotel_clus']=clusters
df3.to_csv("collab.csv", sep=',', encoding='utf-8')
