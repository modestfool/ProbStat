__author__ = 'sreepradhanathirumalaiswami'
import os
import pandas as pd
from random import randint
import ast
#script to obatin results of content based and collaborative filtering and recommed the top 5 results to the user in order
print "content based"
# reads the csv file which contains recommendations from content-based filtering
contentdf=pd.read_csv("Top_10k_users_with_CB.csv")
print "program ended"
argdf=pd.read_csv("Top_10k_users.csv")
#selecting the record(insatnce of user) so as to predict the hotel cluster
rowselection= randint(0,9999)
print "collaborative"

#calls python code for collaborative filtering
os.system("python collaborative.py rowselection")
print "program ended"
#reads file into which the result of collaborative filtering has been written
hotel_clus1=pd.read_csv("collab.csv")
hotel_clus1=hotel_clus1.iloc[:,1]

#print list(ast.literal_eval(contentdf['cb_recommendations'][rowselection]))
matches=[]
for x in ast.literal_eval(hotel_clus1[0]):
    for y in xrange(5):
        #checks for possibility of match between content based(cb) recommendations and collaborative filtering recommendations
        if x in list(ast.literal_eval(contentdf['cb_recommendations'][rowselection]))[y]:
            matches.append(x)
if len(matches)>0:
    print "Found a match between contentbased  and collaborative filtering based recommendation for",list(set(matches))
#with content based filtering giving more accuracy 3 top results of content based are merged with the top two results of collaborative
print "Top recommendations"

for y in xrange(3):
        print list(ast.literal_eval(contentdf['cb_recommendations'][rowselection]))[y][0]
print ast.literal_eval(hotel_clus1[0])
colsel= list(set(ast.literal_eval(hotel_clus1[0])))[:2]
if len(colsel)>1:
    print colsel[1]

