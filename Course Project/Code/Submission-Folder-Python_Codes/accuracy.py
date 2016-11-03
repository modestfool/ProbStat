import pandas as pd
import ast

##Read data
df_collaborative_result=pd.read_csv("collaborativeresult.csv")
pred_result=df_collaborative_result["hotel_clus"]
df_content_result=pd.read_csv("Top_10k_users_with_CB.csv",nrows=100)
hotel_cluster=df_collaborative_result["hotel_cluster"]
hotel_cluster1=df_content_result["cb_recommendations"]

##check and compare content based fitering results and collaborative filtering results and find accuracy of model
i=0
count=0
for x in hotel_cluster:
 j=0
 for y in xrange(len(list(ast.literal_eval(df_content_result["cb_recommendations"][i])))):

  if (x in list(ast.literal_eval(df_collaborative_result["hotel_clus"][i])) or x in list(ast.literal_eval(df_content_result['cb_recommendations'][i]))[j] ):
      j=j+1
      count=count+1
 i=i+1
print count,"percent"














