import pandas as pd
import ast
##READ DATA
df_content_result=pd.read_csv("Top_10k_users_with_CB.csv",nrows=1000)
hotel_cluster=df_content_result["hotel_cluster"]
hotel_cluster1=df_content_result["cb_recommendations"]

j=0
i=0
hit_count={}
miss_count={}
change_bit=0

for x in xrange(1000):
    j=0
    change_bit = 0
    for y in xrange(len(list(ast.literal_eval(df_content_result["cb_recommendations"][i])))):
        t=list(ast.literal_eval(df_content_result['cb_recommendations'][i]))[j][0]


##hit count
        if (t==hotel_cluster[i]):
         change_bit=1
         if (t in hit_count.keys()):
            hit_count[t] += 1
         else:
            hit_count[t] = 0

##miss count
        if(change_bit!=1):
            if (t in miss_count.keys()):
             miss_count[t] += 1
            else:
             miss_count[t] = 0
        j=j+1
    i=i+1

##Bar Chart Plot

x_values=hit_count.keys()
y_values_hit=hit_count.values()
y_values_miss=miss_count.values()

import plotly.plotly as py
import plotly.graph_objs as go

py.sign_in('Subathra24', 'r97ri0w69j')

trace1 = go.Bar(
    x=x_values,
    y=y_values_hit,
    name='HIT'
)
trace2 = go.Bar(
    x=x_values,
    y=y_values_miss,
    name='MISS'
)
data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)
fig = go.Figure(data=data, layout=layout)
plot_url = py.plot(fig, filename='grouped-bar')


##TOTAL NUMBER OF HITS
sum=0
i=0
for x in range(len(y_values_hit)):
    sum=sum+y_values_hit[i]
    i=i+1
print sum