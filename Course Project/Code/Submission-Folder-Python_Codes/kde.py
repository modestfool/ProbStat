import pandas as pd
import matplotlib.pyplot as plt


##READING DATA
data=pd.read_csv("Top_10k_users.csv");

##KDE PLOT
obj=data['num_days_to_checkin'][:100000].plot(kind='kde',label='num_days_to_checkin')
obj1=data["hotel_cluster"][:100000].plot(ax=obj,kind='kde')
obj2=data["hotel_country"][:100000].plot(ax=obj1,kind='kde')
obj3=data["log1p_num_days_to_checkin"][:100000].plot(ax=obj2,kind='kde')
plt.legend()
plt.savefig('kdeplot.png')
plt.show()