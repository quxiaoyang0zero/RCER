import numpy as np
import pandas as pd


london_user_city = pd.read_csv('data/user/london_user_city.csv',delimiter=",")



# london_user_city_num = london_user_city.apply(lambda x: x.sum())
# london_user_city_num.to_csv('data/user/ .csv', index=1)
london_user_city_num = pd.read_csv('data/user/london_user_city_num.csv',delimiter=",")


london_user_city_num1 = london_user_city_num[london_user_city_num['num']>3]


city_feature_list = []

city_feature_list = london_user_city_num1['feature'].values.tolist()


city_new = pd.DataFrame(london_user_city,columns=city_feature_list)
print(city_new)
city_new.to_csv('data/user/london_user_city_new3.csv', index=0)