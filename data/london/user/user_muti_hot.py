import numpy as np
import pandas as pd


london_user_city = pd.read_csv('data/user/london_user_city.csv',delimiter=",")




london_user_city_num = london_user_city.apply(lambda x: x.sum())
london_user_city_num.to_csv('data/item/london_user_city_num_num.csv', index=0)
london_user_city_num = pd.read_csv('data/item/london_user_city_num_num.csv',delimiter=",")
london_user_city_num1 = london_user_city_num[london_user_city_num['1']>3]
print(london_user_city_num1.shape)