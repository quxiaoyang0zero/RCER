import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split

# (16315, 1)
# (16315, 6)
# (16315, 2)
# (16315, 5)
# (16315, 19)
# (16315, 125)
# (16315, 561)
# (16315, 719)
# 718
# # #user id
# user = pd.read_csv('./data/user/london_user.csv',delimiter=",")
# user = user.drop_duplicates(keep = 'first')
# user_id = user['uid_index']
# user_id.columns=['uid']  
# user_id.to_csv('./data/user/london_user_id.csv', index=0)
# #user age
# user_age = user['uage']
# user_age = user_age.str.replace(' ', '_')
# user_age = pd.get_dummies(user_age,prefix='uage')
# user_age.to_csv('./data/user/london_user_age.csv', index=0)

# #user gender
# user_gender = user['ugender']
# user_gender = user_gender.str.replace(' ', '_')
# user_gender = pd.get_dummies(user_gender,prefix='ugender')
# user_gender.to_csv('./data/user/london_user_gender.csv', index=0)

# #user level
# user_level = user['ulevel']
# user_level = user_level.str.replace(' ', '_')
# user_level = pd.get_dummies(user_level,prefix='ulevel')
# user_level.drop(user_level.columns[0],axis = 1,inplace = True)
# user_level.to_csv('./data/user/london_user_level.csv', index=0)

# #user_country
# user_country = user['ucountry']
# user_country = user_country.str.replace(' ', '_')
# user_country = pd.get_dummies(user_country,prefix='ucountry')
# user_country.to_csv('./data/user/london_user_country.csv', index=0)

#user_city

user_city=pd.read_csv('data/user/london_user_city_new3.csv',delimiter=",")

user_style = pd.read_csv('data/user/london_user_style.csv',delimiter=",")

user_id = pd.read_csv('data/user/london_user_id.csv',delimiter=",")
user_gender = pd.read_csv('data/user/london_user_gender.csv',delimiter=",")
user_level = pd.read_csv('data/user/london_user_level.csv',delimiter=",")
user_age = pd.read_csv('data/user/london_user_age.csv',delimiter=",")
user_country = pd.read_csv('data/user/london_user_country.csv',delimiter=",")
user = user_id
user = pd.concat((user,user_style),axis=1)
user = pd.concat((user,user_gender),axis=1)
user = pd.concat((user,user_level),axis=1)
user = pd.concat((user,user_age),axis=1)
user = pd.concat((user,user_country),axis=1)
user = pd.concat((user,user_city),axis=1)
print(user_id.shape)
print(user_age.shape)
print(user_gender.shape)
print(user_level.shape)
print(user_style.shape)
print(user_country.shape)
print(user_city.shape)
print(user.shape)
user_sort = user.sort_values(by='uid')
user_sort.to_csv('./data/user/london_user_list_new3.csv', index=0)

