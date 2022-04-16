import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split

# uid 1 16315
# uage 6
# ugender 2
# ulevel 5
# ustyle    19
# ucountry 125
# ucity 3072
#

#user id
user = pd.read_csv('./data/london_user.csv',delimiter=",")
# user = user.drop_duplicates(keep = 'first')
user_id = user['uid_index']
user_id.columns=['uid']  
#user age
user_age = user['uage']
user_age = user_age.str.replace(' ', '_')
user_age = pd.get_dummies(user_age,prefix='uage')
user_age.to_csv('./data/london_user_age.csv', index=0)

#user gender
user_gender = user['ugender']
user_gender = user_gender.str.replace(' ', '_')
user_gender = pd.get_dummies(user_gender,prefix='ugender')
user_gender.to_csv('./data/london_user_gender.csv', index=0)

#user level
user_level = user['ulevel']
user_level = user_level.str.replace(' ', '_')
user_level = pd.get_dummies(user_level,prefix='ulevel')
user_level.to_csv('./data/london_user_level.csv', index=0)

#user_country
user_country = user['ucountry']
user_country = user_country.str.replace(' ', '_')
user_country = pd.get_dummies(user_country,prefix='ucountry')
user_country.to_csv('./data/london_user_country.csv', index=0)

#user_city
user_city = user['ucity']
user_city = user_city.str.replace(' ', '_')
user_city = pd.get_dummies(user_city,prefix='ucity')
user_city.to_csv('./data/london_user_city.csv', index=0)

# #user style
# user = pd.read_csv('./data/london_user.csv',delimiter=",")
# user = user.drop_duplicates(keep = 'first')
# print(user.shape)
# user_style = user['ustyle']

# user_style = user_style.str.strip(to_strip='[]')

# user_style = user_style.str.split(pat=',', expand=True)

# user_style1 = user_style.copy()         
# col1 = list(user_style.columns)
# user_style[col1]= user_style1[col1].astype(str)       
# for i in col1:
#     user_style[i] = user_style[i].str.strip()
#     user_style[i] = user_style[i].str.replace(' ', '_')
#     user_style[i] = user_style[i].str.strip(to_strip='\'')
# user_style.to_csv('./data/london_user_style0.csv', index=0)


# user_style = pd.read_csv('./data/london_user_style0.csv',delimiter=",")
# user_style.fillna('None',inplace = True)
# col1 = list(user_style.columns)
# print(user_style[:5])
# data = pd.DataFrame()
# print(data[6:10])
# for i in col1:
#     data = pd.concat([data,user_style[i]])
# # user_style.columns=['ustyle','ustyle','ustyle','ustyle','ustyle','ustyle','ustyle','ustyle','ustyle','ustyle','ustyle','ustyle','ustyle','ustyle','ustyle','ustyle','ustyle','ustyle','ustyle']  

# data = data.drop_duplicates(keep = 'first')
# data = data.drop(data.index[0])
# data.columns = ['ustyle']
# data = pd.get_dummies(data)
# col = list(data.columns)
# data1 = pd.DataFrame(columns=col)
# print(data1.shape)

# user_style2 = user_style
# for i in range(user_style2.shape[0]):
#     for j in range(user_style2.shape[1]):
#         if(user_style2.iat[i,j]!='None'):
#             data1.loc[i,'ustyle_'+user_style2.iat[i,j]] = 1
#         else :
#             data1.loc[i,'ustyle_'+user_style2.iat[i,j]] = 0

# data1 = data1.drop('ustyle_None', axis=1)
# print(data1.shape)
# data1.fillna('0',inplace = True)
# data1 = data1.astype(int)
# data1.to_csv('./data/london_user_style.csv', index=0)


user_style = pd.read_csv('data/london_user_style.csv',delimiter=",")

user = user_id
user = pd.concat((user,user_age),axis=1)
user = pd.concat((user,user_gender),axis=1)
user = pd.concat((user,user_level),axis=1)
user = pd.concat((user,user_style),axis=1)
user = pd.concat((user,user_country),axis=1)
user = pd.concat((user,user_city),axis=1)
user.rename(columns={'uid_index':'uid'},inplace=True)
print(user_id.shape)
print(user_age.shape)
print(user_gender.shape)
print(user_level.shape)
print(user_style.shape)
print(user_country.shape)
print(user_city.shape)
print(user.shape)
user_sort = user.sort_values(by='uid')
user_sort.to_csv('./data/london_user_list.csv', index=0)

