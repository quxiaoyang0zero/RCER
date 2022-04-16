import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/london_raw.csv',delimiter="\t")


itemdata = df['iid']
# itemdata = pd.concat((itemdata,df['iattribute']),axis=1)
# itemdata = pd.concat((itemdata,df['itag']),axis=1)
# itemdata = pd.concat((itemdata,df['irating']),axis=1)


userdata = df['uid_index']
userdata = pd.concat((userdata,df['uage']),axis=1)
userdata = pd.concat((userdata,df['ugender']),axis=1)
userdata = pd.concat((userdata,df['ulevel']),axis=1)
userdata = pd.concat((userdata,df['ustyle']),axis=1)
userdata = pd.concat((userdata,df['ucountry']),axis=1)
userdata = pd.concat((userdata,df['ucity']),axis=1)

user_item = df['uid_index']
user_item = pd.concat((user_item,df['iid']),axis=1)
user_item['label'] = 1

userdata = userdata.sort_values(by='uid_index')
userdata.to_csv('./data/london_user.csv', index=0)
# itemdata.to_csv('./data/london_item.csv', index=0)
user_item.rename(columns={'uid_index':'uid'},inplace=True)
user_item = user_item.sort_values(by='uid')
user_item.to_csv('./data/london_u_i.csv', index=0)