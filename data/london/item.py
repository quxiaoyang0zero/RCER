import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split

# iid 878 1
# attribute 87
# tag 4589
# rating 6
# 4682

# item = pd.read_csv('./data/london_item.csv',delimiter=",")
# item = item.drop_duplicates(keep = 'first')
# item.to_csv('./data/london_item.csv', index=0)

# iid
item = pd.read_csv('./data/london_item.csv',delimiter=",")
iid = item['iid']

# item_rating
item_rating = item['irating']
item_rating = pd.get_dummies(item_rating,prefix='irating')
item_rating.to_csv('./data/london_item_rating.csv', index=0)


# #item_tag 
# item = pd.read_csv('./data/london_item.csv',delimiter=",")
# item_tag = item['itag']
# item_tag = item_tag.str.strip(to_strip='[]')
# item_tag = item_tag.str.split(pat=',', expand=True)

# att_col = list(item_tag.columns)
# item_tag= item_tag.astype(str)       
# for i in att_col:
#     item_tag[i] = item_tag[i].str.strip()
#     item_tag[i] = item_tag[i].str.replace(' ', '_')
#     item_tag[i] = item_tag[i].str.strip(to_strip='\'')

# data = pd.DataFrame()
# for i in item_tag:
#     data = pd.concat([data,item_tag[i]])
# data = data.drop_duplicates(keep = 'first')
# print(data.shape)
# data=data[~data[0].isin([''])]
# data=data[~data[0].isin(['None'])]
# print(data.shape)
# data = pd.get_dummies(data,prefix='itag')
# col = list(data.columns)
# data1 = pd.DataFrame(columns=col)

# for i in range(item_tag.shape[0]):
#     print(i)
#     for j in range(item_tag.shape[1]):
#         if(item_tag.iat[i,j]!='None'):
#             data1.loc[i,'itag_'+item_tag.iat[i,j]] = 1
#         else :
#             data1.loc[i,'itag_'+item_tag.iat[i,j]] = 0
# data1 = data1.drop('itag_None', axis=1)
# data1 = data1.drop('itag_', axis=1)
# data1.fillna('0',inplace = True)
# data1 = data1.astype(int)
# print(data1.shape)

# data1.to_csv('./data/london_item_tag.csv', index=0)

# # item att
# item = pd.read_csv('./data/london_item.csv',delimiter=",")
# item_att = item['iattribute']
# item_att = item_att.str.strip(to_strip='[]')
# item_att = item_att.str.split(pat=',', expand=True)

# att_col = list(item_att.columns)
# item_att= item_att.astype(str)       
# for i in att_col:
#     item_att[i] = item_att[i].str.strip()
#     item_att[i] = item_att[i].str.replace(' ', '_')
#     item_att[i] = item_att[i].str.strip(to_strip='\'')

# data = pd.DataFrame()
# for i in item_att:
#     data = pd.concat([data,item_att[i]])
# data = data.drop_duplicates(keep = 'first')
# print(data.shape)
# data=data[~data[0].isin(['None'])]
# data = pd.get_dummies(data,prefix='iatt')
# col = list(data.columns)
# data1 = pd.DataFrame(columns=col)
# print(data1.shape)

# for i in range(item_att.shape[0]):
#     for j in range(item_att.shape[1]):
#         if(item_att.iat[i,j]!='None'):
#             data1.loc[i,'iatt_'+item_att.iat[i,j]] = 1
#         else :
#             data1.loc[i,'iatt_'+item_att.iat[i,j]] = 0
# print(data1.shape)
# data1 = data1.drop('iatt_None', axis=1)
# data1.fillna('0',inplace = True)
# data1 = data1.astype(int)
# print(data1.shape)

# data1.to_csv('./data/london_item_att.csv', index=0)


item_att = pd.read_csv('./data/london_item_att.csv',delimiter=",")
item_tag = pd.read_csv('./data/london_item_tag.csv',delimiter=",")
print(iid.shape)
print(item_att.shape)
print(item_tag.shape)
print(item_rating.shape)
item = iid
item = pd.concat((item,item_att),axis=1)
item = pd.concat((item,item_tag),axis=1)
item = pd.concat((item,item_rating),axis=1)

print(item.shape)
item.to_csv('./data/london_item_list.csv', index=0)

