import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
# (878,)
# (878, 87)
# (878, 604)
# (878, 6)
# (878, 698)
# 697

# iid
item = pd.read_csv('./data/item/london_item.csv',delimiter=",")
iid = item['iid']

# item_rating

item_rating=pd.read_csv('./data/item/london_item_rating.csv',delimiter=",")

item_att = pd.read_csv('./data/item/london_item_att.csv',delimiter=",")
item_tag = pd.read_csv('./data/item/london_item_tag_new3.csv',delimiter=",")
print(iid.shape)
print(item_att.shape)
print(item_tag.shape)
print(item_rating.shape)
item = iid
item = pd.concat((item,item_att),axis=1)
item = pd.concat((item,item_tag),axis=1)
item = pd.concat((item,item_rating),axis=1)

print(item.shape)
item_sort = item.sort_values(by='iid')
item_sort.to_csv('./data/item/london_item_list_new3.csv', index=0)

