import pandas as pd
import numpy as np

item = pd.read_csv('./data/item/london_item_list_new3.csv',delimiter=",")
user = pd.read_csv('./data/user/london_user_list_new3.csv',delimiter=",")
item_max = item.shape[1]-1
item = user
print(item_max)

iid = item['uid']
item = item.drop(columns='uid')

item_row_sum = item.sum(axis=1)
item_max_row_sum = item_row_sum.max()
item_sum_row_sum = item_row_sum.sum()

aitem_index = np.zeros([item_row_sum.shape[0],item_max_row_sum],dtype=np.int32)
aitem_index = aitem_index - 1 

aitem = item.values
item_index = np.where(aitem==1)

pre_num = 0
col_index = 0

for i in range(len(item_index[0])):
    new_num = item_index[0][i]
    if(new_num>pre_num):
        pre_num = new_num
        col_index = 0
    aitem_index[new_num][col_index] = item_index[1][i] + item_max
    col_index += 1

print(aitem_index.max())
item_features = ['user'] * item_max_row_sum
item_features = [col + str(i + 1) for i, col in enumerate(item_features)]   
item_features_list = pd.core.indexes.base.Index(item_features)  
item_index_raw = pd.DataFrame(aitem_index)
item_index_raw.columns = item_features_list

item_index_final = pd.concat((iid,item_index_raw),axis = 1)
item_index_final_sort = item_index_final.sort_values(by='uid')
print(item_index_final_sort)

item_index_final_sort.to_csv('./data/user/london_user_index_final_sort_new3.csv', index=0)