import pandas as pd
import numpy as np

item = pd.read_csv('nyc_item_list.csv',delimiter=",")
user = pd.read_csv('nyc_user_list.csv',delimiter=",")




iid = item['iid']
item = item.drop(columns='iid')
item_max = item.shape[1]
print(item_max)

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
    aitem_index[new_num][col_index] = item_index[1][i]
    col_index += 1

print(aitem_index.max())
item_features = ['item'] * item_max_row_sum
item_features = [col + str(i + 1) for i, col in enumerate(item_features)]   
item_features_list = pd.core.indexes.base.Index(item_features)  
item_index_raw = pd.DataFrame(aitem_index)
item_index_raw.columns = item_features_list

item_index_final = pd.concat((iid,item_index_raw),axis = 1)
item_index_final_sort = item_index_final.sort_values(by='iid')
print(item_index_final_sort)

item_index_final_sort.to_csv('nyc_item_index_final_sort.csv', index=0)



uid = user['uid']
user = user.drop(columns='uid')
user_max = user.shape[1]

user_row_sum = user.sum(axis=1)
user_max_row_sum = user_row_sum.max()
user_sum_row_sum = user_row_sum.sum()

auser_index = np.zeros([user_row_sum.shape[0],user_max_row_sum],dtype=np.int32)
auser_index = auser_index - 1 

auser = user.values
user_index = np.where(auser==1)

pre_num = 0
col_index = 0

for i in range(len(user_index[0])):
    new_num = user_index[0][i]
    if(new_num>pre_num):
        pre_num = new_num
        col_index = 0
    auser_index[new_num][col_index] = user_index[1][i] + item_max
    col_index += 1

print(auser_index.max())
user_features = ['user'] * user_max_row_sum
user_features = [col + str(i + 1) for i, col in enumerate(user_features)]   
user_features_list = pd.core.indexes.base.Index(user_features)  
user_index_raw = pd.DataFrame(auser_index)
user_index_raw.columns = user_features_list

user_index_final = pd.concat((uid,user_index_raw),axis = 1)
user_index_final_sort = user_index_final.sort_values(by='uid')
print(user_index_final_sort)

user_index_final_sort.to_csv('nyc_user_index_final_sort.csv', index=0)