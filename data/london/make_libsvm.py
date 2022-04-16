import numpy as np
import pandas as pd

item = pd.read_csv('data/item/london_item_index_final_sort_new3.csv',delimiter=",")
user = pd.read_csv('data/user/london_user_index_final_sort_new3.csv',delimiter=",")
item_list = pd.read_csv('data/item/london_item_list_new3.csv',delimiter=",")
user_list = pd.read_csv('data/user/london_user_list_new3.csv',delimiter=",")

max_num = item_list.shape[1]+user_list.shape[1]-2


# # train
# print('train')
# train_set_raw = pd.read_csv('new_data/london_u_i_train_set.csv',delimiter=",")
# train_set = pd.merge(train_set_raw,item,how='left')
# train_set = pd.merge(train_set,user,how='left')

# uid = train_set['uid']
# iid = train_set['iid']
# train_label = train_set['label']
# train_set = train_set.drop(columns='uid')
# train_set = train_set.drop(columns='iid')
# id = pd.concat((uid,iid),axis=1)
# train_label.to_csv('new_data/new3/london_train_set_label.csv', index=0)
# id.to_csv('new_data/new3/london_train_set_id.csv', index=0)

# data = train_set.values
# with open("new_data/new3/london_train_set.txt","w") as f:
#     for i in range(data.shape[0]):
#         data_index = ""
#         if i%500==0:
#             print(i)
#         for j in range(data.shape[1]):
#             if j == 0:
#                 data_index = data_index+str(data[i][0])+" "
#             else:
#                 if data[i][j] != -1:
#                     data_index = data_index +str(data[i][j])+":1 "
#         f.write(data_index) 
#         f.write("\n") 

# for i in range(data.shape[0]):
#     if i%500==0:
#         print(i)
#     for j in range(data.shape[1]):
#         if data[i][j] == -1:
#             data[i][j] = max_num

# train_set_raw = pd.DataFrame(data,columns=train_set.columns)
# train_set_raw = train_set_raw.drop(columns='label') 
# train_set_raw.to_csv('new_data/new3/london_train_set.csv', index=0) 
# print('train end')



# # test
# print('test')
# test_set_u_i = pd.read_csv('new_data/london_u_i_test_set.csv',delimiter=",")
# test_set = pd.merge(test_set_u_i,item,how='left')
# test_set = pd.merge(test_set,user,how='left')

# uid = test_set['uid']
# iid = test_set['iid']
# test_label = test_set['label']
# test_set = test_set.drop(columns='uid')
# test_set = test_set.drop(columns='iid')
# id = pd.concat((uid,iid),axis=1)
# test_label.to_csv('new_data/new3/london_test_set_label.csv', index=0)
# id.to_csv('new_data/new3/london_test_set_id.csv', index=0)

# data = test_set.values
# with open("new_data/new3/london_test_set.txt","w") as f:
#     for i in range(data.shape[0]):
#         data_index = ""
#         if i%500==0:
#             print(i)
#         for j in range(data.shape[1]):
#             if j == 0:
#                 data_index = data_index+str(data[i][0])+" "
#             else:
#                 if data[i][j] != -1:
#                     data_index = data_index +str(data[i][j])+":1 "
#         f.write(data_index) 
#         f.write("\n") 

# for i in range(data.shape[0]):
#     if i%500==0:
#         print(i)
#     for j in range(data.shape[1]):
#         if data[i][j] == -1:
#             data[i][j] = max_num

# test_set_raw = pd.DataFrame(data,columns=test_set.columns)
# test_set_raw = test_set_raw.drop(columns='label')
# test_set_raw.to_csv('new_data/new3/london_test_set.csv', index=0)
# print('test end')

#val
print('val')
val_set_u_i = pd.read_csv('new_data/london_u_i_val_set.csv',delimiter=",")

val_set = pd.merge(val_set_u_i,item,how='left')
val_set = pd.merge(val_set,user,how='left')

uid = val_set['uid']
iid = val_set['iid']
val_label = val_set['label']
val_set = val_set.drop(columns='uid')
val_set = val_set.drop(columns='iid')
id = pd.concat((uid,iid),axis=1)
val_label.to_csv('new_data/new3/london_val_set_label.csv', index=0)
id.to_csv('new_data/new3/london_val_set_id.csv', index=0)

data = val_set.values
with open("new_data/new3/london_val_set.txt","w") as f:
    for i in range(data.shape[0]):
        data_index = ""
        if i%500==0:
            print(i)
        for j in range(data.shape[1]):
            if j == 0:
                data_index = data_index+str(data[i][0])+" "
            else:
                if data[i][j] != -1:
                    data_index = data_index +str(data[i][j])+":1 "
        f.write(data_index) 
        f.write("\n") 

for i in range(data.shape[0]):
    if i%500==0:
        print(i)
    for j in range(data.shape[1]):
        if data[i][j] == -1:
            data[i][j] = max_num

val_set_raw = pd.DataFrame(data,columns=val_set.columns)
val_set_raw = val_set_raw.drop(columns='label')
val_set_raw.to_csv('new_data/new3/london_val_set.csv', index=0)
print('val end')




#nohup python -u data/make_libsvm.py > make_dataset_train.txt 2>&1 &
#nohup python -u data/make_libsvm.py > make_dataset_test.txt 2>&1 &
#nohup python -u data/make_libsvm.py > make_dataset_val.txt 2>&1 &


