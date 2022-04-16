import numpy as np
import pandas as pd


item = pd.read_csv('data/item/london_item_list.csv',delimiter=",")
u_i = pd.read_csv('data/london_u_i.csv',delimiter=",")
user = pd.read_csv('data/user/london_user_list.csv',delimiter=",")

uid = user['uid']
iidi = item['iid']
iid = pd.DataFrame(data=iidi)

# #val 1:50
# print("val")
# val_set = pd.read_csv('data/london_u_i_val_set.csv',delimiter=",")
# print(val_set.shape)

# neg_num = 50
# iid_val = val_set['iid']
# uid_val = val_set['uid']
# data =  pd.DataFrame()
# for i in range(val_set.shape[0]):
#     if i%500==0:
#         print(i)

#     datau = pd.DataFrame()
#     uid_i = uid_val[i]
#     iid_i = iid_val[i]
#     u_i_i = u_i.query('uid==@uid_i').reset_index(drop=True)
#     num = u_i_i.shape[0]#观测到的uid iid

#     item = iid
#     for j in range(num):
#         iid_i_all = u_i_i.loc[j,'iid']
#         item = item.query('iid != @iid_i_all')
#         item = item.reset_index(drop=True)
#     item = item.sample(n=neg_num,replace=True).reset_index(drop=True)
#     for k in range(neg_num):
#         item_id = item.loc[k,'iid']
#         d0 = [{'uid':uid_i,'iid':item_id,'label':0}] 
#         data0 = pd.DataFrame(data = d0)
#         datau = pd.concat((datau,data0),axis = 0)
#     d1 = [{'uid':uid_i,'iid':iid_i,'label':1}] 
#     data1 = pd.DataFrame(data = d1)
#     datau = pd.concat((datau,data1),axis = 0)
#     datau=datau.sample(frac=1).reset_index(drop=True)
#     data = pd.concat((data,datau),axis = 0)
# data.to_csv('./new_data/london_u_i_val_set.csv', index=0)
# print("end")


# #test 1:50
# print("test")
# test_set = pd.read_csv('data/london_u_i_test_set.csv',delimiter=",")
# print(test_set.shape)

# neg_num = 50
# iid_test = test_set['iid']
# uid_test = test_set['uid']
# data =  pd.DataFrame()
# for i in range(test_set.shape[0]):
#     if i%500==0:
#         print(i)

#     datau = pd.DataFrame()
#     uid_i = uid_test[i]
#     iid_i = iid_test[i]
#     d1 = [{'uid':uid_i,'iid':iid_i,'label':1}] 
#     u_i_i = u_i.query('uid==@uid_i').reset_index(drop=True)
#     num = u_i_i.shape[0]#观测到的uid iid

#     item = iid
#     for j in range(num):
#         iid_i_all = u_i_i.loc[j,'iid']
#         item = item.query('iid != @iid_i_all')
#         item = item.reset_index(drop=True)
#     item = item.sample(n=neg_num,replace=True).reset_index(drop=True)
#     for k in range(neg_num):
#         item_id = item.loc[k,'iid']
#         d0 = [{'uid':uid_i,'iid':item_id,'label':0}] 
#         data0 = pd.DataFrame(data = d0)
#         datau = pd.concat((datau,data0),axis = 0)
#     data1 = pd.DataFrame(data = d1)
#     datau = pd.concat((datau,data1),axis = 0)
#     datau=datau.sample(frac=1).reset_index(drop=True)
#     data = pd.concat((data,datau),axis = 0)
# data.to_csv('./new_data/london_u_i_test_set.csv', index=0)
# print("end")


#train 1:4
print("train")
train_set = pd.read_csv('data/london_u_i_train_set.csv',delimiter=",")
print(train_set.shape)

neg_num = 4
iid_train = train_set['iid']
uid_train = train_set['uid']
data =  pd.DataFrame()
for i in range(train_set.shape[0]):
    if i%500==0:
        print(i)

    datau = pd.DataFrame()
    uid_i = uid_train[i]
    iid_i = iid_train[i]
    d1 = [{'uid':uid_i,'iid':iid_i,'label':1}] 
    data1 = pd.DataFrame(data = d1)
    datau = pd.concat((datau,data1),axis = 0)
    u_i_i = u_i.query('uid==@uid_i').reset_index(drop=True)
    num = u_i_i.shape[0]#观测到的uid iid

    item = iid
    for j in range(num):
        iid_i_all = u_i_i.loc[j,'iid']
        item = item.query('iid != @iid_i_all')
        item = item.reset_index(drop=True)
    item = item.sample(n=neg_num,replace=True).reset_index(drop=True)
    for k in range(neg_num):
        item_id = item.loc[k,'iid']
        d0 = [{'uid':uid_i,'iid':item_id,'label':0}] 
        data0 = pd.DataFrame(data = d0)
        datau = pd.concat((datau,data0),axis = 0)
    datau = datau.sample(frac=1).reset_index(drop=True)
    data = pd.concat((data,datau),axis = 0)
data = data.sample(frac=1).reset_index(drop=True)
data.to_csv('./new_data/london_u_i_train_set.csv', index=0)



#nohup python -u ./data/make_dataset.py > make_dataset_train.txt 2>&1 &
#nohup python -u ./data/make_dataset.py > make_dataset_test.txt 2>&1 &
#nohup python -u ./data/make_dataset.py > make_dataset_val.txt 2>&1 &
