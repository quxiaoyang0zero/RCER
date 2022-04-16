import torch 
import numpy as np
import pandas as pd
data_path = 'new_data_new3'
if data_path == 'new_data_new3':
    data_pre_name = 'london'
    gbdt_num = 11
    max_feature = 1414+1
elif data_path =='nyc_data':
    data_pre_name = 'nyc' 
    gbdt_num = 12
    max_feature = 259+1

print(data_path)

input_data_path = data_path+"/tem_in/"+data_pre_name
tree_model_path = data_path+"/tree_model"
tree_path = tree_model_path+'/test_'+str(gbdt_num)
max_num = 100



train_raw_set_raw = pd.read_csv(input_data_path + '_train_set.csv',delimiter=",")
train_raw_set_raw = train_raw_set_raw.values

if train_raw_set_raw.shape[1] < max_num:
    sample_num = train_raw_set_raw.shape[0]
    pad_dim = max_num - train_raw_set_raw.shape[1]
    pad_np = np.zeros([sample_num,pad_dim])
    pad_np += max_feature
    train_raw_set_raw = np.concatenate((train_raw_set_raw,pad_np),axis=1)
y_train_raw = pd.read_csv(input_data_path + "_train_set_label.csv",delimiter=",")
y_train_raw = y_train_raw.values
id_train_raw = pd.read_csv(input_data_path + "_train_set_id.csv",delimiter=",")
id_train_raw = id_train_raw.values
np_train_raw = np.load(tree_path+"/train_xgb_np.npy")

label_train_raw1_index,np1 = np.where(y_train_raw==1.0)
train_raw_set_raw1 = train_raw_set_raw[label_train_raw1_index]
np_train_raw1 = np_train_raw[label_train_raw1_index]
id_train_raw1 = id_train_raw[label_train_raw1_index]
y_train_raw1 = y_train_raw[label_train_raw1_index]


label_train_raw0_index,np0 = np.where(y_train_raw==0.0)
train_raw_set_raw0 = train_raw_set_raw[label_train_raw0_index]
np_train_raw0 = np_train_raw[label_train_raw0_index]
id_train_raw0 = id_train_raw[label_train_raw0_index]
y_train_raw0 = y_train_raw[label_train_raw0_index]



for true_pre in [0.5]:

    print(true_pre)

    for train_type in ['1','0','01']:

        for change_pre in [0.5,0.6,0.7,0.8,0.9]:
            print(change_pre)

            for coun_sam_num in [1,4]:

                print(coun_sam_num)

                print('----------------------------')

                counterfactual_data_path = data_path + "/counterfactual_in105_9/"

                counterfactual_data_path = counterfactual_data_path + str(change_pre) + '/'

                counterfactual_data_path = counterfactual_data_path + str(coun_sam_num)


                train_raw_set = np.load(counterfactual_data_path + '/counterfactual_train_raw_set_np' +train_type+'_true_pre'+str(true_pre)+'.npy')
                y_train = np.load(counterfactual_data_path + '/counterfactual_y_train_np' +train_type+'_true_pre'+str(true_pre)+'.npy')
                uid_train = np.load(counterfactual_data_path + '/counterfactual_uid_train_np' +train_type+'_true_pre'+str(true_pre)+'.npy')
                iid_train = np.load(counterfactual_data_path + '/counterfactual_iid_train_np' +train_type+'_true_pre'+str(true_pre)+'.npy')
                np_train = np.load(counterfactual_data_path + '/counterfactual_np_train_np' +train_type+'_true_pre'+str(true_pre)+'.npy')

                id_train = np.concatenate((uid_train,iid_train),axis=1)

                print(train_raw_set.shape)

                np.save(counterfactual_data_path+'/counterfactual'+'/counterfactual_train_raw_set_np'+train_type+'.npy', train_raw_set) 
                np.save(counterfactual_data_path+'/counterfactual'+'/counterfactual_y_train_np'+train_type+'.npy', y_train) 
                np.save(counterfactual_data_path+'/counterfactual'+'/counterfactual_id_train_np'+train_type+'.npy', id_train) 
                np.save(counterfactual_data_path+'/counterfactual'+'/counterfactual_np_train_np'+train_type+'.npy', np_train) 


                train_raw_set_and_all_raw = np.concatenate((train_raw_set,train_raw_set_raw),axis=0)
                np_train_and_all_raw = np.concatenate((np_train,np_train_raw),axis=0)
                id_train_and_all_raw = np.concatenate((id_train,id_train_raw),axis=0)
                y_train_and_all_raw = np.concatenate((y_train,y_train_raw),axis=0)

                randam_index = np.random.permutation(train_raw_set_and_all_raw.shape[0])

                train_raw_set_and_all_raw = train_raw_set_and_all_raw[randam_index, :]
                np_train_and_all_raw = np_train_and_all_raw[randam_index, :]
                id_train_and_all_raw = id_train_and_all_raw[randam_index]
                y_train_and_all_raw = y_train_and_all_raw[randam_index, :]

                print(train_raw_set_and_all_raw.shape)
                print(train_raw_set_raw.shape)

                np.save(counterfactual_data_path+'/counterfactual_and_all_raw'+'/counterfactual_train_raw_set_np'+train_type+'.npy', train_raw_set_and_all_raw) 
                np.save(counterfactual_data_path+'/counterfactual_and_all_raw'+'/counterfactual_y_train_np'+train_type+'.npy', y_train_and_all_raw) 
                np.save(counterfactual_data_path+'/counterfactual_and_all_raw'+'/counterfactual_id_train_np'+train_type+'.npy', id_train_and_all_raw) 
                np.save(counterfactual_data_path+'/counterfactual_and_all_raw'+'/counterfactual_np_train_np'+train_type+'.npy', np_train_and_all_raw) 

                np.save(counterfactual_data_path+'/counterfactual_raw'+'/counterfactual_train_raw_set_np'+train_type+'.npy', train_raw_set_raw) 
                np.save(counterfactual_data_path+'/counterfactual_raw'+'/counterfactual_y_train_np'+train_type+'.npy', y_train_raw) 
                np.save(counterfactual_data_path+'/counterfactual_raw'+'/counterfactual_id_train_np'+train_type+'.npy', id_train_raw) 
                np.save(counterfactual_data_path+'/counterfactual_raw'+'/counterfactual_np_train_np'+train_type+'.npy', np_train_raw) 


                train_raw_set_and_1_raw = np.concatenate((train_raw_set,train_raw_set_raw1),axis=0)
                np_train_and_1_raw = np.concatenate((np_train,np_train_raw1),axis=0)
                id_train_and_1_raw = np.concatenate((id_train,id_train_raw1),axis=0)
                y_train_and_1_raw = np.concatenate((y_train,y_train_raw1),axis=0)

                randam_index = np.random.permutation(train_raw_set_and_1_raw.shape[0])

                train_raw_set_and_1_raw = train_raw_set_and_1_raw[randam_index, :]
                np_train_and_1_raw = np_train_and_1_raw[randam_index, :]
                id_train_and_1_raw = id_train_and_1_raw[randam_index]
                y_train_and_1_raw = y_train_and_1_raw[randam_index, :]

                print(train_raw_set_and_1_raw.shape)

                np.save(counterfactual_data_path+'/counterfactual_and_1_raw'+'/counterfactual_train_raw_set_np'+train_type+'.npy', train_raw_set_and_1_raw) 
                np.save(counterfactual_data_path+'/counterfactual_and_1_raw'+'/counterfactual_y_train_np'+train_type+'.npy', y_train_and_1_raw) 
                np.save(counterfactual_data_path+'/counterfactual_and_1_raw'+'/counterfactual_id_train_np'+train_type+'.npy', id_train_and_1_raw) 
                np.save(counterfactual_data_path+'/counterfactual_and_1_raw'+'/counterfactual_np_train_np'+train_type+'.npy', np_train_and_1_raw) 


                train_raw_set_and_0_raw = np.concatenate((train_raw_set,train_raw_set_raw0),axis=0)
                np_train_and_0_raw = np.concatenate((np_train,np_train_raw0),axis=0)
                id_train_and_0_raw = np.concatenate((id_train,id_train_raw0),axis=0)
                y_train_and_0_raw = np.concatenate((y_train,y_train_raw0),axis=0)

                randam_index = np.random.permutation(train_raw_set_and_0_raw.shape[0])

                train_raw_set_and_0_raw = train_raw_set_and_0_raw[randam_index, :]
                np_train_and_0_raw = np_train_and_0_raw[randam_index, :]
                id_train_and_0_raw = id_train_and_0_raw[randam_index]
                y_train_and_0_raw = y_train_and_0_raw[randam_index, :]

                print(train_raw_set_and_0_raw.shape)

                np.save(counterfactual_data_path+'/counterfactual_and_0_raw'+'/counterfactual_train_raw_set_np'+train_type+'.npy', train_raw_set_and_0_raw) 
                np.save(counterfactual_data_path+'/counterfactual_and_0_raw'+'/counterfactual_y_train_np'+train_type+'.npy', y_train_and_0_raw) 
                np.save(counterfactual_data_path+'/counterfactual_and_0_raw'+'/counterfactual_id_train_np'+train_type+'.npy', id_train_and_0_raw) 
                np.save(counterfactual_data_path+'/counterfactual_and_0_raw'+'/counterfactual_np_train_np'+train_type+'.npy', np_train_and_0_raw) 

