import pandas as pd 
import numpy as np
import torch 
from torch.utils.data import DataLoader , TensorDataset
import torch.nn as nn
import math
import time
import random
from metrics import *
import os
import datetime
from tqdm import tqdm
import argparse
from tree import *
from make_tree_node import *


def max_3(a,b,c):
    if a>b:
        max = a
    else:
        max = b
    if c > max:
        max = c
    return max 

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
def test(model,dl_val,feat_num):


    val_loss = 0.0
    model.eval()
    for step_val, (raw_x_data_val,x_data_gbdt_val,uid_val,iid_val,label_val) in enumerate(dl_val, 1):
        
        x_val = torch.ones(x_data_gbdt_val.shape[0],x_data_gbdt_val.shape[1])
        data_one_hot_val = torch.zeros(x_data_gbdt_val.shape[0],feat_num).scatter(1,x_data_gbdt_val,x_val)
        
        data_neg_val = 1-data_one_hot_val
        preds_val,_,_,_ = model(x_data_gbdt_val,data_neg_val,uid_val,iid_val,raw_x_data_val)
        
        label_val = label_val.to(device)

        y_batch_pred_val = np.array(preds_val.tolist())
        y_label_val = np.array(label_val.tolist())
        



        if step_val == 1:
            y_true_val = y_label_val.flatten()
            y_pred_val = y_batch_pred_val.flatten()
        else:
            y_true_val = np.concatenate([y_true_val, y_label_val.flatten()])
            y_pred_val = np.concatenate([y_pred_val, y_batch_pred_val.flatten()])

    # val_hit, val_ndcg = eval_model_pro(y_true_val, y_pred_val, K=10, row_len=51)



    return y_pred_val
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=1, help='Seed init.')
parser.add_argument('--threads_num', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
parser.add_argument('--device', default='0')

parser.add_argument('--data_path',  default='nyc_data')
parser.add_argument('--gbdt_num', type=int, default=12, help='gbdt data num')


parser.add_argument('--make_counterfactual_batch_size', type=int, default=40000, help='make_counterfactual_batch_size')
parser.add_argument('--step_batch_num', type=int, default=0, help='step_batch_num')

parser.add_argument('--true_pre', type=float, default=0.5, help='true_pre')
parser.add_argument('--counterfactual_num', default='0', help='counterfactual_num')

args = parser.parse_args()
batch_size = args.batch_size
torch.set_num_threads(args.threads_num)
setup_seed(args.seed)
gbdt_num = args.gbdt_num
step_batch_num = args.step_batch_num
make_counterfactual_batch_size = args.make_counterfactual_batch_size
true_pre = args.true_pre
false_pre = 1-true_pre
counterfactual_num= args.counterfactual_num

device = torch.device('cuda:'+args.device if torch.cuda.is_available() else 'cpu')

data_path = args.data_path

if data_path == 'new_data_new3':
    data_pre_name = 'london'
    test_model = torch.load('/storage/ywwei/qxy/tem/new3/new3_test56/7017.pt', map_location=device)
elif data_path =='nyc_data':
    data_pre_name = 'nyc' 
    test_model = torch.load('/storage/ywwei/qxy/tem/nyc/nyc_test12/1848.pt', map_location=device)

input_data_path = data_path+"/tem_in/"+data_pre_name
tree_model_path = data_path+"/tree_model"

train_raw_set = pd.read_csv(input_data_path + '_train_set.csv',delimiter=",")
test_raw_set = pd.read_csv(input_data_path + '_test_set.csv',delimiter=",")
val_raw_set = pd.read_csv(input_data_path + '_val_set.csv',delimiter=",")

train_raw_set = train_raw_set.values
test_raw_set = test_raw_set.values
val_raw_set = val_raw_set.values

raw_x_num = np.max(np.max(train_raw_set,axis=1))+1#attribute总数+1

y_train = pd.read_csv(input_data_path + "_train_set_label.csv",delimiter=",")
y_test = pd.read_csv(input_data_path + "_test_set_label.csv",delimiter=",")
y_val = pd.read_csv(input_data_path + "_val_set_label.csv",delimiter=",")

y_train = y_train.values
y_test = y_test.values
y_val = y_val.values

id_train = pd.read_csv(input_data_path + "_train_set_id.csv",delimiter=",")
id_test = pd.read_csv(input_data_path + "_test_set_id.csv",delimiter=",")
id_val = pd.read_csv(input_data_path + "_val_set_id.csv",delimiter=",")

tree_path = tree_model_path+'/test_'+str(gbdt_num)

tree_brolist_dic = np.load(tree_path+"/tree_brolist_dic.npy",allow_pickle=True).item()#叶子节点兄弟节点按相同父亲个数排序的字典
tree_jump_dic = np.load(tree_path+"/tree_jump_dic.npy",allow_pickle=True).item()#相同父亲个数排序的字典
all_leaf_feature = np.load(tree_path+"/all_leaf_feature.npy")#叶子节点包含的attribute的onehot向量矩阵
all_leaf_feature = torch.tensor(all_leaf_feature,dtype=torch.int16).to(device)

np_train = np.load(tree_path+"/train_xgb_np.npy")
np_val = np.load(tree_path+"/val_xgb_np.npy")
np_test = np.load(tree_path+"/test_xgb_np.npy")


np_train = np_train.astype(np.int64)
np_val = np_val.astype(np.int64)
np_test = np_test.astype(np.int64)
tree_num = np_train.shape[1]#树的棵数

feat_num_train = np_train.max()
feat_num_test = np_test.max()
feat_num_val = np_val.max()

feat_num = int(max_3(feat_num_train,feat_num_test,feat_num_val))+1#最大叶子节点编号

uid_train = id_train['uid'].values
iid_train = id_train['iid'].values

uid_val = id_val['uid'].values
iid_val = id_val['iid'].values

uid_test = id_test['uid'].values
iid_test = id_test['iid'].values

uid_num_train = uid_train.max()
uid_num_test = uid_test.max()
uid_num_val = uid_val.max()

uid_num = int(max_3(uid_num_train,uid_num_test,uid_num_val))+1#最大uid

iid_num_train = iid_train.max()
iid_num_test = iid_test.max()
iid_num_val = iid_val.max()

iid_num = int(max_3(iid_num_train,iid_num_test,iid_num_val))+1#最大iid


print(train_raw_set.shape)
print(np_train.shape)
print(uid_train.shape)
print(iid_train.shape)
print(y_train.shape)
test_model.device = device

print(device)

# test_model.eval()


for true_pre in [0.5]:
    print(true_pre)

    for change_pre in [0.5,0.6,0.7,0.8,0.9]:
        print(change_pre)

        for coun_sam_num in [1,4]:
            print(coun_sam_num)

            for index in range(20):

                
                # print(coun_sam_num)

                # print(index)

                max_feature = 259+1
                # max_feature = 1414+1

                max_num = 100

                counterfactual_train_raw_set_batch = torch.load("./nyc_data/counterfactual_out105/counterfactual_train_raw_set"+str(index)+'_true_pre'+str(true_pre)+".pt", map_location='cpu')
                
                counterfactual_train_raw_set_batch[counterfactual_train_raw_set_batch>1] = 1
                counterfactual_train_raw_set_batch_index = counterfactual_train_raw_set_batch.nonzero(as_tuple=False)
                

                counterfactual_np_train_batch = torch.load("./nyc_data/counterfactual_out105/counterfactual_np_train"+str(index)+'_true_pre'+str(true_pre)+".pt", map_location='cpu') 
                counterfactual_np_train_batch_np = counterfactual_np_train_batch.cpu().numpy()
                counterfactual_uid_train_batch = torch.load("./nyc_data/counterfactual_out105/counterfactual_uid_train"+str(index)+'_true_pre'+str(true_pre)+".pt", map_location='cpu') 
                counterfactual_uid_train_batch = counterfactual_uid_train_batch.view(-1,1)
                counterfactual_uid_train_batch_np = counterfactual_uid_train_batch.cpu().numpy()
                counterfactual_iid_train_batch = torch.load("./nyc_data/counterfactual_out105/counterfactual_iid_train"+str(index)+'_true_pre'+str(true_pre)+".pt", map_location='cpu') 
                counterfactual_iid_train_batch = counterfactual_iid_train_batch.view(-1,1)
                counterfactual_iid_train_batch_np = counterfactual_iid_train_batch.cpu().numpy()
                counterfactual_y_train_batch = torch.load("./nyc_data/counterfactual_out105/counterfactual_y_train"+str(index)+'_true_pre'+str(true_pre)+".pt", map_location='cpu')  
                counterfactual_y_train_batch = counterfactual_y_train_batch.view(-1,1)
                counterfactual_y_train_batch_np = counterfactual_y_train_batch.cpu().numpy()


                counterfactual_train_raw_set_batch_sum = counterfactual_train_raw_set_batch.sum(dim=1)
                # print(counterfactual_y_train_batch_np.shape)

                uid_raw = counterfactual_uid_train_batch_np[0]
                uid_new = 0
                iid_raw = counterfactual_iid_train_batch_np[0]
                iid_new = 0

                coun_num = 0

                sam_num = counterfactual_uid_train_batch_np.shape[0]


                sam_list = []
                coun_sam_list = []


                for i in range(sam_num):
                    uid_new = counterfactual_uid_train_batch_np[i][0]
                    iid_new = counterfactual_iid_train_batch_np[i][0]
                    if uid_new == uid_raw:
                        if iid_new == iid_raw:
                            coun_num += 1
                            if coun_num <= coun_sam_num:
                                coun_sam_list.append(i)

                        else:
                            iid_new = iid_raw
                            sam_list.append(coun_num)
                            coun_num = 1
                            coun_sam_list.append(i)
                            
                    else:
                        uid_raw = uid_new
                        iid_raw = iid_new
                        sam_list.append(coun_num)
                        coun_num = 1
                        coun_sam_list.append(i)

                sam_array = np.array(sam_list)

                coun_sam_array = np.array(coun_sam_list)


                



                src = torch.tensor(counterfactual_train_raw_set_batch_index[:,1].cpu().numpy())

                lists = []
                begin = 0

                for i in range(len(counterfactual_train_raw_set_batch_sum)):

                    num = int(counterfactual_train_raw_set_batch_sum[i])
                    begin = begin
                    end = begin + num 
                    i_src = src[begin:end]
                    lists.append(i_src)
                    begin = end

                
                counterfactual_train_raw_set_batch_final = torch.nn.utils.rnn.pad_sequence(lists,padding_value=max_feature)
                counterfactual_train_raw_set_batch_final = counterfactual_train_raw_set_batch_final.t()
                if counterfactual_train_raw_set_batch_final.shape[1] < max_num:
                    sample_num = counterfactual_train_raw_set_batch_final.shape[0]
                    pad_dim = max_num - counterfactual_train_raw_set_batch_final.shape[1]
                    pad_tensor = torch.zeros([sample_num,pad_dim])
                    pad_tensor += max_feature
                    counterfactual_train_raw_set_batch_final = torch.cat((counterfactual_train_raw_set_batch_final,pad_tensor),dim=1)

                counterfactual_train_raw_set_batch = counterfactual_train_raw_set_batch_final
                counterfactual_train_raw_set_batch_np = counterfactual_train_raw_set_batch.cpu().numpy()


                # print(counterfactual_train_raw_set_batch_np.shape)

                counterfactual_train_raw_set_batch_np = counterfactual_train_raw_set_batch_np[coun_sam_array]
                counterfactual_np_train_batch_np = counterfactual_np_train_batch_np[coun_sam_array]
                counterfactual_uid_train_batch_np = counterfactual_uid_train_batch_np[coun_sam_array]
                counterfactual_iid_train_batch_np = counterfactual_iid_train_batch_np[coun_sam_array]
                counterfactual_y_train_batch_np = counterfactual_y_train_batch_np[coun_sam_array]

                # print(counterfactual_train_raw_set_batch_np.shape)

                counterfactual_uid_train_batch_np = counterfactual_uid_train_batch_np.squeeze(-1)
                counterfactual_iid_train_batch_np = counterfactual_iid_train_batch_np.squeeze(-1)


                dl_test_dataset = TensorDataset(torch.tensor(counterfactual_train_raw_set_batch_np),torch.tensor(counterfactual_np_train_batch_np),
                torch.tensor(counterfactual_uid_train_batch_np),torch.tensor(counterfactual_iid_train_batch_np),torch.tensor(counterfactual_y_train_batch_np).float())
                dl_test = DataLoader(dl_test_dataset,batch_size=batch_size)


                pre_np = test(test_model,dl_test,feat_num)

                

                counterfactual_uid_train_batch_np = counterfactual_uid_train_batch_np.reshape(-1,1)
                counterfactual_iid_train_batch_np = counterfactual_iid_train_batch_np.reshape(-1,1)
                pre_np = pre_np.reshape(-1,1)


                                
                counterfactual_y_train_batch_np0_index,np0 = np.where(pre_np<1-change_pre)
                counterfactual_y_train_batch_np1_index,np1 = np.where(pre_np>change_pre)

                
                counterfactual_train_raw_set_batch_np0 = counterfactual_train_raw_set_batch_np[counterfactual_y_train_batch_np0_index]
                counterfactual_train_raw_set_batch_np1 = counterfactual_train_raw_set_batch_np[counterfactual_y_train_batch_np1_index]

                counterfactual_np_train_batch_np0 = counterfactual_np_train_batch_np[counterfactual_y_train_batch_np0_index]
                counterfactual_np_train_batch_np1 = counterfactual_np_train_batch_np[counterfactual_y_train_batch_np1_index]
                
                counterfactual_uid_train_batch_np0 = counterfactual_uid_train_batch_np[counterfactual_y_train_batch_np0_index]
                counterfactual_uid_train_batch_np1 = counterfactual_uid_train_batch_np[counterfactual_y_train_batch_np1_index]

                counterfactual_iid_train_batch_np0 = counterfactual_iid_train_batch_np[counterfactual_y_train_batch_np0_index]
                counterfactual_iid_train_batch_np1 = counterfactual_iid_train_batch_np[counterfactual_y_train_batch_np1_index]

                counterfactual_y_train_batch_np0 = counterfactual_y_train_batch_np[counterfactual_y_train_batch_np0_index]
                counterfactual_y_train_batch_np1 = counterfactual_y_train_batch_np[counterfactual_y_train_batch_np1_index]

                counterfactual_train_raw_set_batch_np01 = np.concatenate((counterfactual_train_raw_set_batch_np0,counterfactual_train_raw_set_batch_np1),axis=0)
                counterfactual_np_train_batch_np01 = np.concatenate((counterfactual_np_train_batch_np0,counterfactual_np_train_batch_np1),axis=0)
                counterfactual_uid_train_batch_np01 = np.concatenate((counterfactual_uid_train_batch_np0,counterfactual_uid_train_batch_np1),axis=0)
                counterfactual_iid_train_batch_np01 = np.concatenate((counterfactual_iid_train_batch_np0,counterfactual_iid_train_batch_np1),axis=0)
                counterfactual_y_train_batch_np01 = np.concatenate((counterfactual_y_train_batch_np0,counterfactual_y_train_batch_np1),axis=0)
                

                counterfactual_y_train_batch_np0_index,np0 = np.where(counterfactual_y_train_batch_np01==0.0)
                counterfactual_y_train_batch_np1_index,np1 = np.where(counterfactual_y_train_batch_np01==1.0)
                
                counterfactual_train_raw_set_batch_np0 = counterfactual_train_raw_set_batch_np01[counterfactual_y_train_batch_np0_index]
                counterfactual_train_raw_set_batch_np1 = counterfactual_train_raw_set_batch_np01[counterfactual_y_train_batch_np1_index]

                counterfactual_np_train_batch_np0 = counterfactual_np_train_batch_np01[counterfactual_y_train_batch_np0_index]
                counterfactual_np_train_batch_np1 = counterfactual_np_train_batch_np01[counterfactual_y_train_batch_np1_index]
                
                counterfactual_uid_train_batch_np0 = counterfactual_uid_train_batch_np01[counterfactual_y_train_batch_np0_index]
                counterfactual_uid_train_batch_np1 = counterfactual_uid_train_batch_np01[counterfactual_y_train_batch_np1_index]

                counterfactual_iid_train_batch_np0 = counterfactual_iid_train_batch_np01[counterfactual_y_train_batch_np0_index]
                counterfactual_iid_train_batch_np1 = counterfactual_iid_train_batch_np01[counterfactual_y_train_batch_np1_index]

                counterfactual_y_train_batch_np0 = counterfactual_y_train_batch_np01[counterfactual_y_train_batch_np0_index]
                counterfactual_y_train_batch_np1 = counterfactual_y_train_batch_np01[counterfactual_y_train_batch_np1_index]

                counterfactual_train_raw_set_batch_np01 = np.concatenate((counterfactual_train_raw_set_batch_np0,counterfactual_train_raw_set_batch_np1),axis=0)
                counterfactual_np_train_batch_np01 = np.concatenate((counterfactual_np_train_batch_np0,counterfactual_np_train_batch_np1),axis=0)
                counterfactual_uid_train_batch_np01 = np.concatenate((counterfactual_uid_train_batch_np0,counterfactual_uid_train_batch_np1),axis=0)
                counterfactual_iid_train_batch_np01 = np.concatenate((counterfactual_iid_train_batch_np0,counterfactual_iid_train_batch_np1),axis=0)
                counterfactual_y_train_batch_np01 = np.concatenate((counterfactual_y_train_batch_np0,counterfactual_y_train_batch_np1),axis=0)


                if index == 0:

                    counterfactual_train_raw_set_np0 = counterfactual_train_raw_set_batch_np0
                    counterfactual_train_raw_set_np1 = counterfactual_train_raw_set_batch_np1
                    counterfactual_train_raw_set_np01 = counterfactual_train_raw_set_batch_np01

                    counterfactual_np_train_np0 = counterfactual_np_train_batch_np0
                    counterfactual_np_train_np1 = counterfactual_np_train_batch_np1
                    counterfactual_np_train_np01 = counterfactual_np_train_batch_np01
                    
                    counterfactual_uid_train_np0 = counterfactual_uid_train_batch_np0
                    counterfactual_uid_train_np1 = counterfactual_uid_train_batch_np1
                    counterfactual_uid_train_np01 = counterfactual_uid_train_batch_np01

                    counterfactual_iid_train_np0 = counterfactual_iid_train_batch_np0
                    counterfactual_iid_train_np1 = counterfactual_iid_train_batch_np1
                    counterfactual_iid_train_np01 = counterfactual_iid_train_batch_np01

                    counterfactual_y_train_np0 = counterfactual_y_train_batch_np0
                    counterfactual_y_train_np1 = counterfactual_y_train_batch_np1
                    counterfactual_y_train_np01 = counterfactual_y_train_batch_np01

                else :

                    counterfactual_train_raw_set_np0 = np.concatenate((counterfactual_train_raw_set_np0,counterfactual_train_raw_set_batch_np0),axis=0)
                    counterfactual_train_raw_set_np1 = np.concatenate((counterfactual_train_raw_set_np1,counterfactual_train_raw_set_batch_np1),axis=0)
                    counterfactual_train_raw_set_np01 = np.concatenate((counterfactual_train_raw_set_np01,counterfactual_train_raw_set_batch_np01),axis=0)

                    counterfactual_np_train_np0 = np.concatenate((counterfactual_np_train_np0,counterfactual_np_train_batch_np0),axis=0)
                    counterfactual_np_train_np1 = np.concatenate((counterfactual_np_train_np1,counterfactual_np_train_batch_np1),axis=0)
                    counterfactual_np_train_np01 = np.concatenate((counterfactual_np_train_np01,counterfactual_np_train_batch_np01),axis=0)

                    counterfactual_uid_train_np0 = np.concatenate((counterfactual_uid_train_np0,counterfactual_uid_train_batch_np0),axis=0)
                    counterfactual_uid_train_np1 = np.concatenate((counterfactual_uid_train_np1,counterfactual_uid_train_batch_np1),axis=0)
                    counterfactual_uid_train_np01 = np.concatenate((counterfactual_uid_train_np01,counterfactual_uid_train_batch_np01),axis=0)

                    counterfactual_iid_train_np0 = np.concatenate((counterfactual_iid_train_np0,counterfactual_iid_train_batch_np0),axis=0)
                    counterfactual_iid_train_np1 = np.concatenate((counterfactual_iid_train_np1,counterfactual_iid_train_batch_np1),axis=0)
                    counterfactual_iid_train_np01 = np.concatenate((counterfactual_iid_train_np01,counterfactual_iid_train_batch_np01),axis=0)

                    counterfactual_y_train_np0 = np.concatenate((counterfactual_y_train_np0,counterfactual_y_train_batch_np0),axis=0)
                    counterfactual_y_train_np1 = np.concatenate((counterfactual_y_train_np1,counterfactual_y_train_batch_np1),axis=0)
                    counterfactual_y_train_np01 = np.concatenate((counterfactual_y_train_np01,counterfactual_y_train_batch_np01),axis=0)


            print(counterfactual_train_raw_set_np0.shape[0])
            print(counterfactual_train_raw_set_np1.shape[0])
            print(counterfactual_train_raw_set_np01.shape[0])

            # print(counterfactual_np_train_np0.shape)
            # print(counterfactual_np_train_np1.shape)
            # print(counterfactual_np_train_np01.shape)

            # print(counterfactual_uid_train_np0.shape)
            # print(counterfactual_uid_train_np1.shape)
            # print(counterfactual_uid_train_np01.shape)

            # print(counterfactual_iid_train_np0.shape)
            # print(counterfactual_iid_train_np1.shape)
            # print(counterfactual_iid_train_np01.shape)

            # print(counterfactual_y_train_np0.shape)
            # print(counterfactual_y_train_np1.shape)
            # print(counterfactual_y_train_np01.shape)



            np.save('nyc_data/counterfactual_in105_9/'+str(change_pre)+'/'+str(coun_sam_num)+'/counterfactual_train_raw_set_np0'+'_true_pre'+str(true_pre), counterfactual_train_raw_set_np0) 
            np.save('nyc_data/counterfactual_in105_9/'+str(change_pre)+'/'+str(coun_sam_num)+'/counterfactual_train_raw_set_np1'+'_true_pre'+str(true_pre), counterfactual_train_raw_set_np1) 
            np.save('nyc_data/counterfactual_in105_9/'+str(change_pre)+'/'+str(coun_sam_num)+'/counterfactual_train_raw_set_np01'+'_true_pre'+str(true_pre), counterfactual_train_raw_set_np01) 

            np.save('nyc_data/counterfactual_in105_9/'+str(change_pre)+'/'+str(coun_sam_num)+'/counterfactual_np_train_np0'+'_true_pre'+str(true_pre), counterfactual_np_train_np0) 
            np.save('nyc_data/counterfactual_in105_9/'+str(change_pre)+'/'+str(coun_sam_num)+'/counterfactual_np_train_np1'+'_true_pre'+str(true_pre), counterfactual_np_train_np1) 
            np.save('nyc_data/counterfactual_in105_9/'+str(change_pre)+'/'+str(coun_sam_num)+'/counterfactual_np_train_np01'+'_true_pre'+str(true_pre), counterfactual_np_train_np01) 

            np.save('nyc_data/counterfactual_in105_9/'+str(change_pre)+'/'+str(coun_sam_num)+'/counterfactual_uid_train_np0'+'_true_pre'+str(true_pre), counterfactual_uid_train_np0) 
            np.save('nyc_data/counterfactual_in105_9/'+str(change_pre)+'/'+str(coun_sam_num)+'/counterfactual_uid_train_np1'+'_true_pre'+str(true_pre), counterfactual_uid_train_np1) 
            np.save('nyc_data/counterfactual_in105_9/'+str(change_pre)+'/'+str(coun_sam_num)+'/counterfactual_uid_train_np01'+'_true_pre'+str(true_pre), counterfactual_uid_train_np01) 

            np.save('nyc_data/counterfactual_in105_9/'+str(change_pre)+'/'+str(coun_sam_num)+'/counterfactual_iid_train_np0'+'_true_pre'+str(true_pre), counterfactual_iid_train_np0) 
            np.save('nyc_data/counterfactual_in105_9/'+str(change_pre)+'/'+str(coun_sam_num)+'/counterfactual_iid_train_np1'+'_true_pre'+str(true_pre), counterfactual_iid_train_np1) 
            np.save('nyc_data/counterfactual_in105_9/'+str(change_pre)+'/'+str(coun_sam_num)+'/counterfactual_iid_train_np01'+'_true_pre'+str(true_pre), counterfactual_iid_train_np01) 

            np.save('nyc_data/counterfactual_in105_9/'+str(change_pre)+'/'+str(coun_sam_num)+'/counterfactual_y_train_np0'+'_true_pre'+str(true_pre), counterfactual_y_train_np0) 
            np.save('nyc_data/counterfactual_in105_9/'+str(change_pre)+'/'+str(coun_sam_num)+'/counterfactual_y_train_np1'+'_true_pre'+str(true_pre), counterfactual_y_train_np1) 
            np.save('nyc_data/counterfactual_in105_9/'+str(change_pre)+'/'+str(coun_sam_num)+'/counterfactual_y_train_np01'+'_true_pre'+str(true_pre), counterfactual_y_train_np01) 