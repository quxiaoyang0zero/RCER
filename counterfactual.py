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
    
def make_raw_x(xt_np,counterfactual_feature_leaf,all_leaf_feature):

    counterfactual_feature_leaf = counterfactual_feature_leaf.long()
    counterfactual_raw_x = all_leaf_feature[counterfactual_feature_leaf]

    counterfactual_raw_x = counterfactual_raw_x.sum(1)
    counterfactual_raw_x[counterfactual_raw_x>1] = 1

    counterfactual_raw_x_sum = counterfactual_raw_x - xt_np
    counterfactual_raw_x_sum = torch.abs(counterfactual_raw_x_sum)
    counterfactual_raw_x_sum = counterfactual_raw_x_sum.sum(1)


    return counterfactual_raw_x,counterfactual_raw_x_sum

def counterfactual_list(tree_brolist_dic,tree_jump_dic,feature_leaf,feature_tree,model,leaf_index,device):

    p = torch.tensor(1.1).float().to(device)

    sample_num =leaf_index.shape[0]
    tree_num = len(feature_tree)


    counterfactual_feature_leaf = feature_leaf.to(device)
    counterfactual_feature_leaf = counterfactual_feature_leaf.reshape(1,-1)
    counterfactual_feature_leaf = counterfactual_feature_leaf.repeat(sample_num,1)


    tree_list_dic = tree_brolist_dic
    tree_jump_dic = tree_jump_dic

    for list_index in range(len(feature_tree)):

        cur_tree_id = feature_tree[list_index].item()
        cur_leaf_id = feature_leaf[list_index].item()

        
        cur_tree_node = tree_list_dic['tree'+str(cur_tree_id)]#找到对应的树
        cur_tree_jump = tree_jump_dic['tree'+str(cur_tree_id)]


        cur_tree_leaf = torch.tensor(cur_tree_node[:,0]).to(device)
        index = (cur_tree_leaf==cur_leaf_id).nonzero(as_tuple=False)


        cur_leaf_node = cur_tree_node[index]
        cur_leaf_jump = cur_tree_jump[index]


        cur_leaf_node_tensor = torch.tensor(cur_leaf_node).long().to(device)#bro_num
        cur_leaf_node_embed = model.feats_embed_layers[cur_leaf_node_tensor.repeat(sample_num,1)]#sample_num * bro_num * embed


        list_node_tensor = counterfactual_feature_leaf[:,list_index:list_index+tree_num].long().to(device)#sample_num * tree_num
        list_node_embed = model.feats_embed_layers[list_node_tensor]#sample_num * tree_num * embed

        cur_leaf_node_embed = cur_leaf_node_embed.repeat(list_node_embed.shape[1],1,1,1)#tree_num * sample_num * bro_num * embed
        cur_leaf_node_embed = cur_leaf_node_embed.permute(1,2,0,3).float()#sample_num * bro_num * tree_num * embed

        list_node_embed = list_node_embed.repeat(cur_leaf_node_embed.shape[1],1,1,1).float()#bro_num * sample_num * tree_num * embed
        list_node_embed = list_node_embed.permute(1,0,2,3).float()#sample_num * bro_num * tree_num * embed


        cos1 = nn.CosineSimilarity(dim = 3)
        cos_score = cos1(cur_leaf_node_embed,list_node_embed)


        cur_leaf_jump_tensor = torch.tensor(cur_leaf_jump).float().to(device)#bro_num
        cur_leaf_jump_weight = torch.pow(p,cur_leaf_jump_tensor)#bro_num
        
        cos_score_cur = cos_score[:,:,0]#sample_num * bro_num
        cos_score_other = cos_score[:,:,1:]#sample_num * bro_num * tree_num-1


        cos_score_cur = cos_score_cur*cur_leaf_jump_weight
        cos_score_other = cos_score_other.mean(dim=2)

        cos_score = cos_score_cur + cos_score_other#sample_num * bro_num


        # cos_score[:,:,0] = cos_score_cur#sample_num * bro_num
        # cos_score = cos_score.mean(dim=2)


        sorted, indices = torch.sort(cos_score, dim=1, descending=True) 

        index_list = leaf_index[:,list_index].long().reshape(-1,1)
        new_list = indices.gather(1,index_list)
        leaf_id = cur_leaf_node_tensor[new_list]


        counterfactual_feature_leaf = torch.cat((counterfactual_feature_leaf,leaf_id),dim=1)

    counterfactual_feature_leaf = counterfactual_feature_leaf[:,-tree_num:]
    return counterfactual_feature_leaf

def test(model,dl_val,feat_num):

    print('test')
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

    val_hit, val_ndcg = eval_model_pro(y_true_val, y_pred_val, K=10, row_len=51)



    return val_ndcg,val_hit

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

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
    print(y_train.shape)
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

    dl_train_dataset_all = TensorDataset(torch.tensor(train_raw_set),torch.tensor(np_train),torch.tensor(uid_train),torch.tensor(iid_train),torch.tensor(y_train).float())
    dl_train_all = DataLoader(dl_train_dataset_all,batch_size=make_counterfactual_batch_size,num_workers = 1)

    dl_test_dataset = TensorDataset(torch.tensor(test_raw_set),torch.tensor(np_test),torch.tensor(uid_test),torch.tensor(iid_test),torch.tensor(y_test).float())
    dl_test = DataLoader(dl_test_dataset,batch_size=batch_size)

    test_model.device = device

    print(device)

    test_model.eval()

    # val_ndcg,val_hit = test(test_model,dl_test,feat_num)
    # print(val_ndcg,val_hit)
    # exit()
    with torch.no_grad():

        #保存生成的反事实的数据

        counterfactual_train_raw_set_all = {}
        counterfactual_np_train_all = {}
        counterfactual_uid_train_all = {}
        counterfactual_iid_train_all = {}
        counterfactual_y_train_all = {}

        uid_pre = torch.tensor([]).to(device)
        iid_pre = torch.tensor([]).to(device)
        train_pre = torch.tensor([]).to(device)
        coun_sample_pre = torch.tensor([]).to(device)


        counterfactual_train_raw_set_all[str(true_pre)] = torch.tensor([]).to(device)
        counterfactual_np_train_all[str(true_pre)] = torch.tensor([]).to(device)
        counterfactual_uid_train_all[str(true_pre)] = torch.tensor([]).to(device)
        counterfactual_iid_train_all[str(true_pre)] = torch.tensor([]).to(device)
        counterfactual_y_train_all[str(true_pre)] = torch.tensor([]).to(device)


        for step_coun_batch, (raw_x_data_coun_batch,x_data_gbdt_coun_batch,uid_coun_batch,iid_coun_batch,label_coun_batch) in enumerate(dl_train_all, 0):

            print(raw_x_data_coun_batch.shape)

            if step_coun_batch != step_batch_num:
                continue

            print(step_coun_batch)

            dl_train_dataset = TensorDataset(raw_x_data_coun_batch,x_data_gbdt_coun_batch,uid_coun_batch,iid_coun_batch,label_coun_batch)
            dl_train = DataLoader(dl_train_dataset,batch_size=batch_size,num_workers = 1)

            step_sum = 0
            sample_sum = 0

            for step, (raw_x_data,x_data_gbdt,uid,iid,label) in enumerate(dl_train, 0):



                
                #打印平均做了几次循环
                if step % 100 == 1:
                    outfile = open("./"+data_path+'/counterfactual_out'+str(counterfactual_num)+'/txt/counterfactual_txt'+str(step_coun_batch)+'.txt', 'w')   

                    if sample_sum == 0:
                        print(step)
                        print('0')
                        outfile.write('{0}\t{1}\n'.format(step,sample_sum) )

                    else:
                        print(step)
                        step_avg = step_sum/sample_sum
                        print(step_avg)
                        outfile.write('{0}\t{1}\n'.format(step,step_avg) )

                #得到当前样本的预测值和 attention
                
                x_new = torch.ones(x_data_gbdt.shape[0],x_data_gbdt.shape[1])
                data_one_hot = torch.zeros(x_data_gbdt.shape[0],feat_num).scatter(1,x_data_gbdt,x_new)
                data_neg = 1-data_one_hot

                preds_new,_,_,weight = test_model(x_data_gbdt,data_neg,uid,iid,raw_x_data)
                preds = preds_new[0]

                sorted, indices = torch.sort(weight, dim=1, descending=True)    


                #按attention大小排序的树编号和叶子节点编号
                indices = indices.view(-1)
                attention_tree = indices

                feature_leaf = x_data_gbdt[0]
                feature_leaf = feature_leaf[indices]

                #不用预测错误的样本
                if (label[0].item() == 1) & (preds.item()<0.5):
                    continue
                
                elif (label[0].item() == 0) & (preds.item()>0.5):
                    continue

                coun_sample = 0
                #维护的反事实叶子节点
                index_last = torch.zeros([1,tree_num])
                
                for cur_tree_index in range(tree_num):
                    
                    index_matrix = index_last.clone()

                    cur_tree = attention_tree[cur_tree_index].item()
                    cur_tree_list = tree_brolist_dic['tree'+str(cur_tree)]
                    tree_max_num = cur_tree_list.shape[1]

                    index_matrix = index_matrix.repeat(tree_max_num,1)
                    index_cur_range = torch.arange(0,tree_max_num,1)

                    index_matrix[:,cur_tree_index] = index_cur_range

                    leaf_index = index_matrix.int().to(device)
                    sample_num = leaf_index.shape[0]

            
                    counterfactual_feature_leaf = counterfactual_list(tree_brolist_dic,tree_jump_dic,feature_leaf,attention_tree,test_model,leaf_index,device)


                    raw_x = raw_x_data[0]
                    
                    xt = raw_x.to(device)
                    xt = torch.nn.functional.one_hot(xt,num_classes = raw_x_num)
                    xt = torch.sum(xt, dim = 0)
                    xt = xt[:xt.shape[0]-1]
                
                    counterfactual_raw_x,counterfactual_raw_x_sum = make_raw_x(xt,counterfactual_feature_leaf,all_leaf_feature)

                    uid_new = uid[0]
                    iid_new = iid[0]
                    label_new = 1 - label[0]
                    preds_raw = preds_new[0]
                    coun_sample_tensor = torch.tensor(coun_sample)

                    uid_new = uid_new.repeat(sample_num)
                    iid_new = iid_new.repeat(sample_num)
                    label_new = label_new.repeat(sample_num)
                    preds_raw = preds_raw.repeat(sample_num)
                    coun_sample_tensor = coun_sample_tensor.repeat(sample_num)



                    x_new = torch.ones(counterfactual_feature_leaf.shape[0],counterfactual_feature_leaf.shape[1]).to(device)
                    data_one_hot_new = torch.zeros(counterfactual_feature_leaf.shape[0],feat_num).to(device).scatter(1,counterfactual_feature_leaf,x_new)
                    data_neg_new = 1-data_one_hot_new


                    preds_new,_,_,_ = test_model(counterfactual_feature_leaf,data_neg_new,uid_new,iid_new,counterfactual_raw_x)

                    preds_new = preds_new.squeeze(-1)

                    preds_new_now = preds_new.clone()



                    if label[0] == 1:

                        counterfactual_index_last = torch.argmin(preds_new_now)
                        counterfactual_pre = torch.min(preds_new_now)
                        index_last[0,cur_tree_index] = counterfactual_index_last

                        if counterfactual_pre >= false_pre:
                            continue

                        counterfactual_index_list = (preds_new_now<false_pre).nonzero(as_tuple=False)

                    elif (label[0] == 0):

                        counterfactual_index_last = torch.argmax(preds_new_now)
                        counterfactual_pre = torch.max(preds_new_now)
                        index_last[0,cur_tree_index] = counterfactual_index_last

                        if counterfactual_pre <= true_pre:
                            continue

                        counterfactual_index_list = (preds_new_now>true_pre).nonzero(as_tuple=False)


                    counterfactual_index = counterfactual_index_list[0]
                    counterfactual_index = counterfactual_index.view(-1)
                    
                    if counterfactual_train_raw_set_all[str(true_pre)].shape == torch.Size([0]):
                        counterfactual_train_raw_set_all[str(true_pre)] = counterfactual_raw_x[counterfactual_index]
                        counterfactual_np_train_all[str(true_pre)] = counterfactual_feature_leaf[counterfactual_index]
                        counterfactual_uid_train_all[str(true_pre)] = uid_new[counterfactual_index]
                        counterfactual_iid_train_all[str(true_pre)] = iid_new[counterfactual_index]
                        counterfactual_y_train_all[str(true_pre)] = label_new[counterfactual_index]
                    else:
                        counterfactual_train_raw_set_all[str(true_pre)] = torch.cat((counterfactual_train_raw_set_all[str(true_pre)],counterfactual_raw_x[counterfactual_index]),dim = 0)
                        counterfactual_np_train_all[str(true_pre)] = torch.cat((counterfactual_np_train_all[str(true_pre)],counterfactual_feature_leaf[counterfactual_index]),dim = 0)
                        counterfactual_uid_train_all[str(true_pre)] = torch.cat((counterfactual_uid_train_all[str(true_pre)],uid_new[counterfactual_index]),dim = 0)
                        counterfactual_iid_train_all[str(true_pre)] = torch.cat((counterfactual_iid_train_all[str(true_pre)],iid_new[counterfactual_index]),dim = 0)
                        counterfactual_y_train_all[str(true_pre)] = torch.cat((counterfactual_y_train_all[str(true_pre)],label_new[counterfactual_index]),dim = 0)

                    if uid_pre.shape == torch.Size([0]):
                        uid_pre = uid_new[counterfactual_index]
                        iid_pre = iid_new[counterfactual_index]
                        train_pre = preds_raw[counterfactual_index]
                        coun_sample_pre = coun_sample_tensor[counterfactual_index]
                    else:
                        train_pre = torch.cat((train_pre,preds_raw[counterfactual_index]),dim = 0)
                        uid_pre = torch.cat((uid_pre,uid_new[counterfactual_index]),dim = 0)
                        iid_pre = torch.cat((iid_pre,iid_new[counterfactual_index]),dim = 0)
                        coun_sample_pre = torch.cat((coun_sample_pre,coun_sample_tensor[counterfactual_index]),dim = 0)


                    # print(counterfactual_train_raw_set_all[str(true_pre)].shape)
                    # print(counterfactual_np_train_all[str(true_pre)].shape)
                    # print(counterfactual_uid_train_all[str(true_pre)].shape)
                    # print(counterfactual_iid_train_all[str(true_pre)].shape)
                    # print(counterfactual_y_train_all[str(true_pre)].shape)

                    step_num = cur_tree_index + 1
                    step_sum += step_num
                    sample_sum += 1

                    coun_sample += 1
                    
                    if coun_sample == 5:
                        break

                    

                

            print(true_pre)
            print(false_pre)
            print(counterfactual_train_raw_set_all[str(true_pre)].shape)


            torch.save(counterfactual_train_raw_set_all[str(true_pre)], "./"+data_path+"/counterfactual_out"+counterfactual_num+"/counterfactual_train_raw_set"+str(step_coun_batch)+"_true_pre"+str(true_pre)+".pt")
            torch.save(counterfactual_np_train_all[str(true_pre)], "./"+data_path+"/counterfactual_out"+counterfactual_num+"/counterfactual_np_train"+str(step_coun_batch)+"_true_pre"+str(true_pre)+".pt")
            torch.save(counterfactual_uid_train_all[str(true_pre)], "./"+data_path+"/counterfactual_out"+counterfactual_num+"/counterfactual_uid_train"+str(step_coun_batch)+"_true_pre"+str(true_pre)+".pt")
            torch.save(counterfactual_iid_train_all[str(true_pre)], "./"+data_path+"/counterfactual_out"+counterfactual_num+"/counterfactual_iid_train"+str(step_coun_batch)+"_true_pre"+str(true_pre)+".pt")
            torch.save(counterfactual_y_train_all[str(true_pre)], "./"+data_path+"/counterfactual_out"+counterfactual_num+"/counterfactual_y_train"+str(step_coun_batch)+"_true_pre"+str(true_pre)+".pt")

            torch.save(train_pre, "./"+data_path+"/counterfactual_out"+counterfactual_num+"/train_pre_new"+str(step_coun_batch)+".pt")
            torch.save(uid_pre, "./"+data_path+"/counterfactual_out"+counterfactual_num+"/uid_pre_new"+str(step_coun_batch)+".pt")
            torch.save(iid_pre, "./"+data_path+"/counterfactual_out"+counterfactual_num+"/iid_pre_new"+str(step_coun_batch)+".pt")
            torch.save(coun_sample_pre, "./"+data_path+"/counterfactual_out"+counterfactual_num+"/coun_sample_pre"+str(step_coun_batch)+".pt")

        print('end')

        
            

