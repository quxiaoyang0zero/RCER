import pandas as pd 
import numpy as np
import torch 
from torch.utils.data import DataLoader , TensorDataset
import torch.nn as nn
import math
import time
import random
from metrics import *
from model_wei import *
import os
import datetime
from tqdm import tqdm
import argparse
from tree import *
from make_tree_node import *
import argparse
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score


class EarlyStopping:
    
    def __init__(self, patience ,verbose=False, delta=0.000001):
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0

        self.best_val_ndcg = None
        self.best_val_hit = None
        self.best_val_loss = None
        self.best_val_auc = None


        self.best_test_ndcg = None
        self.best_test_hit = None
        self.best_test_loss = None
        self.best_test_auc = None


        self.early_stop = False


        self.val_loss_min = np.Inf
        self.val_socre_max = 99


        self.delta = delta

    def __call__(self,val_ndcg,val_hit,val_loss,val_auc,model_name,dl_test, model,max_leaf,loss_func):

        if self.best_val_loss is None:
            test_ndcg,test_hit,test_loss,test_auc = test(model_name,model,dl_test,max_leaf,loss_func)
            print('test_ndcg,test_hit,test_loss,test_auc')
            print(test_ndcg,test_hit,test_loss,test_auc)  
            
            self.best_val_ndcg = val_ndcg
            self.best_val_hit = val_hit
            self.best_val_loss = val_loss
            self.best_val_auc = val_auc

            self.best_test_ndcg = test_ndcg
            self.best_test_hit = test_hit
            self.best_test_loss = test_loss
            self.best_test_auc = test_auc

            self.save_model(val_ndcg, model)

        elif val_loss >= self.best_val_loss + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            test_ndcg,test_hit,test_loss,test_auc = test(model_name,model,dl_test,max_leaf,loss_func)
            print('test_ndcg,test_hit,test_loss,test_auc')
            print(test_ndcg,test_hit,test_loss,test_auc) 

            self.best_val_ndcg = val_ndcg
            self.best_val_hit = val_hit
            self.best_val_loss = val_loss
            self.best_val_auc = val_auc

            self.best_test_ndcg = test_ndcg
            self.best_test_hit = test_hit
            self.best_test_loss = test_loss
            self.best_test_auc = test_auc

            self.save_model(val_ndcg, model)
            self.counter = 0

    def save_model(self, model_score, model):
        # torch.save(model, 'nyc/nyc_test12'+'/'+'1848.pt')
        if type == 'loss':
                self.val_loss_min = model_score
        if type == 'score':
            self.val_socre_max = model_score           


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

def test(model_name,model,dl_val,max_leaf,loss_func):

    model.eval()
    for step_val, (raw_x_data_val,x_data_gbdt_val,uid_val,iid_val,label_val) in enumerate(dl_val, 1):

        if model_name == 'contrastive':
            x_val = torch.ones(x_data_gbdt_val.shape[0],x_data_gbdt_val.shape[1])
            data_one_hot_val = torch.zeros(x_data_gbdt_val.shape[0],max_leaf).scatter(1,x_data_gbdt_val,x_val)
            data_neg_val = 1-data_one_hot_val


            preds_val,_,_,_ = model(x_data_gbdt_val,data_neg_val,uid_val,iid_val,raw_x_data_val)
            # preds_val,_,_,_ = model(x_data_gbdt_val,data_neg_val,uid_val,iid_val,raw_x_data_val)

        label_val = label_val.to(device)
        y_batch_pred_val = np.array(preds_val.tolist())
        y_label_val = np.array(label_val.tolist())

        if step_val == 1:
            y_true_val = y_label_val.flatten()
            y_pred_val = y_batch_pred_val.flatten()
        else:
            y_true_val = np.concatenate([y_true_val, y_label_val.flatten()])
            y_pred_val = np.concatenate([y_pred_val, y_batch_pred_val.flatten()])

    # print('---------------------------')
    # print(np.shape(y_true_val))
    # print(np.shape(y_pred_val))
            
    val_hit, val_ndcg = eval_model_pro(y_true_val, y_pred_val, K=10, row_len=51)
    y_pred_val = torch.tensor(y_pred_val).squeeze(-1)
    y_true_val = torch.tensor(y_true_val).squeeze(-1)
    val_loss = loss_func(y_pred_val, y_true_val)
    label_numpy = y_true_val.cpu().numpy()
    preds_numpy = y_pred_val.detach().cpu().numpy()
    val_auc = roc_auc_score(label_numpy, preds_numpy)

    # print(y_pred_val.sum())
    return val_ndcg,val_hit,val_loss,val_auc

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    parser.add_argument('--threads_num', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--model_name', default='constast', help='model name')
    parser.add_argument('--gbdt_num', type=int, default=11, help='gbdt data num')
    parser.add_argument('--embed_dim', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--tem_mode', default='avg')
    parser.add_argument('--pos_feat', default='min')
    parser.add_argument('--all_loss', default='all')
    parser.add_argument('--device', default='0')

    parser.add_argument('--pos_pro', type=float, default=0.9)
    parser.add_argument('--con_weight', type=float, default=0.1)
    parser.add_argument('--lr_num', type=float, default=0.01)
    parser.add_argument('--l2_weight', type=float, default=1e-5)

    parser.add_argument('--true_pre', type=float, default=0.9)
    parser.add_argument('--train_type', default='0')
    parser.add_argument('--data_path',  default='new_data_new3')

    parser.add_argument('--counterfactual_num', default='1')
    parser.add_argument('--type',  default='counterfactual_and_all_raw')
    parser.add_argument('--coun_sam_num',  default='5')
    

    args = parser.parse_args()
    device = torch.device('cuda:'+args.device if torch.cuda.is_available() else 'cpu')

    con_weight = args.con_weight
    l2_weight = args.l2_weight
    lr_num= args.lr_num
    true_pre = args.true_pre
    train_type = args.train_type
    coun_sam_num = args.coun_sam_num

    batch_size = 1024
    torch.set_num_threads(2)
    setup_seed(1)
    patience = 5



    data_path = args.data_path

    if data_path == 'new_data_new3':
        data_pre_name = 'london'
        model = torch.load('/storage/ywwei/qxy/tem/new3/new3_test56/7017.pt', map_location=device)
    elif data_path =='nyc_data':
        data_pre_name = 'nyc' 
        model = torch.load('/storage/ywwei/qxy/tem/nyc/nyc_test12/1848.pt', map_location=device)
    input_data_path = data_path+"/tem_in/"+data_pre_name
    tree_model_path = data_path+"/tree_model"

    gbdt_num = args.gbdt_num
    counterfactual_num = args.counterfactual_num
    type = args.type
    
    print(train_type)

    # if train_type == '01':
    #     type = 'counterfactual_and_all_raw'
    # if train_type == '0':
    #     type = 'counterfactual_and_1_raw'
    # if train_type == '1':
    #     type = 'counterfactual_and_0_raw'

    

    counterfactual_data_path = data_path + "/counterfactual_in"+str(counterfactual_num)+'/'+str(coun_sam_num)+'/'
    print(counterfactual_data_path)

    print(type)

    train_raw_set = np.load(counterfactual_data_path + type +'/counterfactual_train_raw_set_np' +train_type+'.npy')
    test_raw_set = pd.read_csv(input_data_path + '_test_set.csv',delimiter=",")
    val_raw_set = pd.read_csv(input_data_path + '_val_set.csv',delimiter=",")

    test_raw_set = test_raw_set.values
    val_raw_set = val_raw_set.values

    raw_x_num = test_raw_set.max()+1
    print(raw_x_num)

    y_train = np.load(counterfactual_data_path + type + '/counterfactual_y_train_np' +train_type+'.npy')
    y_test = pd.read_csv(input_data_path + "_test_set_label.csv",delimiter=",")
    y_val = pd.read_csv(input_data_path + "_val_set_label.csv",delimiter=",")


    y_test = y_test.values
    y_val = y_val.values

    id_train = np.load(counterfactual_data_path + type + '/counterfactual_id_train_np' +train_type+'.npy')
    id_test = pd.read_csv(input_data_path + "_test_set_id.csv",delimiter=",")
    id_val = pd.read_csv(input_data_path + "_val_set_id.csv",delimiter=",")

    tree_path = tree_model_path+'/test_'+str(gbdt_num)


    tree_brolist_dic = np.load(tree_path+"/tree_brolist_dic.npy",allow_pickle=True).item()
    tree_jump_dic = np.load(tree_path+"/tree_jump_dic.npy",allow_pickle=True).item()
    all_leaf_feature = np.load(tree_path+"/all_leaf_feature.npy")
    tree_leaf_dic = np.load(tree_path+"/tree_leaf_dic.npy",allow_pickle=True).item()

    all_leaf_feature = all_leaf_feature.astype(np.int16)



    np_train = np.load(counterfactual_data_path + type + '/counterfactual_np_train_np' +train_type+'.npy')

    np_train_raw = np.load(tree_path+"/train_xgb_np.npy")
    np_val = np.load(tree_path+"/val_xgb_np.npy")
    np_test = np.load(tree_path+"/test_xgb_np.npy")


    np_train = np_train.astype(np.int64)
    np_val = np_val.astype(np.int64)
    np_test = np_test.astype(np.int64)

    tree_num = np_train.shape[1]

    max_leaf_train = np_train.max()
    max_leaf_test = np_test.max()
    max_leaf_val = np_val.max()

    max_leaf = int(max_3(max_leaf_train,max_leaf_test,max_leaf_val))+1
    print(max_leaf)
    


    uid_train = id_train[:,0]
    iid_train = id_train[:,1]

    uid_val = id_val['uid'].values
    iid_val = id_val['iid'].values

    uid_test = id_test['uid'].values
    iid_test = id_test['iid'].values


    uid_num_train = uid_train.max()
    uid_num_test = uid_test.max()
    uid_num_val = uid_val.max()
    
    uid_num = int(max_3(uid_num_train,uid_num_test,uid_num_val))+1

    iid_num_train = iid_train.max()
    iid_num_test = iid_test.max()
    iid_num_val = iid_val.max()
    
    iid_num = int(max_3(iid_num_train,iid_num_test,iid_num_val))+1

    tree_num = np_train.shape[1]


    randam_index = np.random.permutation(train_raw_set.shape[0])

    train_raw_set = train_raw_set[randam_index, :]
    np_train = np_train[randam_index, :]
    uid_train = uid_train[randam_index]
    iid_train = iid_train[randam_index]
    y_train = y_train[randam_index, :]

    dl_train_dataset = TensorDataset(torch.tensor(train_raw_set),torch.tensor(np_train),torch.tensor(uid_train),torch.tensor(iid_train),torch.tensor(y_train).float())
    dl_train = DataLoader(dl_train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)#, pin_memory=True)

    dl_val_dataset = TensorDataset(torch.tensor(val_raw_set),torch.tensor(np_val),torch.tensor(uid_val),torch.tensor(iid_val),torch.tensor(y_val).float())
    dl_val = DataLoader(dl_val_dataset, batch_size=batch_size, num_workers=4)#, pin_memory=True)

    dl_test_dataset = TensorDataset(torch.tensor(test_raw_set),torch.tensor(np_test),torch.tensor(uid_test),torch.tensor(iid_test),torch.tensor(y_test).float())
    dl_test = DataLoader(dl_test_dataset,batch_size=batch_size, num_workers=4)#, pin_memory=True)

    model.device = device
    loss_func = nn.BCELoss().to(device)


    print(device)

    model_name = 'contrastive'

    
    # model.eval()
    # with torch.no_grad():
    #     val_ndcg,val_hit,val_loss,val_auc = test(model_name,model,dl_test,max_leaf,loss_func)
    #     print(val_ndcg,val_hit,val_loss,val_auc) 

    # exit()

    sample_num = train_raw_set.shape[0]
    step_num_max = int(sample_num / batch_size)
    test_num = int(step_num_max / 20)

    embed_dim = 20
    mode = 'avg'
    pos_feat = 'min'

    # con_weight = 0.1
    # l2_weight = 0.001
    # lr_num= 0.001

    # pos_pro = 0.9

    constast_num = 2

    best_val_ndcg = 0.0
    best_val_hit =0.0
    best_val_loss = 0.0
    best_val_auc = 0.0


    best_test_ndcg = 0.0
    best_test_hit = 0.0
    best_test_loss = 0.0
    best_test_auc = 0.0




    epochs = 250


    early_stopping = EarlyStopping(patience=patience, verbose=True)
    # model = tem_con(embed_dim,max_leaf,uid_num,iid_num,mode,tree_num,raw_x_num,device,pos_feat,pos_pro, temperature).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_num,weight_decay=0)
    loss_func = nn.BCELoss().to(device)

    print('con_weight,l2_weight,lr_num')
    print(con_weight,l2_weight,lr_num)
    for epoch in range(1, epochs+1):


        # 训练阶段
        print("train")
        print(epoch)
        model.train()
        train_loss_sum = 0.0
        tem_loss_sum = 0.0
        con_loss_sum = 0.0
        reg_sum = 0.0
        step = 0 

        for step_train, (raw_x_data,x_data_gbdt,uid,iid,label) in enumerate(dl_train, 1):

            model.train()

            label = label.to(device)
            optimizer.zero_grad()
            if constast_num ==2:

                x = torch.ones(x_data_gbdt.shape[0],x_data_gbdt.shape[1])#.to(device)
                data_one_hot = torch.zeros(x_data_gbdt.shape[0],max_leaf).scatter(1,x_data_gbdt,x).to(device)
                data_neg = 1-data_one_hot
                preds,l2_regularization,con_loss,_ = model(x_data_gbdt,data_neg,uid,iid,raw_x_data)
                tem_loss = loss_func(preds, label)
                train_loss = (1-con_weight)*tem_loss + con_weight*con_loss + l2_weight*l2_regularization #L2 正则化

                train_loss_sum += train_loss
                tem_loss_sum += tem_loss
                con_loss_sum += con_loss
                reg_sum += l2_regularization
                step += 1.0

            train_loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            print("val and test")
            with torch.no_grad():

                val_ndcg,val_hit,val_loss,val_auc = test(model_name,model,dl_val,max_leaf,loss_func)
                early_stopping(val_ndcg,val_hit,val_loss,val_auc,model_name,dl_test,model,max_leaf,loss_func)

                print('val_ndcg,val_hit,val_loss,val_auc')
                print(val_ndcg,val_hit,val_loss,val_auc) 

                if early_stopping.early_stop:

                    if val_ndcg > best_val_ndcg:
                        
                        best_test_ndcg = early_stopping.best_test_ndcg
                        best_test_hit = early_stopping.best_test_hit
                        best_test_loss = early_stopping.best_test_loss
                        best_test_auc = early_stopping.best_test_auc

                        best_val_ndcg = early_stopping.best_val_ndcg
                        best_val_hit = early_stopping.best_val_hit
                        best_val_loss = early_stopping.best_val_loss
                        best_val_auc = early_stopping.best_val_auc
                    print("best_test_ndcg,best_test_hit,best_test_loss,best_test_auc")
                    print(best_test_ndcg,best_test_hit,best_test_loss,best_test_auc)   
                    print("best_val_ndcg,best_val_hit,best_val_loss,best_val_auc")
                    print(best_val_ndcg,best_val_hit,best_val_loss,best_val_auc)

                    break

