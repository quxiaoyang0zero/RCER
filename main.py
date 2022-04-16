import argparse
import pandas as pd 
import numpy as np
import torch 
from torch.utils.data import DataLoader , TensorDataset
import torch.nn as nn
import math
import time
import random
from metrics import *
from model_RCER import *
from model_tem import *
from baseline_Model_1220 import *
import os
import datetime
from tqdm import tqdm
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
        # torch.save(model, 'new3/new3_test56'+'/'+'7017.pt')
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
    for step_val, (raw_x_data_val,x_data_gbdt_val,uid_val,iid_val,label_val,exist_val) in enumerate(dl_val, 1):

        if model_name == 'contrastive':
            x_val = torch.ones(x_data_gbdt_val.shape[0],x_data_gbdt_val.shape[1])
            data_one_hot_val = torch.zeros(x_data_gbdt_val.shape[0],max_leaf).scatter(1,x_data_gbdt_val,x_val)
            data_neg_val = 1-data_one_hot_val


            preds_val,_,_,_ = model(x_data_gbdt_val,data_neg_val,uid_val,iid_val,raw_x_data_val)
            # preds_val,_,_,_ = model(x_data_gbdt_val,data_neg_val,uid_val,iid_val,raw_x_data_val)

        elif model_name == 'tem':
            preds_val,_ = model(x_data_gbdt_val,uid_val,iid_val,raw_x_data_val)
        elif model_name == 'widedeep' or model_name == 'fm' or model_name == 'afm' or model_name == 'nfm' or model_name == 'afn' or model_name == 'lfm':
            preds_val,_ = model(raw_x_data_val,exist_val,uid_val,iid_val)

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
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--model_name', default='contrastive', help='model name')
    
    parser.add_argument('--embed_dim', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--tem_mode', default='avg')
    parser.add_argument('--pos_feat', default='min')
    parser.add_argument('--all_loss', default='all')
    parser.add_argument('--device', default='cpu')

    parser.add_argument('--pos_pro', type=float, default=0.9)
    parser.add_argument('--con_weight', type=float, default=0.1)
    parser.add_argument('--lr_num', type=float, default=0.01)
    parser.add_argument('--l2_weight', type=float, default=1e-5)

    parser.add_argument('--data_path',  default='nyc_data')
    parser.add_argument('--gbdt_num', type=int, default=12, help='gbdt data num')
    
    args = parser.parse_args()
    batch_size = args.batch_size
    torch.set_num_threads(args.threads_num)
    setup_seed(args.seed)
    patience = args.patience
    model_name = args.model_name
    gbdt_num= args.gbdt_num
    all_loss = args.all_loss

    

    print(all_loss)
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:'+args.device if torch.cuda.is_available() else 'cpu')
    mode = args.tem_mode
    pos_feat = args.pos_feat

    
    pos_pro = args.pos_pro

    embed_dim = args.embed_dim
    con_weight = args.con_weight
    temperature = args.temperature
    lr_num = args.lr_num
    l2_weight = args.l2_weight

    pos_feat = 'min'

    data_path = args.data_path
    print(data_path)
    print(gbdt_num)


    if data_path == 'new_data_new3':
        data_pre_name = 'london'
        index_begin = [87,604,6,19,2,5,6,125,561]

    elif data_path =='nyc_data':
        data_pre_name = 'nyc'   
        index_begin = [7,3,100,6,2,5,19,118]


    tree_model_path = data_path+"/tree_model"
    input_data_path = data_path+"/tem_in/"+data_pre_name

    print(tree_model_path)

    
    train_raw_set = pd.read_csv(input_data_path + '_train_set.csv',delimiter=",")
    test_raw_set = pd.read_csv(input_data_path + '_test_set.csv',delimiter=",")
    val_raw_set = pd.read_csv(input_data_path + '_val_set.csv',delimiter=",")



    train_raw_set = train_raw_set.values
    test_raw_set = test_raw_set.values
    val_raw_set = val_raw_set.values
    
    max_num = np.max(np.max(train_raw_set,axis=1))+1
    print(max_num)

    feat_max = train_raw_set.shape[1]
    print(feat_max)


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

    np_train = np.load(tree_path+"/train_xgb_np.npy")
    np_val = np.load(tree_path+"/val_xgb_np.npy")
    np_test = np.load(tree_path+"/test_xgb_np.npy")
    np_leaf_value = np.load(tree_path+'/all_leaf_pre_np.npy')
    leaf_value = torch.tensor(np_leaf_value)

    # print(np_leaf_value)

    print(np_train.shape)


    np_train = np_train.astype(np.int64)
    np_val = np_val.astype(np.int64)
    np_test = np_test.astype(np.int64)

    tree_num = np_train.shape[1]

    max_leaf_train = np_train.max()
    max_leaf_test = np_test.max()
    max_leaf_val = np_val.max()

    max_leaf = int(max_3(max_leaf_train,max_leaf_test,max_leaf_val))+1
    print(max_leaf)

    max_feat = max_num-1
    
    train_one = np.array(train_raw_set)
    train_one[train_one < max_feat] = -1
    train_one[train_one > 0] = 0
    train_one[train_one < 0] = 1

    test_one = np.array(test_raw_set)
    test_one[test_one < max_feat] = -1
    test_one[test_one > 0] = 0
    test_one[test_one < 0] = 1

    val_one = np.array(val_raw_set)
    val_one[val_one < max_feat] = -1
    val_one[val_one > 0] = 0
    val_one[val_one < 0] = 1

    uid_train = id_train['uid'].values
    iid_train = id_train['iid'].values
    
    uid_val = id_val['uid'].values
    iid_val = id_val['iid'].values

    uid_test = id_test['uid'].values
    iid_test = id_test['iid'].values

    uid_num_train = uid_train.max()
    uid_num_test = uid_test.max()
    uid_num_val = uid_val.max()
    
    uid_num = int(max_3(uid_num_train,uid_num_test,uid_num_val))+1
    print(uid_num)

    iid_num_train = iid_train.max()
    iid_num_test = iid_test.max()
    iid_num_val = iid_val.max()
    
    iid_num = int(max_3(iid_num_train,iid_num_test,iid_num_val))+1
    print(iid_num)


    tree_leaf_num = np.load(tree_path+"/tree_leaf_num.npy")
    tree_range = np.arange(0,tree_num)

    tree_leaf_index = np.expand_dims(np.repeat(tree_range,tree_leaf_num,axis = 0),0)


    tree_leaf_index = torch.tensor(tree_leaf_index)

    max_leaf_num = tree_leaf_num.max()

    best_val_ndcg = 0.0
    best_val_hit =0.0
    best_val_loss = 0.0
    best_val_auc = 0.0


    best_test_ndcg = 0.0
    best_test_hit = 0.0
    best_test_loss = 0.0
    best_test_auc = 0.0


    woutput1 = 0.0001

    mode = 'avg'              


    print('con_weight,l2_weight,lr_num,temperature,embed_dim,pos_pro')
    print(con_weight,l2_weight,lr_num,temperature,embed_dim,pos_pro)

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    print(train_one.shape)
    print(train_raw_set.shape)

    dl_train_dataset = TensorDataset(torch.tensor(train_raw_set),torch.tensor(np_train),torch.tensor(uid_train),torch.tensor(iid_train),torch.tensor(y_train).float(),torch.tensor(train_one))
    dl_train = DataLoader(dl_train_dataset, shuffle=True, batch_size=batch_size, num_workers=1)

    dl_val_dataset = TensorDataset(torch.tensor(val_raw_set),torch.tensor(np_val),torch.tensor(uid_val),torch.tensor(iid_val),torch.tensor(y_val).float(),torch.tensor(val_one))
    dl_val = DataLoader(dl_val_dataset, batch_size=batch_size, num_workers=4)

    dl_test_dataset = TensorDataset(torch.tensor(test_raw_set),torch.tensor(np_test),torch.tensor(uid_test),torch.tensor(iid_test),torch.tensor(y_test).float(),torch.tensor(test_one))
    dl_test = DataLoader(dl_test_dataset,batch_size=batch_size, num_workers=4)
    
    if model_name == 'contrastive':
        model = tem_con(embed_dim,max_leaf,uid_num,iid_num,mode,tree_num,max_num,device,pos_feat,pos_pro,temperature).to(device)
    elif model_name == 'tem':
        model = tem(embed_dim,max_leaf,uid_num,iid_num,mode,tree_num,max_num,device,woutput1).to(device)
    elif model_name == 'widedeep':
        hidden_units = [8]
        model = WideDeep(max_num,hidden_units,embed_dim,uid_num,iid_num,feat_max,device).to(device)
    elif model_name == 'fm':
        hidden_units = [8]
        model = FM(max_num,hidden_units,embed_dim,uid_num,iid_num,feat_max,device).to(device)
    elif model_name == 'nfm':
        hidden_units = [8]
        model = NFM(max_num,hidden_units,embed_dim,uid_num,iid_num,feat_max,device).to(device)
    elif model_name == 'afm':
        hidden_units = [8]
        model = AFM(max_num,hidden_units,embed_dim,uid_num,iid_num,feat_max,device).to(device)
    elif model_name == 'afn':
        hidden_units = [8]
        model = AFN(max_num,hidden_units,embed_dim,uid_num,iid_num,feat_max,device,index_begin).to(device)
    elif model_name == 'lfm':
        hidden_units = [8]
        model = LorentzFM(max_num,hidden_units,embed_dim,uid_num,iid_num,feat_max,device,index_begin).to(device)

    # optimizer = torch.optim.Adagrad(model.parameters(), lr=lr_num)#,weight_decay=0)
    optimizer = torch.optim.Adam(model.parameters(), lr= lr_num)
    loss_func = nn.BCELoss().to(device)

    epochs = 250
    
    for epoch in range(1, epochs+1):
        print("train")
        print(epoch)
        model.train()
        train_loss_sum = 0.0
        model_loss_sum = 0.0
        con_loss_sum = 0.0
        reg_sum = 0.0
        step = 0 
        # pbar = tqdm(total=len(dl_train_dataset))

        for step_train, (raw_x_data,x_data_gbdt,uid,iid,label,exist) in enumerate(dl_train, 1):




            label = label.to(device)
            optimizer.zero_grad()

            if model_name == 'contrastive':

                x = torch.ones(x_data_gbdt.shape[0],x_data_gbdt.shape[1])
                data_one_hot = torch.zeros(x_data_gbdt.shape[0],max_leaf).scatter(1,x_data_gbdt,x).to(device)
                data_neg = 1-data_one_hot
                preds, l2_regularization, con_loss,_ = model(x_data_gbdt,data_neg,uid,iid,raw_x_data)
                # preds, l2_regularization,loss_1,loss_2 = model(x_data_gbdt,data_neg,uid,iid,raw_x_data)
                model_loss = loss_func(preds, label)
                train_loss = (1-con_weight)*model_loss + con_weight*con_loss + l2_weight*l2_regularization #L2 正则化
                con_loss_sum += con_loss

            elif model_name == 'tem':
                preds, l2_regularization = model(x_data_gbdt,uid,iid,raw_x_data)
                model_loss = loss_func(preds, label)
                train_loss = model_loss + l2_weight*l2_regularization #L2 正则化
            elif model_name == 'widedeep' or model_name == 'fm' or model_name == 'afm' or model_name == 'nfm' or model_name == 'afn' or model_name == 'lfm':
                preds,l2_regularization = model(raw_x_data,exist,uid,iid)
                model_loss = loss_func(preds, label)
                train_loss = model_loss + l2_weight*l2_regularization

            train_loss_sum += train_loss
            model_loss_sum += model_loss
            reg_sum += l2_regularization
            # pbar.update(batch_size)
            step += 1.0

            train_loss.backward()
            optimizer.step()
        # pbar.close()

        # print('----------------- loss value:{}  model_loss value:{} contrastive_loss value:{} reg_loss value:{} --------------'.format(train_loss_sum/step, model_loss_sum/step, con_loss_sum/step, reg_sum/step))
        #验证

        
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

            
print('all end')


