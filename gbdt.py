import argparse
import pandas as pd 
import numpy as np
from sklearn.metrics import log_loss,roc_auc_score, accuracy_score
import xgboost as xgb
from metrics import *
import os 
import math

def mkdir(path):

    folder = os.path.exists(path)
    if not folder:
            os.makedirs(path)
            print("---  new folder...  ---")
            print("---  OK  ---")

    else:
            print("---  There is this folder!  ---")
def max_3(a,b,c):
    if a>b:
        max = a
    else:
        max = b
    if c > max:
        max = c
    return max 

def evaluate(y_true, y_pred, K, row_len):
        
    tmp_true = np.array(y_true.tolist())
    tmp_pred = np.array(y_pred.tolist())

    tmp_true = tmp_true.flatten()
    tmp_pred = tmp_pred.flatten()

    test_hit, test_ndcg = eval_model_pro(tmp_true, tmp_pred, K=K, row_len=row_len)
    return test_hit, test_ndcg


if __name__ == '__main__':

    testnum = 14

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',  default='nyc_data')
    parser.add_argument('--lr_num', type=float, default=0.01)
    parser.add_argument('--l2_weight', type=float, default=1e-5)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', default='1')



    args = parser.parse_args()


    data_path = args.data_path
    lr_num = args.lr_num
    seed = args.seed
    l2_weight = args.l2_weight
    device = args.device
    if data_path == 'new_data_new3':
        data_pre_name = 'london'
        index_begin = [87,604,6,19,2,5,6,125,561]

    elif data_path =='nyc_data':
        data_pre_name = 'nyc'   
        index_begin = [7,3,100,6,2,5,19,118]

    tree_model_path = data_path+"/tree_model"

    input_data_path = data_path+"/gbdt_in/"+data_pre_name

    dm_train = xgb.DMatrix(input_data_path + "_train_set.txt")
    dm_test = xgb.DMatrix(input_data_path + "_test_set.txt")
    dm_val = xgb.DMatrix(input_data_path + "_val_set.txt")


    y_train = pd.read_csv(input_data_path + "_train_set_label.csv",delimiter=",")
    y_test = pd.read_csv(input_data_path + "_test_set_label.csv",delimiter=",")
    y_val = pd.read_csv(input_data_path + "_val_set_label.csv",delimiter=",")

    y_val = y_val.values
    y_test = y_test.values
    evallist = [(dm_train, 'train'), (dm_val, 'valid'), (dm_test, 'test')]

    best_val_ndcg = 0.0
    best_test_ndcg = 0

    best_val_hit = 0.0
    best_test_hit = 0

    tree_max_depth = 4  #[3,4,5,6] 6
    xgb_num_round = 50   #[100,200,300,400,500]

    reg_lambda = l2_weight
    learning_rate = lr_num

    print('learning_rate,reg_lambda') 
    print(learning_rate,reg_lambda) 
    

    neg_num = 512


    ind_params = {
                'learning_rate': learning_rate,
                # 'scale_pos_weight': scale_pos_weight,

                'max_depth': tree_max_depth,

                # 'min_child_weight': min_child_weight,
                # 'gamma': gamma,
                
                # 'subsample': subsample,
                # 'colsample_bytree': colsample_bytree,
                
                # 'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,

                'booster':'gbtree',
                'objective': 'binary:logistic',
                'seed': seed,
                'nthread': 6,
                'base_score': 0.5,
                'eval_metric': 'logloss',
                'tree_method':'gpu_hist',
                'gpu_id':device,
                }

    xgb_model = xgb.train(ind_params, dm_train, xgb_num_round, evallist, early_stopping_rounds=10, verbose_eval=20)

    xgb_model.set_param
    y_train_pred = xgb_model.predict(dm_train)
    y_val_pred = xgb_model.predict(dm_val)
    y_test_pred = xgb_model.predict(dm_test)

    valid_log_loss = log_loss(y_val, y_val_pred)
    test_log_loss = log_loss(y_test, y_test_pred)
    val_hit, val_ndcg = evaluate(y_val, y_val_pred, K=10, row_len=51)
    test_hit, test_ndcg = evaluate(y_test, y_test_pred, K=10, row_len=51)

    valid_auc = roc_auc_score(y_val, y_val_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)

    print("test_ndcg,test_hit,test_log_loss,test_auc")
    print(test_ndcg,test_hit,test_log_loss,test_auc)     

    # print("best_test_ndcg,best_val_ndcg,best_val_hit,best_test_hit,best_val_loss,best_test_loss")# 0.6740
    # print(best_test_ndcg,best_val_ndcg,best_val_hit,best_test_hit,best_val_loss,best_test_loss) 
    print('end')

    #SAVE


    gbdt_model_out_path = tree_model_path+'/test_'+str(testnum)
    mkdir(gbdt_model_out_path)
            
    xgb_model.save_model(gbdt_model_out_path + "/xgbst.model")
    xgb_model.dump_model(gbdt_model_out_path + "/xgbst.dump.raw.txt")
    xgb_model.dump_model(gbdt_model_out_path + "/xgbst.dump.nice.txt",  data_path+"/feature_map_new3.txt")

    np_train = xgb_model.predict(dm_train, pred_leaf=True)
    np_test = xgb_model.predict(dm_test, pred_leaf=True)
    np_val = xgb_model.predict(dm_val, pred_leaf=True)




    # # begin
    add_num = math.pow(2,tree_max_depth+1)
    
    
    for col in range(np_train.shape[1]):
        if col>0:
            np_train[:,col] = np_train[:,col]+col*add_num
            np_test[:,col] = np_test[:,col]+col*add_num
            np_val[:,col] = np_val[:,col]+col*add_num
    np_train = np_train.astype(int)
    np_test = np_test.astype(int)
    np_val = np_val.astype(int)
    np.save(gbdt_model_out_path+"/raw_train_xgb_np", np_train)
    np.save(gbdt_model_out_path+"/raw_test_xgb_np", np_test)
    np.save(gbdt_model_out_path+"/raw_val_xgb_np", np_val)
    # end



