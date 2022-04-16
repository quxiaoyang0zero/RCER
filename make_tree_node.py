import re
import pandas as pd
import numpy as np
from pandas.core.indexing import IndexSlice
from torch import tensor
from tree import *
import torch 
from torch.utils.data import DataLoader , TensorDataset
import torch.nn as nn

def get_tree_str(lines):

    tree_str = np.empty(shape=[0, 4])
    tree_leaf = []
    leaf_pre = []

    



    for node in lines:
        
        if "leaf=" in node:
            value = int(node.split(":")[0])
            tree_leaf.append(value)
            pre = float(node.split("=")[1])
            leaf_pre.append(pre)


        else:
            fa_id = node.split(":")[0]
            feature_id = re.split("f", node)[1]
            feature_id = re.split("<", feature_id)[0]
            child_no = re.split("yes=", node)[1]
            child_no = re.split(",", child_no)[0]
            child_yes = re.split("no=", node)[1]
            child_yes = re.split(",", child_yes)[0]
            str = [int(fa_id),int(feature_id),int(child_yes),int(child_no)]
            str = np.reshape(str,(1,4)) 
            tree_str = np.append(tree_str, str, axis=0)

    return tree_str,tree_leaf,leaf_pre

def parse_trees(tree_filepath,tree_max_num):
    lines = []
    for line in open(tree_filepath):
        lines.append(line.strip())

    all_tree_str = {}
    all_tree_leaf = {}
    all_leaf_pre = {}
    all_leaf_value = {}

    tree_id = 0
    tmp_tree_lines = []
    for line in lines[1:]:
        if "booster" in line:
            tree_str,tree_leaf,leaf_pre = get_tree_str(tmp_tree_lines)

            tree_str = np.array(tree_str).astype(int)
            tree_leaf = np.array(tree_leaf).astype(int)
            leaf_pre = np.array(leaf_pre).astype(float)

            
            all_tree_str['tree'+str(tree_id)] = tree_str + tree_id*tree_max_num
            all_tree_leaf['tree'+str(tree_id)] = tree_leaf + tree_id*tree_max_num
            for leaf_index in range(len(tree_leaf)):
                all_leaf_value['leaf'+str(tree_leaf[leaf_index])] = leaf_pre[leaf_index]
            all_leaf_value['tree'+str(tree_id)] = all_leaf_value
            Z = zip(tree_leaf,leaf_pre)
            Z = sorted(Z,reverse=False)
            tree_leaf_sort,leaf_pre_new = zip(*Z)
            all_leaf_pre['tree'+str(tree_id)] = leaf_pre_new
            tmp_tree_lines = []
            tree_id = int(re.split(r"[\[\]]", line)[1])
            
        else:
            tmp_tree_lines.append(line)

    # process the last tree
    tree_str,tree_leaf,leaf_pre = get_tree_str(tmp_tree_lines)
    tree_str = np.array(tree_str).astype(int)
    tree_leaf = np.array(tree_leaf).astype(int)
    leaf_pre = np.array(leaf_pre).astype(float)
    all_tree_str['tree'+str(tree_id)] = tree_str + tree_id*tree_max_num
    all_tree_leaf['tree'+str(tree_id)] = tree_leaf + tree_id*tree_max_num
    for leaf_index in range(len(tree_leaf)):
        all_leaf_value['leaf'+str(tree_leaf[leaf_index])] = leaf_pre[leaf_index]
    all_leaf_value['tree'+str(tree_id)] = all_leaf_value
    Z = zip(tree_leaf,leaf_pre)
    Z = sorted(Z,reverse=False)
    tree_leaf_sort,leaf_pre_new = zip(*Z)
    all_leaf_pre['tree'+str(tree_id)] = leaf_pre_new

    return all_tree_str,all_tree_leaf,all_leaf_pre

def make_tree_dic(all_tree_str,tree_max_num,tree_num):

    all_tree_dic = {}

    for tree_id in range(tree_num):

        tree_dic = {}
        tree_str_np = all_tree_str['tree'+str(tree_id)]

        a0 = TreeNode('node'+str(tree_id*tree_max_num)) 
        tree_dic['node'+str(tree_id*tree_max_num)] = a0

        for np_id in range(tree_str_np.shape[0]):
            str_np = tree_str_np[np_id]
            yes_id = str_np[2]
            no_id = str_np[3]
            feat_id = str_np[1] - (tree_id*tree_max_num)
            father_id = str_np[0]
            b0 = tree_dic['node'+str(int(father_id))]
            b1 = b0.add_child('node'+str(int(yes_id)),feat_id,1)#yes
            b2 = b0.add_child('node'+str(int(no_id)),feat_id,0)
            tree_dic['node'+str(int(yes_id))] = b1
            tree_dic['node'+str(int(no_id))] = b2
        all_tree_dic['tree'+str(tree_id)] = tree_dic

    return all_tree_dic


def make_brolist(all_tree_dic,all_tree_leaf,tree_leaf_dic,tree_num,all_leaf_pre):



    tree_brolist_dic = {}
    tree_jump_dic = {}
    tree_leaf_num = []
    tree_value_dic = {}

    for tree_id in range(tree_num):
        tree_dic = all_tree_dic['tree'+str(tree_id)]
        leaf_value = all_leaf_pre['tree'+str(tree_id)]
        tree_leaf_node = []
        
        tree_leaf = all_tree_leaf['tree'+str(tree_id)]
        finall_node = np.empty(shape=[0, len(tree_leaf)])
        finall_jump = np.empty(shape=[0, len(tree_leaf)])
    

        for leaf_id_index in range(len(tree_leaf)):         
            tree_leaf_node.append(tree_dic['node'+str(int(tree_leaf[leaf_id_index]))])
        for leaf_id_index in range(len(tree_leaf)): 

            curr_leaf = tree_leaf_node[leaf_id_index]
            curr_leaf_list = curr_leaf.parent_list
            curr_leaf_list.append(leaf_id_index)

            curr_leaf_value = curr_leaf

            tree_leaf_sort= []
            tree_leaf_sort2 = []
            tree_leaf_sort_value = []

            for other_leaf_id_index in range(len(tree_leaf)): 
                if other_leaf_id_index != leaf_id_index:

                    other_leaf_list = tree_leaf_node[other_leaf_id_index].parent_list
                    other_leaf_list.append(other_leaf_id_index)
                    same_list = list(set(curr_leaf_list)&set(other_leaf_list))
                    same_list2 = list(set(curr_leaf_list)|set(other_leaf_list))
                    tree_leaf_sort.append(len(same_list))
                    tree_leaf_sort2.append(len(same_list2)-len(same_list))
                else:
                    tree_leaf_sort.append(len(curr_leaf_list))
                    tree_leaf_sort2.append(0)

            Z = zip(tree_leaf_sort,tree_leaf)
            Z = sorted(Z,reverse=True)
            tree_leaf_sort_new,tree_leaf_new = zip(*Z)

            # Z = zip(tree_leaf_sort2,tree_leaf)
            # Z = sorted(Z,reverse=False)
            # tree_leaf_sort_new,tree_leaf_new = zip(*Z)

            tree_leaf_sort_new_np = np.array(list(tree_leaf_sort_new))
            tree_leaf_new_np = np.array(list(tree_leaf_new))

            tree_leaf_new_list = []
            for node in tree_leaf_new_np:
                tree_leaf_new_list.append(tree_leaf_dic[node])
            tree_leaf_new_np = np.array(tree_leaf_new_list)

            tree_leaf_new_np = np.reshape(tree_leaf_new_np,(1,-1))
            tree_leaf_sort_new_np = np.reshape(tree_leaf_sort_new_np,(1,-1))
            
            finall_jump = np.append(finall_jump, tree_leaf_sort_new_np, axis=0)
            finall_node = np.append(finall_node, tree_leaf_new_np, axis=0)
        tree_jump_dic['tree'+str(tree_id)] = finall_jump
        tree_brolist_dic['tree'+str(tree_id)] = finall_node
        tree_leaf_num.append(finall_node.shape[0])

    return tree_jump_dic,tree_brolist_dic,tree_leaf_num

def change_node_np(node_np,tree_leaf_dic):

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    batch_size = 20480
    num =(node_np.shape[0]-(node_np.shape[0]%batch_size))/batch_size+1
    num = int(num)
    print(num)
    finall_node_np = np.empty(shape=node_np.shape)
    key = list(tree_leaf_dic.keys())
    value = list(tree_leaf_dic.values())
    key = np.array(key)
    value = np.array(value)
    max_num = np.max(key)+1

    key_one_hot = torch.zeros(max_num)
    key_one_hot = key_one_hot-1
    key_index = 0
    for index in range(max_num):
        if index == key[key_index]:
            key_one_hot[index] = value[key_index]
            key_index += 1
    key_one_hot = key_one_hot+1

    for index in range(num):

        print(index)

        begin = index*batch_size
        end = (index+1)*batch_size
        if end>node_np.shape[0]:
            end = node_np.shape[0]
        batch_np = node_np[begin:end,:]
        batch_np = batch_np.astype(int)


        batch_ten = torch.tensor(batch_np)

        x = torch.ones(batch_ten.shape[0],batch_ten.shape[1])
        data_one_hot = torch.zeros(batch_ten.shape[0],max_num).scatter(1,batch_ten,x)

        data_one_hot = data_one_hot.float().to(device)
        batch_size = data_one_hot.shape[0]


        feats_embed_layers = key_one_hot.repeat(batch_size,1).to(device)
        pos_embed = data_one_hot * feats_embed_layers

        pos_embed = torch.masked_select(pos_embed, pos_embed!=0)
        pos_embed = pos_embed.reshape([batch_size,-1])
        pos_embed = pos_embed-1
        pos_embed_np = pos_embed.cpu().numpy()

        finall_node_np[begin:end,:] = pos_embed_np

    return finall_node_np

def make_raw_x(all_tree_dic,feature_num,tree_leaf_dic,all_tree_leaf):


    leaf_num = len(tree_leaf_dic)

    all_leaf_feature = np.zeros([leaf_num,feature_num])

    for tree_index in range(len(all_tree_dic)):

        print('tree')
        print(tree_index)

        tree_leaf = all_tree_leaf['tree'+str(tree_index)]
        tree_dic = all_tree_dic['tree'+str(tree_index)]

        for leaf_index in range(len(tree_leaf)):

            leaf_feature = np.zeros([feature_num])
            leaf_id = tree_leaf[leaf_index]
            new_leaf_id = tree_leaf_dic[leaf_id]

            node = tree_dic['node'+str(leaf_id)]

            feat = node.feat
            feat_yes = node.feat_yes
        
            for feat_index in range(len(feat)):
                leaf_feature[feat[feat_index]] = feat_yes[feat_index]

            all_leaf_feature[new_leaf_id] = leaf_feature


    return all_leaf_feature



if __name__ == '__main__':


    tree_max_num = 32
    testnum = 14

    data_path = 'new_data_new3'
    input_data_path = data_path+"/tem_in"

    nice_csv = data_path+'/tree_model/test_'+str(testnum)+'/xgbst.dump.nice.txt'
    raw_csv = data_path+'/tree_model/test_'+str(testnum)+'/xgbst.dump.raw.txt'

    all_tree_str,all_tree_leaf,all_leaf_pre = parse_trees(raw_csv,tree_max_num)
    np_train = np.load(data_path+"/tree_model/test_"+str(testnum)+"/raw_train_xgb_np.npy")
    np_test = np.load(data_path+"/tree_model/test_"+str(testnum)+"/raw_test_xgb_np.npy")
    np_val = np.load(data_path+"/tree_model/test_"+str(testnum)+"/raw_val_xgb_np.npy")

    train_raw_set = pd.read_csv(input_data_path + '/london_train_set.csv',delimiter=",")
    max_num = np.max(np.max(train_raw_set,axis=1))
    print(max_num)

    
    tree_num = np_train.shape[1]
    tree_leaf_list = []
    tree_leaf_dic = {}

    for index in range(len(all_tree_leaf)):
        cur_leaf = np.array(all_tree_leaf["tree"+str(index)])
        cur_leaf.sort()
        tree_leaf_list = np.append(tree_leaf_list, cur_leaf, axis=0)
    tree_leaf_list = tree_leaf_list.astype(int)
    for index in range(len(tree_leaf_list)):
        tree_leaf_dic[tree_leaf_list[index]] = index

    all_tree_dic = make_tree_dic(all_tree_str,tree_max_num,tree_num)
    all_leaf_feature = make_raw_x(all_tree_dic,max_num,tree_leaf_dic,all_tree_leaf)
    tree_jump_dic,tree_brolist_dic,tree_leaf_num = make_brolist(all_tree_dic,all_tree_leaf,tree_leaf_dic,tree_num,all_leaf_pre)

    


    np_train_new = change_node_np(np_train,tree_leaf_dic)
    print(np_train.shape)
    print(np_train_new.shape)
    np_test_new = change_node_np(np_test,tree_leaf_dic)
    print(np_test.shape)
    print(np_test_new.shape) 
    np_val_new = change_node_np(np_val,tree_leaf_dic)
    print(np_val.shape)
    print(np_val_new.shape)
    np.save(data_path+"/tree_model/test_"+str(testnum)+"/train_xgb_np", np_train_new)
    np.save(data_path+"/tree_model/test_"+str(testnum)+"/test_xgb_np", np_test_new)
    np.save(data_path+"/tree_model/test_"+str(testnum)+"/val_xgb_np", np_val_new)
    print(" end")
    for index in range(tree_num):
        tree_leaf_raw = all_tree_leaf["tree"+str(index)]
        tree_leaf_new = []
        for node in tree_leaf_raw:
            tree_leaf_new.append(tree_leaf_dic[node])
        tree_leaf_new = np.array(tree_leaf_new)
        all_tree_leaf["tree"+str(index)] = tree_leaf_new
    print( type(tree_brolist_dic))
    print( type(tree_jump_dic))
    print( type(all_tree_leaf))

    all_leaf_pre_list = []
    for k in all_leaf_pre.values():
        k = list(k)
        all_leaf_pre_list += k
    all_leaf_pre_np = np.array(all_leaf_pre_list)
    # all_leaf_pre_np = all_leaf_pre_np.reshape(-1) 

    np.save(data_path+"/tree_model/test_"+str(testnum)+'/tree_brolist_dic.npy', tree_brolist_dic) 
    np.save(data_path+"/tree_model/test_"+str(testnum)+'/tree_jump_dic.npy', tree_jump_dic) 
    np.save(data_path+"/tree_model/test_"+str(testnum)+'/all_tree_leaf.npy', all_tree_leaf) 
    np.save(data_path+"/tree_model/test_"+str(testnum)+'/tree_leaf_dic.npy', tree_leaf_dic) 
    np.save(data_path+"/tree_model/test_"+str(testnum)+'/tree_leaf_num.npy', tree_leaf_num)
    np.save(data_path+"/tree_model/test_"+str(testnum)+'/all_leaf_pre_np.npy', all_leaf_pre_np) 
    np.save(data_path+"/tree_model/test_"+str(testnum)+'/all_leaf_feature.npy', all_leaf_feature) 



    print(" all end")




        
        



