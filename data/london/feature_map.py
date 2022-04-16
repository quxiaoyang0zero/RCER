import pandas as pd 


# data = pd.read_csv('data/london_data_1_5_user5.csv',delimiter=",")
# print(data.shape)
# data.drop(['uid'], axis = 1, inplace = True)
# data.drop(['iid'], axis = 1, inplace = True)
# data.drop(['label'], axis = 1, inplace = True)
# print(data.shape)
# features = data.columns.values
# outfile = open('feature_map.txt', 'w')    
# for i, feat in enumerate(features): 
#     feat = feat.replace(' ','_')    
#     outfile.write('{0}\t{1}\ti\n'.format(i, feat))       
# outfile.close()


# new3
user = pd.read_csv('data/user/london_user_list_new3.csv',delimiter=',')
item = pd.read_csv('data/item/london_item_list_new3.csv',delimiter=',')
u_i = pd.read_csv('data/london_u_i_train_set.csv',delimiter=",")
data = pd.merge(u_i,item,how='left')
data = pd.merge(data,user,how='left')
print(data.shape)
data.drop(['uid'], axis = 1, inplace = True)
data.drop(['iid'], axis = 1, inplace = True)
data.drop(['label'], axis = 1, inplace = True)
print(data.shape)
features = data.columns.values
outfile = open('feature_map_new3.txt', 'w')    
feature_map_dic = {}
for i, feat in enumerate(features): 
    feature_map_dic[i] = feat
    feat = feat.replace(' ','_')    
    outfile.write('{0}\t{1}\ti\n'.format(i, feat))       
outfile.close()
