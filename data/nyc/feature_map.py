import pandas as pd 
import numpy as np
user = pd.read_csv('nyc_user_list.csv',delimiter=',')
item = pd.read_csv('nyc_item_list.csv',delimiter=',')
u_i = pd.read_csv('nyc_u_i_train_set.csv',delimiter=",")
data = pd.merge(u_i,item,how='left')
data = pd.merge(data,user,how='left')
print(data.shape)
data.drop(['uid'], axis = 1, inplace = True)
data.drop(['iid'], axis = 1, inplace = True)
print(data.shape)
features = data.columns.values
outfile = open('feature_map_nyc.txt', 'w')    
feature_map_dic = {}
for i, feat in enumerate(features): 
    feature_map_dic[i] = feat
    feat = feat.replace(' ','_')    
    outfile.write('{0}\t{1}\ti\n'.format(i, feat))       
outfile.close()
np.save('feature_map_dic_nyc.npy', feature_map_dic) 