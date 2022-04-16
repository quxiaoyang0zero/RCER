import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split

def data_london():
    
    item = pd.read_csv('data/london_item_list.csv',delimiter=",")
    u_i = pd.read_csv('data/london_u_i_1_5_user5.csv',delimiter=",")
    user = pd.read_csv('data/london_user_list.csv',delimiter=",")
    data = pd.merge(u_i,item,how='left')
    print(data.shape)
    data = pd.merge(data,user,how='left')
    print(data.shape)
    print('over')

    return data