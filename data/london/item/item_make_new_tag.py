import numpy as np
import pandas as pd


london_item_tag = pd.read_csv('data/item/london_item_tag.csv',delimiter=",")



# london_item_tag_num = london_item_tag.apply(lambda x: x.sum())
# london_item_tag_num.to_csv('data/item/london_item_tag_num.csv', index=1)
london_item_tag_num = pd.read_csv('data/item/london_item_tag_num.csv',delimiter=",")


london_item_tag_num1 = london_item_tag_num[london_item_tag_num['num']>3]


tag_feature_list = []

tag_feature_list = london_item_tag_num1['feature'].values.tolist()


tag_new = pd.DataFrame(london_item_tag,columns=tag_feature_list)
print(tag_new)
tag_new.to_csv('data/item/london_item_tag_new3.csv', index=0)