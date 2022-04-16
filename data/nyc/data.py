import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# nyc = pd.read_csv('/storage/ywwei/qxy/tem/nyc_data/New_York_City_Restaurant_Complete_Review.csv',delimiter='\t')


# user = nyc[['uid_index','uage','ugender','ulevel','ustyle','ucountry']]
# item = nyc[['iid','iattribute','iprice','irating']]
# u_i = nyc[['uid_index','iid']]

# u_i = u_i.drop_duplicates(keep = 'first')
# user = user.drop_duplicates(keep = 'first')
# item = item.drop_duplicates(keep = 'first')

# u_i.to_csv('nyc_u_i.csv', index=0)
# user.to_csv('nyc_user.csv', index=0)
# item.to_csv('nyc_item.csv', index=0)

# u_i = pd.read_csv('nyc_u_i.csv',delimiter=',')
# u_i.rename(columns={'uid_index':'uid'},inplace=True)
# u_i.to_csv('nyc_u_i.csv', index=0)
# (15232,)
# (15232, 6)
# (15232, 2)
# (15232, 5)
# (15232, 19)
# (15232, 118)
# (15232, 151)


# #user
# user= pd.read_csv('nyc_user.csv',delimiter=',')
# user.rename(columns={'uid_index':'uid'},inplace=True)
# #user id
# user_id = user['uid']
# #user age
# user_age = user['uage']
# user_age = user_age.str.replace(' ', '_')
# user_age = pd.get_dummies(user_age,prefix='uage')
# user_age.to_csv('./nyc_user_age.csv', index=0)
# print(user_age.shape)
# print(user_age.columns)
# #user gender
# user_gender = user['ugender']
# user_gender = user_gender.str.replace(' ', '_')
# user_gender = pd.get_dummies(user_gender,prefix='ugender')
# user_gender.to_csv('./nyc_user_gender.csv', index=0)
# print(user_gender.shape)
# print(user_gender.columns)
# #user level
# user_level = user['ulevel']
# user_level = user_level.str.replace(' ', '_')
# user_level = pd.get_dummies(user_level,prefix='ulevel')
# user_level.to_csv('./nyc_user_level.csv', index=0)
# print(user_level.shape)
# print(user_level.columns)
# #user_country
# user_country = user['ucountry']
# user_country = user_country.str.replace(' ', '_')
# user_country = pd.get_dummies(user_country,prefix='ucountry')
# user_country.to_csv('./nyc_user_country.csv', index=0)
# print(user_country.shape)
# print(user_country.columns)

# # #user style
# # print(user.shape)
# # user_style = user['ustyle']


# # user_style = user_style.str.strip(to_strip='[]')
# # user_style = user_style.str.split(pat=',', expand=True)

# # user_style1 = user_style.copy()         
# # col1 = list(user_style.columns)
# # user_style[col1]= user_style1[col1].astype(str)       
# # for i in col1:
# #     user_style[i] = user_style[i].str.strip()
# #     user_style[i] = user_style[i].str.replace(' ', '_')
# #     user_style[i] = user_style[i].str.strip(to_strip='\'')

# # user_style.to_csv('./nyc_user_style0.csv', index=0)
# # user_style = pd.read_csv('./nyc_user_style0.csv',delimiter=",")
# # user_style.fillna('None',inplace = True)
# # col1 = list(user_style.columns)
# # data = pd.DataFrame()
# # for i in col1:
# #     data = pd.concat([data,user_style[i]])

# # data = data.drop_duplicates(keep = 'first')
# # data.columns = ['ustyle']
# # data = data[data['ustyle']!='None']
# # data = pd.get_dummies(data)
# # col = list(data.columns)

# # data_style  = pd.DataFrame(columns=col)
# # print(user_style)
# # for i in range(user_style.shape[0]):
# #     for j in range(user_style.shape[1]):
# #         if i%1000 == 1:
# #             print(i)
# #         if(user_style.iat[i,j]!='None'):
# #             data_style.loc[i,'ustyle_'+user_style.iat[i,j]] = 1
# #         else :
# #             data_style.loc[i,'ustyle_'+user_style.iat[i,j]] = 0
# # data_style = data_style.drop('ustyle_None', axis=1)
# # print(data_style.shape)
# # print(data_style)
# # data_style.fillna('0',inplace = True)
# # data_style = data_style.astype(int)
# # data_style.to_csv('./nyc_user_style.csv', index=0)

# user_style = pd.read_csv('nyc_user_style.csv',delimiter=',')
# user_style.to_csv('./nyc_user_style.csv', index=0)
# user = user_id
# user = pd.concat((user,user_age),axis=1)
# user = pd.concat((user,user_gender),axis=1)
# user = pd.concat((user,user_level),axis=1)
# user = pd.concat((user,user_style),axis=1)
# user = pd.concat((user,user_country),axis=1)
# print(user_id.shape)
# print(user_age.shape)
# print(user_gender.shape)
# print(user_level.shape)
# print(user_style.shape)
# print(user_country.shape)
# print(user.shape)

# print(user_age)
# print(user)
# user_sort = user.sort_values(by='uid')
# print(user_sort)

# user_sort.to_csv('nyc_user_list.csv', index=0)


# (6258,)
# (6258, 7)
# (6258, 3)
# (6258, 100)
# (6258, 111)

[7,3,100,111,6,2,5,19,118]
# # item 

# item= pd.read_csv('nyc_item.csv',delimiter=',')

# # item id
# iid = item['iid']
# # item_rating
# item_rating = item['irating']
# item_rating = pd.get_dummies(item_rating,prefix='irating')
# item_rating.to_csv('nyc_item_rating.csv', index=0)
# print(item_rating.shape)
# print(item_rating.columns)
# # item_price
# item_price = item['iprice']
# item_price = pd.get_dummies(item_price,prefix='iprice')
# item_price.to_csv('nyc_item_price.csv', index=0)
# print(item_price.shape)
# # print(item_price.columns)


# # # item_attribute
# # item_attribute = item['iattribute']


# # item_attribute = item_attribute.str.strip(to_strip='[]')
# # item_attribute = item_attribute.str.split(pat=',', expand=True)

# # item_attribute1 = item_attribute.copy()         
# # col1 = list(item_attribute.columns)
# # item_attribute[col1]= item_attribute1[col1].astype(str)       
# # for i in col1:
# #     item_attribute[i] = item_attribute[i].str.strip()
# #     item_attribute[i] = item_attribute[i].str.replace(' ', '_')
# #     item_attribute[i] = item_attribute[i].str.strip(to_strip='\'')
# # for i in range(item_attribute.shape[0]):
# #     if i%1000 == 1:
# #         print(i)
# #     for j in range(item_attribute.shape[1]):
# #         if(("of" in item_attribute.iat[i,j]) | ("trkP" in item_attribute.iat[i,j]) | ("overrideIndex_" in item_attribute.iat[i,j]) | ("_in_" in item_attribute.iat[i,j])):
# #             item_attribute.iat[i,j] = 'None'

# # item_attribute.to_csv('./nyc_item_attribute0.csv', index=0)
# # item_attribute = pd.read_csv('./nyc_item_attribute0.csv',delimiter=",")
# # item_attribute.fillna('None',inplace = True)
# # col1 = list(item_attribute.columns)
# # data = pd.DataFrame()
# # for i in col1:
# #     data = pd.concat([data,item_attribute[i]])

# # data = data.drop_duplicates(keep = 'first')
# # data.columns = ['iattribute']
# # data = data[data['iattribute']!='None']
# # print(data.shape)
# # print(data)
# # data = data.sort_values(by='iattribute')
# # data = pd.get_dummies(data)
# # col = list(data.columns)

# # data_attribute  = pd.DataFrame(columns=col)
# # print(col)
# # for i in range(item_attribute.shape[0]):
# #     if i%1000 == 1:
# #         print(i)
# #     for j in range(item_attribute.shape[1]):
# #         if(item_attribute.iat[i,j]!='None'):
# #             data_attribute.loc[i,'iattribute_'+item_attribute.iat[i,j]] = 1
# #         else :
# #             data_attribute.loc[i,'iattribute_'+item_attribute.iat[i,j]] = 0
# # data_attribute = data_attribute.drop('iattribute_None', axis=1)
# # print(data_attribute.shape)
# # print(data_attribute)
# # data_attribute.fillna('0',inplace = True)
# # data_attribute = data_attribute.astype(int)
# # data_attribute.to_csv('./nyc_item_attribute.csv', index=0)


# item_attribute = pd.read_csv('nyc_item_attribute.csv',delimiter=',')
# item_attribute = item_attribute.drop('iattribute_None', axis=1)
# item_attribute.to_csv('nyc_item_attribute.csv', index=0)
# item = iid
# item = pd.concat((item,item_rating),axis=1)
# item = pd.concat((item,item_price),axis=1)
# item = pd.concat((item,item_attribute),axis=1)
# print(iid.shape)
# print(item_rating.shape)
# print(item_price.shape)
# print(item_attribute.shape)
# print(item.shape)
# item_sort = item.sort_values(by='iid')
# item_sort.to_csv('nyc_item_list.csv', index=0)


# u_i

u_i = pd.read_csv('nyc_u_i.csv',delimiter=',')
user = pd.read_csv('nyc_user_list.csv',delimiter=',')

uid = user['uid']

train_set = pd.DataFrame()
test_set = pd.DataFrame()
val_set = pd.DataFrame()
for i in range(user.shape[0]):

    if i%1000 == 0:
        print(i)

    uid_index = uid[i]
    u_i_i = u_i.query('uid==@uid_index').reset_index(drop=True)
    num = u_i_i.shape[0]#观测到的uid iid
    train_set_u, test_set_u = train_test_split(u_i_i, test_size = 0.2, random_state=2021)
    if train_set_u.shape[0]!=1:
        train_set_u, val_set_u = train_test_split(train_set_u, test_size = 0.125, random_state=2021)
        train_set = pd.concat((train_set,train_set_u),axis = 0)
        test_set = pd.concat((test_set,test_set_u),axis = 0)
        val_set = pd.concat((val_set,val_set_u),axis = 0)
    else :
        train_set = pd.concat((train_set,train_set_u),axis = 0)
        test_set = pd.concat((test_set,test_set_u),axis = 0)




train_set = train_set.sample(frac=1).reset_index(drop=True)
test_set = test_set.sample(frac=1).reset_index(drop=True)
val_set = val_set.sample(frac=1).reset_index(drop=True)
train_set.to_csv("nyc_u_i_train_set.csv",index=0)
test_set.to_csv("nyc_u_i_test_set.csv",index=0)
val_set.to_csv("nyc_u_i_val_set.csv",index=0)