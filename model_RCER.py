import torch 
import time
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class tem_con(nn.Module):
    def __init__(self,embed_dim,feat_num,uid_num,iid_num,mode,tree_num,raw_x_num,device,pos_feat,pos_pro,temperature):
        super(tem_con, self).__init__()

        print('model_1128')
        self.temperature = temperature
        self.mode = mode 
        self.tree_num = tree_num
        self.raw_x_num = raw_x_num
        self.device = device
        self.pos_pro = pos_pro
        self.feat_num = feat_num

        self.uid_embed_layers = nn.init.xavier_normal_(nn.Parameter(torch.rand([uid_num, embed_dim])))
        self.iid_embed_layers = nn.init.xavier_normal_(nn.Parameter(torch.rand([iid_num, embed_dim])))
        self.feats_embed_layers = nn.init.xavier_normal_(nn.Parameter(torch.rand([feat_num, embed_dim])))
        
        self.tem_b0 = nn.Parameter(torch.rand([1,1]))
        self.tem_bt = nn.init.xavier_normal_(nn.Parameter(torch.rand([self.raw_x_num-1, 1])))
        self.tem_r_id = nn.init.xavier_normal_(nn.Parameter(torch.rand([1, embed_dim])))
        self.tem_r_feats = nn.init.xavier_normal_(nn.Parameter(torch.rand([embed_dim, 1])))

        self.atten_w = nn.init.xavier_normal_(nn.Parameter(torch.rand([1, 2*embed_dim])))


        self.head_layer_1 = nn.Linear(embed_dim, embed_dim)
        self.head_layer_2 = nn.Linear(embed_dim, embed_dim)
        self.head_layer_3 = nn.Linear(embed_dim, embed_dim)

        self.trasformer_layer = nn.TransformerEncoderLayer(embed_dim, nhead=2, dim_feedforward=embed_dim)

    def att_layers(self, feats, id_mul):
        id_mul = id_mul.repeat(self.tree_num,1,1) #tree_num*batch*embed_dim
        id_mul = id_mul.permute(1,0,2) #batch*tree_num*embed_dim
        score = torch.cat((feats,id_mul),axis = 2) #batch*tree_num*(2*embed_dim)
        score = torch.tensordot(score,self.atten_w,dims=([-1],[-1]))#batch*tree_num*att_dim

        weight = F.softmax(score, dim=1)
        return weight

    def att_layers2(self, feats, id_mul):
        id_mul = id_mul.unsqueeze(2)#batch*embed_dim*1
        score = torch.matmul(feats, id_mul)
        weight = F.softmax(score, dim=1)
        # print(wieght.size())
        return weight

    def att_layers3(self, feats, id_concat):#batch*2embed_dim
        id_concat = id_concat.unsqueeze(1).repeat(1, self.tree_num, 1)
        temp_concat = torch.cat((feats, id_concat), dim=-1)
        score = torch.matmul(temp_concat, self.atten_w)
        weight = F.softmax(score, dim=1)
        return weight


    def head(self, feat):
        temp = self.head_layer_1(feat)
        temp = F.leaky_relu(temp)
        temp = self.head_layer_2(temp)

        return temp

    def contrstive_loss_f(self, anchor, pos, neg_matrix):
        all_feats = self.feats_embed_layers
        anchor = self.head(anchor)
        pos = self.head(pos)
        all_feats = self.head(all_feats)
        anchor = F.normalize(anchor)
        pos = F.normalize(pos)
        all_feats = F.normalize(all_feats)
        pos_score = torch.sum(anchor*pos, dim=1)/self.temperature
        pos_score = torch.exp(pos_score)
        all_score = torch.matmul(anchor, all_feats.t())
        all_score = torch.exp(all_score/self.temperature)
        neg_score = torch.sum(all_score*neg_matrix,dim=1)
        loss_con = (-torch.log(pos_score/(pos_score+neg_score))).mean()
        return loss_con

    def simsiam_loss(self, anchor, pos):
        anchor_head = self.head(anchor)
        pos_head = self.head(pos)
        anchor = F.normalize(anchor).detach()
        pos = F.normalize(pos).detach()
        anchor_head = F.normalize(anchor_head)
        pos_head = F.normalize(pos_head)

        loss_1 = -torch.sum(anchor_head*pos, dim=1).mean()
        loss_2 = -torch.sum(pos_head*anchor, dim=1).mean()
        loss_con = loss_1/2 + loss_2/2
        return loss_con


    def forward(self,pos, neg, uid, iid, raw_x):
        pos = pos.long().to(self.device)
        neg = neg.to(self.device)
        uid = uid.long().to(self.device)
        iid = iid.long().to(self.device)
        ##################################################################################################
        xt = raw_x.to(self.device)
        xt = xt.long()
        if raw_x.shape[1] == self.raw_x_num-1:
            xt = xt.float()
            output1 = self.tem_b0 + torch.mm(xt, self.tem_bt)
        else :
            xt = torch.nn.functional.one_hot(xt,num_classes = self.raw_x_num)
            xt = torch.sum(xt, dim = 1)
            xt = xt[:,:xt.shape[1]-1]
            xt = xt.float()
            output1 = self.tem_b0 + torch.mm(xt, self.tem_bt)
        ##################################################################################################
        uid_embed = self.uid_embed_layers[uid]
        iid_embed = self.iid_embed_layers[iid]
        id_mul = torch.mul(uid_embed,iid_embed)
        # output2 = torch.sum(id_mul, dim=1, keepdim=True)
        output2 = torch.tensordot(id_mul, self.tem_r_id, dims=([1],[1]))
        ##################################################################################################
        activted_feat = self.feats_embed_layers[pos]

        transfered_activted_feat = self.trasformer_layer(activted_feat)
        # transfered_activted_feat = activted_feat

        weight = self.att_layers(transfered_activted_feat, id_mul)
        # id_concat = torch.cat((uid_embed, iid_embed), dim=-1)
        # weight = self.att_layers3(transfered_activted_feat, id_concat)
        att_feat = weight * transfered_activted_feat
        att_feat = torch.sum(att_feat,dim = 1)
        output3 = torch.matmul(att_feat, self.tem_r_feats)
        ##################################################################################################
        output = torch.sigmoid(output1+output2+output3)


        ##################################################################################################
        if self.training is True:
            anchor_f = torch.mean(transfered_activted_feat, dim=1)
            pos_index = torch.rand([pos.shape[0],pos.shape[1],1]).to(self.device)
            pos_index[pos_index < self.pos_pro] = -1
            pos_index[pos_index > self.pos_pro] = 0
            pos_index[pos_index == -1] = 1
            pos_f = transfered_activted_feat * pos_index
            pos_f = torch.mean(pos_f, dim=1)

            con_loss = self.simsiam_loss(anchor_f, pos_f)
            ##################################################################################################
            reg_loss = torch.norm(self.uid_embed_layers, 2, dim=1).mean() + torch.norm(self.iid_embed_layers, 2, dim=1).mean() + torch.norm(self.feats_embed_layers, 2, dim=1).mean() 
            reg_loss += torch.norm(self.head_layer_1.weight, 2).mean() + torch.norm(self.head_layer_2.weight, 2).mean() + torch.norm(self.head_layer_3.weight, 2).mean()
            for param in self.trasformer_layer.parameters():
                reg_loss += torch.norm(param, 2).mean()
            ##################################################################################################
        else:
            reg_loss = con_loss = None
        return output, reg_loss, con_loss,weight





