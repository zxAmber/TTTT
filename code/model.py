import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
from sklearn.cluster import KMeans
from time import time
import scipy.sparse as sp


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class CGCN(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(CGCN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['CGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when CGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        # H
        self.all_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        self.Graph = self.dataset.getDistantMatrix(self.all_emb)
        print(f"cgcn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        # users_emb = self.embedding_user.weight
        # items_emb = self.embedding_item.weight
        # all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [self.all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], self.all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                self.all_emb = side_emb
            else:
                self.all_emb = torch.sparse.mm(g_droped, self.all_emb)
            embs.append(self.all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getKmeans(self, x):
        # x shape: [2048, 150, 64]
        batch_size = x.shape[0]  # 2048

        centroids_list = []
        for i in range(batch_size):
            x_np = x[i].cpu().detach().numpy()  # Shape: [150, 64]
            kmeans = KMeans(n_clusters=self.config['num_clusters'], random_state=0).fit(x_np)
            centroids = kmeans.cluster_centers_
            centroids_tensor = torch.tensor(centroids, dtype=torch.float32).to(x.device)
            centroids_list.append(centroids_tensor)

        centroids = torch.stack(centroids_list)  # Shape: [2048, num_clusters, 64]
        return centroids

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)  # batch_size*1*dim
        pos_emb_ego = self.embedding_item(pos_items)  # batch_size*n*dim
        # ----------------------------- todo add some CLUSTER algo -----------------------------
        # means(batchsize*n*dim) -> batchsize*1*dim
        #
        # 神经网络聚类
        pos_emb = torch.mean(pos_emb, dim=1, keepdim=False)
        # pos_emb_ego = torch.mean(pos_emb_ego, dim=1, keepdim=False)
        ## ----------------------------- KMEAN 聚合模块 -----------------------------
        kmeansResult = self.getKmeans(pos_emb_ego)

        # TODO 其他距离的方案
        users_emb_ego_exp = users_emb_ego[:, None, :]
        user_posi_dist = 1.0 / torch.sqrt(torch.sum((users_emb_ego_exp - kmeansResult) ** 2, dim=-1)) + 1e-8
        # 权重归一化 --论文中要提出
        user_pos_weight = user_posi_dist / torch.sum(user_posi_dist, dim=1, keepdim=True)
        pos_emb_ego = torch.sum(kmeansResult * user_pos_weight[:, :, None], dim=1)  # 对聚类中心加权求平均
        #
        # ==============================================================
        neg_emb_ego = self.embedding_item(neg_items)  # batch_size*1*dim
        # return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        # L2正则化项
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))

        # reg_loss = (1 / 2) * (userPosEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / float(len(users))

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    # def forward(self, users, items):
    #     # compute embedding
    #     all_users, all_items = self.computer()
    #     # print('forward')
    #     # all_users, all_items = self.computer()
    #     users_emb = all_users[users]
    #     items_emb = all_items[items]
    #     inner_pro = torch.mul(users_emb, items_emb)
    #     gamma = torch.sum(inner_pro, dim=1)
    #     return gamma


# class PureMF(BasicModel):
#     def __init__(self,
#                  config: dict,
#                  dataset: BasicDataset):
#         super(PureMF, self).__init__()
#         self.num_users = dataset.n_users
#         self.num_items = dataset.m_items
#         self.latent_dim = config['latent_dim_rec']
#         self.f = nn.Sigmoid()
#         self.__init_weight()
#
#     def __init_weight(self):
#         self.embedding_user = torch.nn.Embedding(
#             num_embeddings=self.num_users, embedding_dim=self.latent_dim)
#         self.embedding_item = torch.nn.Embedding(
#             num_embeddings=self.num_items, embedding_dim=self.latent_dim)
#         print("using Normal distribution N(0,1) initialization for PureMF")
#
#     def getUsersRating(self, users):
#         users = users.long()
#         users_emb = self.embedding_user(users)
#         items_emb = self.embedding_item.weight
#         scores = torch.matmul(users_emb, items_emb.t())
#         return self.f(scores)
#
#     def bpr_loss(self, users, pos, neg):
#         users_emb = self.embedding_user(users.long())
#         pos_emb = self.embedding_item(pos.long())
#         neg_emb = self.embedding_item(neg.long())
#         pos_scores = torch.sum(users_emb * pos_emb, dim=1)
#         neg_scores = torch.sum(users_emb * neg_emb, dim=1)
#         loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
#         reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
#                               pos_emb.norm(2).pow(2) +
#                               neg_emb.norm(2).pow(2)) / float(len(users))
#         return loss, reg_loss
#
#     def forward(self, users, items):
#         users = users.long()
#         items = items.long()
#         users_emb = self.embedding_user(users)
#         items_emb = self.embedding_item(items)
#         scores = torch.sum(users_emb * items_emb, dim=1)
#         return self.f(scores)
