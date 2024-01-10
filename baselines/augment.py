from model import LightGCN
import numpy as np
from utils import randint_choice
import scipy.sparse as sp
import world
import torch
import torch.nn.functional as F
from precalcul import precalculate
import time
import torch_sparse
from scipy.sparse import csr_matrix
from dataloader import dataset
import torch_geometric
import copy


class ED_Uniform():
    def __init__(self, config, model:LightGCN, precal:precalculate):
        self.config = config
        self.model = model
        self.precal = precal
        self.augAdjMatrix1 = None
        self.augAdjMatrix2 = None

    def Edge_drop_random(self, p_drop):
        n_nodes = self.model.num_users + self.model.num_items
        trainUser = self.model.dataset.trainUser.copy()
        trainItem = self.model.dataset.trainItem.copy()
        keep_idx = randint_choice(len(self.model.dataset.trainUser), size=int(len(self.model.dataset.trainUser) * (1 - p_drop)), replace=False)
        user_np = trainUser[keep_idx]
        item_np = trainItem[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.model.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        if self.config['if_big_matrix']:
            g = self.model.dataset._split_matrix(adj_matrix)
            for fold in g:
                fold.requires_grad = False
        else:
            g = self.model.dataset._convert_sp_mat_to_sp_tensor(adj_matrix).coalesce().to(world.device)
            g.requires_grad = False
        return g

    def get_augAdjMatrix(self):
        p_drop = world.config['p_drop']
        self.augAdjMatrix1 =  self.Edge_drop_random(p_drop)
        self.augAdjMatrix2 =  self.Edge_drop_random(p_drop)


class RW_Uniform(ED_Uniform):
    def __init__(self, config, model, precal):
        super(RW_Uniform, self).__init__(config, model, precal)

    def Random_Walk(self, p_drop):
        aug_g = []
        for layer in range(self.config['num_layers']):
            aug_g.append(self.Edge_drop_random(p_drop))
        return aug_g

    def get_augAdjMatrix(self):
        p_drop = world.config['p_drop']
        self.augAdjMatrix1 =  self.Random_Walk(p_drop)
        self.augAdjMatrix2 =  self.Random_Walk(p_drop)

