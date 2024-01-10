import os
from os.path import join
import random
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import torch
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import world
from world import cprint
from time import time
import networkx as nx
from torch_geometric.data import Data

class dataset():

    def __init__(self, config = world.config, path="../data/yelp2018"):
        # train or test
        print(f'loading [{path}]')
        self.n_user = 0
        self.m_item = 0
        if config['if_valid']:
            train_file = path + '/train_7.txt'
            valid_file = path + '/valid_1.txt'
            test_file = path + '/test.txt'
        else:
            train_file = path + '/train.txt'
            valid_file = path + '/valid.txt'
            test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        validUniqueUsers, validItem, validUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0
        self.validDataSize = 0

        self._TrainPop_item = {}#item's popularity (degree) in the training dataset
        self._TrainPop_user = {}#user's popularity (degree) in the training dataset
        
        self._allPos = {}
        self._allPos_item = {}

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
                    #================Pop=================#
                    for item in items:
                        if item in self._TrainPop_item.keys():
                            self._TrainPop_item[item] += 1
                        else:
                            self._TrainPop_item[item] = 1
                        
                        if item in self._allPos_item.keys():
                            self._allPos_item[item].append(uid)
                        else:
                            self._allPos_item[item] = [uid]

                    if uid in self._TrainPop_user.keys():
                        self._TrainPop_user[uid] += len(items)
                    else:
                        self._TrainPop_user[uid] = len(items)

                    self._allPos[uid] = items
                    #================Pop=================#
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if l[1]:
                        items = [int(i) for i in l[1:]]
                        uid = int(l[0])
                        testUniqueUsers.append(uid)
                        testUser.extend([uid] * len(items))
                        testItem.extend(items)
                        self.m_item = max(self.m_item, max(items))
                        self.n_user = max(self.n_user, uid)
                        self.testDataSize += len(items)
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        
        if os.path.exists(valid_file):
            with open(valid_file) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        if l[1]:
                            items = [int(i) for i in l[1:]]
                            uid = int(l[0])
                            validUniqueUsers.append(uid)
                            validUser.extend([uid] * len(items))
                            validItem.extend(items)
                            self.m_item = max(self.m_item, max(items))
                            self.n_user = max(self.n_user, uid)
                            self.validDataSize += len(items)
            self.validUniqueUsers = np.array(validUniqueUsers)
            self.validUser = np.array(validUser)
            self.validItem = np.array(validItem)
        
        self.m_item += 1
        self.n_user += 1
        
        for i in range(self.m_item):
            if i not in self._TrainPop_item.keys():
                self._TrainPop_item[i] = 1 
        for i in range(self.n_user):
            if i not in self._TrainPop_user.keys():
                self._TrainPop_user[i] = 1 



        self.Graph = None
        print(f"{self.n_user} users and {self.m_item} items")
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{config['dataset']} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)), shape=(self.n_user, self.m_item))
        # pre-calculate
        # self._allPos = self.getUserPosItems(list(range(self.n_user)))
        # self._allPos_item = self.getItemPosUsers(list(range(self.m_item)))
        self.__testDict = self.__build_test()
        if world.config['if_valid']:
            self.__validDict = self.__build_valid()
        # self._edge_indices = self.get_edge_indices()
        self.getSparseGraph()
        self.get_edge_index()

        print(f"{config['dataset']} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def _trainUser(self):
        return self.trainUser
    
    @property
    def _trainItem(self):
        return self.trainItem
    
    @property
    def testDict(self):
        return self.__testDict
    
    @property
    def validDict(self):
        return self.__validDict

    @property
    def allPos(self):
        return self._allPos
    
    @property
    def allPos_item(self):
        return self._allPos_item

    @property
    def TrainPop_item(self):
        '''
        dict of items' popularity(degree) in training set
        '''
        return self._TrainPop_item

    @property
    def TrainPop_user(self):
        '''
        dict of users' popularity(degree) in training set
        '''
        return self._TrainPop_user
    
    @property
    def edge_indices(self):
        '''
        Edge's indice start from 1.\n
        Minus 1 while using, so that -1 means no edge.\n
        It's sparse, .to_dense() if many indices are needed.
        '''
        return self._edge_indices
    
    def get_edge_index(self):
        '''
        graph: (n_user+n_item) * (n_user+n_item)
        '''
        self.edge_index = torch.LongTensor([list(np.append(self.trainUser, self.trainItem+self.n_user)), 
                                        list(np.append(self.trainItem+self.n_user, self.trainUser))]).to(world.device)
        self.graph_pyg = Data(edge_index=self.edge_index.contiguous())
        return self.edge_index



    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        """
        Graph = \n
        D^(-1/2) @ A @ D^(-1/2) \n
        A = \n
        |0,   R|\n
        |R.T, 0|\n
        """
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                cprint('Remember to delete this pre-calculed mat while changing data split !')
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix --- All train data")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                batch_size = 30000 
                num_batches = int(np.ceil(self.n_users / batch_size))

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, self.n_users)
                    batch_R = self.UserItemNet[start_idx:end_idx].tolil()
                    adj_mat[start_idx:end_idx, self.n_users:] = batch_R
                    adj_mat[self.n_users:, start_idx:end_idx] = batch_R.T

                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(world.device)
            self.nx_Graph = nx.from_scipy_sparse_matrix(norm_adj)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def __build_valid(self):
        """
        return:
            dict: {user: [items]}
        """
        valid_data = {}
        for i, item in enumerate(self.validItem):
            user = self.validUser[i]
            if valid_data.get(user):
                valid_data[user].append(item)
            else:
                valid_data[user] = [item]
        return valid_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems 
        
    def getItemPosUsers(self, items):
        posUsers = []
        for item in items:
            posUsers.append(self.UserItemNet[:,item].nonzero()[0])
        return posUsers 

    def get_edge_indices(self):
        '''
        edge's indices start from 1, so that 0 means no edge\n
        -1 when use this index
        '''
        index = torch.stack([torch.tensor(self.trainUser).to(torch.int64), torch.tensor(self.trainItem).to(torch.int64)])
        val =torch.arange(len(self.trainItem)) + 1
        edge_indice = torch.sparse.FloatTensor(index, val, (self.n_user, self.m_item))
        edge_indice = edge_indice.coalesce()
        return edge_indice
