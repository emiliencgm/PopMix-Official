
import os
import world
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from dataloader import dataset
from precalcul import precalculate
from gtn_propagation import GeneralPropagation
import torch_sparse
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.nn import LGConv
import torch.nn.functional as F
from torch.nn import ModuleList
#=============================================================Basic LightGCN============================================================#
class LightGCN(nn.Module):
    def __init__(self, config, dataset:dataset, precal:precalculate):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.precal = precal
        self.__init_weight()

        if config['augment'] in ['SVD']:
            #LightGCL----------SVD
            l = world.config['num_layers']
            self.E_u_list = [None] * (l+1)
            self.E_i_list = [None] * (l+1)
            self.E_u_list[0] = self.embedding_user.weight
            self.E_i_list[0] = self.embedding_item.weight
            self.Z_u_list = [None] * (l+1)
            self.Z_i_list = [None] * (l+1)
            self.G_u_list = [None] * (l+1)
            self.G_i_list = [None] * (l+1)
            self.G_u_list[0] = self.embedding_user.weight
            self.G_i_list[0] = self.embedding_item.weight

            self.u_mul_s = self.precal.SVD_Graph.u_mul_s
            self.v_mul_s = self.precal.SVD_Graph.v_mul_s
            self.ut = self.precal.SVD_Graph.ut
            self.vt = self.precal.SVD_Graph.vt

            self.adj_norm_UI = self.precal.SVD_Graph.adj_norm

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        print("user:{}, item:{}".format(self.num_users, self.num_items))
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['num_layers']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        
        if self.config['loss'] in ['BC', 'Causal_pop']:
            #For BC loss
            hotest_user = self.precal.popularity.UserPopGroupDict[self.config['pop_group']-1][-1]
            hotest_user = int(hotest_user)
            self.max_pop_user = self.precal.popularity.user_pop_degree_label[hotest_user]
            hotest_item = self.precal.popularity.ItemPopGroupDict[self.config['pop_group']-1][-1]
            hotest_item = int(hotest_item)
            self.max_pop_item = self.precal.popularity.item_pop_degree_label[hotest_item]
            self.embed_user_pop = nn.Embedding(self.max_pop_user+1, self.latent_dim)
            self.embed_item_pop = nn.Embedding(self.max_pop_item+1, self.latent_dim)
            nn.init.xavier_normal_(self.embed_user_pop.weight)
            nn.init.xavier_normal_(self.embed_item_pop.weight)        
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)            
        self.f = nn.Sigmoid()
        if self.dataset.Graph is None:
            self.Graph = self.dataset.getSparseGraph()
        else:
            self.Graph = self.dataset.Graph

        print(f"GCL Model is ready to go!")

    def computer(self):
        """
        vanilla LightGCN. No dropout used, return final embedding for rec. 
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        graph=self.Graph    
        for layer in range(self.n_layers):
            if world.config['if_big_matrix']:
                temp_emb = []
                for i_fold in range(len(graph)):
                    temp_emb.append(torch.sparse.mm(graph[i_fold], all_emb))
                all_emb = torch.cat(temp_emb, dim=0)
            else:
                all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    

    def view_computer(self, graph):
        """
        vanilla LightGCN. No dropout used, return final embedding for rec. 
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]    
        for layer in range(self.n_layers):
            if world.config['if_big_matrix']:
                temp_emb = []
                for i_fold in range(len(graph)):
                    temp_emb.append(torch.sparse.mm(graph[i_fold], all_emb))
                all_emb = torch.cat(temp_emb, dim=0)
            else:
                all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def LightGCL_svd_computer(self):
        for layer in range(1,world.config['num_layers']+1):
            # GNN propagation
            self.Z_u_list[layer] = (torch.spmm(self.adj_norm_UI, self.E_i_list[layer-1]))
            self.Z_i_list[layer] = (torch.spmm(self.adj_norm_UI.transpose(0,1), self.E_u_list[layer-1]))

            # svd_adj propagation
            vt_ei = self.vt @ self.E_i_list[layer-1]
            self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
            ut_eu = self.ut @ self.E_u_list[layer-1]
            self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

            # aggregate
            self.E_u_list[layer] = self.Z_u_list[layer]
            self.E_i_list[layer] = self.Z_i_list[layer]

        #TODO sum --> mean ???
        self.G_u = sum(self.G_u_list)
        self.G_i = sum(self.G_i_list)

        # aggregate across layers
        self.E_u = sum(self.E_u_list)
        self.E_i = sum(self.E_i_list)

        return self.G_u, self.G_i, self.E_u, self.E_i
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        if world.config['loss'] == 'Causal_pop':
            elu = torch.nn.ELU()
            return elu(rating)
        return rating

    #================Pop=================#
    def getItemRating(self):
        itemsPopDict = self.precal.popularity.ItemPopGroupDict
        all_users, all_items = self.computer()
        items_embDict = {}
        for group in range(world.config['pop_group']):
            items_embDict[group] = all_items[itemsPopDict[group].long()]
        users_emb = all_users
        rating_Dict = {}
        for group in range(world.config['pop_group']):
            rating_Dict[group] = torch.matmul(items_embDict[group], users_emb.t())
            rating_Dict[group] = torch.mean(rating_Dict[group], dim=1)
            rating_Dict[group] = torch.mean(rating_Dict[group])
        return rating_Dict
    #================Pop=================#

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        All_embs = [all_users, all_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, All_embs
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, _) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))

        if(torch.isnan(loss).any().tolist()):
            print("user emb")
            print(userEmb0)
            print("pos_emb")
            print(posEmb0)
            print("neg_emb")
            print(negEmb0)
            print("neg_scores")
            print(neg_scores)
            print("pos_scores")
            print(pos_scores)
            return None, None
        return loss, reg_loss


#=============================================================GTN Encoder============================================================#
class GTN(LightGCN):
    def __init__(self, config, dataset:dataset, precal:precalculate):
        super(GTN, self).__init__(config, dataset, precal)

        self.gp = GeneralPropagation(config['GTN_K'], config['GTN_alpha'], cached=True, args=config)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        graph=self.Graph #Normalized
        # from GTN
        x = all_emb
        rc = graph.indices()
        r = rc[0]#row
        c = rc[1]#col
        val = torch.ones(graph.values().shape[0]).to(world.device)
        num_nodes = graph.shape[0]
        edge_index = torch_sparse.SparseTensor(row=r, col=c, value=val, sparse_sizes=(num_nodes, num_nodes))
        emb, embs = self.gp.forward(x, edge_index)#TODO
        light_out = emb

        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

#=============================================================SGL ED & RW============================================================#
class SGL(LightGCN):
    def __init__(self, config, dataset:dataset, precal):
        super(SGL, self).__init__(config, dataset, precal)

    def view_computer(self, augmentGraph):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        if world.config['model'] == 'SGL':
            embs = [all_emb]
            if world.config['augment']=='RW':
                for layer in range(self.n_layers):
                    if world.config['if_big_matrix']:
                        temp_emb = []
                        for i_fold in range(len(augmentGraph[layer])):
                            temp_emb.append(torch.sparse.mm(augmentGraph[layer][i_fold], all_emb))
                        all_emb = torch.cat(temp_emb, dim=0)
                    else:
                        all_emb = torch.sparse.mm(augmentGraph[layer], all_emb)
                    embs.append(all_emb)
            elif world.config['augment']=='ED':
                for layer in range(self.n_layers):
                    if world.config['if_big_matrix']:
                        temp_emb = []
                        for i_fold in range(len(augmentGraph)):
                            temp_emb.append(torch.sparse.mm(augmentGraph[i_fold], all_emb))
                        all_emb = torch.cat(temp_emb, dim=0)
                    else:
                        all_emb = torch.sparse.mm(augmentGraph, all_emb)
                    embs.append(all_emb)
        else:
            raise TypeError('model-mode')
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items


#=============================================================SimGCL============================================================#
class SimGCL(LightGCN):
    def __init__(self, config, dataset:dataset, precal):
        super(SimGCL, self).__init__(config, dataset, precal)
        
    def view_computer(self):
        raw_graph = self.Graph
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        if world.config['model'] in ['SimGCL']:
            embs = []
            if world.config['model']=='SimGCL':
                for layer in range(self.n_layers):
                    if world.config['if_big_matrix']:
                        temp_emb = []
                        for i_fold in range(len(raw_graph)):
                            temp_emb.append(torch.sparse.mm(raw_graph[i_fold], all_emb))
                        all_emb = torch.cat(temp_emb, dim=0)
                    else:
                        all_emb = torch.sparse.mm(raw_graph, all_emb)
                    low = torch.zeros_like(all_emb).float()
                    high = torch.ones_like(all_emb).float()
                    random_noise = torch.distributions.uniform.Uniform(low, high).sample()
                    noise = torch.mul(torch.sign(all_emb),torch.nn.functional.normalize(random_noise, dim=1)) * world.config['eps_SimGCL']
                    all_emb += noise
                    embs.append(all_emb)
            else:
                raise TypeError('model-mode')
        else:
            raise TypeError('model-mode')
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    

# class LightGCN_PyG(nn.Module):
#     def __init__(self, config, dataset:dataset, precal:precalculate):
#         super(LightGCN_PyG, self).__init__()
#         self.config = config
#         self.dataset = dataset
#         self.precal = precal 
#         self.lightConv = LGConv()

#         self.__init_weight()

#         if config['augment'] in ['SVD']:
#             #LightGCL----------SVD
#             l = world.config['num_layers']
#             self.E_u_list = [None] * (l+1)
#             self.E_i_list = [None] * (l+1)
#             self.E_u_list[0] = self.embedding_user.weight
#             self.E_i_list[0] = self.embedding_item.weight
#             self.Z_u_list = [None] * (l+1)
#             self.Z_i_list = [None] * (l+1)
#             self.G_u_list = [None] * (l+1)
#             self.G_i_list = [None] * (l+1)
#             self.G_u_list[0] = self.embedding_user.weight
#             self.G_i_list[0] = self.embedding_item.weight

#             self.u_mul_s = self.precal.SVD_Graph.u_mul_s
#             self.v_mul_s = self.precal.SVD_Graph.v_mul_s
#             self.ut = self.precal.SVD_Graph.ut
#             self.vt = self.precal.SVD_Graph.vt

#             self.adj_norm_UI = self.precal.SVD_Graph.adj_norm

#     def __init_weight(self):
#         self.num_users  = self.dataset.n_users
#         self.num_items  = self.dataset.m_items
#         print("user:{}, item:{}".format(self.num_users, self.num_items))
#         self.latent_dim = self.config['latent_dim_rec']
#         self.n_layers = self.config['num_layers']
#         self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
#         self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        
#         if self.config['loss'] in ['BC', 'Causal_pop']:
#             #For BC loss
#             hotest_user = self.precal.popularity.UserPopGroupDict[self.config['pop_group']-1][-1]
#             hotest_user = int(hotest_user)
#             self.max_pop_user = self.precal.popularity.user_pop_degree_label[hotest_user]
#             hotest_item = self.precal.popularity.ItemPopGroupDict[self.config['pop_group']-1][-1]
#             hotest_item = int(hotest_item)
#             self.max_pop_item = self.precal.popularity.item_pop_degree_label[hotest_item]
#             self.embed_user_pop = nn.Embedding(self.max_pop_user+1, self.latent_dim)
#             self.embed_item_pop = nn.Embedding(self.max_pop_item+1, self.latent_dim)
#             nn.init.xavier_normal_(self.embed_user_pop.weight)
#             nn.init.xavier_normal_(self.embed_item_pop.weight)        
#         nn.init.normal_(self.embedding_user.weight, std=0.1)
#         nn.init.normal_(self.embedding_item.weight, std=0.1)
#         self.f = nn.Sigmoid()
#         if self.dataset.Graph is None:
#             self.Graph = self.dataset.getSparseGraph()
#         else:
#             self.Graph = self.dataset.Graph
        
#         self.edge_index = torch.tensor([list(np.append(self.dataset.trainUser, self.dataset.trainItem)), list(np.append(self.dataset.trainItem, self.dataset.trainUser))])
        

#         print(f"GCL Model is ready to go!")
    
#     def pyg_data(self):
#         users_emb0 = self.embedding_user.weight
#         items_emb0 = self.embedding_item.weight
#         x = torch.cat([users_emb0, items_emb0])
#         data_origin = torch_geometric.data.Data(x=x, edge_index=self.edge_index.contiguous())
#         return data_origin

#     def computer(self):
#         """
#         vanilla LightGCN. No dropout used, return final embedding for rec. 
#         """
#         users_emb0 = self.embedding_user.weight
#         items_emb0 = self.embedding_item.weight
#         x = torch.cat([users_emb0, items_emb0])
#         x, edge_index = x.to(world.device), self.edge_index.to(world.device)
#         embs = [x]
#         for layer in range(self.n_layers):
#             x = self.lightConv(x=x, edge_index=edge_index)
#             embs.append(x)
#         embs = torch.stack(embs, dim=1)
#         light_out = torch.mean(embs, dim=1)
#         users, items = torch.split(light_out, [self.num_users, self.num_items])
#         return users, items
    

#     def view_computer(self, data):
#         """
#         vanilla LightGCN. No dropout used, return final embedding for rec. 
#         """       
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#         x, edge_index, edge_weight = x.to(world.device), edge_index.to(world.device), edge_weight.to(world.device)
#         embs = [x]
#         for layer in range(self.n_layers):
#             x = self.lightConv(x=x, edge_index=edge_index, edge_weight=edge_weight)
#             embs.append(x)
#         embs = torch.stack(embs, dim=1)
#         light_out = torch.mean(embs, dim=1)
#         users, items = torch.split(light_out, [self.num_users, self.num_items])
#         return users, items
    

#     def LightGCL_svd_computer(self):
#         for layer in range(1,world.config['num_layers']+1):
#             # GNN propagation
#             self.Z_u_list[layer] = (torch.spmm(self.adj_norm_UI, self.E_i_list[layer-1]))
#             self.Z_i_list[layer] = (torch.spmm(self.adj_norm_UI.transpose(0,1), self.E_u_list[layer-1]))

#             # svd_adj propagation
#             vt_ei = self.vt @ self.E_i_list[layer-1]
#             self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
#             ut_eu = self.ut @ self.E_u_list[layer-1]
#             self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

#             # aggregate
#             self.E_u_list[layer] = self.Z_u_list[layer]
#             self.E_i_list[layer] = self.Z_i_list[layer]

#         #TODO sum --> mean ???
#         self.G_u = sum(self.G_u_list)
#         self.G_i = sum(self.G_i_list)

#         # aggregate across layers
#         self.E_u = sum(self.E_u_list)
#         self.E_i = sum(self.E_i_list)

#         return self.G_u, self.G_i, self.E_u, self.E_i
    
#     def getUsersRating(self, users):
#         all_users, all_items = self.computer()
#         users_emb = all_users[users.long()]
#         items_emb = all_items
#         rating = self.f(torch.matmul(users_emb, items_emb.t()))
#         if world.config['loss'] == 'Causal_pop':
#             elu = torch.nn.ELU()
#             return elu(rating)
#         return rating

#     #================Pop=================#
#     def getItemRating(self):
#         itemsPopDict = self.precal.popularity.ItemPopGroupDict
#         all_users, all_items = self.computer()
#         items_embDict = {}
#         for group in range(world.config['pop_group']):
#             items_embDict[group] = all_items[itemsPopDict[group].long()]
#         users_emb = all_users
#         rating_Dict = {}
#         for group in range(world.config['pop_group']):
#             rating_Dict[group] = torch.matmul(items_embDict[group], users_emb.t())
#             rating_Dict[group] = torch.mean(rating_Dict[group], dim=1)
#             rating_Dict[group] = torch.mean(rating_Dict[group])
#         return rating_Dict
#     #================Pop=================#

#     def getEmbedding(self, users, pos_items, neg_items):
#         all_users, all_items = self.computer()
#         users_emb = all_users[users]
#         pos_emb = all_items[pos_items]
#         neg_emb = all_items[neg_items]
#         users_emb_ego = self.embedding_user(users)
#         pos_emb_ego = self.embedding_item(pos_items)
#         neg_emb_ego = self.embedding_item(neg_items)
#         All_embs = [all_users, all_items]
#         return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, All_embs
    
#     def bpr_loss(self, users, pos, neg):
#         (users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, _) = self.getEmbedding(users.long(), pos.long(), neg.long())
#         reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2))
#         pos_scores = torch.mul(users_emb, pos_emb)
#         pos_scores = torch.sum(pos_scores, dim=1)
#         neg_scores = torch.mul(users_emb, neg_emb)
#         neg_scores = torch.sum(neg_scores, dim=1)
#         loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))
#         return loss, reg_loss
    
    
class LightGCN_PyG(nn.Module):
    def __init__(self, config, dataset:dataset, precal:precalculate):
        super(LightGCN_PyG, self).__init__()
        self.config = config
        self.dataset = dataset
        self.precal = precal 
        self.__init_weight()
        self.convs = ModuleList([LGConv() for _ in range(self.n_layers)])
        self.alpha = 1. / (self.n_layers + 1)

        self.projector_user = Projector()
        self.projector_item = Projector()


        #LightGCL----------SVD
        l = world.config['num_layers']
        self.E_u_list = [None] * (l+1)
        self.E_i_list = [None] * (l+1)
        self.E_u_list[0] = self.embedding_user.weight
        self.E_i_list[0] = self.embedding_item.weight
        self.Z_u_list = [None] * (l+1)
        self.Z_i_list = [None] * (l+1)
        self.G_u_list = [None] * (l+1)
        self.G_i_list = [None] * (l+1)
        self.G_u_list[0] = self.embedding_user.weight
        self.G_i_list[0] = self.embedding_item.weight

        self.u_mul_s = self.precal.SVD_Graph.u_mul_s
        self.v_mul_s = self.precal.SVD_Graph.v_mul_s
        self.ut = self.precal.SVD_Graph.ut
        self.vt = self.precal.SVD_Graph.vt

        self.adj_norm_UI = self.precal.SVD_Graph.adj_norm


    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        print("user:{}, item:{}".format(self.num_users, self.num_items))
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['num_layers']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)        
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)        
        self.f = nn.Sigmoid()
        self.edge_index = self.dataset.edge_index

        print(f"GCL Model is ready to go!")

    def computer(self):
        """
        vanilla LightGCN. No dropout used, return final embedding for rec. 
        """
        users_emb0 = self.embedding_user.weight
        items_emb0 = self.embedding_item.weight
        x = torch.cat([users_emb0, items_emb0])
        out = x * self.alpha
        for i in range(self.n_layers):
            x = self.convs[i](x, self.edge_index)
            out = out + x * self.alpha
        users, items = torch.split(out, [self.num_users, self.num_items])
        return users, items
    
    def computer_per_layer(self):
        """
        vanilla LightGCN. No dropout used, return final embedding for rec. 
        """
        users_emb0 = self.embedding_user.weight
        items_emb0 = self.embedding_item.weight
        x = torch.cat([users_emb0, items_emb0])
        embs_per_layer = []
        embs_per_layer.append(x)
        out = x * self.alpha
        for i in range(self.n_layers):
            x = self.convs[i](x, self.edge_index)
            embs_per_layer.append(x)
            out = out + x * self.alpha
        users, items = torch.split(out, [self.num_users, self.num_items])
        return users, items, embs_per_layer

    def view_computer(self, x, edge_index, edge_weight=None):
        try:
            x, edge_index, edge_weight = x.to(world.device), edge_index.to(world.device), edge_weight.to(world.device)
        except:
            x, edge_index = x.to(world.device), edge_index.to(world.device)

        out = x * self.alpha
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            out = out + x * self.alpha
        users, items = torch.split(out, [self.num_users, self.num_items])
        return users, items
    
    def LightGCL_svd_computer(self):
        for layer in range(1,world.config['num_layers']+1):
            # GNN propagation
            self.Z_u_list[layer] = (torch.spmm(self.adj_norm_UI, self.E_i_list[layer-1]))
            self.Z_i_list[layer] = (torch.spmm(self.adj_norm_UI.transpose(0,1), self.E_u_list[layer-1]))

            # svd_adj propagation
            vt_ei = self.vt @ self.E_i_list[layer-1]
            self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
            ut_eu = self.ut @ self.E_u_list[layer-1]
            self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

            # aggregate
            self.E_u_list[layer] = self.Z_u_list[layer]
            self.E_i_list[layer] = self.Z_i_list[layer]

        #TODO sum --> mean ???
        self.G_u = sum(self.G_u_list)
        self.G_i = sum(self.G_i_list)

        # aggregate across layers
        self.E_u = sum(self.E_u_list)
        self.E_i = sum(self.E_i_list)

        return self.G_u, self.G_i, self.E_u, self.E_i
    
    #================Pop=================#
    def getItemRating(self):
        itemsPopDict = self.precal.popularity.ItemPopGroupDict
        all_users, all_items = self.computer()
        items_embDict = {}
        for group in range(world.config['pop_group']):
            items_embDict[group] = all_items[itemsPopDict[group].long()]
        users_emb = all_users
        rating_Dict = {}
        for group in range(world.config['pop_group']):
            rating_Dict[group] = torch.matmul(items_embDict[group], users_emb.t())
            rating_Dict[group] = torch.mean(rating_Dict[group], dim=1)
            rating_Dict[group] = torch.mean(rating_Dict[group])
        return rating_Dict
    #================Pop=================#
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        All_embs = [all_users, all_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, All_embs
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, _) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))
        return loss, reg_loss
    


class Projector(torch.nn.Module):
    def __init__(self, output_dim=world.config['latent_dim_rec'], input_dim=world.config['latent_dim_rec']):
        super(Projector, self).__init__()        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim*4),
            torch.nn.BatchNorm1d(input_dim*4),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(input_dim*4, input_dim*4),
            torch.nn.BatchNorm1d(input_dim*4), 
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(input_dim*4, output_dim)
        ).to(world.device)

    def forward(self, embs):
        return self.net(embs)



class Classifier(torch.nn.Module):
    def __init__(self, input_dim, out_dim, precal:precalculate):
        super(Classifier, self).__init__()
        self.input_dim = input_dim

        self.all_label = precal.popularity.item_pop_group_label
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim*4),
            torch.nn.BatchNorm1d(input_dim*4),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(input_dim*4, input_dim*4),
            torch.nn.BatchNorm1d(input_dim*4), 
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(input_dim*4, out_dim),
            torch.nn.Softmax(dim=-1)
        ).to(world.device)

        self.criterion = nn.CrossEntropyLoss()
    
    def cal_loss_and_test(self, inputs, batch_item):
        '''
        return loss and test accuracy of the same batch before update
        '''
        batch_item = batch_item.cpu()
        batch_label = torch.tensor(self.all_label[batch_item], dtype=torch.long).to(world.device)
        outputs = self.net(inputs)
        CE_loss = self.criterion(outputs, batch_label)

        predicted_labels = torch.argmax(outputs, dim=1)
        accuracy = torch.mean((predicted_labels == batch_label).float())

        return CE_loss, accuracy