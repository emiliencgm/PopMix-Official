import world
import torch
from torch import nn
from dataloader import dataset
from precalcul import precalculate
from torch_geometric.nn import LGConv
from torch_geometric.nn import GCNConv
from torch.nn import ModuleList
import numpy as np

    
class LightGCN(nn.Module):
    def __init__(self, config, dataset:dataset, precal:precalculate):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.precal = precal 
        self.__init_weight()
        self.convs = ModuleList([LGConv() for _ in range(self.n_layers)])
        self.alpha = 1. / (self.n_layers + 1)

        self.projector_user = Projector()
        self.projector_item = Projector()

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
        self.graph = self.dataset.graph_pyg

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