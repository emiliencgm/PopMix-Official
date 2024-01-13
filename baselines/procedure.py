import torch
import numpy as np
import world
import utils
import multiprocessing
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import LightGCN
from collections import OrderedDict
import loss

class Train():
    def __init__(self, loss_cal):
        self.loss = loss_cal
        self.test = Test()
        self.temp = world.config['temp_tau']

    def train(self, dataset, Recmodel, augmentation, epoch, optimizer, pop_class):
        Recmodel = Recmodel
        Recmodel.train()
        batch_size = world.config['batch_size']
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

        total_batch = len(dataloader)
        aver_loss = 0.
        aver_pop_acc = 0.

        for batch_i, train_data in tqdm(enumerate(dataloader), desc='training'):
            batch_users = train_data[0].long().to(world.device)
            batch_pos = train_data[1].long().to(world.device)
            batch_neg = train_data[2].long().to(world.device)

            if world.config['loss'] == 'BPR':
                l_all, pop_acc = self.BPR_train(Recmodel, batch_users, batch_pos, batch_neg, augmentation, pop_class, epoch)
                
            elif world.config['loss'] == 'PDA':
                l_all, pop_acc = self.PDA_train(Recmodel, batch_users, batch_pos, batch_neg, augmentation, pop_class, epoch)          
            
            elif world.config['loss'] == 'BPR_Contrast':
                l_all, pop_acc = self.BPR_Contrast_train(Recmodel, batch_users, batch_pos, batch_neg, augmentation, pop_class, epoch)
            
            elif world.config['loss'] == 'BC':
                l_all, pop_acc = self.BC_train(Recmodel, batch_users, batch_pos, batch_neg, augmentation, pop_class, epoch)            
            
            elif world.config['loss'] == 'Adaptive':                
                l_all, pop_acc = self.Adaloss_train(Recmodel, batch_users, batch_pos, batch_neg, augmentation, pop_class, epoch)

            else:
                l_all = None
                raise TypeError('No demanded loss')
            
            optimizer.zero_grad()
            l_all.backward()
            optimizer.step()
            
            aver_loss += l_all.cpu().item()
            if pop_acc:
                aver_pop_acc += pop_acc.cpu().item()
            else:
                aver_pop_acc = 0
            
        aver_loss = aver_loss / (total_batch)
        aver_pop_acc = aver_pop_acc / (total_batch)
        print(f'EPOCH[{epoch}]:loss {aver_loss:.3f}    pop_classifier_acc: {aver_pop_acc}')
        return aver_loss, aver_pop_acc
    

    def BPR_train(self, Recmodel, batch_users, batch_pos, batch_neg, augmentation, pop_class, epoch):

        users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, embs_per_layer_or_all_embs= Recmodel.getEmbedding(batch_users.long(), batch_pos.long(), batch_neg.long())

        l_all = self.loss.bpr_loss(users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0)

        classifier_loss, classifier_acc = pop_class['classifier'].cal_loss_and_test(pos_emb.detach(), batch_pos)
        pop_class['optimizer'].zero_grad()
        classifier_loss.backward()
        pop_class['optimizer'].step()


        return l_all, classifier_acc

    def BPR_Contrast_train(self, Recmodel, batch_users, batch_pos, batch_neg, augmentation, pop_class, epoch):

        if world.config['augment'] in ['SVD']:
            G_u, G_i, E_u, E_i = Recmodel.LightGCL_svd_computer()
            users_emb = E_u[batch_users]
            pos_emb = E_i[batch_pos]
            neg_emb = E_i[batch_neg]
            users_emb_ego = Recmodel.embedding_user(batch_users)
            pos_emb_ego1 = Recmodel.embedding_item(batch_pos)
            neg_emb_ego = Recmodel.embedding_item(batch_neg)

            iids = torch.concat([batch_pos, batch_neg], dim=0)

            # reg = (1/6)*(users_emb_ego.norm(2).pow(2) + pos_emb_ego1.norm(2).pow(2) + neg_emb_ego.norm(2).pow(2))/len(batch_users)
            reg = (0.5 * torch.norm(users_emb_ego) ** 2 + len(batch_users) * 0.5 * torch.norm(pos_emb_ego1) ** 2)/len(batch_users)

            # cl loss
            G_u_norm = G_u
            E_u_norm = E_u
            G_i_norm = G_i
            E_i_norm = E_i
            neg_score = torch.log(torch.exp(G_u_norm[batch_users] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
            neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score = (torch.clamp((G_u_norm[batch_users] * E_u_norm[batch_users]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp,-5.0,5.0)).mean()
            loss_cl = -pos_score + neg_score

            loss_bpr = self.bpr_loss(users_emb, pos_emb, neg_emb)

            loss_all = world.config['weight_decay']*reg + loss_bpr + world.config['lambda1']*loss_cl

            classifier_loss, classifier_acc = pop_class['classifier'].cal_loss_and_test(pos_emb.detach(), batch_pos)
            pop_class['optimizer'].zero_grad()
            classifier_loss.backward()
            pop_class['optimizer'].step()

            return loss_all, classifier_acc
        
        else:
            users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, embs_per_layer_or_all_embs= Recmodel.getEmbedding(batch_users.long(), batch_pos.long(), batch_neg.long())

            if world.config['model'] in ['SGL']:
                aug_users1, aug_items1 = Recmodel.view_computer(augmentation.augAdjMatrix1)
                aug_users2, aug_items2 = Recmodel.view_computer(augmentation.augAdjMatrix2)
            elif world.config['model'] in ['SimGCL']:
                aug_users1, aug_items1 = Recmodel.view_computer()
                aug_users2, aug_items2 = Recmodel.view_computer()

            l_all = self.loss.bpr_contrast_loss(users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, batch_users, batch_pos, batch_neg, aug_users1, aug_items1, aug_users2, aug_items2)


            classifier_loss, classifier_acc = pop_class['classifier'].cal_loss_and_test(pos_emb.detach(), batch_pos)
            pop_class['optimizer'].zero_grad()
            classifier_loss.backward()
            pop_class['optimizer'].step()


            return l_all, classifier_acc
        
    def bpr_loss(self, u_emb, pos_emb, neg_emb):
        # pos_scores = torch.mul(users_emb, pos_emb)
        # pos_scores = torch.sum(pos_scores, dim=1)
        # neg_scores = torch.mul(users_emb, neg_emb)
        # neg_scores = torch.sum(neg_scores, dim=1)
        # # mean or sum
        # loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))#TODO SOFTPLUS()!!!
        # return loss/world.config['batch_size']
        pos_scores = (u_emb * pos_emb).sum(-1)
        neg_scores = (u_emb * neg_emb).sum(-1)
        loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

        return loss_r

    def PDA_train(self, Recmodel, batch_users, batch_pos, batch_neg, augmentation, pop_class, epoch):

        users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, embs_per_layer_or_all_embs= Recmodel.getEmbedding(batch_users.long(), batch_pos.long(), batch_neg.long())

        l_all = self.loss.causal_popularity_bpr_loss(users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, batch_pos, batch_neg)

        classifier_loss, classifier_acc = pop_class['classifier'].cal_loss_and_test(pos_emb.detach(), batch_pos)
        pop_class['optimizer'].zero_grad()
        classifier_loss.backward()
        pop_class['optimizer'].step()


        return l_all, classifier_acc

    def BC_train(self, Recmodel, batch_users, batch_pos, batch_neg, augmentation, pop_class, epoch):

        users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, embs_per_layer_or_all_embs= Recmodel.getEmbedding(batch_users.long(), batch_pos.long(), batch_neg.long())

        if epoch < world.config['epoch_only_pop_for_BCloss']:
            mode = 'only_pop'
        else:
            mode = 'pop_bc'

        l_all = self.loss.bc_loss(users_emb, pos_emb, userEmb0, posEmb0, batch_users, batch_pos, mode)

        classifier_loss, classifier_acc = pop_class['classifier'].cal_loss_and_test(pos_emb.detach(), batch_pos)
        pop_class['optimizer'].zero_grad()
        classifier_loss.backward()
        pop_class['optimizer'].step()


        return l_all, classifier_acc

    def Adaloss_train(self, Recmodel, batch_users, batch_pos, batch_neg, augmentation, pop_class, epoch):

        users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, embs_per_layer_or_all_embs= Recmodel.getEmbedding(batch_users.long(), batch_pos.long(), batch_neg.long())

        if world.config['model'] in ['SGL']:
            aug_users1, aug_items1 = Recmodel.view_computer(augmentation.augAdjMatrix1)
            aug_users2, aug_items2 = Recmodel.view_computer(augmentation.augAdjMatrix2)
        elif world.config['model'] in ['SimGCL']:
            aug_users1, aug_items1 = Recmodel.view_computer()
            aug_users2, aug_items2 = Recmodel.view_computer()
        elif world.config['model'] in ['LightGCN', 'GTN', 'LightGCN_PyG']:
            aug_users1, aug_items1 = None, None
            aug_users2, aug_items2 = None, None

        l_all = self.loss.adaptive_softmax_loss(users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, batch_users, batch_pos, batch_neg, aug_users1, aug_items1, aug_users2, aug_items2, epoch)


        classifier_loss, classifier_acc = pop_class['classifier'].cal_loss_and_test(pos_emb.detach(), batch_pos)
        pop_class['optimizer'].zero_grad()
        classifier_loss.backward()
        pop_class['optimizer'].step()


        return l_all, classifier_acc


class Test():
    def __init__(self):
        pass
    
    def test_one_batch(self, X):
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        #================Pop=================#
        groundTrue_popDict = X[2]#{0: [ [items of u1], [items of u2] ] }
        r, r_popDict = utils.getLabel(groundTrue, groundTrue_popDict, sorted_items)
        #================Pop=================#
        pre, recall, recall_pop, recall_pop_Contribute, ndcg = [], [], {}, {}, []
        num_group = world.config['pop_group']
        for group in range(num_group):
                recall_pop[group] = []
        for group in range(num_group):
                recall_pop_Contribute[group] = []

        for k in world.config['topks']:
            ret = utils.RecallPrecision_ATk(groundTrue, groundTrue_popDict, r, r_popDict, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])

            num_group = world.config['pop_group']
            for group in range(num_group):
                recall_pop[group].append(ret['recall_popDIct'][group])
            for group in range(num_group):
                recall_pop_Contribute[group].append(ret['recall_Contribute_popDict'][group])

            ndcg.append(utils.NDCGatK_r(groundTrue,r,k))

        
        for group in range(num_group):
            recall_pop[group] = np.array(recall_pop[group])
        for group in range(num_group):
            recall_pop_Contribute[group] = np.array(recall_pop_Contribute[group])

        return {'recall':np.array(recall), 
                'recall_popDict':recall_pop,
                'recall_Contribute_popDict':recall_pop_Contribute,
                'precision':np.array(pre), 
                'ndcg':np.array(ndcg)}


    def test(self, dataset, Recmodel, precal, epoch, multicore=0):
        u_batch_size = world.config['test_u_batch_size']
        testDict: dict = dataset.testDict
        testDict_pop = precal.popularity.testDict_PopGroup
        max_K = max(world.config['topks'])
        CORES = multiprocessing.cpu_count() // 2
        # CORES = multiprocessing.cpu_count()
        if multicore == 1:
            pool = multiprocessing.Pool(CORES)
        results = {'precision': np.zeros(len(world.config['topks'])),
                'recall': np.zeros(len(world.config['topks'])),
                'recall_pop': {},
                'recall_pop_Contribute': {},
                'ndcg': np.zeros(len(world.config['topks']))}
        num_group = world.config['pop_group']
        for group in range(num_group):
            results['recall_pop'][group] = np.zeros(len(world.config['topks']))
            results['recall_pop_Contribute'][group] = np.zeros(len(world.config['topks']))

        with torch.no_grad():
            Recmodel = Recmodel.eval()
            #================Pop=================#
            RatingsPopDict = Recmodel.getItemRating()
            #================Pop=================#
            users = list(testDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            groundTrue_list_pop = []
            # auc_record = []
            # ratings = []
            total_batch = len(users) // u_batch_size + 1
            for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                allPos = dataset.getUserPosItems(batch_users)
                groundTrue = [testDict[u] for u in batch_users]
                #================Pop=================#
                groundTrue_pop = {}
                for group, ground in testDict_pop.items():
                    groundTrue_pop[group] = [ground[u] for u in batch_users]
                #================Pop=================#
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)

                rating = Recmodel.getUsersRating(batch_users_gpu)
                #rating = rating.cpu()
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1<<10)
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
                #================Pop=================#
                groundTrue_list_pop.append(groundTrue_pop)
                #================Pop=================#
            assert total_batch == len(users_list)
            X = zip(rating_list, groundTrue_list, groundTrue_list_pop)
            if multicore == 1:
                pre_results = pool.map(self.test_one_batch, X)
            else:
                pre_results = []
                for x in X:
                    pre_results.append(self.test_one_batch(x))
            scale = float(u_batch_size/len(users))
                
            for result in pre_results:
                results['recall'] += result['recall']
                for group in range(num_group):
                    results['recall_pop'][group] += result['recall_popDict'][group]
                    results['recall_pop_Contribute'][group] += result['recall_Contribute_popDict'][group]
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            for group in range(num_group):
                results['recall_pop'][group] /= float(len(users))
                results['recall_pop_Contribute'][group] /= float(len(users))

            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            
            if multicore == 1:
                pool.close()
            print(results)
            return results
    

    def valid_one_batch(self, X):
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        r= utils.getLabel_Valid(groundTrue, sorted_items)
        pre, recall, ndcg = [], [], []

        for k in world.config['topks']:
            ret = utils.RecallPrecision_ATk_Valid(groundTrue, r, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])
            ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
        return {'recall':np.array(recall),
                'precision':np.array(pre), 
                'ndcg':np.array(ndcg)}
    
    def valid_one_batch_batch(self, X):
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        r= utils.getLabel_Valid(groundTrue, sorted_items)

        k = world.config['topks'][0]
        ret = utils.RecallPrecision_ATk_Valid(groundTrue, r, k)
        recall = ret['recall']
        return recall

    def valid(self, dataset, Recmodel, multicore=0, if_print=True):
        u_batch_size = world.config['test_u_batch_size']
        validDict: dict = dataset.validDict
        Recmodel = Recmodel.eval()
        max_K = max(world.config['topks'])
        CORES = multiprocessing.cpu_count() // 2
        # CORES = multiprocessing.cpu_count()
        if multicore == 1:
            pool = multiprocessing.Pool(CORES)
        results = {'precision': np.zeros(len(world.config['topks'])),
                'recall': np.zeros(len(world.config['topks'])),
                'ndcg': np.zeros(len(world.config['topks']))}

        with torch.no_grad():
            users = list(validDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                if if_print:
                    print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            total_batch = len(users) // u_batch_size + 1
            for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                allPos = dataset.getUserPosItems(batch_users)
                groundTrue = [validDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)

                rating = Recmodel.getUsersRating(batch_users_gpu)
                #rating = rating.cpu()
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1<<10)
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
            assert total_batch == len(users_list)
            X = zip(rating_list, groundTrue_list)
            if multicore == 1:
                pre_results = pool.map(self.valid_one_batch, X)
            else:
                pre_results = []
                for x in X:
                    pre_results.append(self.valid_one_batch(x))
            scale = float(u_batch_size/len(users))
                
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            if multicore == 1:
                pool.close()
            if if_print:
                print('VALID',results)
            return results


    def valid_batch(self, dataset, Recmodel, batch_users):
        batch_users = batch_users.cpu()
        validDict: dict = dataset.validDict
        Recmodel = Recmodel.eval()
        max_K = max(world.config['topks'])

        with torch.no_grad():
            users = list(batch_users)
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [validDict[u.item()] for u in batch_users]
            batch_users_gpu = batch_users.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            recall = self.valid_one_batch_batch([rating_K.cpu(), groundTrue])
            return recall/float(len(users))
