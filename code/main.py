import dataloader
import precalcul
import world
from world import cprint
from world import cprint_rare
import model
import loss
import procedure
import torch
from os.path import join
import time
import visual
from pprint import pprint
import utils
import wandb
import math
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from torch_geometric import seed_everything
import json

def grouped_recall(epoch, result):
    current_best_recall_group = np.zeros((world.config['pop_group'], len(world.config['topks'])))
    for i in range(len(world.config['topks'])):
        k = world.config['topks'][i]
        for group in range(world.config['pop_group']):
            current_best_recall_group[group, i] = result['recall_pop_Contribute'][group][i]
    return current_best_recall_group

def NDCG20_with_best_recall(epoch, result):
    return result['ndcg'][0]

def main():
    print('DEVICE:',world.device, world.args.cuda)
    #print(torch.cuda.get_device_name(torch.cuda.current_device()))
    print(torch.cuda.get_device_name(world.device))

    project = world.config['project']
    name = world.config['name']
    tag = world.config['tag']
    notes = world.config['notes']
    group = world.config['group']
    job_type = world.config['job_type']
    # os.environ['WANDB_MODE'] = 'dryrun'
    wandb.init(project=project, name=name, tags=tag, group=group, job_type=job_type, config=world.config, save_code=True, sync_tensorboard=False, notes=notes)
    wandb.define_metric("epoch")
    wandb.define_metric(f"{world.config['dataset']}"+'/loss', step_metric='epoch')
    for k in world.config['topks']:
        wandb.define_metric(f"{world.config['dataset']}"+f'/recall@{str(k)}', step_metric='epoch')
        wandb.define_metric(f"{world.config['dataset']}"+f'/ndcg@{str(k)}', step_metric='epoch')
        wandb.define_metric(f"{world.config['dataset']}"+f'/precision@{str(k)}', step_metric='epoch')
        wandb.define_metric(f"{world.config['dataset']}"+f'/valid_recall@{str(k)}', step_metric='epoch')
        wandb.define_metric(f"{world.config['dataset']}"+f'/valid_ndcg@{str(k)}', step_metric='epoch')
        wandb.define_metric(f"{world.config['dataset']}"+f'/valid_precision@{str(k)}', step_metric='epoch')
        for group in range(world.config['pop_group']):
            wandb.define_metric(f"{world.config['dataset']}"+f"/groups/recall_group_{group+1}@{str(k)}", step_metric='epoch')
    wandb.define_metric(f"{world.config['dataset']}"+f"/training_time", step_metric='epoch')
    
    wandb.define_metric(f"{world.config['dataset']}"+'/pop_classifier_acc', step_metric='epoch')

    for group in range(world.config['pop_group']):
        wandb.define_metric(f"{world.config['dataset']}"+f"/groups/Rating_group_{group+1}", step_metric='epoch')


    world.make_print_to_file()

    seed_everything(seed=world.config['seed'])

    print('==========config==========')
    pprint(world.config)
    print('==========config==========')

    cprint('[DATALOADER--START]')
    datasetpath = join(world.DATA_PATH, world.config['dataset'])
    dataset = dataloader.dataset(world.config, datasetpath)
    cprint('[DATALOADER--END]')

    cprint('[PRECALCULATE--START]')
    start = time.time()
    precal = precalcul.precalculate(world.config, dataset)
    end = time.time()
    print('precal cost : ',end-start)
    cprint('[PRECALCULATE--END]')

    cprint('[SAMPLER--START]')
    sampler = precalcul.sampler(dataset=dataset, precal=precal)
    cprint('[SAMPLER--END]')
    

    models = {'LightGCN':model.LightGCN}
    Recmodel = models[world.config['model']](world.config, dataset, precal).to(world.device)

    classifier = model.Classifier(input_dim=world.config['latent_dim_rec'], out_dim=world.config['pop_group'], precal=precal)

    augmentation = None
    

    losss = {'Adaptive':loss.Adaptive_loss}
    total_loss = losss[world.config['loss']](world.config, Recmodel, precal)

    train = procedure.Train(total_loss)
    test = procedure.Test()
    
    emb_optimizer = torch.optim.Adam(Recmodel.parameters(), lr=world.config['lr'])
    emb_optimizer.add_param_group({'params':total_loss.MLP_model.parameters()})
    pop_optimizer = torch.optim.Adam(classifier.parameters(), lr=world.config['lr'])
    optimizer = {'emb':emb_optimizer, 'pop':pop_optimizer}

    quantify = visual.Quantify(dataset, Recmodel, precal)


    try:
        best_result_recall = 0.
        best_result_ndcg = 0.
        stopping_step = 0
        best_result_recall_group = None
        ndcg_with_best_recall_20 = None
        Recall_40 = []
        NDCG_40 = []
        finished = 0
        if world.config['if_valid']:
            best_valid_recall = 0.
            stopping_valid_step = 0


        for epoch in range(world.config['epochs']):
            torch.cuda.empty_cache()
            wandb.log({"epoch": epoch})
            start = time.time()
            #====================VISUAL====================
            if world.config['if_visual'] == 1 and epoch % world.config['visual_epoch'] == 0:
                cprint("[Visualization]")
                if world.config['if_tsne'] == 1:
                    quantify.visualize_tsne(epoch)
                if world.config['if_double_label'] == 1:
                    quantify.visualize_double_label(epoch)
            #====================AUGMENT====================
            #None
            #====================TRAIN====================                    
            cprint('[TRAIN]')
            start_train = time.time()
            avg_loss, avg_pop_acc = train.train(sampler, Recmodel, epoch, optimizer, classifier)
            end_train = time.time()
            wandb.log({ f"{world.config['dataset']}"+'/loss': avg_loss})
            wandb.log({f"{world.config['dataset']}"+f"/training_time": end_train - start_train})

            wandb.log({ f"{world.config['dataset']}"+'/pop_classifier_acc': avg_pop_acc})

            with torch.no_grad():
                if epoch % 1== 0:
                    #====================VALID====================
                    if world.config['if_valid']:
                        cprint("[valid]")
                        result = test.valid(dataset, Recmodel, multicore=world.config['if_multicore'])
                        if result["recall"][0] > best_valid_recall:#early stop
                            stopping_valid_step = 0
                            advance = (result["recall"][0] - best_valid_recall)
                            best_valid_recall = result["recall"][0]
                            cprint_rare("find a better valid recall", str(best_valid_recall), extra='++'+str(advance))
                            wandb.run.summary['best valid recall'] = best_valid_recall  
                        else:
                            stopping_valid_step += 1
                            if stopping_valid_step >= world.config['early_stop_steps']:
                                print(f"early stop triggerd at epoch {epoch}, best valid recall: {best_valid_recall}")
                                break
                        for i in range(len(world.config['topks'])):
                            k = world.config['topks'][i]
                            wandb.log({ f"{world.config['dataset']}"+f'/valid_recall@{str(k)}': result["recall"][i],
                                        f"{world.config['dataset']}"+f'/valid_ndcg@{str(k)}': result["ndcg"][i],
                                        f"{world.config['dataset']}"+f'/valid_precision@{str(k)}': result["precision"][i]})
                            
                    #====================TEST====================
                    cprint("[TEST]")
                    result = test.test(dataset, Recmodel, precal, epoch, world.config['if_multicore'])

                    Recall_40.append(result["recall"][1])
                    NDCG_40.append(result["ndcg"][1])

                    if result["recall"][0] > best_result_recall:
                        stopping_step = 0
                        advance = (result["recall"][0] - best_result_recall)
                        best_result_recall = result["recall"][0]
                        cprint_rare("find a better recall", str(best_result_recall), extra='++'+str(advance))
                        best_result_recall_group = grouped_recall(epoch, result)
                        ndcg_with_best_recall_20 = NDCG20_with_best_recall(epoch, result)
                        wandb.run.summary['best test recall'] = best_result_recall  
                    else:
                        stopping_step += 1
                        if stopping_step >= world.config['early_stop_steps']:
                            print(f"early stop triggerd at epoch {epoch}, best recall: {best_result_recall}, in group: {best_result_recall_group}")

                            finished = 1

                            max_index = Recall_40.index(max(Recall_40))

                            my_table = wandb.Table(columns=["Name",               "Tag",               "Dataset",                "Best Recall@20",     "NDCG@20 for best Recall@20",   "Best Recall@40 before Early Stop", "NDCG@40 for best Recall@40", "Recall(1)@20", "Recall(2)@20", "Recall(3)@20", "Recall(4)@20", "Recall(5)@20", "Recall(6)@20", "Recall(7)@20", "Recall(8)@20", "Recall(9)@20", "Recall(10)@20"], 
                                                    data= [[world.config['name'], world.config['tag'][0], world.config['dataset'],  best_result_recall,   ndcg_with_best_recall_20,       Recall_40[max_index],                NDCG_40[max_index],           best_result_recall_group[0,0],best_result_recall_group[1,0],best_result_recall_group[2,0],best_result_recall_group[3,0],best_result_recall_group[4,0],best_result_recall_group[5,0],best_result_recall_group[6,0],best_result_recall_group[7,0],best_result_recall_group[8,0],best_result_recall_group[9,0]]] )
                            wandb.log({"Summary Table": my_table})   

                            break
                    
                    if world.config['if_visual'] == 1:
                        Ratings_group = Recmodel.getItemRating()
                        for group in range(world.config['pop_group']):
                            wandb.log({f"{world.config['dataset']}"+f"/groups/Rating_group_{group+1}": Ratings_group[group]})


                    for i in range(len(world.config['topks'])):
                        k = world.config['topks'][i]
                        wandb.log({ f"{world.config['dataset']}"+f'/recall@{str(k)}': result["recall"][i],
                                    f"{world.config['dataset']}"+f'/ndcg@{str(k)}': result["ndcg"][i],
                                    f"{world.config['dataset']}"+f'/precision@{str(k)}': result["precision"][i]})
                        for group in range(world.config['pop_group']):
                            wandb.log({f"{world.config['dataset']}"+f"/groups/recall_group_{group+1}@{str(k)}": result['recall_pop_Contribute'][group][i]})

            during = time.time() - start
            print(f"total time cost of epoch {epoch}: ", during)
                

    finally:
        if finished == 0:
            api = wandb.Api()
            run = api.run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
            meta = json.load(run.file("wandb-metadata.json").download(replace=True))
            program = ["python"] + [meta["program"]] + meta["args"]
            rerun_cmd = str(' '.join(program))

            debug_table = wandb.Table(
                            columns=["Name",               "Tag",               "Dataset",                 "rerun_cmd"], 
                            data=   [[world.config['name'], world.config['tag'][0], world.config['dataset'],   rerun_cmd]] )
            wandb.log({"Debug Re-run": debug_table})

            
        cprint(world.config['c'])
        wandb.finish()
        cprint(world.config['c'])


if __name__ == '__main__':
    main()