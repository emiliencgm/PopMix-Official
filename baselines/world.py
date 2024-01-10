import os
from os.path import join
import torch
from parse import parse_args
import multiprocessing
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

config = {}
config['temp_tau'] = args.temp_tau
config['edge_drop_prob'] = args.edge_drop_prob
config['latent_dim_rec'] = args.latent_dim_rec
config['num_layers'] = args.num_layers
config['dataset'] = args.dataset
config['lr'] = args.lr
config["weight_decay"] = args.weight_decay
config['seed'] = args.seed
config['model'] = args.model
config['epochs'] = args.epochs
config['if_multicore'] = args.if_multicore
config['early_stop_steps'] = args.early_stop_steps
config['batch_size'] = args.batch_size
config['lambda1'] = args.lambda1
config['topks'] = eval(args.topks)
config['test_u_batch_size'] = args.test_u_batch_size
config['pop_group'] = args.pop_group
config['if_big_matrix'] = args.if_big_matrix
config['n_fold'] = args.n_fold
config['p_drop'] = args.p_drop
config['perplexity'] = args.perplexity
config['visual_epoch'] = args.visual_epoch
config['if_double_label'] = args.if_double_label
config['if_tsne'] = args.if_tsne 
config['tsne_group'] = eval(args.tsne_group)
config['eps_SimGCL'] = args.eps_SimGCL
config['tsne_points'] = args.tsne_points
config['loss'] = args.loss
config['augment'] = args.augment
config['alpha'] = args.alpha
config['epoch_only_pop_for_BCloss'] = args.epoch_only_pop_for_BCloss
config['centroid_mode'] = args.centroid_mode
config['commonNeighbor_mode'] = args.commonNeighbor_mode
config['adaptive_method'] = args.adaptive_method
config['if_visual'] = args.if_visual
config['GTN_K'] = args.GTN_K
config['GTN_alpha'] = args.GTN_alpha
config['pop_gamma'] = args.pop_gamma
config['if_valid'] = args.if_valid
config['temp_tau_pop'] = args.temp_tau_pop
config['comment'] = args.comment
#WandB
config['project'] = args.project
config['name'] = args.name
config['tag'] = args.tag
config['notes'] = args.notes
config['group'] = args.group
config['job_type'] = args.job_type
config['c'] = args.c

log = {}

LogItems = ['model', 'loss', 'alpha', 'temp_tau', 'comment']
ArchitectureItems = ['model', 'dataset', 'seed', 'loss', 'augment', 'centroid_mode', 'commonNeighbor_mode', 'adaptive_method', 'init_method', 'perplexity']
HyperparameterItems = ['temp_tau', 'alpha', 'lr', 'weight_decay', 'lambda1', 'n_cluster', 'sigma_gausse', 'eps_SimGCL', 'epsilon_GCLRec', 'w_GCLRec', 'k_aug', 'epoch_only_pop_for_BCloss']
for key in LogItems:
    log[key] = config[key]

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
CODE_PATH = join(ROOT_PATH, 'baselines')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, f"runs/{config['dataset']}")
FILE_PATH = join(CODE_PATH, 'checkpoints')
LOG_FILE = join(CODE_PATH, 'Log')
FIG_FILE = join(CODE_PATH, 'Fig')
FIG_FILE = join(FIG_FILE, config['dataset'])
RESULT_FILE = join(CODE_PATH, 'result')
RESULT_FILE = join(RESULT_FILE, config['dataset'])

date = datetime.datetime.now().strftime(f"%m_%d_%Hh%Mm%Ss-")
#FIG_FILE = join(FIG_FILE, date)
NOHUPPATH = None
PRECALPATH = join(DATA_PATH, f"precalculate/{config['dataset']}")


import sys
sys.path.append(join(CODE_PATH, 'sources'))

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)
if not os.path.exists(LOG_FILE):
    os.makedirs(LOG_FILE, exist_ok=True)
if not os.path.exists(PRECALPATH):
    os.makedirs(PRECALPATH, exist_ok=True)


GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
config['device'] = device

CORES = multiprocessing.cpu_count() // 2
config['cores'] = CORES
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

def cprint_rare(desc, val, extra=None):
    if extra is None:
        print(f"\033[5;37;44m{desc}\033[0m", f"\033[0;37;44m{val}\033[0m")
    else:
        print(f"\033[5;37;44m{desc}\033[0m", f"\033[0;37;44m{val}\033[0m", f"\033[0;37;44m{extra}\033[0m")



def make_print_to_file(path=LOG_FILE):
    import sys
    import os
    #import config_file as cfg_file
    import sys

    class Logger(object):
        def __init__(self, filename="Default.log", path=path):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime(f"%m_%d_%Hh%Mm%Ss-{config['comment']}")
    sys.stdout = Logger(fileName + f"-{config['dataset']}-" +'.log', path=path)

