import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Go GCLRec")
    parser.add_argument('--dataset', type=str, default='yelp2018', help="dataset")
    parser.add_argument('--method', type=str, default='PopMix', help="method: PopMix")
    parser.add_argument('--device', type=int, default=0, help="device")
    parser.add_argument('--visual', type=int, default=0, help="visualization")
    parser.add_argument('--temp_tau', type=float, default=0.1, help="temp_tau")
    parser.add_argument('--lambda1', type=float, default=0.1, help="lambda1")
    return parser.parse_args()

args = parse_args()
project = 'TestCode'
temp_tau = args.temp_tau
lambda1 = args.lambda1


if args.method == 'PopMix':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book']:
        centroid = 'eigenvector'
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag PopMix --group ours --job_type {args.dataset} --model LightGCN --loss Adaptive --augment No --lambda1 {lambda1} --temp_tau {temp_tau} --centroid_mode {centroid}\
                    --sampling uii --dataset {args.dataset} --cuda {args.device} --if_visual {args.visual} --visual_epoch 5')
    
    if args.dataset in ['ifashion', 'last-fm']:
        centroid = 'pagerank'
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag PopMix --group ours --job_type {args.dataset} --model LightGCN --loss Adaptive --augment No --lambda1 {lambda1} --temp_tau {temp_tau} --centroid_mode {centroid}\
                    --sampling uii --dataset {args.dataset} --cuda {args.device} --if_visual {args.visual} --visual_epoch 5')

else:
    print('#=====================#')
    print(args.method)
    print('#=====================#')
