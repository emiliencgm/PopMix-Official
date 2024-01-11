import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Go GCLRec")
    parser.add_argument('--dataset', type=str, default='yelp2018', help="dataset")
    parser.add_argument('--method', type=str, default='LightGCN', help="method: LightGCN, GTN, SGL-ED, SGL-RW, SimGCL, PDA, BC")
    parser.add_argument('--device', type=int, default=0, help="device")
    parser.add_argument('--visual', type=int, default=0, help="visualization")
    parser.add_argument('--temp_tau', type=float, default=0.1, help="temp_tau")
    parser.add_argument('--lambda1', type=float, default=0.1, help="lambda1")
    parser.add_argument('--num_layers', type=int, default=3, help="num_layers of LightGCL's SVD")
    return parser.parse_args()

args = parse_args()
project = 'TestCode'
temp_tau = args.temp_tau
lambda1 = args.lambda1
num_layers = args.num_layers

if args.method == 'LightGCN':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion']:
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag {args.method} --group baseline --job_type {args.dataset} --model LightGCN --loss BPR --dataset {args.dataset} --cuda {args.device} --if_visual {args.visual} --visual_epoch 5')

elif args.method == 'GTN':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion']:
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag {args.method} --group baseline --job_type {args.dataset} --model GTN --loss BPR --dataset {args.dataset} --cuda {args.device} --if_visual {args.visual} --visual_epoch 5')

elif args.method == 'SGL-ED':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion']:
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag SGL --group baseline --job_type {args.dataset} --model SGL --loss BPR_Contrast --augment ED --lambda1 {lambda1} --temp_tau {temp_tau} \
                    --dataset {args.dataset} --cuda {args.device} --if_visual {args.visual} --visual_epoch 1')
        
elif args.method == 'SGL-RW':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion']:
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag SGL --group baseline --job_type {args.dataset} --model SGL --loss BPR_Contrast --augment RW --lambda1 {lambda1} --temp_tau {temp_tau} \
                    --dataset {args.dataset} --cuda {args.device} --if_visual {args.visual} --visual_epoch 1')
        
elif args.method == 'SimGCL':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion']:
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag {args.method} --group baseline --job_type {args.dataset} --model SimGCL --loss BPR_Contrast --augment No --lambda1 {lambda1} --temp_tau {temp_tau} \
                    --dataset {args.dataset} --cuda {args.device} --if_visual {args.visual} --visual_epoch 1')
        
elif args.method == 'PDA':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion']:
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag {args.method} --group baseline --job_type {args.dataset} --model LightGCN --loss PDA --augment No \
                    --dataset {args.dataset} --cuda {args.device} --if_visual {args.visual} --visual_epoch 4')

elif args.method == 'LightGCL':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion']:
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag {args.method} --group baseline --job_type {args.dataset} --model LightGCN_PyG --loss BPR_Contrast --augment SVD --lambda1 {lambda1} --temp_tau {temp_tau}\
                    --num_layers {num_layers} --dataset {args.dataset} --cuda {args.device} --if_visual {args.visual} --visual_epoch 3')

elif args.method == 'BC':
    if args.dataset in ['yelp2018', 'gowalla', 'amazon-book', 'last-fm', 'ifashion']:
        os.system(f'python main.py --project {project} --name {args.method} --notes _ --tag {args.method} --group baseline --job_type {args.dataset} --model LightGCN --loss BC --augment No --lambda1 {lambda1} --temp_tau_pop 0.1 --temp_tau {temp_tau} \
                    --dataset {args.dataset} --cuda {args.device} --if_visual {args.visual} --visual_epoch 5')

else:
    print('#=====================#')
    print(args.method)
    print('#=====================#')
