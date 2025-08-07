#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import os
import sys
import json
import random
import copy
import pickle
import numpy as np
import pandas as pd
import medmnist
from medmnist import INFO
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from models import get_model
from fl_methods import get_fl_method_class
from query_strategies import random_query_samples, algo_query_samples
from util.args import args_parser
from util.path import set_result_dir, set_dict_user_path
from util.data_simulator import shard_balance, dir_balance
from util.longtail_dataset import IMBALANCECIFAR10, IMBALANCECIFAR100
from util.misc import adjust_learning_rate
from data.fedisic_loader import load_csv_splits_feal_style as load_csv_splits


def get_dataset(args):
    #Normalisation
    MEAN = {'mnist': (0.1307,), 'fmnist': (0.5,), 'emnist': (0.5,), 'svhn': [0.4376821, 0.4437697, 0.47280442], 
            'cifar10': [0.485, 0.456, 0.406], 'cifar100': [0.507, 0.487, 0.441], 'pathmnist': (0.5,), 
            'octmnist': (0.5,), 'organamnist': (0.5,), 'dermamnist': (0.5,), 'bloodmnist': (0.5,),
            'fedisic': [0.485, 0.456, 0.406]}
    STD = {'mnist': (0.3081,), 'fmnist': (0.5,), 'emnist': (0.5,), 'svhn': [0.19803012, 0.20101562, 0.19703614], 
           'cifar10': [0.229, 0.224, 0.225], 'cifar100': [0.267, 0.256, 0.276], 'pathmnist': (0.5,),
           'octmnist': (0.5,), 'organamnist': (0.5,), 'dermamnist': (0.5,), 'bloodmnist': (0.5,),
           'fedisic': [0.229, 0.224, 0.225]}
    
    if 'lt' not in args.dataset:
        noaug = [transforms.ToTensor(),
                 transforms.Normalize(mean=MEAN[args.dataset], std=STD[args.dataset])]
        
        weakaug = [transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize(mean=MEAN[args.dataset], std=STD[args.dataset])]
        
        trans_noaug = transforms.Compose(noaug)
        trans_weakaug = transforms.Compose(weakaug)
        
    # standard benchmarks
    print('Load Dataset {}'.format(args.dataset))
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST(args.data_dir, train=True, download=True, transform=trans_weakaug)
        dataset_query = datasets.MNIST(args.data_dir, train=True, download=True, transform=trans_noaug)
        dataset_test = datasets.MNIST(args.data_dir, train=False, download=True, transform=trans_noaug)
    
    elif args.dataset == "fmnist":
        dataset_train = datasets.FashionMNIST(args.data_dir, download=True, train=True, transform=trans_weakaug)
        dataset_query = datasets.FashionMNIST(args.data_dir, download=True, train=True, transform=trans_noaug)
        dataset_test = datasets.FashionMNIST(args.data_dir, download=True, train=False, transform=trans_noaug)

    elif args.dataset == 'emnist':
        dataset_train = datasets.EMNIST(args.data_dir, split='byclass', train=True, download=True, transform=trans_weakaug)
        dataset_query = datasets.EMNIST(args.data_dir, split='byclass', train=True, download=True, transform=trans_noaug)
        dataset_test = datasets.EMNIST(args.data_dir, split='byclass', train=False, download=True, transform=trans_noaug)

    elif args.dataset == 'svhn':
        dataset_train = datasets.SVHN(args.data_dir, 'train', download=True, transform=trans_weakaug)
        dataset_query = datasets.SVHN(args.data_dir, 'train', download=True, transform=trans_noaug)
        dataset_test = datasets.SVHN(args.data_dir, 'test', download=True, transform=trans_noaug)
            
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=trans_weakaug)
        dataset_query = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=trans_noaug)
        dataset_test = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=trans_noaug)
            
    elif args.dataset == 'cifar10_lt':
        dataset_train = IMBALANCECIFAR10('train', args.imb_ratio, args.data_dir)
        dataset_query = IMBALANCECIFAR10('train', args.imb_ratio, args.data_dir, train_aug=False)
        dataset_test = IMBALANCECIFAR10('test', args.imb_ratio, args.data_dir)
        
    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100(args.data_dir, train=True, download=True, transform=trans_weakaug)
        dataset_query = datasets.CIFAR100(args.data_dir, train=True, download=True, transform=trans_noaug)
        dataset_test = datasets.CIFAR100(args.data_dir, train=False, download=True, transform=trans_noaug)
            
    elif args.dataset == 'cifar10_lt':
        dataset_train = IMBALANCECIFAR100('train', args.imb_ratio, args.data_dir)
        dataset_query = IMBALANCECIFAR100('train', args.imb_ratio, args.data_dir, train_aug=False)
        dataset_test = IMBALANCECIFAR100('test', args.imb_ratio, args.data_dir)

    # medical benchmarks
    elif args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
        DataClass = getattr(medmnist, INFO[args.dataset]['python_class'])
        
        dataset_train = DataClass(download=True, split='train', transform=trans_weakaug)
        dataset_query = DataClass(download=True, split='train', transform=trans_noaug)
        dataset_test = DataClass(download=True, split='test', transform=trans_noaug)

    elif args.dataset == 'fedisic':
        from data.fedisic_loader import get_fedisic_loaders

        print('Loading FedISIC dataset...')

        # csv_dir = os.path.join(args.data_dir, 'splits')
        # csv_splits = load_csv_splits(csv_dir)
        csv_path = os.path.join(args.data_dir, 'train_test_split.csv')
        csv_splits = load_csv_splits(csv_path)

        config = {
            'dataset': {'data_root': os.path.join(args.data_dir, 'FedISIC_npy')},
            'initial_label_rate': args.query_ratio,
            'batch_size': args.local_bs,
            'num_clients': args.num_users
        }

        dataset_train, dict_users_train_total, test_loader = get_fedisic_loaders(config, csv_splits)

        dataset_query = dataset_train
        dataset_test = test_loader.dataset

        args.dataset_train = dataset_train
        args.total_data = len(dataset_train)
        args.test_loader = test_loader

    else:
        exit('Error: unrecognized dataset')
        
    args.dataset_train = dataset_train
    args.total_data = len(dataset_train)

    if args.partition == "shard_balance":
        dict_users_train_total = shard_balance(dataset_train, args)
        dict_users_test_total = shard_balance(dataset_test, args)
    elif args.partition == "dir_balance":
        dict_users_train_total, sample = dir_balance(dataset_train, args)
        dict_users_test_total, _ = dir_balance(dataset_test, args, sample)
    #added for data partition
    elif args.partition == "dirichlet":
        from util.partition import partition_data_dirichlet
        dict_users_train_total, dict_users_test_total = partition_data_dirichlet(dataset_train, dataset_test, args)
    

    args.n_query = round(args.total_data, -2) * args.query_ratio #total samples taken for query
    args.n_data = round(args.total_data, -2) * args.current_ratio
    
    return dataset_train, dataset_query, dataset_test, dict_users_train_total, dict_users_test_total, args
    #added


def train_test(net_glob, dataset_train, dataset_test, dict_users_train_label, args):
    results_save_path = os.path.join(args.result_dir, 'results.csv')

    fl_method = get_fl_method_class(args.fl_algo)(args, dict_users_train_label)
    if args.fl_algo == 'scaffold':
        fl_method.init_c_nets(net_glob)

    results = []   
    for round in range(args.rounds):
        w_glob = None
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #helps to stabilize training, prevent model oscillations
        lr = adjust_learning_rate(args, round)
        print("Round {}, lr: {:.6f}, momentum:{}, weight decay:{}, idx_users: {}".format(round+1, lr, args.momentum, args.weight_decay, idxs_users))

        #total_data_num = sum([len(dict_users_train_label[idx]) for idx in idxs_users])
        #added compute total data being used
        total_data_num = sum([len(dict_users_train_label[int(idx)]) for idx in idxs_users if int(idx) in dict_users_train_label])
        #added
        
        fl_method.on_round_start(net_glob=net_glob)
        
        # for idx in idxs_users:
        #     fl_method.on_user_iter_start(dataset_train, idx)
            
        #     net_local = copy.deepcopy(net_glob)
        #     w_local, loss = fl_method.train(net=net_local.to(args.device), 
        #                                     user_idx=idx,
        #                                     lr=lr,
        #                                     momentum=args.momentum,
        #                                     weight_decay=args.weight_decay)            
        #     loss_locals.append(copy.deepcopy(loss))
            
        #     fl_method.on_user_iter_end()

        #     w_glob = fl_method.aggregate(w_glob=w_glob, w_local=w_local, idx_user=(idx), total_data_num=total_data_num)
        
        #added
        for idx in idxs_users:
            idx = int(idx)  # ensure idx is integer
        
            # Skip clients with no data explicitly
            if idx not in dict_users_train_label or len(dict_users_train_label[idx]) == 0:
                print(f"Client {idx} skipped due to no data.")
                continue
            
            fl_method.on_user_iter_start(dataset_train, idx)

            net_local = copy.deepcopy(net_glob)

            # Additional safety: double-check again before creating loader
            if len(dict_users_train_label[idx]) == 0:
                print(f"Client {idx} has no data at loader creation. Skipping client.")
                continue

            w_local, loss = fl_method.train(net=net_local.to(args.device), 
                                            user_idx=idx,
                                            lr=lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)            
            loss_locals.append(copy.deepcopy(loss))
            
            fl_method.on_user_iter_end()

            w_glob = fl_method.aggregate(
                w_glob=w_glob, 
                w_local=w_local, 
                idx_user=idx, 
                total_data_num=total_data_num
            )    
        # added      

        fl_method.on_round_end(idxs_users)
                
        net_glob.load_state_dict(w_glob, strict=False)
        acc_test, loss_test = fl_method.test(net_glob, dataset_test)

        # loss_avg = sum(loss_locals) / len(loss_locals)
        # print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
        #     round+1, loss_avg, loss_test, acc_test))
        #added
        if len(loss_locals) > 0:
            loss_avg = sum(loss_locals) / len(loss_locals)
        else:
            loss_avg = float('nan')
            print("All clients skipped this round, no loss computed.")

        print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
            round+1, loss_avg, loss_test, acc_test))
        #added
        results.append(np.array([round, loss_avg, loss_test, acc_test]))
    
    last_save_path = os.path.join(args.result_dir, 'last.pt')
    torch.save(net_glob.state_dict(), last_save_path)
    
    final_results = np.array(results)
    final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test'])
    final_results.to_csv(results_save_path, index=False)
            
    return net_glob.state_dict()
        

if __name__ == '__main__':
    start_time = time.time()
    print("===== STARTING TRAINING =====")
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # print("device:", args.device)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)
        
    args = set_result_dir(args) 
    args = set_dict_user_path(args)

    # total dataset computed for each client
    dataset_train, dataset_query, dataset_test, dict_users_train_total, dict_users_test_total, args = get_dataset(args)
    dict_users_train_label = None
    
    while round(args.current_ratio, 2) <= args.end_ratio:
        print('[Current data ratio] %.3f' % args.current_ratio)

        net_glob = get_model(args)
   
        # if args.query_ratio == args.current_ratio:
        #     dict_users_train_label, args = random_query_samples(dict_users_train_total, dict_users_test_total, args)
        # else:
        #     if dict_users_train_label is None:
        #         path = os.path.join(args.dict_user_path, 'dict_users_train_label_{:.3f}.pkl'.format(args.current_ratio - args.query_ratio))
        #         with open(path, 'rb') as f:
        #             dict_users_train_label = pickle.load(f)
        #         args.dict_users_total_path = os.path.join(args.dict_user_path, 'dict_users_train_test_total.pkl'.format(args.seed))
                
        #         last_ckpt = torch.load(args.query_model)
                            
        #     print("Load Total Data Idxs from {}".format(args.dict_users_total_path))
        #     with open(args.dict_users_total_path, 'rb') as f:
        #         dict_users_train_total, dict_users_test_total = pickle.load(f) 
                
        #     dict_users_train_label = algo_query_samples(dataset_train, dataset_query, dict_users_train_total, args)
        
        #added
        if round(args.current_ratio, 3) <= round(args.query_ratio, 3):
            print("Initial query round, sampling randomly.")
            dict_users_train_label, args = random_query_samples(dict_users_train_total, dict_users_test_total, args)
        else:
            previous_ratio = args.current_ratio - args.query_ratio
            path = os.path.join(args.dict_user_path, f'dict_users_train_label_{previous_ratio:.3f}.pkl')
            print(f"Loading previous labels from: {path}")
            with open(path, 'rb') as f:
                dict_users_train_label = pickle.load(f)

            args.dict_users_total_path = os.path.join(args.dict_user_path, 'dict_users_train_test_total.pkl')
            print("Load Total Data Idxs from {}".format(args.dict_users_total_path))
            with open(args.dict_users_total_path, 'rb') as f:
                dict_users_train_total, dict_users_test_total = pickle.load(f)

            # Automatically update args.query_model from last round
            if args.current_ratio > args.query_ratio:
                prev_round = args.current_ratio - args.query_ratio
                model_path = os.path.join(args.result_dir.replace(f"{args.current_ratio:.3f}", f"{prev_round:.3f}"), 'last.pt')
                print(f"Auto-setting args.query_model to: {model_path}")
                args.query_model = model_path
                if args.reset != 'random_init':
                    args.reset = 'continue'
            dict_users_train_label = algo_query_samples(dataset_train, dataset_query, dict_users_train_total, args)
        #added
                        
        if args.reset == 'continue' and args.query_model:
            query_net_state_dict = torch.load(args.query_model)
            net_glob.load_state_dict(query_net_state_dict)

        last_ckpt = train_test(net_glob, dataset_train, dataset_test, dict_users_train_label, args)
        
        args.current_ratio += args.query_ratio
        
        # update path
        args = set_result_dir(args) 
        args = set_dict_user_path(args)
    end_time = time.time()
    elapsed = end_time - start_time
    elapsed_minutes = elapsed / 60
    elapsed_hours = elapsed / 3600

    from datetime import datetime
    run_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("===== TRAINING COMPLETE =====")
    print(f"Run finished at {run_time_str}")
    print(f"Total Time Elapsed: {elapsed:.2f} seconds")
    print(f"                 or: {elapsed_minutes:.2f} minutes")
    print(f"                 or: {elapsed_hours:.2f} hours")

    with open("/mnt/sdz/adbi01_data/Aditya/Logo_B0_FedISIC/save/logs/runtime_log.txt", "a") as f:
        f.write(f"[{run_time_str}] Total run time: {elapsed:.2f} seconds "
                f"({elapsed_minutes:.2f} minutes, {elapsed_hours:.2f} hours)\n")