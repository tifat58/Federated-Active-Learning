import numpy as np
import torch.nn as nn
import pdb
import torch
import torch.nn.functional as F
from utils.loss_func import EDL_Loss
from utils.loss_func import EDL_Loss, EDL_Dice_Loss, Dice_Loss

def train(round_idx, client_idx, model, dataloader, optimizer, num_per_class, args):
    model.train()

    if args.dataset == 'FedISIC':
        max_epoch = 1
        if args.al_method == 'FEAL':
            criterion = EDL_Loss(
                prior=num_per_class / num_per_class.sum(),
                kl_weight=args.kl_weight,
                annealing_step=args.annealing_step,
            )
        else:
            criterion = nn.CrossEntropyLoss(weight=2 - num_per_class / num_per_class.sum())
    else:  # FedPolyp → segmentation
        max_epoch = args.local_epoch if hasattr(args, 'local_epoch') else 2
        if args.al_method == 'FEAL':
            criterion = EDL_Dice_Loss(
                kl_weight=args.kl_weight,
                annealing_step=args.annealing_step,
            )
        else:
            criterion = Dice_Loss()
    iters = 0
    for epoch in range(max_epoch):
        for _, (_, data) in enumerate(dataloader):
            
            iters += 1
            optimizer.zero_grad()
            image, label = data['image'], data['label']
            image = image.cuda()
            label = label.cuda()
            logit = model(image)[0]    # logit, pred, embedding, block_output

            if args.al_method == 'FEAL':
                loss = criterion(logit, label, round_idx) 
            else:
                loss = criterion(logit, label)
                
            if iters % args.display_freq == 0:
                print('iter {}: {}'.format(iters, loss.item()))

            loss.backward()
            optimizer.step()            

