import copy
import numpy as np
from copy import deepcopy
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        out = self.dataset[self.idxs[item]]
        if len(out) == 2:
            image, label = out
        else:
            image, label, _ = out  # Ignore index
        return image, label


class Strategy:
    def __init__(self, dataset_query, dataset_train, net, args):
        self.dataset_query = dataset_query
        self.dataset_train = dataset_train
        self.net = net
        self.args = args
        self.local_net_dict = {}
        self.loss_func = nn.CrossEntropyLoss()
        
    def query(self, label_idx, unlabel_idx):
        pass

    def predict_prob(self, unlabel_idxs, net=None):
        loader_te = DataLoader(DatasetSplit(self.dataset_query, unlabel_idxs), shuffle=False)
        
        if net is None:
            net = self.net
            
        net.eval()
        probs = torch.zeros([len(unlabel_idxs), self.args.num_classes])
        with torch.no_grad():
            for batch in loader_te:
                if len(batch) == 3:
                    x, y, idxs = batch
                else:
                    x, y = batch
                    idxs = torch.arange(len(y))

                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
                output, _ = net(x)
                probs[idxs] = F.softmax(output, dim=1).cpu().data
        return probs

    def get_embedding(self, data_idxs, net=None):
        loader_te = DataLoader(DatasetSplit(self.dataset_query, data_idxs), shuffle=False)
        
        if net is None:
            net = self.net
        
        net.eval()
        embedding = torch.zeros([len(data_idxs), net.get_embedding_dim()])
        with torch.no_grad():
            for batch in loader_te:
                if len(batch) == 3:
                    x, y, idxs = batch
                else:
                    x, y = batch
                    idxs = torch.arange(len(y))

                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
                _, e1 = net(x)
                embedding[idxs] = e1.data.cpu()
        
        return embedding

    def get_grad_embedding(self, data_idxs, net=None):
        if net is None:
            net = self.net
            
        embDim = net.get_embedding_dim()
        net.eval()
        
        nLab = self.args.num_classes 
        embedding = np.zeros([len(data_idxs), embDim * nLab])
        loader_te = DataLoader(DatasetSplit(self.dataset_query, data_idxs), shuffle=False)
        
        with torch.no_grad():
            for batch in loader_te:
                if len(batch) == 3:
                    x, y, idxs = batch
                else:
                    x, y = batch
                    idxs = torch.arange(len(y))

                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
                cout, out = net(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)

    def get_grad_embedding_maxInd(self, data_idxs, net=None):
        if net is None:
            net = self.net
            
        embDim = net.get_embedding_dim() #Gets the embedding vector dimension (from EfficientNet’s last hidden layer)
        net.eval() #Evaluation mode
        
        nLab = self.args.num_classes 
        embedding = np.zeros([len(data_idxs), embDim]) #array to store the samples embeddings
        loader_te = DataLoader(DatasetSplit(self.dataset_query, data_idxs), shuffle=False) #Builds a DataLoader for the selected indices from self.dataset_query(unlabelled samples) using DatasetSplit.
        
        with torch.no_grad():
            for batch in loader_te:
                if len(batch) == 3:
                    x, y, idxs = batch
                else:
                    x, y = batch
                    idxs = torch.arange(len(y))

                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
                cout, out = net(x)
                out = out.data.cpu().numpy() #Converts embeddings and predicted probabilities to NumPy
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1) #predicted class with maximum softmax score for each sample.
                
                for j in range(len(y)):
                    if len(embedding.shape) == 2:
                        embedding[idxs[j]] = deepcopy(out[j]) * (1 - batchProbs[j][maxInds[j]]) # (1 - max class probability) = model’s uncertainty
            return torch.Tensor(embedding)

    def training_local_only(self, label_idxs, finetune=False):
        finetune_ep = 50
        local_net = deepcopy(self.net)

        if not finetune: 
            local_net.load_state_dict(self.args.raw_ckpt)

        label_train = DataLoader(DatasetSplit(self.dataset_train, label_idxs), batch_size=self.args.local_bs, shuffle=True)
        
        optimizer = torch.optim.SGD(local_net.parameters(), 
                                    lr=self.args.lr, 
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(finetune_ep * 3 / 4)], gamma=self.args.lr_decay)
        
        for epoch in range(finetune_ep):
            local_net.train()
            for batch in label_train:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch

                if self.args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
                    labels = labels.squeeze().long()
                    
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                output, emb = local_net(images)
                
                if output.shape[0] == 1:
                    labels = labels.reshape(1,)

                loss = self.loss_func(output, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            correct, cnt = 0., 0.
            local_net.eval()
            with torch.no_grad():
                for batch in label_train:
                    if len(batch) == 3:
                        images, labels, _ = batch
                    else:
                        images, labels = batch

                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    output, _ = local_net(images)
                    
                    y_pred = output.data.max(1, keepdim=True)[1]
                    correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
                    cnt += len(labels)
        
                acc = correct / cnt
                if acc >= 0.99:
                    break

        return local_net