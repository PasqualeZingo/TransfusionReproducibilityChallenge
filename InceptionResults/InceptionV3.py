import os

import torch
from torch.nn import functional as F
import torch.nn as nn
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

import pytorch_lightning as pl

class InceptionV3(pl.LightningModule):

    def __init__(self, transfer):
        super(InceptionV3, self).__init__()
        print("INIT MODEL")
        
        self.model = models.inception_v3(pretrained = transfer)
        self.model.fc = nn.Linear(2048, 5)
        
        num_train = 35126
        indices = list(range(num_train))
        valid_size = 0.1
        split = int(np.floor(valid_size * num_train))
        shuffle = True
        if shuffle:
            np.random.seed(0)
            np.random.shuffle(indices)

        self.train_idx, self.valid_idx = indices[split:], indices[:split]



    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x).logits
        y_true = torch.zeros(len(y_hat), 5, device='cuda')
        for ind, val in enumerate(y_true):
            y_true[ind][y[ind]] = 1
            
        loss = F.binary_cross_entropy_with_logits(y_hat, y_true)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        
        y_true = torch.zeros(len(y_hat), 5, device='cuda')
        for ind, val in enumerate(y_true):
            y_true[ind][y[ind]] = 1
        y_class2 = torch.Tensor([1 if t == 2 else 0 for t in y])
        
        return {'val_loss': F.binary_cross_entropy_with_logits(y_hat, y_true),
                'y_preds': y_hat,
                'y_true' : y_class2}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y_preds = torch.cat([x['y_preds'][:,2] for x in outputs])
        y_true = torch.cat([x['y_true'] for x in outputs])

        y_true, y_preds = y_true.cpu(), y_preds.cpu()
        
        auc = roc_auc_score(y_true, y_preds)
        tensorboard_logs = {'val_loss': avg_loss,
                            'AUC': auc}
        return {'val_loss': avg_loss, 'AUC': auc, 'log': tensorboard_logs}

    def configure_optimizers(self):
        print("CONFIGURING OPTIMIZER")
        optim = torch.optim.Adam(self.model.parameters(), lr = 0.001)
        return optim
    
    @pl.data_loader
    def train_dataloader(self):
        
        print("INITIALIZING TRAIN DATALOADER")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        datadir = '/data/fundus/train'
        dataset = datasets.ImageFolder(
            datadir,
            transforms.Compose([
                transforms.Resize((587, 587)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        train_dataset = torch.utils.data.Subset(dataset, self.train_idx)
        dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        
        train_DL = DataLoader(train_dataset, sampler=dist_sampler, batch_size=16, num_workers=4)
        return train_DL
        
    @pl.data_loader
    def val_dataloader(self):
        print("INITIALIZING TRAIN DATALOADER")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        datadir = '/data/fundus/train'
        dataset = datasets.ImageFolder(
            datadir,
            transforms.Compose([
                transforms.Resize((587, 587)),
                transforms.ToTensor(),
                normalize,
            ]))
        valid_dataset = torch.utils.data.Subset(dataset, self.valid_idx)
        dist_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
        valid_DL = DataLoader(valid_dataset, sampler=dist_sampler, batch_size=16, num_workers=4)
        return valid_DL
