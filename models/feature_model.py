"""
Wrapper for feature extraction
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import models.resnet
import models.kimianet

class Feature_extract(pl.LightningModule):
    def __init__(self,model="KimiaNet",pretrained_output_channel=3):
        super(Feature_extract, self).__init__()
        if model=="ResNet":
            self.model = models.resnet.resnet50_baseline(pretrained=True)
        elif model=="KimiaNet":
            self.model = models.kimianet.KimiaNet(imagenet=False, pretrained_output_channel=pretrained_output_channel)
        elif model=="DenseNet":
            self.model = models.kimianet.KimiaNet(imagenet=True, pretrained_output_channel=pretrained_output_channel)
        else:
            raise NotImplementedError

    def forward(self, x):
        feature = self.model(x)
        return feature

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])
