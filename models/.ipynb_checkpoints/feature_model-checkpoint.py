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
            self.model = models.resnet.resnet50_baseline(pretrained=True, pretrained_output_channel=pretrained_output_channel)
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


if __name__ == "__main__":
    model_name = "ResNet"  
    model = Feature_extract(model=model_name)
    
    print("Model Summary:")
    print(model)

    from torchview import draw_graph
    import matplotlib.pyplot as plt
    dummy_input = torch.randn(1, 3, 256, 256)
    
    model_graph = draw_graph(model, input_size=(1, 3, 256, 256), expand_nested=True)
    visual_graph = model_graph.visual_graph

