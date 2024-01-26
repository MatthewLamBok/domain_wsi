import torch
import torchvision.models as models
import torch.nn as nn
import pytorch_lightning as pl


class KimiaNet(nn.Module):
    def __init__(self, imagenet=False, pretrained_output_channel=1):
        super(KimiaNet, self).__init__()

        # Load the pre-trained DenseNet121 model
        original_model = models.densenet121(pretrained=True)

        # If the input channel is not 3 (typical RGB image), adjust the first convolutional layer
        if pretrained_output_channel == 1:
            # Get the original first conv layer parameters
            original_first_conv = original_model.features[0]
            new_first_conv = nn.Conv2d(
                pretrained_output_channel,
                original_first_conv.out_channels,
                kernel_size=original_first_conv.kernel_size,
                stride=original_first_conv.stride,
                padding=original_first_conv.padding,
                bias=original_first_conv.bias,
            )
            # Replace the first conv layer
            original_model.features[0] = new_first_conv

        # Create the modified model with potentially adjusted first layer
        self.model = nn.Sequential(
            original_model.features,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1, -1)
        )

        if not imagenet:
            state_dict = torch.load('saved_models/KimiaNet.pth')
            if pretrained_output_channel == 1:
                state_dict['0.conv0.weight'] = state_dict['0.conv0.weight'].mean(1, keepdim=True)
            self.model.load_state_dict(state_dict)


    def forward(self, x):
        feature = self.model(x)
        return feature

    
if __name__ == "__main__":
    kimianet = KimiaNet(imagenet=False, pretrained_output_channel=3)
    print(kimianet)