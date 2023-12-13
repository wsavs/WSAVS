import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import Bottleneck, BasicBlock


class ResNet(torchvision.models.resnet.ResNet):
    def forward(self, x, return_embs=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if return_embs:
            return x4, [x1, x2, x3, x4]
        else:
            return x4


def resnet50(pretrained='', **kwargs):
    in_chans = 3
    if 'in_chans' in kwargs and kwargs['in_chans'] != 3:
        in_chans = kwargs['in_chans']
    del kwargs['in_chans']

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if in_chans != 3:
        model.conv1 = nn.Conv2d(in_chans, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    if pretrained == 'supervised':
        url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")

    elif pretrained == 'mocov3':
        url = 'https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar'
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")['state_dict']
        state_dict = {k.replace('module.base_encoder.', ''): state_dict[k] for k in state_dict
                      if k.startswith('module.base_encoder.')}

    if in_chans != 3:
        conv1 = state_dict['conv1.weight'].mean(1, keepdims=True)
        state_dict['conv1.weight'] = conv1.repeat((1, in_chans, 1, 1))
    msg = model.load_state_dict(state_dict, strict=False)
    assert all([k.startswith('fc') for k in msg.missing_keys])
    assert all([k.startswith('fc') for k in msg.unexpected_keys])

    return model