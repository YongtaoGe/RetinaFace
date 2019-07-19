from torchvision.models import resnet as vrn
import torch.utils.model_zoo as model_zoo
# from .utils import register

class ResNet(vrn.ResNet):
    'Deep Residual Network - https://arxiv.org/abs/1512.03385'

    def __init__(self, layers=[3, 4, 6, 3], bottleneck=vrn.Bottleneck, outputs=[5], url=None):
        self.stride = 128
        self.bottleneck = bottleneck
        self.outputs = outputs
        self.url = url
        super().__init__(bottleneck, layers)

    def initialize(self):
        if self.url:
            self.load_state_dict(model_zoo.load_url(self.url))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outputs = []
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            level = i + 2
            if level > max(self.outputs):
                break
            x = layer(x)
            if level in self.outputs:
                outputs.append(x)

        return outputs

# @register
def ResNet18C4():
    return ResNet(layers=[2, 2, 2, 2], bottleneck=vrn.BasicBlock, outputs=[4], url=vrn.model_urls['resnet18'])

# @register
def ResNet34C4():
    return ResNet(layers=[3, 4, 6, 3], bottleneck=vrn.BasicBlock, outputs=[4], url=vrn.model_urls['resnet34'])