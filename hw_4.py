import torch
import torch.nn as nn
import torchvision as tv
from typing import Tuple


class UnetFromPretrained(torch.nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int):
        """
        :param encoder: nn.Sequential, pretrained encoder
        :param num_classes: Python int, number of segmentation classes
        """
        super(UnetFromPretrained, self).__init__()
        self.num_classes = num_classes
        self.encoder, self.decoder, features_out = create_features(encoder)
        self.classifier = nn.Conv2d(features_out, num_classes, (1, 1), (1, 1))

    def forward(self, x):
        skip_connections = []
        for module in self.encoder:
            if isinstance(module, nn.MaxPool2d):
                skip_connections.append(x)
            x = module(x)

        for module in self.decoder:
            if isinstance(module, nn.Sequential):
                x = torch.cat([x, skip_connections.pop()], dim=1)
            x = module(x)
        x = self.classifier(x)
        return x


def save_model(model, destination):
    torch.save(model.state_dict(), destination)


def load_model() -> Tuple[nn.Module, str]:
    """
    :return: model: your trained NN; encoder_name: name of NN, which was used to create your NN
    """
    vgg19_bn = tv.models.vgg19_bn(True)
    num_classes = 6
    model = UnetFromPretrained(vgg19_bn.features, num_classes)
    # model.load_state_dict(torch.load(f'unet_0.7891678389344641.pth', map_location=torch.device('cpu')))
    encoder_name = 'vgg19_bn'
    return model, encoder_name


def create_features(input_encoder: nn.Module) -> (nn.Module, nn.Module, int):
    decoder_modules = []
    channels = [input_encoder[0].out_channels]
    encoder_modules = []
    encoder_layers = []
    decoder_layers = []

    # create ordered dict of decoder layers

    for i, m in enumerate(input_encoder.children()):
        if isinstance(m, nn.Conv2d):
            encoder_modules.append(m)
            if i == 0:
                decoder_modules.append(
                    nn.Conv2d(m.out_channels, m.out_channels, m.kernel_size, m.stride, m.padding, m.dilation))

            else:
                decoder_modules.append(nn.Conv2d(m.out_channels, m.in_channels, m.kernel_size,
                                                 m.stride, m.padding, m.dilation))
            channels.append(m.out_channels)
        elif isinstance(m, nn.MaxPool2d):
            encoder_layers.append(nn.Sequential(*encoder_modules))
            encoder_modules = []
            encoder_layers.append(m)

            decoder_layers.append(decoder_sequential(decoder_modules))
            decoder_modules = []
            decoder_layers.append(nn.ConvTranspose2d(channels[-1], channels[-1], m.kernel_size, m.stride))
        else:
            encoder_modules.append(m)

    if len(encoder_modules) > 0:
        encoder_layers.append(nn.Sequential(*encoder_modules))

    if isinstance(encoder_layers[-1], nn.MaxPool2d):
        encoder_layers = encoder_layers[:-1]
        decoder_layers = decoder_layers[:-2]

    encoder = nn.Sequential(*encoder_layers)
    decoder = nn.Sequential(*reversed(decoder_layers))

    return encoder, decoder, channels[0]


def decoder_sequential(decoder_modules: list):
    modules = []
    for i, m in enumerate(reversed(decoder_modules)):
        if i == 0:
            modules.append(nn.Conv2d(m.in_channels * 2, m.out_channels, m.kernel_size, m.stride, m.padding, m.dilation))
        else:
            modules.append(m)
        modules.append(nn.BatchNorm2d(m.out_channels))
        modules.append(nn.ReLU(inplace=True))

    return nn.Sequential(*modules)


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    @staticmethod
    def forward(x):
        print(x.shape)
        return x


if __name__ == '__main__':
    model, encoder_name = load_model()
