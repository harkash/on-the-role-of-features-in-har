import torch.nn as nn
import torch
import torch.nn.functional as F
from building_blocks import Vggish
from torch.nn import init

# ----------------------------------------------------------------------------------------------------------------------

class MLP_Classifier(nn.Module):
    def __init__(self, input_size):
        super(MLP_Classifier, self).__init__()
        self.linear_1 = nn.Linear(in_features=input_size, out_features=2048)
        self.linear_2 = nn.Linear(in_features=2048, out_features=512)
        self.out = nn.Linear(in_features=512, out_features=18)

        self.bn_1 = nn.BatchNorm1d(2048)
        self.bn_2 = nn.BatchNorm1d(512)

        self.dropout_1 = nn.Dropout(0.4)
        self.dropout_2 = nn.Dropout(0.4)

        # Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear or nn.GRU or nn.LSTM):
                init.xavier_normal_(m.weight)
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, inputs):
        linear_1 = self.dropout_1(F.relu(self.bn_1(self.linear_1(inputs))))
        linear_2 = self.dropout_2(F.relu(self.bn_2(self.linear_2(linear_1))))
        out = self.out(linear_2)

        return out


class Convolutional_Autoencoder_Flat(nn.Module):
    def __init__(self, dataset, conv_size, latent_size):
        """
        Creating a convolutional autoencoder model
        :param dataset: choice of dataset. E.g.: Opportunity, Skoda etc
        :param conv_size: flattened size of the convolutional layers
        :param latent_size: latent representation size
        """
        super(Convolutional_Autoencoder_Flat, self).__init__()
        # Convolution
        self.up_conv_1 = Vggish(in_channels=1, out_channels=64)
        self.up_conv_2 = Vggish(in_channels=64, out_channels=128)

        if dataset == 'usc_had':
            self.up_conv_3 = Vggish(in_channels=128, out_channels=256, pool=False)
        else:
            self.up_conv_3 = Vggish(in_channels=128, out_channels=256)

        if dataset == 'daphnet' or dataset == 'usc_had':
            self.up_conv_4 = Vggish(in_channels=256, out_channels=512, pool=False)
        else:
            self.up_conv_4 = Vggish(in_channels=256, out_channels=512)

        # Flattening
        self.embedding = nn.Linear(conv_size, latent_size)
        self.de_embedding = nn.Linear(latent_size, conv_size)
        self.bn_1 = nn.BatchNorm1d(latent_size)
        self.bn_2 = nn.BatchNorm1d(conv_size)

        # Deconvolution
        self.down_conv_4 = Vggish(in_channels=512, out_channels=256, pool=False)
        self.down_conv_3 = Vggish(in_channels=256, out_channels=128, pool=False)
        self.down_conv_2 = Vggish(in_channels=128, out_channels=64, pool=False)
        self.down_conv_1 = Vggish(in_channels=64, out_channels=1, pool=False, last=True)

        self.dataset = dataset

        # Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear or nn.GRU or nn.LSTM):
                init.xavier_normal_(m.weight)
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, inputs):
        conv_1 = self.up_conv_1(inputs)  # 64x12x38
        conv_2 = self.up_conv_2(conv_1)  # 128x6x19
        conv_3 = self.up_conv_3(conv_2)  # 256x3x9
        conv_4 = self.up_conv_4(conv_3)  # 512x1x4

        # Vectorizing conv_4 and reducing size to the bottleneck
        rep = conv_4.view(conv_4.shape[0], -1)
        embedding_out = self.embedding(rep)
        embedding = F.relu(self.bn_1(embedding_out))
        de_embedding = F.relu(self.bn_2(self.de_embedding(embedding)))
        conv_back = de_embedding.view(conv_4.shape)

        # Performing the decoding
        if self.dataset == 'daphnet' or self.dataset == 'usc_had':
            pad_4 = conv_back
        else:
            inter_4 = F.interpolate(conv_back, scale_factor=2, mode='nearest')
            if self.dataset == 'pamap2':
                pad_4 = F.pad(inter_4, (0, 0, 0, 1), 'constant', 0)
            else:
                # Adding a zero row to the bottom, zero column to the right
                pad_4 = F.pad(inter_4, (0, 1, 0, 1), 'constant', 0)
        de_conv_4 = self.down_conv_4(pad_4)

        if self.dataset == 'usc_had':
            pad_3 = de_conv_4
        else:
            inter_3 = F.interpolate(de_conv_4, scale_factor=2, mode='nearest')  # Upscaling by 2x2, 256x6x18
            if self.dataset == 'daphnet':
                pad_3 = F.pad(inter_3, (0, 0, 0, 1), 'constant', 0)  # Adding a zero row to the bottom
            else:
                # Adding a zero row to the bottom, zero column to the right
                pad_3 = F.pad(inter_3, (0, 1, 0, 1), 'constant', 0)
        de_conv_3 = self.down_conv_3(pad_3)  # 128x7x19

        inter_2 = F.interpolate(de_conv_3, scale_factor=2, mode='nearest')  # Upscaling by 2x2, 128x12x38
        if self.dataset == 'usc_had':
            pad_2 = F.pad(inter_2, (0, 1, 0, 1), 'constant', 0)
        else:
            pad_2 = F.pad(inter_2, (0, 0, 0, 1), 'constant', 0)
        de_conv_2 = self.down_conv_2(pad_2)  # 64x12x38

        inter_1 = F.interpolate(de_conv_2, scale_factor=2, mode='nearest')  # Upscaling by 2x2, 64x24x76
        if self.dataset == 'opportunity' or self.dataset == 'daphnet':
            # Due to the size of opportunity and daphnet, this is needed. Not otherwise.
            inter_1 = F.pad(inter_1, (0, 1), 'constant', 0)  # 64x24x77

        de_conv_1 = self.down_conv_1(inter_1)  # 1x24x77

        return de_conv_1, embedding_out