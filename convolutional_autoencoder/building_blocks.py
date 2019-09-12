import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init


# Creating the basic VGG like block. Can pass in the input and output channels, along with the pooling strategy
class Vggish(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True, last=False, stride=1):
        """
        Initializing the structure of the basic VGG-like block
        :param in_channels: the number of input channels
        :param out_channels: the number of output channels
        :param pool: flag to perform max pooling in the end
        :param last: flag to check if the block is the last layer in the autoencoder
        :param stride: stride for the convolutional layers
        """
        super(Vggish, self).__init__()
        # The two convolutional layers
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                stride=(stride, stride), padding=(1, 1), bias=False)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3),
                                stride=(stride, stride), padding=(1, 1), bias=False)

        # Batch normalization
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels)

        # Flags
        self.last = last
        self.pool = pool

        # Setting up the max pooling
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)

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
        """
        Performing the forward pass
        :param inputs: the actual data from the batch
        :return: output after the forward pass
        """
        x = F.relu(self.bn_1(self.conv_1(inputs)))

        # If it is the last layer, use sigmoid activation instead of hyperbolic tangent
        if self.last:
            x = F.tanh(self.bn_2(self.conv_2(x)))
        else:
            x = F.relu(self.bn_2(self.conv_2(x)))

        # Performing max pooling if needed
        if self.pool:
            x, indices = self.max_pool(x)

        return x