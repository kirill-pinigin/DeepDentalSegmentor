import torch
from torch import nn
import copy

from NeuralBlocks import *


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 33,
                 n_blocks: int = 5,
                 start_filters: int = 64,
                 activation: str = 'relu',
                 normalization: str = 'batch',
                 conv_mode: str = 'same',
                 up_mode: str = 'transposed'
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.start_filters = start_filters
        self.activation = activation
        self.normalization = normalization
        self.conv_mode = conv_mode
        self.up_mode = up_mode

        self.encoder_blocks = []
        self.decoder_blocks = []

        # create encoder path
        for i in range(self.n_blocks):
            num_filters_in = self.in_channels if i == 0 else num_filters_out
            num_filters_out = self.start_filters * (2 ** i)
            pooling = True if i < self.n_blocks - 1 else False

            encoder_block = EncoderBlock(in_channels=num_filters_in,
                                   out_channels=num_filters_out,
                                   pooling=pooling,
                                   activation=self.activation,
                                   normalization=self.normalization,
                                   conv_mode=self.conv_mode)

            self.encoder_blocks.append(encoder_block)

        # create decoder path (requires only n_blocks-1 blocks)
        for i in range(n_blocks - 1):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            decoder_block = DecoderBlock(in_channels=num_filters_in,
                               out_channels=num_filters_out,
                               activation=self.activation,
                               normalization=self.normalization,
                               conv_mode=self.conv_mode,
                               up_mode=self.up_mode)

            self.decoder_blocks.append(decoder_block)

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

        # final convolution
        self.head = nn.Sequential(
            get_conv_layer(num_filters_out, num_filters_out, kernel_size=3, stride=1, padding=1, bias=True)
            , nn.ReLU()
            , get_conv_layer(num_filters_out, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # initialize the weights
        self.initialize_parameters()

    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.weight, **kwargs)  # weights

    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.bias, **kwargs)  # bias

    def initialize_parameters(self,
                              method_weights=nn.init.xavier_uniform_,
                              method_bias=nn.init.zeros_,
                              kwargs_weights={},
                              kwargs_bias={}
                              ):

        for module in self.modules():
            self.weight_init(module, method_weights, **kwargs_weights)  # initialize weights
            self.bias_init(module, method_bias, **kwargs_bias)  # initialize bias

    def evaluate(self, x: torch.tensor):
        encoder_output = []
        # Encoder pathway
        for module in self.encoder_blocks:
            x, before_pooling = module(x)
            encoder_output.append(before_pooling)

        # Decoder pathway
        for i, module in enumerate(self.decoder_blocks):
            before_pool = encoder_output[-(i + 2)]
            x = module(before_pool, x)
        return x, encoder_output[-1]

    def forward(self, x: torch.tensor):
        x,_ = self.evaluate(x)
        out = self.head(x)
        return torch.softmax(out, dim=1)

    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys() if '_' not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'


class JointDetectionUNet(UNet):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 33,
                 n_blocks: int = 5,
                 start_filters: int = 64,
                 activation: str = 'relu',
                 normalization: str = 'batch',
                 conv_mode: str = 'same',
                 up_mode: str = 'transposed',
                 points_dimension = 320,
                 ):
        super().__init__(in_channels,out_channels, n_blocks, start_filters,activation, normalization, conv_mode, up_mode)

        bottleneck_dimension = self.encoder_blocks[-1].out_channels
        self.points_predictor = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1))
                        , get_conv_layer(bottleneck_dimension, bottleneck_dimension, kernel_size=1, stride=1, padding=0,bias=True)
                        , nn.ReLU()
                        , get_conv_layer(bottleneck_dimension, bottleneck_dimension, kernel_size=1, stride=1, padding=0, bias=True)
                        , nn.ReLU()
                        , get_conv_layer(bottleneck_dimension, points_dimension, kernel_size=1, stride=1, padding=0,bias=True)
                        )

    def forward(self, x: torch.tensor):
        x, botlle_neck = self.evaluate(x)
        segments = self.head(x)
        points = self.points_predictor(botlle_neck)
        return torch.softmax(segments, dim=1), torch.sigmoid(points)


class VariationalUNet(UNet):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 33,
                 n_blocks: int = 5,
                 start_filters: int = 64,
                 activation: str = 'relu',
                 normalization: str = 'batch',
                 conv_mode: str = 'same',
                 up_mode: str = 'transposed',
                 points_dimension = 320,
                 ):
        super().__init__(in_channels,out_channels, n_blocks, start_filters,activation, normalization, conv_mode, up_mode)

        varitional_encoder = VariationalEncoderBlock(self.encoder_blocks[-1].in_channels,self.encoder_blocks[-1].out_channels)
        self.encoder_blocks[-1] = varitional_encoder
        self.reconstructor_blocks = copy.deepcopy(self.decoder_blocks)
        dimension = self.decoder_blocks[-1].out_channels

        # final convolution
        self.reconstruction_head = nn.Sequential(
            get_conv_layer(dimension, dimension, kernel_size=3, stride=1, padding=1, bias=True)
            , nn.ReLU()
            , get_conv_layer(dimension, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x: torch.tensor):
        encoder_output = []
        # Encoder pathway
        for module in self.encoder_blocks:
            x, before_pooling = module(x)
            encoder_output.append(before_pooling)

        # Decoder pathway
        for i, module in enumerate(self.decoder_blocks):
            before_pool = encoder_output[-(i + 2)]
            r = self.reconstructor_blocks[i](before_pool, x)
            x = module(before_pool, x)

        segments = self.head(x)
        reconstruction = self.reconstruction_head(r)
        kl = self.encoder_blocks[-1].kl_loss()
        return torch.softmax(segments, dim=1), torch.sigmoid(reconstruction), kl


class RecurentVariationalUNet(VariationalUNet):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 33,
                 n_blocks: int = 5,
                 start_filters: int = 64,
                 activation: str = 'relu',
                 normalization: str = 'batch',
                 conv_mode: str = 'same',
                 up_mode: str = 'transposed',
                 points_dimension=320,
                 ):
        super().__init__(in_channels, out_channels, n_blocks, start_filters, activation, normalization, conv_mode,
                         up_mode)
        self.steps = 3
        varitional_encoder = RecurrentVariationalEncoderBlock(self.encoder_blocks[-1].in_channels, self.encoder_blocks[-1].out_channels)
        self.encoder_blocks[-1] = varitional_encoder
        self.reconstructor_blocks = copy.deepcopy(self.decoder_blocks)
        dimension = self.decoder_blocks[-1].out_channels

        # final convolution
        self.reconstruction_head = nn.Sequential(
            get_conv_layer(dimension, dimension, kernel_size=3, stride=1, padding=1, bias=True)
            , nn.ReLU()
            , get_conv_layer(dimension, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x: torch.tensor):
        encoder_output = []
        # Encoder pathway
        for module in self.encoder_blocks:
            x, before_pooling = module(x)
            encoder_output.append(before_pooling)

        # Decoder pathway
        for i, module in enumerate(self.decoder_blocks):
            before_pool = encoder_output[-(i + 2)]
            r = self.reconstructor_blocks[i](before_pool, x)
            x = module(before_pool, x)

        segments = self.head(x)
        reconstruction = self.reconstruction_head(r)
        kl = self.encoder_blocks[-1].kl_loss()
        return torch.softmax(segments, dim=1), torch.sigmoid(reconstruction), kl