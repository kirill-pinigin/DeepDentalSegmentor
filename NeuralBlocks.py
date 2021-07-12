import torch
from torch import nn


@torch.jit.script
def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
    if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
        ds = encoder_layer.shape[2:]
        es = decoder_layer.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        encoder_layer = encoder_layer[
                        :,
                        :,
                        ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                        ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)
                        ]
    return encoder_layer, decoder_layer

def get_conv_layer(in_channels: int,
                   out_channels: int,
                   kernel_size: int = 3,
                   stride: int = 1,
                   padding: int = 1,
                   bias: bool = True,
                   ):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=bias)

def get_up_layer(in_channels: int,
                 out_channels: int,
                 kernel_size: int = 2,
                 stride: int = 2,
                 up_mode: str = 'transposed',
                 ):
    if up_mode == 'transposed':
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    else:
        return nn.Upsample(scale_factor=2.0, mode=up_mode)

def get_maxpool_layer(kernel_size: int = 2,
                      stride: int = 2,
                      padding: int = 0):
    return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)


def get_activation(activation: str):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'elu':
        return nn.ELU()


def get_normalization(normalization: str,
                      num_channels: int,):
    if normalization == 'batch':
        return nn.BatchNorm2d(num_channels)
    elif normalization == 'instance':
        return nn.InstanceNorm2d(num_channels)
    elif 'group' in normalization:
        num_groups = int(normalization.partition('group')[-1])  # get the group size from string
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, layer_1, layer_2):
        x = torch.cat((layer_1, layer_2), 1)
        return x


class EncoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pooling: bool = True,
                 activation: str = 'relu',
                 normalization: str = None,
                 conv_mode: str = 'same'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.normalization = normalization
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.activation = activation

        self.conv1 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,  bias=True)
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding, bias=True)

        # pooling layer
        if self.pooling:
            self.pool = get_maxpool_layer(kernel_size=2, stride=2, padding=0)

        # activation layers
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        # normalization layers
        if self.normalization:
            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels)
            self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels)

    def forward(self, x):
        y = self.conv1(x)
        if self.normalization:
            y = self.norm1(y)  # normalization 1
        y = self.act1(y)
        y = self.conv2(y)  # convolution 2
        if self.normalization:
            y = self.norm2(y)  # normalization 2
        y = self.act2(y)  # activation 2
        before_pooling = y  # save the outputs before the pooling operation
        if self.pooling:
            y = self.pool(y)  # pooling
        return y, before_pooling


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: str = 'relu',
                 normalization: str = None,
                 conv_mode: str = 'same',
                 up_mode: str = 'transposed'
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.activation = activation
        self.up_mode = up_mode

        self.up = get_up_layer(self.in_channels, self.out_channels, kernel_size=2, stride=2, up_mode=self.up_mode)

        self.conv0 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0,bias=True)
        self.conv1 = get_conv_layer(2 * self.out_channels, self.out_channels, kernel_size=3, stride=1,padding=self.padding, bias=True)
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,bias=True)

        self.act0 = get_activation(self.activation)
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        if self.normalization:
            self.norm0 = get_normalization(normalization=self.normalization, num_channels=self.out_channels)
            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels)
            self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels)

        self.concat = Concatenate()

    def forward(self, encoder_layer, decoder_layer):
        up_layer = self.up(decoder_layer)  # up-convolution/up-sampling
        cropped_encoder_layer, dec_layer = autocrop(encoder_layer, up_layer)  # cropping

        if self.up_mode != 'transposed':
            up_layer = self.conv0(up_layer)

        if self.normalization:
            up_layer = self.norm0(up_layer)

        up_layer = self.act0(up_layer)

        merged_layer = self.concat(up_layer, cropped_encoder_layer)
        y = self.conv1(merged_layer)
        if self.normalization:
            y = self.norm1(y)
        y = self.act1(y)
        y = self.conv2(y)
        if self.normalization:
            y = self.norm2(y)
        y = self.act2(y)
        return y

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl


class ModuleWrapper(nn.Module):
    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        for module in self.children():
            x = module(x)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return x, kl

class BayesianConvolution(ModuleWrapper):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, priors=None):

        super(BayesianConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = torch.nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
        self.W_rho = torch.nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))

        if self.use_bias:
            self.bias_mu = torch.nn.Parameter(torch.empty((out_channels)))
            self.bias_rho = torch.nn.Parameter(torch.empty((out_channels)))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        if self.training or sample:
           return self.sample(input)
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return torch.nn.functional.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def sample(self, input):
        W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(input.device)
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        weight = self.W_mu + W_eps * self.W_sigma

        if self.use_bias:
            bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(input.device)
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_eps * self.bias_sigma
        else:
            bias = None

        return torch.nn.functional.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def kl_loss(self):
        kl = calculate_kl(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += calculate_kl(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl


class BaysesianPerceptron(ModuleWrapper):
    def __init__(self, in_features, out_features, bias=True, priors=None):
        super(BaysesianPerceptron, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = torch.nn.Parameter(torch.empty((out_features, in_features), device=self.device))
        self.W_rho = torch.nn.Parameter(torch.empty((out_features, in_features), device=self.device))

        if self.use_bias:
            self.bias_mu = torch.nn.Parameter(torch.empty((out_features), device=self.device))
            self.bias_rho = torch.nn.Parameter(torch.empty((out_features), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        if self.training or sample:
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return torch.nn.functional.linear(input, weight, bias)

    def kl_loss(self):
        kl = calculate_kl(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += calculate_kl(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl


class VariationalEncoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pooling: bool = True,
                 activation: str = 'relu',
                 normalization: str = "batch",
                 conv_mode: str = 'same'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.normalization = normalization
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.activation = activation

        self.conv1 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True)
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True)

        # pooling layer
        self.pool = get_maxpool_layer(kernel_size=2, stride=2, padding=0)

        # activation layers
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        # normalization layers
        self.norm1 = get_normalization(normalization=normalization, num_channels=self.out_channels)
        self.norm2 = get_normalization(normalization=normalization, num_channels=self.out_channels)

        self.variation = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1))
            ,BayesianConvolution(self.out_channels, self.out_channels, kernel_size=1,stride=1, padding=0, bias=False)
            )

    def kl_loss(self):
        return self.variation[1].kl_loss()

    def sample(self, x):
        return self.variation[1](x)

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)  # normalization 1
        y = torch.nn.functional.relu(y)
        y = self.conv2(y)  # convolution 2
        y = self.norm2(y)  # normalization 2
        y = torch.nn.functional.relu(y)  # activation 2

        before_pooling = y  # save the outputs before the pooling operation

        z = self.variation(y)
        z = z.expand_as(y)
        return z, before_pooling


class RecurrentVariationalEncoderBlock(VariationalEncoderBlock):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pooling: bool = True,
                 activation: str = 'relu',
                 normalization: str = "batch",
                 conv_mode: str = 'same'):
        super().__init__(in_channels, out_channels, pooling, activation, normalization, conv_mode)
        self.conv3 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True)
        self.conv4 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True)

        # pooling layer
        self.pool = get_maxpool_layer(kernel_size=2, stride=2, padding=0)

        # activation layers
        self.act3 = get_activation(self.activation)
        self.act4 = get_activation(self.activation)

        # normalization layers
        self.norm3 = get_normalization(normalization=normalization, num_channels=self.out_channels)
        self.norm4 = get_normalization(normalization=normalization, num_channels=self.out_channels)
        self.hidden = None


    def forward(self, x, hidden_state = None):
        y = self.conv1(x)
        y = self.norm1(y)  # normalization 1
        y = torch.nn.functional.relu(y)
        y = self.conv2(y)  # convolution 2
        y = self.norm2(y)  # normalization 2
        y = torch.nn.functional.relu(y)  # activation 2
        update = torch.sigmoid(y)
        y = self.conv3(update)
        y = self.norm3(y)  # normalization 1
        y = torch.nn.functional.relu(y)
        y = self.conv4(y)  # convolution 2
        y = self.norm4(y)  # normalization 2
        y = torch.nn.functional.relu(y)  # activation 2
        output = torch.tanh(y)
        before_pooling = y  # save the outputs before the pooling operation

        if hidden_state is None:
            hidden_state = torch.zeros_like(y)

        y = hidden_state * (1 - update) + output * update
        hidden_state = y
        z = self.variation(y)
        z = z.expand_as(y)
        return z, before_pooling, hidden_state
