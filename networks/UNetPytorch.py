import torch
import torch.nn as nn
import torch.nn.functional as F

################### Basic Modules

class DoubleConv(nn.Module):
    """Two convolutional layers: (Conv => [BN2d] => ReLU/LReLU)*2
    * Attention: In the original UNet structure, DoubleConv only uses ReLU.
    And the default negative slope should be 0.01 if LReLU"""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        activation="relu",
        dropout_rate=None,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if activation == "lrelu":
            activation_function = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif activation == "relu":
            activation_function = nn.ReLU(inplace=True)
        elif activation == "elu":
            activation_function = nn.ELU(inplace=True)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            activation_function,
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            activation_function,
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

    def forward(self, x):
        x = self.double_conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Down(nn.Module):
    """Downscaling : maxpool then double conv"""

    def __init__(self, in_channels, out_channels, activation="relu", dropout_rate=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(
                in_channels,
                out_channels,
                activation=activation,
                dropout_rate=dropout_rate,
            ),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DownSkip(nn.Module):
    """Downscaling : maxpool then double conv with skipping connections"""

    def __init__(self, in_channels, out_channels, activation="relu", dropout_rate=None):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(
            in_channels,
            out_channels // 2,
            activation=activation,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        xp = self.maxpool(x)
        x = self.conv(xp)
        x = torch.cat([xp, x], dim=1)
        return x


class Up(nn.Module):
    """Upscaling : upsampling then double conv with skip connections"""

    def __init__(
        self,
        in_channels,
        out_channels,
        bilinear=True,
        activation="relu",
        dropout_rate=None,
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels,
                out_channels,
                in_channels // 2,
                activation=activation,
                dropout_rate=dropout_rate,
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv(
                in_channels,
                out_channels,
                activation=activation,
                dropout_rate=dropout_rate,
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW (channel - height - width)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Conv layer with kernel size 1"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


def init_weights(layers, kernel_initializer):
    if kernel_initializer == "xavier_normal":
        layers.apply(init_weights_xavier_normal)
    if kernel_initializer == "kaiming_normal":
        layers.apply(init_weights_kaiming_normal)


def init_weights_kaiming_normal(m):
    if hasattr(m, "weight") and m.weight is not None:
        torch.nn.init.kaiming_normal_(m.weight)


def init_weights_xavier_normal(m):
    if hasattr(m, "weight") and m.weight is not None:
        torch.nn.init.xavier_normal_(m.weight)


################### UNet

class UNet(nn.Module):
    """U-Net architecture for the 2nd evaluator/predictor "epsilon_y" :
    Encoder: (Conv => [BN2d] => ReLU/LReLU)*2 => { Maxpool => [(Conv => [BN2d] => ReLU/LReLU)*2] }*ndepth => ...
    Decoder: ... => { Upsampling/DeConv => [(Conv => [BN2d] => ReLU/LReLU)*2] }*ndepth => Conv
    - initial_channels, n_depth and bilinear can be changed while training
    - n_channels=1 for input channel(s) and n_classes=1 for output channel(s) by defaut
    """

    def __init__(
        self,
        n_channels,
        n_output_channels,
        initial_channels,
        ndepth,
        bilinear=False,
        activation="relu",
        dropout_rate=None,
        final_activation=None,
    ):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_output_channels = n_output_channels
        self.bilinear = bilinear
        self.final_activation = final_activation

        c = initial_channels
        cn = 2 * initial_channels
        factor = 2 if bilinear else 1

        # DoubleConv + Downsampling(n_depth-1): all outputs in "list_downs" are to be concat. with inputs in "list_ups"
        list_downs = []

        # (Input channels, Output channels)[result]: (1, 64)[x1]
        list_downs.append(
            DoubleConv(
                n_channels,
                c,
                activation=activation,
                dropout_rate=dropout_rate,
            )
        )

        # (64, 128)[x2]-(128, 256)[x3]-(256, 512)[x4]
        for i in range(ndepth - 1):
            list_downs.append(
                Down(c, cn, activation=activation, dropout_rate=dropout_rate)
            )
            c = cn
            cn *= 2
        self.downs = nn.ModuleList(list_downs)

        # The last downsampling layer: (512, 1024//f)[x5]
        self.down_last = Down(
            c, cn // factor, activation=activation, dropout_rate=dropout_rate
        )

        # Upsampling
        list_ups = []

        # (1024, 512//f, b)[cat x4]-(512, 256//f, b)[cat x3]-(256, 128//f, b)[cat x2]-(128, 64, b)[cat x1]
        for i in range(ndepth):
            if i != ndepth - 1:
                list_ups.append(
                    Up(
                        cn,
                        c // factor,
                        bilinear,
                        activation=activation,
                        dropout_rate=dropout_rate,
                    )
                )
                cn = c
                c = c // 2
            else:
                list_ups.append(
                    Up(
                        cn,
                        c,
                        bilinear,
                        activation=activation,
                        dropout_rate=dropout_rate,
                    )
                )
        self.ups = nn.ModuleList(list_ups)

        # (64, 1)[logits]
        self.outc = OutConv(c, n_output_channels)

    def forward(self, x):

        # Store feature maps in each downsampling operation (the first DoubleConv also included)
        features = []
        for down in self.downs:
            x = down(x)
            features.append(x)

        # Latent space feature maps
        x = self.down_last(x)

        # Match properly the concatenation in each upsampling operation
        for i, up in enumerate(self.ups):
            x = up(x, features[len(features) - i - 1])

        # Output
        x = self.outc(x)
        if self.final_activation == "softmax":
            x = torch.nn.functional.softmax(x, dim=1)
        return x

    def use_checkpointing(self):
        self.downs = torch.utils.checkpoint(self.downs)
        self.down_last = torch.utils.checkpoint(self.down_last)
        self.ups = torch.utils.checkpoint(self.ups)
        self.outc = torch.utils.checkpoint(self.outc)

