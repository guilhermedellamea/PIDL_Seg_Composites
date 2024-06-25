import torch
import torch.nn as nn
import torch.nn.functional as F

################### Basic Modules

class DoubleConv(nn.Module):
    """A module comprising two convolutional layers each followed by an activation function.
    
    This module performs two consecutive convolution operations, each followed by a ReLU, LeakyReLU, 
    or ELU activation function. Batch normalization is optional.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (int, optional): Number of intermediate channels. If None, it defaults to out_channels.
        activation (str): Type of activation function to use ("relu", "lrelu", "elu"). Default is "relu".
        dropout_rate (float, optional): Dropout rate. If None, no dropout is applied.
    
    Attributes:
        double_conv (nn.Sequential): Sequential container of layers for the double convolution.
        dropout (nn.Dropout or None): Dropout layer, if dropout_rate is specified.
    """

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
    """Downscaling with maxpool followed by a DoubleConv.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): Type of activation function to use in DoubleConv ("relu", "lrelu", "elu"). Default is "relu".
        dropout_rate (float, optional): Dropout rate for DoubleConv. If None, no dropout is applied.
    
    Attributes:
        maxpool_conv (nn.Sequential): Sequential container of layers for downsampling.
    """

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
    """Downscaling with maxpool followed by a DoubleConv, including skip connections.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): Type of activation function to use in DoubleConv ("relu", "lrelu", "elu"). Default is "relu".
        dropout_rate (float, optional): Dropout rate for DoubleConv. If None, no dropout is applied.
    
    Attributes:
        maxpool (nn.MaxPool2d): Max pooling layer.
        conv (DoubleConv): Double convolutional layer with activation function.
    """

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
    """Upscaling then double conv with skip connections.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bilinear (bool): If True, use bilinear upsampling. Otherwise, use ConvTranspose2d.
        activation (str): Type of activation function to use in DoubleConv ("relu", "lrelu", "elu"). Default is "relu".
        dropout_rate (float, optional): Dropout rate for DoubleConv. If None, no dropout is applied.
    
    Attributes:
        up (nn.Module): Upsampling layer (either Upsample or ConvTranspose2d).
        conv (DoubleConv): Double convolutional layer with activation function.
    """

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
    """Output convolutional layer with kernel size 1.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    
    Attributes:
        conv (nn.Conv2d): Convolutional layer.
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)



################### UNet

class UNet(nn.Module):
    """U-Net architecture for image segmentation or similar tasks.
    
    The U-Net consists of an encoder and a decoder. The encoder downsamples the input image, 
    capturing context through a series of downsampling and convolutional layers. The decoder 
    upsamples the feature maps, reconstructing the spatial dimensions through a series of 
    upsampling and convolutional layers, incorporating skip connections from the encoder.

    Args:
        n_channels (int): Number of input channels.
        n_output_channels (int): Number of output channels.
        initial_channels (int): Number of channels for the initial convolutional layer.
        ndepth (int): Depth of the U-Net, defining the number of downsampling/upsampling operations.
        bilinear (bool): If True, use bilinear upsampling. Otherwise, use ConvTranspose2d.
        activation (str): Type of activation function to use in convolutional layers ("relu", "lrelu", "elu"). Default is "relu".
        dropout_rate (float, optional): Dropout rate for convolutional layers. If None, no dropout is applied.
        final_activation (str, optional): Final activation function ("softmax" or None).
    
    Attributes:
        n_channels (int): Number of input channels.
        n_output_channels (int): Number of output channels.
        bilinear (bool): If True, use bilinear upsampling.
        final_activation (str or None): Final activation function.
        downs (nn.ModuleList): List of downsampling layers.
        down_last (Down): Final downsampling layer.
        ups (nn.ModuleList): List of upsampling layers.
        outc (OutConv): Output convolutional layer.
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

        # DoubleConv + Downsampling(n_depth-1): all outputs in "list_downs" are to be concatenated with inputs in "list_ups"
        list_downs = []

        # (Input channels, Output channels)[result]: (1, initial_channels)[x1]
        list_downs.append(
            DoubleConv(
                n_channels,
                c,
                activation=activation,
                dropout_rate=dropout_rate,
            )
        )

        # Downsampling layers: (initial_channels, 2*initial_channels)[x2], (2*initial_channels, 4*initial_channels)[x3], etc.
        for i in range(ndepth - 1):
            list_downs.append(
                Down(c, cn, activation=activation, dropout_rate=dropout_rate)
            )
            c = cn
            cn *= 2
        self.downs = nn.ModuleList(list_downs)

        # The last downsampling layer: (current_channels, next_channels/factor)[x5]
        self.down_last = Down(
            c, cn // factor, activation=activation, dropout_rate=dropout_rate
        )

        # Upsampling layers
        list_ups = []

        # (next_channels, current_channels/factor)[cat x4], (current_channels, prev_channels/factor)[cat x3], etc.
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

        # Output layer: (final_channels, n_output_channels)
        self.outc = OutConv(c, n_output_channels)

    def forward(self, x):
        # Store feature maps from each downsampling operation
        features = []
        for down in self.downs:
            x = down(x)
            features.append(x)

        # Latent space feature maps
        x = self.down_last(x)

        # Match the concatenation with feature maps from the encoder
        for i, up in enumerate(self.ups):
            x = up(x, features[len(features) - i - 1])

        # Output layer
        x = self.outc(x)
        if self.final_activation == "softmax":
            x = torch.nn.functional.softmax(x, dim=1)
        return x

    def use_checkpointing(self):
        """Enable gradient checkpointing to save memory."""
        self.downs = torch.utils.checkpoint(self.downs)
        self.down_last = torch.utils.checkpoint(self.down_last)
        self.ups = torch.utils.checkpoint(self.ups)
        self.outc = torch.utils.checkpoint(self.outc)

