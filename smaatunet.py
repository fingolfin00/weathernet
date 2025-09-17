'''
Description: UNet architecture with CBAM. This script is straight from  SmaAT-UNet:
    https://github.com/HansBambel/SmaAt-UNet/tree/master/models
'''
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from metrics import CustomLightningModule

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels),
        )

    def forward(self, x):
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out

class DepthwiseSeparableConv(CustomLightningModule):
    def __init__(self, in_channels, output_channels, kernel_size, extra_logger, padding=0, kernels_per_layer=1):
        super(DepthwiseSeparableConv, self).__init__(extra_logger=extra_logger)
        # In Tensorflow DepthwiseConv2D has depth_multiplier instead of kernels_per_layer
        self.extra_logger.debug("DepthwiseSeparableConv params:")
        self.extra_logger.debug(f" kernel_size: {kernel_size}")
        self.extra_logger.debug(f" padding: {padding}")
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        self.extra_logger.debug(f"Depthwise input: shape={x.shape}, dtype={x.dtype}, device={x.device}")
        # self.extra_logger.debug(x)
        if (torch.isnan(x).any() or torch.isinf(x).any()):
             self.extra_logger.error("Depthwise input has NaN/Inf!")
        
        x = self.depthwise(x)
        self.extra_logger.debug(f"Depthwise output: shape={x.shape}, dtype={x.dtype}, device={x.device}")

        self.extra_logger.debug(f"Pointwise input: shape={x.shape}, dtype={x.dtype}, device={x.device}")
        x = self.pointwise(x)
        self.extra_logger.debug(f"Pointwise output: shape={x.shape}, dtype={x.dtype}, device={x.device}")
        return x

class DoubleConvDS(CustomLightningModule):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, extra_logger, mid_channels=None, kernels_per_layer=1):
        super().__init__(extra_logger=extra_logger)
        if not mid_channels:
            mid_channels = out_channels
        self.extra_logger.debug("DoubleConvDS params:")
        self.extra_logger.debug(f" in_channels: {in_channels}")
        self.extra_logger.debug(f" mid_channels: {mid_channels}")
        self.extra_logger.debug(f" out_channels: {out_channels}")
        self.extra_logger.debug(f" kernels_per_layer: {kernels_per_layer}")
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels,
                mid_channels,
                kernel_size=3,
                extra_logger=extra_logger,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            DepthwiseSeparableConv(
                mid_channels,
                out_channels,
                kernel_size=3,
                extra_logger=extra_logger,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class DownDS(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, extra_logger, kernels_per_layer=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDS(in_channels, out_channels, extra_logger, kernels_per_layer=kernels_per_layer),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpDS(CustomLightningModule):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, extra_logger, bilinear=True, kernels_per_layer=1):
        super().__init__(extra_logger=extra_logger)

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConvDS(
                in_channels,
                out_channels,
                extra_logger,
                in_channels // 2, # mid channels
                kernels_per_layer=kernels_per_layer,
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, extra_logger, kernels_per_layer=kernels_per_layer)

    def forward(self, x1, x2):
        self.extra_logger.debug(f"UpDS.forward - x1 (from deeper layer): shape={x1.shape}, dtype={x1.dtype}, device={x1.device}")
        self.extra_logger.debug(f"UpDS.forward - x2 (skip connection): shape={x2.shape}, dtype={x2.dtype}, device={x2.device}")

        x1 = self.up(x1)
        self.extra_logger.debug(f"UpDS.forward - x1 after upsample: shape={x1.shape}, dtype={x1.dtype}, device={x1.device}")
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffX < 0 or diffY < 0:
            self.extra_logger.error(f"UpDS.forward - Negative padding needed! x1:{x1.shape}, x2:{x2.shape}")
            # This indicates an issue with dimensions being larger than expected,
            # which could lead to problematic padding arguments.
            # You might want to raise an error or adjust logic here.

        self.extra_logger.debug(f"UpDS.forward - padding: [{diffX // 2}, {diffX - diffX // 2}, {diffY // 2}, {diffY - diffY // 2}]")
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        self.extra_logger.debug(f"UpDS.forward - x1 after padding: shape={x1.shape}, dtype={x1.dtype}, device={x1.device}")

        x = torch.cat([x2, x1], dim=1)
        self.extra_logger.debug(f"UpDS.forward - x after concatenation: shape={x.shape}, dtype={x.dtype}, device={x.device}")
        
        return self.conv(x) # This calls DoubleConvDS, which in turn calls DepthwiseSeparableConv

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SmaAt_UNet(CustomLightningModule):
    def __init__(
        self, loss, learning_rate, norm, supervised, extra_logger,
        n_channels=1, n_classes=1,
        kernels_per_layer=2,
        bilinear=True,
        reduction_ratio=16,
    ):
        super(SmaAt_UNet, self).__init__(extra_logger=extra_logger)
        
        self.extra_logger = extra_logger
        self.loss = loss
        self.learning_rate = learning_rate
        self.supervised = supervised
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        self.inc = DoubleConvDS(self.n_channels, 64, self.extra_logger, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, self.extra_logger, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, self.extra_logger, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, self.extra_logger, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, extra_logger, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024// factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, self.extra_logger, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.extra_logger, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.extra_logger, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.extra_logger, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        # Inside your smaatunet.py, e.g., in the forward method of DoubleConv or Unet
        if torch.isnan(x).any() or torch.isinf(x).any():
            self.extra_logger.debug("Input tensor contains NaN or Inf values!")
            # Add more debugging info:
            self.extra_logger.debug(f"Input shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
            # Potentially break here to inspect further
            exit()
        
        # You can also check model parameters
        for name, param in self.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                self.extra_logger.debug(f"Model parameter {name} contains NaN or Inf values!")
                exit()
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        self.extra_logger.debug(f"Before up1: x5Att shape: {x5Att.shape}, x4Att shape: {x4Att.shape}")
        x = self.up1(x5Att, x4Att)
        self.extra_logger.debug(f"After up1: x shape: {x.shape}, x3Att shape: {x3Att.shape}")
        x = self.up2(x, x3Att)
        self.extra_logger.debug(f"After up2: x shape: {x.shape}, x2Att shape: {x2Att.shape}")
        x = self.up3(x, x2Att)
        self.extra_logger.debug(f"After up3: x shape: {x.shape}, x1Att shape: {x1Att.shape}")
        x = self.up4(x, x1Att)
        self.extra_logger.debug(f"After up4: x shape: {x.shape}")
        logits = self.outc(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=4, verbose=True),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
