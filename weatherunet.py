import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as L
from metrics import CustomLightningModule

class ResNetUNetEncoder(CustomLightningModule):
    """
    ResNet18-based encoder for UNet.
    Input: (B, 1, H, W)  -> Output: list of features at multiple scales
    Works for non-square H, W.
    """
    def __init__(self, norm, extra_logger):
        super().__init__(extra_logger=extra_logger)
        self.extra_logger = extra_logger

        base_model = models.resnet18(norm_layer=norm)

        # First conv: adapt ResNet18 to 1 input channel
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool  # exact halving

        # ResNet layers
        self.layer1 = base_model.layer1  # 1/4 size
        self.layer2 = base_model.layer2  # 1/8 size
        self.layer3 = base_model.layer3  # 1/16 size
        self.layer4 = base_model.layer4  # 1/32 size

    def forward(self, x):
        # Save input size for decoder
        input_size = x.shape[-2:]

        # Initial stem
        x0 = self.conv1(x)   # H/2, W/2
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x1 = self.maxpool(x0)  # H/4, W/4

        # Encoder path
        x2 = self.layer1(x1)   # H/4
        x3 = self.layer2(x2)   # H/8
        x4 = self.layer3(x3)   # H/16
        x5 = self.layer4(x4)   # H/32

        return [x0, x1, x2, x3, x4, x5], input_size

class DecoderBlock(CustomLightningModule):
    """
    Decoder block: upsample (ConvTranspose2d) + conv layers.
    """
    def __init__(self, in_channels, skip_channels, out_channels, extra_logger, norm=nn.BatchNorm2d):
        super().__init__(extra_logger=extra_logger)
        # Exact doubling
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)  # 2x exact
        if skip is not None:
            # Make x match skip (prefer cropping x over altering skip features)
            _, _, H, W = skip.shape
            x = self.match_spatial(x,  H, W)
            # If due to rounding the other way around, ensure skip also matches x
            if skip.shape[-2:] != x.shape[-2:]:
                _, _, H, W = x.shape
                skip = self.match_spatial(skip, H, W)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetUNetDecoder(CustomLightningModule):
    """
    UNet decoder with shape-preserving design.
    Always outputs same H×W as input.
    """
    def __init__(self, norm, extra_logger):
        super().__init__(extra_logger=extra_logger)
        self.extra_logger = extra_logger

        # Mirror channels from ResNet18 encoder
        self.decoder4 = DecoderBlock(512, 256, 256, self.extra_logger, norm)  # 1/32 -> 1/16
        self.decoder3 = DecoderBlock(256, 128, 128, self.extra_logger, norm)  # 1/16 -> 1/8
        self.decoder2 = DecoderBlock(128, 64, 64, self.extra_logger, norm)    # 1/8 -> 1/4
        self.decoder1 = DecoderBlock(64, 64, 64, self.extra_logger, norm)     # 1/4 -> 1/2
        self.decoder0 = DecoderBlock(64, 64, 32, self.extra_logger, norm)     # 1/2 -> 1/1

        self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # Final prediction
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, encoder_features):
        [x0, x1, x2, x3, x4, x5], input_size = encoder_features

        d4 = self.decoder4(x5, x4)   # 1/32 -> 1/16
        d3 = self.decoder3(d4, x3)   # 1/16 -> 1/8
        d2 = self.decoder2(d3, x2)   # 1/8  -> 1/4
        d1 = self.decoder1(d2, x1)   # 1/4  -> 1/2
        d0 = self.decoder0(d1, x0)   # 1/2  -> 1/1 (full resolution)

        d0 = self.final_upsample(d0)
        output = self.final_conv(d0)
        if output.shape[-2:] != input_size:
            self.extra_logger.debug(f"Shape mismatch: got {output.shape[-2:]}, expected {input_size}")
            output = self.match_spatial(output, input_size[-2], input_size[-1])
            self.extra_logger.debug(f"New size: {output.shape[-2:]}")
        return output

class WeatherResNetUNet(CustomLightningModule):
    """
    UNet model for weather
    """
    def __init__(self, loss, learning_rate, norm, supervised, extra_logger):
        super().__init__(extra_logger=extra_logger)
        self.loss = loss
        self.learning_rate = learning_rate
        self.supervised = supervised

        # Components
        self.encoder = ResNetUNetEncoder(norm=norm, extra_logger=extra_logger)
        self.decoder = ResNetUNetDecoder(norm=norm, extra_logger=extra_logger)

    def forward(self, x):
        """
        Forward pass through complete UNet
        Args:
            x: Input tensor (B, 1, 256/128, 256/128)
        Returns:
            output: Prediction (B, 1, 256/128, 256/128)
        """
        # data_size = min(x.shape[-1], x.shape[-2])
        encoder_features = self.encoder(x)
        output = self.decoder(encoder_features)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=4, verbose=True),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
