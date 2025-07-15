import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import lightning as L
from torchmetrics.image import SpatialCorrelationCoefficient

class CustomLightningModule(L.LightningModule):
    def __init__(self, extra_logger):
        super().__init__()
        self.extra_logger = extra_logger

    def _squeeze_and_add_log_img(self, x, c, n, s, tl):
        if not self.trainer.sanity_checking and self.global_step % 100 == 0:
            sq_x = torch.squeeze(x[n:n+1, c:c+1, :, :], 0)
            self.extra_logger.debug(f"{s} x, squeezed, channel {c}, sample {n}: {sq_x.shape}")
            tl.add_image(f"{s} c:{c}, n:{n}", sq_x, self.global_step)

    def _print_debug(self, x, s):
        self.extra_logger.debug(f"{s} x: {x.shape}, device: {x.device}")

class ResNetUNetEncoder(CustomLightningModule):
    """
    ResNet18-based encoder for UNet
    Input: 256x256 -> Output features at multiple scales
    """
    def __init__(self, norm, extra_logger):
        super().__init__(extra_logger=extra_logger)

        # Load ResNet18 and modify for single channel input
        base_model = models.resnet18(norm_layer=norm)

        # Input processing (256x256 -> 128x128)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool  # 128x128 -> 64x64

        # Encoder blocks
        self.layer1 = base_model.layer1  # 64x64 -> 64x64 (64 channels)
        self.layer2 = base_model.layer2  # 64x64 -> 32x32 (128 channels)
        self.layer3 = base_model.layer3  # 32x32 -> 16x16 (256 channels)
        self.layer4 = base_model.layer4  # 16x16 -> 8x8 (512 channels)

        # Channel dimensions for decoder
        self.channels = [64, 128, 256, 512]

    def forward(self, x):
        """
        Forward pass through encoder
        Args:
            x: Input tensor (B, 1, 256, 256)
        Returns:
            features: List of feature maps at different scales
        """
        # Initial processing
        x = self.conv1(x)      # 256x256 -> 128x128
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # 128x128 -> 64x64

        # Encoder features (save for skip connections)
        x1 = self.layer1(x)    # 64x64, 64 channels
        x2 = self.layer2(x1)   # 32x32, 128 channels  
        x3 = self.layer3(x2)   # 16x16, 256 channels
        x4 = self.layer4(x3)   # 8x8, 512 channels

        return [x1, x2, x3, x4]

class DecoderBlock(nn.Module):
    """
    Single decoder block with upsampling and skip connection
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Convolution after concatenation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        """
        Args:
            x: Input from previous decoder layer
            skip: Skip connection from encoder
        """
        # Upsample to match skip connection size
        x = self.upsample(x)
        # Concatenate with skip connection
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        # Process through convolutions
        x = self.conv(x)
        return x

class ResNetUNetDecoder(CustomLightningModule):
    """
    UNet decoder with explicit size handling
    """
    def __init__(self, extra_logger):
        super().__init__(extra_logger=extra_logger)

        # Decoder blocks (explicit channel management)
        self.decoder4 = DecoderBlock(512, 256, 256)  # 8x8 -> 16x16
        self.decoder3 = DecoderBlock(256, 128, 128)  # 16x16 -> 32x32
        self.decoder2 = DecoderBlock(128, 64, 64)    # 32x32 -> 64x64
        self.decoder1 = DecoderBlock(64, 0, 64)      # 64x64 -> 128x128 (no skip)

        # Final output layer
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: List [x1, x2, x3, x4] from encoder
        Returns:
            output: Final prediction (B, 1, 256, 256)
        """
        x1, x2, x3, x4 = encoder_features

        # Decoder path with skip connections
        d4 = self.decoder4(x4, x3)  # 8x8 -> 16x16
        d3 = self.decoder3(d4, x2)  # 16x16 -> 32x32
        d2 = self.decoder2(d3, x1)  # 32x32 -> 64x64
        d1 = self.decoder1(d2, None)  # 64x64 -> 128x128 (no skip)

        # Final upsampling to original resolution
        d1 = self.final_upsample(d1)  # 128x128 -> 256x256
        output = self.final_conv(d1)  # 256x256 -> 256x256 (1 channel)

        return output

class WeatherResNetUNet(L.LightningModule):
    """
    Complete UNet model with clean architecture
    """
    def __init__(self, loss, learning_rate, norm, supervised, extra_logger):
        super().__init__()
        self.extra_logger = extra_logger
        self.loss = loss
        self.learning_rate = learning_rate
        self.supervised = supervised

        # Components
        self.encoder = ResNetUNetEncoder(norm=norm, extra_logger=extra_logger)
        self.decoder = ResNetUNetDecoder(extra_logger=extra_logger)

        # Metrics
        self.scc = SpatialCorrelationCoefficient()

        # Metrics storage
        self.test_step_outputs = []
        self.val_step_outputs = []

        # Log model info
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.extra_logger.info(f"Trainable parameters: {trainable_params:,}")

    def forward(self, x):
        """
        Forward pass through complete UNet
        Args:
            x: Input tensor (B, 1, 256, 256)
        Returns:
            output: Prediction (B, 1, 256, 256)
        """
        encoder_features = self.encoder(x)
        output = self.decoder(encoder_features)
        return output

    def training_step(self, batch, batch_idx):
        if self.supervised:
            x, y = batch
            pred = self(x)
            loss = self.loss(pred, y)
        else:
            x, _ = batch
            pred = self(x)
            loss = self.loss(pred, x)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "val_loss", self.val_step_outputs)

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "test_loss", self.test_step_outputs)

    def _shared_eval_step(self, batch, batch_idx, loss_name, outputs_list):
        """Shared logic for validation and test steps"""
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)

        # Calculate metrics
        mae = F.l1_loss(pred, y, reduction='mean')
        rmse = torch.sqrt(F.mse_loss(pred, y, reduction='mean'))
        scc = self.scc(pred, y)

        # Store results
        result = {
            loss_name: loss.detach(),
            "preds": pred.detach().cpu(),
            "targets": y.detach().cpu(),
            "mae": mae.item(),
            "rmse": rmse.item(),
            "scc": scc.item()
        }
        outputs_list.append(result)

        # Log metrics
        self.log(loss_name, loss, prog_bar=True)
        self.log("mae", mae, prog_bar=True)
        self.log("rmse", rmse, prog_bar=True)
        self.log("scc", scc, prog_bar=True)

        return result

    def on_validation_epoch_end(self):
        """Process validation results"""
        if self.val_step_outputs:
            avg_mae = sum(out["mae"] for out in self.val_step_outputs) / len(self.val_step_outputs)
            avg_rmse = sum(out["rmse"] for out in self.val_step_outputs) / len(self.val_step_outputs)
            avg_scc = sum(out["scc"] for out in self.val_step_outputs) / len(self.val_step_outputs)

            self.log("val_mae_epoch", avg_mae)
            self.log("val_rmse_epoch", avg_rmse)
            self.log("val_scc_epoch", avg_scc)

            self.val_step_outputs.clear()

    def on_test_epoch_end(self):
        """Process test results and store for external access"""
        if self.test_step_outputs:
            # Store predictions and targets
            self.test_preds = torch.cat([out["preds"] for out in self.test_step_outputs], dim=0)
            self.test_targets = torch.cat([out["targets"] for out in self.test_step_outputs], dim=0)

            # Calculate and store metrics
            self.test_mae = [out["mae"] for out in self.test_step_outputs]
            self.test_rmse = [out["rmse"] for out in self.test_step_outputs]
            self.test_scc = [out["scc"] for out in self.test_step_outputs]

            # Log summary
            avg_mae = sum(self.test_mae) / len(self.test_mae)
            avg_rmse = sum(self.test_rmse) / len(self.test_rmse)
            avg_scc = sum(self.test_scc) / len(self.test_scc)

            self.extra_logger.info(f"Test Results - MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}, SCC: {avg_scc:.4f}")

            self.test_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
