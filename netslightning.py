import os
import torch
torch.set_float32_matmul_precision('medium')
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torchvision.models as models
from torchmetrics.image import SpatialCorrelationCoefficient
import lightning as L

class CustomLightningModule (L.LightningModule):
    def __init__ (self, debug):
        super().__init__()
        self.debug = debug

    def _squeeze_and_add_log_img (self, x, c, n, s, tl):
        if not self.trainer.sanity_checking:
            sq_x = torch.squeeze(x[n:n+1,c:c+1,:,:],0)
            if self.debug:
                print(f"[DEBUG] {s} x, squeezed, channel {c}, sample {n}: {sq_x.shape}, x.device: {sq_x.device}")
            tl.add_image(f"{s} c:{c}, n:{n}", sq_x, self.global_step)

    def _print_debug (self, x, s):
        if self.debug:
            print(f"[DEBUG] {s} x: {x.shape}, x.device: {x.device}")
    
class ResNetUNetEncoder(CustomLightningModule):
    def __init__(self, norm, debug):
        super().__init__(debug=debug)
        # self.debug = debug
        # Load base ResNet18 model
        # base_model = models.resnet18(weights=None)
        base_model = models.resnet18(norm_layer=norm)

        # Replace first conv layer for single-channel input (grayscale)
        self.encoder_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder_norm = base_model.bn1  # batch norm -> instance norm
        
        self.encoder_relu = base_model.relu
        self.encoder_maxpool = base_model.maxpool  # downsample: 128 -> 64, or 256 -> 128

        # Encoder layers from ResNet
        self.encoder1 = base_model.layer1  # 64x64 -> 64x64
        self.encoder2 = base_model.layer2  # 64x64 -> 32x32
        self.encoder3 = base_model.layer3  # 32x32 -> 16x16
        self.encoder4 = base_model.layer4  # 16x16 -> 8x8
        
    def forward(self, x):
        tensorboard_logger = self.logger.experiment
        c, n = 0, 0 # selected channel and sample in batch for logging
        # Initial conv
        self._print_debug(x, "init")
        x = self.encoder_conv1(x)
        self._print_debug(x, "conv1")
        x = self.encoder_norm(x)
        self._print_debug(x, "norm")
        x = self.encoder_relu(x)
        x = self.encoder_maxpool(x)  # 128 → 64 or 256 -> 256
        self._print_debug(x, "maxpool")
        self._squeeze_and_add_log_img(x, c, n, "maxpool", tensorboard_logger)

        # Encoder
        x1 = self.encoder1(x)   # 64x64
        self._print_debug(x1, "encoder 1")
        self._squeeze_and_add_log_img(x1, c, n, "encoder 1", tensorboard_logger)
        x2 = self.encoder2(x1)  # 32x32
        self._print_debug(x2, "encoder 2")
        self._squeeze_and_add_log_img(x2, c, n, "encoder 2", tensorboard_logger)
        x3 = self.encoder3(x2)  # 16x16
        self._print_debug(x3, "encoder 3")
        self._squeeze_and_add_log_img(x3, c, n, "encoder 3", tensorboard_logger)
        x4 = self.encoder4(x3)  # 8x8
        self._print_debug(x4, "encoder 4")
        self._squeeze_and_add_log_img(x4, c, n, "encoder 4", tensorboard_logger)

        return x1, x2, x3, x4 

class ResNetUNetDecoder(CustomLightningModule):
    def __init__(self, debug):
        super().__init__(debug=debug)
        # self.debug = debug
        # Decoder path
        self.decoder4 = self._upsample_block(512, 256)                  # 8x8 or 16x16 -> up
        self.decoder3 = self._upsample_block(256 + 256, 128)            # -> up
        self.decoder2 = self._upsample_block(128 + 128, 64)             # -> up
        self.decoder1 = self._upsample_block(64 + 64, 64)               # -> up
        self.decoder0 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)  # Output same shape as input
        )

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3, x4):
        tensorboard_logger = self.logger.experiment
        c, n = 0, 0 # selected channel and sample in batch for logging
        d4 = self.decoder4(x4)
        self._squeeze_and_add_log_img(d4, c, n, "decoder 4", tensorboard_logger)
        d4 = F.interpolate(d4, size=x3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.decoder3(torch.cat([d4, x3], dim=1))
        self._squeeze_and_add_log_img(d3, c, n, "decoder 3", tensorboard_logger)

        d3 = F.interpolate(d3, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.decoder2(torch.cat([d3, x2], dim=1))
        self._squeeze_and_add_log_img(d2, c, n, "decoder 2", tensorboard_logger)

        d2 = F.interpolate(d2, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.decoder1(torch.cat([d2, x1], dim=1))
        self._squeeze_and_add_log_img(d1, c, n, "decoder 1", tensorboard_logger)

        # Final layer to restore original resolution
        # out = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.decoder0(d1)
        self._squeeze_and_add_log_img(out, c, n, "out", tensorboard_logger)
        return out


class LitAutoEncoder(L.LightningModule):
    def __init__(self, loss, learning_rate, norm, supervised, debug):
        super().__init__()
        self.encoder = ResNetUNetEncoder(norm=norm, debug=debug)
        self.decoder = ResNetUNetDecoder(debug=debug)
        self.loss = loss
        self.norm = norm
        self.learning_rate = learning_rate
        self.supervised = supervised
        self.test_step_outputs = []
        print("Trainable parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        output = self.decoder(x1, x2, x3, x4)
        return output

    def training_step(self, batch, batch_idx):
        if self.supervised:
            # Supervised
            x, y = batch
            # x = x.view(x.size(0), -1)
            z1, z2, z3, z4 = self.encoder(x)
            x_hat = self.decoder(z1, z2, z3, z4)
            # print(f"[DEBUG] x_hat: {x_hat.shape}, x: {x.shape}, x_hat.device: {x_hat.device}, x.device: {x.device}")
            loss = self.loss(x_hat, y)
        else:
            # Unsupervised
            x, _ = batch
            # x = x.view(x.size(0), -1)
            z1, z2, z3, z4 = self.encoder(x)
            x_hat = self.decoder(z1, z2, z3, z4)
            # print(f"[DEBUG] x_hat: {x_hat.shape}, x: {x.shape}, x_hat.device: {x_hat.device}, x.device: {x.device}")
            loss = self.loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _val_step (self, batch, batch_idx, loss_name):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        scc = SpatialCorrelationCoefficient().to(x.device)
        MAE = F.l1_loss(preds, y, reduction='mean').item()
        RMSE = F.mse_loss(preds, y, reduction='mean').item()
        SCC = scc(preds, y)
        out = {
            loss_name: loss.detach(),
            "preds": preds.detach().cpu(),
            "MAE": MAE,
            "RMSE": RMSE,
            "SCC": SCC
        }
        self.test_step_outputs.append(out)
        self.log(loss_name, loss, prog_bar=True)
        # self.log("preds", preds, prog_bar=True)
        self.log("MAE", MAE, prog_bar=True)
        self.log("RMSE", RMSE, prog_bar=True)
        self.log("SCC", SCC, prog_bar=True)
        return out
        
    def validation_step(self, batch, batch_idx):
        return self._val_step(batch, batch_idx, "val_loss")

    def test_step(self, batch, batch_idx):
        return self._val_step(batch, batch_idx, "test_loss")
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_test_epoch_end(self):
        # all_preds = torch.stack(self.test_step_outputs)
        preds = [out["preds"] for out in self.test_step_outputs]
        MAE = [out["MAE"] for out in self.test_step_outputs]
        RMSE = [out["RMSE"] for out in self.test_step_outputs]
        SCC = [out["SCC"] for out in self.test_step_outputs]
        # store for external access
        self.test_preds = torch.cat(preds, dim=0)
        self.test_MAE = MAE
        self.test_RMSE = RMSE
        self.test_SCC = SCC
        print(f"MAE: {MAE}")
        print(f"RMSE: {RMSE}")
        print(f"SCC: {SCC}")
