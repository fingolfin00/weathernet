import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import MeanAbsoluteError, MeanSquaredError, Metric
from torchmetrics.image import SpatialCorrelationCoefficient
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import Optional

class AnomalyCorrelationCoefficient(Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True # Set as a class attribute directly

    def __init__(self):
        super().__init__()

        # Initialize state variables
        self.add_state("sum_anomaly_product", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_pred_anomaly_sq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_target_anomaly_sq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_elements", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Ensure preds and target are float32
        preds = preds.float()
        target = target.float()

        # Flatten the tensors for ACC calculation (assuming images/grids)
        # Handle potential batch dimension of 1 for certain cases if needed
        if preds.dim() == 4: # Assuming NCHW
            preds_flat = preds.reshape(preds.shape[0], -1)
            target_flat = target.reshape(target.shape[0], -1)
        elif preds.dim() == 2: # Assuming N, num_elements_per_sample (already flattened)
            preds_flat = preds
            target_flat = target
        else: # Handle other dimensions if necessary or raise error
             raise ValueError(f"Unsupported tensor dimension for ACC: {preds.dim()}")

        # Calculate batch-wise means
        pred_mean_batch = preds_flat.mean(dim=-1, keepdim=True)
        target_mean_batch = target_flat.mean(dim=-1, keepdim=True)

        # Calculate anomalies
        pred_anomaly = preds_flat - pred_mean_batch
        target_anomaly = target_flat - target_mean_batch

        # Accumulate sums for numerator and denominator across batch and elements
        self.sum_anomaly_product += (pred_anomaly * target_anomaly).sum()
        self.sum_pred_anomaly_sq += (pred_anomaly**2).sum()
        self.sum_target_anomaly_sq += (target_anomaly**2).sum()
        self.num_elements += preds_flat.numel() # Total elements in the batch

    def compute(self):
        numerator = self.sum_anomaly_product
        denominator = torch.sqrt(self.sum_pred_anomaly_sq * self.sum_target_anomaly_sq)

        # Handle cases where denominator might be zero (e.g., flat predictions/targets)
        # Add a small epsilon to avoid division by zero and handle cases with no variance
        epsilon = 1e-6
        if denominator.abs() < epsilon: # Check if close to zero
            return torch.tensor(0.0, device=numerator.device)
        return numerator / denominator

class CustomLightningModule(L.LightningModule):
    def __init__(self, extra_logger):
        super().__init__()
        self.extra_logger = extra_logger

        # Metrics
        self.train_mae = MeanAbsoluteError()
        self.train_rmse = MeanSquaredError(squared=False) # squared=False for RMSE
        self.train_scc = SpatialCorrelationCoefficient()
        self.train_acc = AnomalyCorrelationCoefficient()

        self.val_mae = MeanAbsoluteError()
        self.val_rmse = MeanSquaredError(squared=False) # squared=False for RMSE
        self.val_scc = SpatialCorrelationCoefficient() # Your custom SCC
        self.val_acc = AnomalyCorrelationCoefficient() # Your custom ACC

        self.test_mae = MeanAbsoluteError()
        self.test_rmse = MeanSquaredError(squared=False)
        self.test_scc = SpatialCorrelationCoefficient()
        self.test_acc = AnomalyCorrelationCoefficient()

        # Metrics storage
        self.test_step_outputs = []
        self.pic_log_interval = 1 # set to 1 to log at every epoch
        self.last_val_pred = None
        self.last_val_target = None

    def _squeeze_and_add_log_img(self, img_tensor, name_tag, logger_instance, colormap=None, vmin=None, vmax=None):
        """
        Logs a single 2D image to the logger, with optional colormap.
        img_tensor: A 2D tensor (H, W) for grayscale, or 3D tensor (C, H, W) for RGB/RGBA.
                    For colormapping, it should typically be a 2D grayscale tensor.
        name_tag: The name to appear in the logger (e.g., "Validation/Prediction_Sample").
        logger_instance: The logger object (e.g., self.logger).
        colormap: String name of a matplotlib colormap (e.g., 'viridis', 'jet', 'gray').
                  If None, the image is logged as is (assuming it's already in the correct format).
        vmin, vmax: Optional min/max values for normalization before applying colormap.
                    If None, min/max of the tensor will be used.
        """
        if self.trainer.sanity_checking:
            return # Don't log images during sanity check

        # Detach from graph and move to CPU if not already
        img_tensor = img_tensor.detach().cpu()

        if colormap:
            # Apply colormap
            if img_tensor.dim() not in [2, 3]: # Should be 2D (H,W) or (1,H,W) for colormapping
                self.extra_logger.warning(f"Colormap applied to unexpected tensor dim: {img_tensor.shape}. Expecting 2D or (1,H,W).")
                return

            # If it's (1, H, W), squeeze to (H, W) for colormapping
            if img_tensor.dim() == 3 and img_tensor.shape[0] == 1:
                img_tensor = img_tensor.squeeze(0)
            
            # Convert to numpy array
            np_img = img_tensor.numpy()

            # Normalize values before colormapping if vmin/vmax are provided, or use auto-scaling
            if vmin is None:
                vmin = np_img.min()
            if vmax is None:
                vmax = np_img.max()

            # Get the colormap
            cmap = cm.get_cmap(colormap)
            
            # Apply colormap. cmap returns RGBA (H, W, 4) in float [0, 1]
            # Convert to RGB (H, W, 3) and then to uint8 (0-255)
            # TensorBoard expects uint8 for image summaries for better visualization range
            color_mapped_np = (cmap(np_img / (vmax - vmin) - vmin / (vmax - vmin))[:, :, :3] * 255).astype(np.uint8)

            # Convert back to torch.Tensor and permute to (C, H, W)
            log_tensor = torch.from_numpy(color_mapped_np).permute(2, 0, 1) # HWC -> CHW
            
        else:
            # No colormap, assume image is already in suitable format (e.g., grayscale 1xHxW or RGB 3xHxW)
            if img_tensor.dim() == 2:
                log_tensor = img_tensor.unsqueeze(0) # Grayscale (H, W) -> (1, H, W)
            elif img_tensor.dim() == 3:
                log_tensor = img_tensor # Already (C, H, W)
            else:
                self.extra_logger.warning(f"Unexpected tensor dimension for image logging without colormap: {img_tensor.shape}. Skipping.")
                return

        logger_instance.experiment.add_image(name_tag, log_tensor, self.global_step)
        self.extra_logger.debug(f"Logged image '{name_tag}' at step {self.global_step}")


    def _log_prediction(self, pred_tensor, target_tensor, tag_prefix, logger_instance, colormap='bwr'):
        """
        Logs a sample prediction and its corresponding target.
        pred_tensor: The prediction tensor (e.g., from self.last_val_pred). Expected 4D (N, C, H, W).
        target_tensor: The target tensor (e.g., from self.last_val_target). Expected 4D (N, C, H, W).
        tag_prefix: Base string for the log name (e.g., "Val").
        logger_instance: The logger object (e.g., self.logger).
        colormap: The colormap to apply. Defaults to 'bwr'.
        """
        if self.trainer.sanity_checking:
            return

        if pred_tensor is None or target_tensor is None:
            self.extra_logger.warning("Attempted to log image, but prediction or target tensor is None.")
            return
        
        if pred_tensor.dim() != 4 or pred_tensor.shape[0] == 0:
            self.extra_logger.warning(f"Prediction tensor not 4D (N,C,H,W) or empty for logging: {pred_tensor.shape}. Skipping.")
            return
        if target_tensor.dim() != 4 or target_tensor.shape[0] == 0:
            self.extra_logger.warning(f"Target tensor not 4D (N,C,H,W) or empty for logging: {target_tensor.shape}. Skipping.")
            return

        # Choose the first sample (index 0) and the first channel (index 0)
        sample_idx = 0 
        channel_idx = 0 

        # Extract the single image slice (H, W)
        # pred_img_slice = pred_tensor[sample_idx, channel_idx, :, :]
        # target_img_slice = target_tensor[sample_idx, channel_idx, :, :]
        pred_img_slice = pred_tensor.mean(axis=(0,1))
        target_img_slice = target_tensor.mean(axis=(0,1))

        # Determine vmin/vmax for consistent color mapping across prediction and target
        # This is important for comparing them visually
        all_values = target_img_slice.flatten()-pred_img_slice.flatten()
        vmin = all_values.min().item()
        vmax = all_values.max().item()

        self._squeeze_and_add_log_img(target_img_slice-pred_img_slice, 
                                      tag_prefix, 
                                      logger_instance, 
                                      colormap=colormap, 
                                      vmin=vmin, vmax=vmax)

    def training_step(self, batch, batch_idx):
        if self.supervised:
            x, y = batch
        else:
            x, _ = batch
            y = x
        pred = self(x)
        # ensure contiguous
        pred = pred.contiguous()
        y = y.contiguous()
        loss = self.loss(pred, y)

        self.train_mae.update(pred, y)
        self.train_rmse.update(pred, y)
        self.train_scc.update(pred, y)
        self.train_acc.update(pred, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_mae", self.train_mae, on_step=False, on_epoch=True)
        self.log("train_rmse", self.train_rmse, on_step=False, on_epoch=True)
        self.log("train_scc", self.train_scc, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.supervised:
            x, y = batch
        else:
            x, _ = batch
            y = x

        pred = self(x)
        # ensure contiguous
        pred = pred.contiguous()
        y = y.contiguous()
        loss = self.loss(pred, y)

        self.val_mae.update(pred, y)
        self.val_rmse.update(pred, y)
        self.val_scc.update(pred, y)
        self.val_acc.update(pred, y)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) # Corrected: use `loss`
        self.log("val_mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_scc", self.val_scc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.last_val_pred = pred.detach().cpu()
        self.last_val_target = y.detach().cpu()

    def test_step(self, batch, batch_idx):
        if self.supervised:
            x, y = batch
        else:
            x, _ = batch
            y = x
        pred = self(x)
        # ensure contiguous
        pred = pred.contiguous()
        y = y.contiguous()
        loss = self.loss(pred, y) # Corrected: use `loss`

        self.test_mae.update(pred, y)
        self.test_rmse.update(pred, y)
        self.test_scc.update(pred, y)
        self.test_acc.update(pred, y)

        # Log per-batch loss. Metrics will be logged at epoch end using their aggregated state.
        self.log("test_loss", loss, on_step=True, on_epoch=True, logger=True) # Corrected: use `loss`
        self.log("test_mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_rmse", self.test_rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_scc", self.test_scc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.test_step_outputs.append({"preds": pred.detach().cpu(), "targets": y.detach().cpu()})

    def on_train_epoch_start(self):
        # Access the optimizer's learning rate
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]['lr']
        self.log('lr', current_lr, on_step=False, on_epoch=True)
        self.extra_logger.info(f"Epoch {self.current_epoch}: learning Rate = {current_lr}")

    def on_validation_epoch_end(self):
        """Process validation results, including logging a sample image"""
        if self.current_epoch % self.pic_log_interval == 0:
            self._log_prediction(self.last_val_pred, self.last_val_target, "Prediction Error", self.logger)
        # Clean up the stored tensors to prevent accidental reuse or memory issues
        self.last_val_pred = None
        self.last_val_target = None

    def on_test_epoch_end(self):
        """Process test results and store for external access"""
        # If you need to access the final aggregated metric values from the Test stage
        # *after* trainer.test() has completed, you can get them from the logger or from trainer.callback_metrics.
        final_test_mae = self.test_mae.compute()
        final_test_rmse = self.test_rmse.compute()
        final_test_scc = self.test_scc.compute()
        final_test_acc = self.test_acc.compute()
        
        self.extra_logger.info(f"Test Results - MAE: {final_test_mae:.4f}, RMSE: {final_test_rmse:.4f}, SCC: {final_test_scc:.4f}, ACC: {final_test_acc:.4f}")

        # Store all test_preds/targets for post-test analysis
        if self.test_step_outputs: # Check if there were any batches
            self.test_preds = torch.cat([out["preds"] for out in self.test_step_outputs], dim=0)
            self.test_targets = torch.cat([out["targets"] for out in self.test_step_outputs], dim=0)
        self.test_step_outputs.clear()
