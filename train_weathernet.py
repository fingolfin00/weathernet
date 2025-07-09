import sys, os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch import nn
from pathlib import Path
from torch.utils.data import random_split, Dataset, DataLoader
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from netsutils import WeatherDataset, Common, WeatherUtils

def train_weathernet():
    # Init data object
    wd = WeatherUtils("weather.toml")
    wd.setup_logger()
    wd.log_global_setup()
    wd.log_train_setup()
    # SSL stuff for cartopy (if ever needed)
    wd.config_ssl_env(Path(sys.executable).parents[1])
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wd.logger.info(wd.logger)
    model = wd.Net(
        learning_rate=wd.learning_rate,
        loss=wd.criterion,
        norm=wd.norm, 
        supervised=wd.supervised,
        extra_logger=wd.logger
    ).to(device)
    num_workers = min(os.cpu_count(), 8)  # safe default
    # Tensorboard
    tl_logger = TensorBoardLogger(wd.run_path+"lightning_logs", name=wd.run_name)

    # Prepare data if necessary
    wd.prepare_data("train")
    # Load data for training
    X_np, y_np, _ = wd.load_data(model, 'train')
    
    # Normalize data (example: min-max scaling to [0, 1])
    # X_np, _, _ = Common.rescale(X_np)
    # y_np, _, _ = Common.rescale(y_np)
    X_np, _ = wd.normalize(X_np)
    y_np, _ = wd.normalize(y_np)
    
    dataset = WeatherDataset(X_np, y_np)
    
    # Split into train (90%) and test (10%)
    num_samples = X_np.shape[0]
    test_size = int(0.1 * num_samples)
    train_size = num_samples - test_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, test_size])
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=wd.batch_size, num_workers=num_workers)
    validation_dataloader = DataLoader(validation_dataset, batch_size=wd.batch_size, shuffle=False, num_workers=num_workers)
    # for X, y in train_dataloader:
    #     print(f"Shape of input  X (forecast) training set [N, C, H, W]: {X.shape}")
    #     print(f"Shape of target y (analysis) training set [N, C, H, W]: {y.shape} {y.dtype}")
    #     break
    # for X, y in validation_dataloader:
    #     print(f"Shape of input  X (forecast) validation set [N, C, H, W]: {X.shape}")
    #     print(f"Shape of target y (analysis) validation set [N, C, H, W]: {y.shape} {y.dtype}")
    #     break


    # DEBUG if crazy errors show up
    # x, y = next(iter(train_dataloader))
    # x, y = x.cuda(), y.cuda()  # if using CUDA
    
    # out = model(x)
    # loss = model.loss(out, y)
    # try:
    #     loss.backward()
    # except Exception as e:
    #     print("Backward error:", e)
    #     print("Loss value:", loss)
    #     print("out stats:", out.min(), out.max(), out.mean())
    #     print("target stats:", y.min(), y.max(), y.mean())
    #     raise

    # Train
    trainer = L.Trainer(max_epochs=wd.epochs, log_every_n_steps=1, logger=tl_logger)
    trainer.fit(model, train_dataloader, validation_dataloader)
    
    # Save model weights
    wd.save_weights(model)
    
    # for epoch in range(num_epochs):
    #     for batch_forecasts, batch_analyses in train_dataloader:
    #         # Move data to device
    #         batch_forecasts = batch_forecasts.to(device)
    #         batch_analyses = batch_analyses.to(device)
            
    #         # Forward pass
    #         outputs = model(batch_forecasts)
    #         loss = criterion(outputs, batch_analyses)
            
    #         # Backward pass
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
        
    #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    #     # Validation
    #     with torch.no_grad():
    #         total_mse = 0.0
    #         total_mae = 0.0
    #         total_pixels = 0
        
    #         for batch_forecasts, batch_analyses in test_dataloader:
    #             batch_forecasts = batch_forecasts.to(device)
    #             batch_analyses = batch_analyses.to(device)
        
    #             outputs = model(batch_forecasts)
        
    #             mse = F.mse_loss(outputs, batch_analyses, reduction='sum').item()
    #             mae = F.l1_loss(outputs, batch_analyses, reduction='sum').item()
        
    #             total_mse += mse
    #             total_mae += mae
    #             total_pixels += batch_analyses.numel()
        
    #         print(f"Validation MSE: {total_mse / total_pixels:.6f}")
    #         print(f"Validation MAE: {total_mae / total_pixels:.6f}")
    

if __name__ == "__main__":
    train_weathernet()
