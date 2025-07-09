import torch, sys, os
import pandas as pd
import numpy as np
from torch import nn
from pathlib import Path
from torch.utils.data import random_split, Dataset, DataLoader
import lightning as L
from netsutils import WeatherDataset, Common, WeatherUtils

def test_weathernet():
    # Init data object
    wd = WeatherUtils("weather.toml")
    # SSL stuff for cartopy
    wd.config_ssl_env(Path(sys.executable).parents[1])
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = wd.Net(
        learning_rate=wd.learning_rate,
        loss=wd.criterion,
        norm=wd.norm, 
        supervised=wd.supervised,
        debug = wd.debug
    ).to(device)
    num_workers = min(os.cpu_count(), 8)  # safe default

    # Prepare data if necessary
    wd.prepare_data("test")
    # Load data for testing
    X_np_test, y_np_test, date_range = wd.load_data(model, 'test')
    
    # Normalize data
    # X_np_test, X_max, X_min = Common.rescale(X_np_test)
    # y_np_test, y_max, y_min = Common.rescale(y_np_test)
    X_np_test, X_scaler = wd.normalize(X_np_test)
    y_np_test, y_scaler = wd.normalize(y_np_test)
    
    test_dataset = WeatherDataset(X_np_test, y_np_test)
    
    # Create data loaders
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=num_workers, shuffle=False)
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    # Load weights
    print(f"Load weights from file: {wd.get_weights_fn(model)}")
    model.load_state_dict(torch.load(f"{wd.weights_folder}{wd.get_weights_fn(model)}", map_location=device))

    trainer = L.Trainer(max_epochs=wd.epochs, log_every_n_steps=1)
    output_d = trainer.test(model, dataloaders=test_dataloader)
    all_outputs = model.test_step_outputs
    print(all_outputs[-1].keys())
    # print(type(all_outputs[0]))
    # print(len(all_outputs[0]))
    # print(all_outputs[0])

    # Diffusion
    # all_outputs = torch.cat([r[0]["x0_recon"] for r in output_d], dim=0)
    # outputs = all_outputs[-1]['x0_recon']
    inputs, targets, outputs = [], [], []
    for idx, (input, target) in enumerate(test_dataloader): # input=forecast, target=analysis + x days
        inputs.append(input[0,:,:,:].cpu())
        targets.append(target[0,:,:,:].cpu())
        outputs.append(all_outputs[idx]['preds'][0,:,:,:])
    
    # Denormalize
    inputs = wd.denormalize(np.array(inputs), X_scaler)
    targets = wd.denormalize(np.array(targets), y_scaler)
    predictions = wd.denormalize(np.array(outputs), X_scaler)
    # predictions = wd.denormalize(outputs.cpu().numpy(), X_scaler)
    
    # Plot
    date_range = pd.date_range(start=wd.test_start_date, end=wd.test_end_date, freq=wd.config[wd.source]["origin_frequency"]).to_pydatetime().tolist()
    for idx, (date, i, t, o) in enumerate(zip(date_range, inputs, targets, predictions)):
        wd.plot_figures(model, date, i, t, o)
        
    # inputs, targets = test_dataloader
    
    # all_outputs = torch.cat(torch.FloatTensor(r) for r in output_d)
    # all_outputs = torch.cat([r[0]["preds"] for r in output_d], dim=0)

    

    # Plot
    
    # print(model)
    # t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device)
    # Testing
    #model.eval()  # Set model to evaluation mode
    #
    #test_loss = 0.0
    #with torch.no_grad():  # Disable gradient computation
    #    for inputs, targets in test_dataloader: # input=forecast, target=analysis + x days
    #        inputs = inputs.to(device)
    #        targets = targets.to(device)
    #        
    #        # outputs = model(inputs) # UNet
    #        outputs = model(inputs, t) # DIffusion
    #        loss = criterion(outputs, targets)
    #        test_loss += loss.item()
    #
    #test_loss /= len(test_dataloader)
    #print(f"Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    test_weathernet()
