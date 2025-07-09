import glob, os, sys, torch, datetime, cfgrib, time, toml, importlib
from torch import nn
from torch.utils.data import random_split, Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from scipy.signal import detrend
import xarray as xr
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import urllib.request, ssl, certifi
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
# from nets import WeatherCNN, MiniResNet18, ResNetUNet, UNetConvLSTM, SpatioTemporalForecastDataset
from netslightning import LitAutoEncoder
from diffusion import DiffusionUNet, DiffusionNoiseScheduler, LitDiffusion

class Common:
    @staticmethod
    def normalize(data):
        data_min = data.min()
        data_max = data.max()
        return (data - data_min) / (data_max - data_min), data_max, data_min

    @staticmethod
    def denormalize(norm_data, data_max, data_min):
        return data_min + norm_data * (data_max - data_min)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def trend(data):
        # Reshape to (T, nxn) to apply regression across all pixels
        reshaped = data.reshape(data.shape[0], -1)  # Shape: (T, n*n)
        
        # Preallocate arrays for slope and intercept
        slopes = np.zeros(reshaped.shape[1])
        intercepts = np.zeros(reshaped.shape[1])

        # Compute slope and intercept per pixel
        t = np.arange(data.shape[0])
        for i in range(reshaped.shape[1]):
            slope, intercept, *_ = linregress(t, reshaped[:, i])
            slopes[i] = slope
            intercepts[i] = intercept

        # Reconstruct trend line for each pixel over time
        # trend = np.outer(t, slopes) + intercepts  # shape: (T, 16384)
        return slopes.reshape(data.shape[-2], data.shape[-1]), intercepts.reshape(data.shape[-2], data.shape[-1])

        # # Compute slope (trend) per pixel
        # slopes = np.apply_along_axis(lambda ts: linregress(np.arange(data.shape[0]), ts).slope, axis=0, arr=reshaped)
        # # Reshape back to (n, n) grid
        # trend_map = slopes.reshape(data.shape[-2], data.shape[-1])
        # return trend_map

class WeatherUtils:
    def __init__ (self, tomlfn):
        self.tomlfn = tomlfn
        # User configuration
        self.config = toml.load(tomlfn)
        # Global
        self.run_name                = self.config["global"]["run_name"]
        self.work_root_path          = self.config["global"]["work_root_path"]
        self.run_path                = self.config["global"]["run_root_path"] + self.run_name + "/"
        self.netname                 = self.config["global"]["net"]
        self.Net                     = globals()[self.netname]
        self.save_data_folder        = self.work_root_path + self.config["global"]["save_data_path"]
        self.extra_data_folder       = self.run_path + self.config["global"]["extra_data_path"]
        self.fig_folder              = self.run_path + self.config["global"]["figure_path"]
        self.weights_folder          = self.work_root_path + self.config["global"]["weights_path"]
        self.folder_freq             = "1d"
        self.print_date_strformat    = self.config["global"]["print_date_strformat"]
        self.folder_date_strformat   = "%Y%m%d"
        self.forecast_date_strformat = "%m%d%H%M"
        self.new_date_strformat      = "%Y%m%d%H%M"
        self.plot_date_strformat     = "%Y%m%d-%H%M"
        self.data_extension          = ".npy.npz"
        self.debug                   = self.config["global"]["debug"]
        # Data
        self.var_forecast            = self.config["data"]["var_forecast"]
        self.var_analysis            = self.config["data"]["var_analysis"]
        self.levhpa                  = self.config["data"]["levhpa"]
        self.lonini, self.lonfin     = self.config["data"]["lonini"], self.config["data"]["lonfin"]
        self.latini, self.latfin     = self.config["data"]["latini"], self.config["data"]["latfin"]
        self.anomaly                 = self.config["data"]["anomaly"]
        self.detrend                 = self.config["data"]["detrend"]
        self.size_an                 = self.config["data"]["domain_size"] # set a square domain starting from lonini and latini, lonfin and latfin are ignored
        self.source                  = self.config["data"]["source"]
        self.forecast_delta          = self.config["data"]["forecast_delta"]
        # self.acq_freq              = self.config["data"]["acquisition_frequency"]
        # Train
        self.start_date              = datetime.datetime.fromisoformat(self.config["train"]["train_start_date"])
        self.end_date                = datetime.datetime.fromisoformat(self.config["train"]["train_end_date"])
        # Test
        self.test_start_date         = datetime.datetime.fromisoformat(self.config["test"]["test_start_date"])
        self.test_end_date           = datetime.datetime.fromisoformat(self.config["test"]["test_end_date"])
        self.cmap                    = self.config["test"]["cmap"]
        self.cmap_anomaly            = self.config["test"]["cmap_anomaly"]
        self.cmap_error              = self.config["test"]["cmap_error"]
        # Make folders if necessary
        os.makedirs(self.save_data_folder, exist_ok=True)
        os.makedirs(self.extra_data_folder, exist_ok=True)
        os.makedirs(self.fig_folder, exist_ok=True)
        os.makedirs(self.weights_folder, exist_ok=True)
        # Suffices for file names
        self.suffix = "_anomaly" if self.anomaly else ""
        self.suffix += "_detrend" if self.detrend else ""
        # Hyperparameters
        self.learning_rate           = self.config["hyper"]["learning_rate"]
        self.batch_size              = self.config["hyper"]["batch_size"]
        self.epochs                  = self.config["hyper"]["epochs"]
        self.loss                    = self.config["hyper"]["loss"]
        self.norm_strategy           = self.config["hyper"]["norm_strategy"]
        self.supervised              = self.config["hyper"]["supervised"]
        self.supervised_str = "supervised" if self.supervised else "unsupervised"
        criterion = getattr(nn, self.loss)
        self.criterion = criterion()
        self.norm = getattr(nn, self.norm_strategy)

        
        # self.norm = norm()

    def _get_base_data_fn (self, start_date, end_date):
        return f"{self.save_data_folder}{self.source}_{self.var_forecast}-{self.var_analysis}_{self.forecast_delta}_{start_date.strftime(self.folder_date_strformat)}_{end_date.strftime(self.folder_date_strformat)}"
        
    def _get_train_data_fn (self):
        return self._get_base_data_fn(self.start_date, self.end_date)
    
    def _get_test_data_fn (self):
        return self._get_base_data_fn(self.test_start_date, self.test_end_date)

    def _get_data_fn_from_type (self, type):
        if type == 'train':
            data_fn = self._get_train_data_fn() + self.data_extension
        elif type == 'test':
            data_fn = self._get_test_data_fn() + self.data_extension
        else:
            print(f"ERROR: unsupported type {type}")
            data_fn = None
        return data_fn

    def config_ssl_env (self, env_base_path):
        os.environ["SSL_CERT_FILE"] = str(env_base_path / 'ssl' / 'tls-ca-bundle.pem')
        os.environ["SSL_CERT_DIR"] = str(env_base_path / 'ssl')
        os.environ["REQUESTS_CA_BUNDLE"] = str(env_base_path / 'ssl' / 'tls-ca-bundle.pem')

    def prepare_data (self, type):
        print(f"Prepare {type} data")
        data_fn = self._get_data_fn_from_type(type)
        if type == 'train':
            start_date = self.start_date
            end_date = self.end_date
            coords_d = self.start_date
        elif type == 'test':
            start_date = self.test_start_date
            end_date = self.test_end_date
            coords_d = self.test_start_date
        else:
            print(f"ERROR: unsupported type {type}")
            coords_d = None
        # Prepare variables
        if self.source == "ecmwf":
            en_datatype = self.config["ecmwf"]["ensemble_dataype"]
            head_forecast_fn = "forecast-"
            head_analysis_fn = "analysis-"
            forecast_root_path = self.config["data"]["download_path"]
            analysis_root_path = forecast_root_path
            orig_freq = self.config["ecmwf"]["origin_frequency"]
            grib_dict = {'dataType': en_datatype, 'cfVarName': self.var_forecast}
        elif self.source == "cmcc":
            head_forecast_fn = "JLS"
            head_analysis_fn = "JLD"
            forecast_root_path = f"{self.config["cmcc"]["forecast_path"]}{self.config["cmcc"]["prod_freq_forecast"]}/{self.config["cmcc"]["file_format_forecast"]}/"
            # analysis_root_path = f"/data/inputs/METOCEAN/rolling/model/atmos/ECMWF/IFS_010/3.1analysis/{folder_acq_freq_analysis}/netcdf/"
            analysis_root_path = f"{self.config["cmcc"]["analysis_path"]}{self.config["cmcc"]["prod_freq_analysis"]}/{self.config["cmcc"]["file_format_analysis"]}/"
            orig_freq = self.config["cmcc"]["origin_frequency"]
            grib_dict = {'cfVarName': self.var_forecast}
        else:
            print("ERROR: Unsupported source type. Aborting...")
            quit()

        # Get the data
        date_range = pd.date_range(start=start_date, end=end_date, freq=orig_freq).to_pydatetime().tolist()
        print(f"Get {self.var_forecast} (forecast), {self.var_analysis} (analysis) from {start_date.strftime(self.print_date_strformat)} to {end_date.strftime(self.print_date_strformat)}")
        var_d = {}
        for d in date_range:
            forecast_d = d - datetime.timedelta(days=self.forecast_delta)
            print(f"Analysis: {d}")
            print(f"Forecast: {forecast_d}")
            # print(d)
            # print(forecast_d)
            # acq_range = pd.date_range(start=d, end=d+datetime.timedelta(days=1), freq=acq_freq).to_pydatetime().tolist()
            if self.source == "ecmwf":
                analysis_path = analysis_root_path
                forecast_path = forecast_root_path
                analysis_fn_glob = f"{head_analysis_fn}{self.var_analysis}-{d.strftime(self.new_date_strformat)}-{d.strftime(self.new_date_strformat)}*"
                forecast_fn_glob = f"{head_forecast_fn}{self.var_forecast}-{forecast_d.strftime(self.new_date_strformat)}-{d.strftime(self.new_date_strformat)}*"
            elif self.source == "cmcc":
                # analysis_path = f"{analysis_root_path}{d.strftime(folder_date_strformat)}/"
                analysis_path = f"{analysis_root_path}/{d.strftime("%Y")}/{d.strftime("%m")}/"
                forecast_path = f"{forecast_root_path}{forecast_d.strftime(self.folder_date_strformat)}/"
                analysis_fn_glob = f"{head_analysis_fn}{d.strftime(self.forecast_date_strformat)}{d.strftime(self.forecast_date_strformat)}*"
                # analysis_fn_glob = f"{d.strftime(folder_date_strformat)}-ECMWF---AM0100-MEDATL-b{d.strftime(folder_date_strformat)}_an{h.strftime("%M")}-fv11.00.nc"
                forecast_fn_glob = f"{head_forecast_fn}{forecast_d.strftime(self.forecast_date_strformat)}{d.strftime(self.forecast_date_strformat)}*"
            
            # print(forecast_fn_glob)
            # print(analysis_fn_glob)
            errormsg = None
            nearest_forecast_flag = False
            try:
                forecast_fn = glob.glob(forecast_path + forecast_fn_glob)[0]
            except:
                errormsg = f"Couldn't find forecast file {forecast_path + forecast_fn_glob}"
                if source == "cmcc":
                    # If exact +deltaforecast is not present attempt to estimate with nearest +-1h forecasts
                    minus1h_d = d - datetime.timedelta(hours=1)
                    plus1h_d = d + datetime.timedelta(hours=1)
                    forecast_minus1h_fn_glob = f"{head_forecast_fn}{forecast_d.strftime(self.forecast_date_strformat)}{minus1h_d.strftime(self.forecast_date_strformat)}*"
                    forecast_plus1h_fn_glob = f"{head_forecast_fn}{forecast_d.strftime(self.forecast_date_strformat)}{plus1h_d.strftime(self.forecast_date_strformat)}*"
                    print("Try nearest +-1h forecasts...")
                    try:
                        forecast_minus1h_fn = glob.glob(forecast_path + forecast_minus1h_fn_glob)[0]
                        forecast_plus1h_fn = glob.glob(forecast_path + forecast_plus1h_fn_glob)[0]
                        nearest_forecast_flag = True
                    except:
                        print(f"ERROR: {errormsg}")
                        errormsg = f"Couldn't find forecast -1h file {forecast_path + forecast_minus1h_fn_glob}"
                        print(f"ERROR: {errormsg}")
                        errormsg = f"Couldn't find forecast +1h file {forecast_path + forecast_plus1h_fn_glob}"
                        print(f"ERROR: {errormsg}. Skipping...")
                        continue
                # print(f"ERROR: {errormsg}. Skipping...")
                continue
            try:
                analysis_fn = glob.glob(analysis_path + analysis_fn_glob)[0]
            except:
                errormsg = f"Couldn't find analysis file {analysis_path + analysis_fn_glob}"
                print(f"ERROR: {errormsg}. Skipping...")
                continue
            try:
                print(analysis_fn)
                ds_analysis = xr.open_dataset(
                    analysis_fn, engine="cfgrib", indexpath="", decode_timedelta=True,
                    backend_kwargs={'filter_by_keys': grib_dict}
                )
                # ds_analysis = nc.Dataset(analysis_fn)
            except:
                errormsg = f"Couldn't open analysis file {analysis_fn}"
                print(f"ERROR: {errormsg}. Skipping...")
                continue
            try:
                print(forecast_fn)
                if nearest_forecast_flag:
                    ds_forecast_minus1h = xr.open_dataset(
                        forecast_minus1h_fn, engine="cfgrib", indexpath="", decode_timedelta=True,
                        backend_kwargs={'filter_by_keys': grib_dict}
                    )
                    ds_forecast_plus1h = xr.open_dataset(
                        forecast_plus1h_fn, engine="cfgrib", indexpath="", decode_timedelta=True,
                        backend_kwargs={'filter_by_keys': grib_dict}
                    )
                    fc_values = np.mean( np.array([ ds_forecast_plus1h.variables[var_forecast].values, ds_forecast_minus1h.variables[var_forecast].values ]), axis=0 )
                else:
                    ds_forecast = xr.open_dataset(
                        forecast_fn, engine="cfgrib", indexpath="", decode_timedelta=True,
                        backend_kwargs={'filter_by_keys': grib_dict}
                    )
                    fc_values = ds_forecast.variables[self.var_forecast].values
                # ds_forecast = xr.open_dataset(forecast_fn, engine="cfgrib", indexpath="")
                # ds_forecast = cfgrib.open_datasets(forecast_fn, indexpath="/tmp/tempgrib.{short_hash}.idx")
            except:
                errormsg = f"Couldn't open forecast file {forecast_fn}"
                print(f"ERROR: {errormsg}. Skipping...")
                continue
            try:
                var_d[d] = {
                    'forecast': fc_values,
                    # 'analysis': ds_analysis.variables[self.var_analysis][:]
                    'analysis': ds_analysis.variables[self.var_analysis].values
                }
            except:
                print(f"ERROR: Data not available. Skipping...")
        
        # Get coordinate data        
        if self.source == "ecmwf":
            analysis_fn_coords_glob = f"analysis-{self.var_analysis}-{coords_d.strftime(self.new_date_strformat)}-{coords_d.strftime(self.new_date_strformat)}*"
            forecast_fn_coords_glob = f"forecast-{self.var_forecast}-{(coords_d-datetime.timedelta(days=self.forecast_delta)).strftime(self.new_date_strformat)}-{coords_d.strftime(self.new_date_strformat)}*"
            forecast_coords_path = forecast_path
            analysis_coords_path = analysis_path
        elif self.source == "cmcc":
            forecast_fn_coords_glob = f"{head_forecast_fn}{coords_d.strftime(self.forecast_date_strformat)}*"
            analysis_fn_coords_glob = f"{head_analysis_fn}{coords_d.strftime(self.forecast_date_strformat)}*"
            # analysis_fn_coords_glob = f"{coords_d.strftime(folder_date_strformat)}-ECMWF---AM0100-MEDATL-b{coords_d.strftime(folder_date_strformat)}_an00-fv11.00.nc"
            forecast_coords_path = f"{forecast_root_path}{coords_d.strftime(self.folder_date_strformat)}/"
            analysis_coords_path = f"{analysis_root_path}/{coords_d.strftime("%Y")}/{coords_d.strftime("%m")}/"
        forecast_coords_fn = glob.glob(forecast_coords_path + forecast_fn_coords_glob)[0]
        analysis_coords_fn = glob.glob(analysis_coords_path + analysis_fn_coords_glob)[0]
        
        if self.source == "ecmwf":
            ds_forecast_coords = xr.open_dataset(
                forecast_coords_fn, engine="cfgrib", indexpath="", decode_timedelta=True,
                backend_kwargs={'filter_by_keys': {'dataType': en_datatype, 'cfVarName': self.var_forecast}}
            )
            ds_analysis_coords = xr.open_dataset(
                analysis_coords_fn, engine="cfgrib", indexpath="", decode_timedelta=True,
                backend_kwargs={'filter_by_keys': {'dataType': en_datatype, 'cfVarName': self.var_analysis}}
            )
        elif self.source == "cmcc":
            ds_forecast_coords = xr.open_dataset(
                forecast_coords_fn, engine="cfgrib", indexpath="", decode_timedelta=True,
                backend_kwargs={'filter_by_keys': {'cfVarName': self.var_forecast}}
            )
            ds_analysis_coords = xr.open_dataset(
                analysis_coords_fn, engine="cfgrib", indexpath="", decode_timedelta=True,
                backend_kwargs={'filter_by_keys': {'cfVarName': self.var_analysis}}
            )
        
        lon_fc_full = ds_forecast_coords['longitude'].values
        lat_fc_full = ds_forecast_coords['latitude'].values
        lev_fc_full = ds_forecast_coords['isobaricInhPa'].values
        lon_an_full = ds_analysis_coords['longitude'].values
        lat_an_full = ds_analysis_coords['latitude'].values
        lev_an_full = ds_analysis_coords['isobaricInhPa'].values
        
        var_d['lon'] = {'analysis': lon_an_full}
        var_d['lat'] = {'analysis': lat_an_full}
        var_d['lev'] = {'analysis': lev_an_full}
        var_d['lon']['forecast'] = lon_fc_full
        var_d['lat']['forecast'] = lat_fc_full
        var_d['lev']['forecast'] = lev_fc_full
        
        # Save data
        print(f"Saving data in {data_fn}")
        with open(f"{data_fn}", 'wb') as f:
            np.savez(f"{data_fn}", var_d, allow_pickle=True)

    def get_averages_from_fn (self, average_data_fn):
        with open(f"{average_data_fn}", 'rb') as f:
            average_d = np.load(f, allow_pickle=True)['arr_0'].item()
        return average_d

    def load_data (self, model, type):
        data_fn = self._get_data_fn_from_type(type)
        average_data_fn = self._get_average_fn(model)
        trend_data_fn = self._get_trend_fn(model)
        print(f"Loading {type} data from {data_fn}")
        with open(f"{data_fn}", 'rb') as f:
            var_d = np.load(f, allow_pickle=True)['arr_0'].item()
        
        lon_an_full = var_d['lon']['analysis']
        lat_an_full = var_d['lat']['analysis']
        lev_an_full = var_d['lev']['analysis']
        lon_fc_full = var_d['lon']['forecast']
        lat_fc_full = var_d['lat']['forecast']
        lev_fc_full = var_d['lev']['forecast']
        
        print(f"lon_an[0], lon_an[-1]: {lon_an_full[0]}, {lon_an_full[-1]}")
        print(f"lon_fc[0], lon_fc[-1]: {lon_fc_full[0]}, {lon_fc_full[-1]}")
        print(f"lat_an[0], lat_an[-1]: {lat_an_full[0]}, {lat_an_full[-1]}")
        print(f"lat_fc[0], lat_fc[-1]: {lat_fc_full[0]}, {lat_fc_full[-1]}")
        lonfin = self.lonfin
        latfin = self.latfin
        lonini_fc_idx = (np.abs(lon_fc_full - self.lonini)).argmin()
        lonfin_fc_idx = (np.abs(lon_fc_full - lonfin)).argmin()
        lonini_an_idx = (np.abs(lon_an_full - self.lonini)).argmin()
        lonfin_an_idx = (np.abs(lon_an_full - lonfin)).argmin()
        latini_fc_idx = (np.abs(lat_fc_full - self.latini)).argmin()
        latfin_fc_idx = (np.abs(lat_fc_full - latfin)).argmin()
        latini_an_idx = (np.abs(lat_an_full - self.latini)).argmin()
        latfin_an_idx = (np.abs(lat_an_full - latfin)).argmin()
        lev_analysis = (np.abs(lev_an_full - self.levhpa)).argmin()
        lev_forecast = (np.abs(lev_fc_full - self.levhpa)).argmin()
        
        if self.size_an:
            print(f"Selected regular square size for analysis: {self.size_an}x{self.size_an}")
            print(f"Ignoring latfin: {latfin} and lonfin: {lonfin}")
            # print(f"Lonfin before adding size_an: {lonfin_an_idx}")
            # print(f"Latfin before adding size_an: {latfin_an_idx}")
            lonfin_an_idx = lonini_an_idx + self.size_an
            latfin_an_idx = latini_an_idx + self.size_an
            # print(f"Lonfin after adding size_an: {lonfin_an_idx}")
            # print(f"Latfin after adding size_an: {latfin_an_idx}")
            lonfin = lon_an_full[lonfin_an_idx]
            latfin = lat_an_full[latfin_an_idx]
            lonfin_fc_idx = (np.abs(lon_fc_full - lonfin)).argmin()
            latfin_fc_idx = (np.abs(lat_fc_full - latfin)).argmin()
            
        self.lonfc = lon_fc_full[lonini_fc_idx:lonfin_fc_idx]
        self.latfc = lat_fc_full[latini_fc_idx:latfin_fc_idx]
        self.lonan = lon_an_full[lonini_an_idx:lonfin_an_idx]
        self.latan = lat_an_full[latini_an_idx:latfin_an_idx]
        
        print(f"lat ini {self.latini}, lat fin {latfin} (forecast ds) = {latini_fc_idx}, {latfin_fc_idx}")
        print(f"lat ini {self.latini}, lat fin {latfin} (analysis ds) = {latini_an_idx}, {latfin_an_idx}")
        print(f"lon ini {self.lonini}, lon fin {lonfin} (forecast ds) = {lonini_fc_idx}, {lonfin_fc_idx}")
        print(f"lon ini {self.lonini}, lon fin {lonfin} (analysis ds) = {lonini_an_idx}, {lonfin_an_idx}")
        print(f"lev {self.levhpa} (analysis ds) = {lev_analysis}")
        print(f"lev {self.levhpa} (forecast ds) = {lev_forecast}")
        print(f"Full {self.var_forecast} shape: {var_d[next(iter(var_d))]['forecast'].shape}")
        print(f"Full {self.var_analysis} shape: {var_d[next(iter(var_d))]['analysis'].shape}")

        # Lists to hold all samples
        forecast_data = []
        analysis_data = []
        
        # Extract data from dictionary
        for key in var_d.keys():  # Loop through all timesteps
            if key not in ['lon', 'lat', 'lev']:
                # Get 2D fields
                forecast = var_d[key]["forecast"]  # Shape: (height, width)
                analysis = var_d[key]["analysis"]  # Shape: (height, width)
            
                if len(analysis.shape) == 3:
                    analysis = analysis[lev_analysis,latini_an_idx:latfin_an_idx,lonini_an_idx:lonfin_an_idx]
                elif len(analysis.shape) == 2:
                    analysis = analysis[latini_an_idx:latfin_an_idx,lonini_an_idx:lonfin_an_idx]
                else:
                    print(f"ERROR: unsupported dimensions {analysis.shape} for {self.var_analysis}")
            
                if len(forecast.shape) == 3:
                    forecast = forecast[lev_forecast,latini_fc_idx:latfin_fc_idx,lonini_fc_idx:lonfin_fc_idx]
                elif len(forecast.shape) == 2:
                    forecast = forecast[latini_fc_idx:latfin_fc_idx,lonini_fc_idx:lonfin_fc_idx]
                else:
                    print(f"ERROR: unsupported dimensions {forecast.shape} for {self.var_forecast}")
            
                # if self.anomaly:
                #     forecast = np.mean(forecast) - forecast
                #     analysis = np.mean(analysis) - analysis
                # Add channel dimension (PyTorch expects [channels, height, width])
                forecast = np.expand_dims(forecast, axis=0)  # New shape: (1, height, width)
                analysis = np.expand_dims(analysis, axis=0)  # New shape: (1, height, width)
                
                forecast_data.append(forecast)
                analysis_data.append(analysis)
        
        # Convert to numpy arrays (shape: [num_samples, 1, height, width])
        X_np = np.stack(forecast_data, axis=0)  # Stack along sample dimension
        y_np = np.stack(analysis_data, axis=0)
        
        average_data_path = f"{self.extra_data_folder}{average_data_fn}"
        if type == 'train':
            print(f"Save training period average in {average_data_fn}")
            average_fc = X_np.mean(axis=0, keepdims=True)
            average_an = y_np.mean(axis=0, keepdims=True)
            # print(f"average_fc shape: {np.shape(average_fc)}")
            average_d = {"forecast": average_fc, "analysis": average_an}
            with open(average_data_path, 'wb') as f:
                np.savez(average_data_path, average_d, allow_pickle=True)
        elif type == 'test':
            # Load anomaly data
            print(f"Load training period average from {average_data_fn}")
            average_d = self.get_averages_from_fn(average_data_path)
            average_fc = average_d["forecast"]
            average_an = average_d["analysis"]
        if self.anomaly:
            X_np -= average_fc
            y_np -= average_an
        
        if self.detrend:
            # Load trend data
            print(f"Loading trend from {trend_data_fn}")
            with open(f"{trend_data_fn}", 'rb') as f:
                trend_d = np.load(f, allow_pickle=True)['arr_0'].item()
                slopes_fc, intercepts_fc = trend_d['slopes']['forecast'], trend_d['intercepts']['forecast']
                slopes_an, intercepts_an = trend_d['slopes']['analysis'], trend_d['intercepts']['analysis']
                dim_an, dim_fc = trend_d['dims']['analysis'], trend_d['dims']['forecast']
            trend_fc = dim_fc*slopes_fc.flatten() + intercepts_fc.flatten()
            trend_an = dim_an*slopes_an.flatten() + intercepts_an.flatten()
            X_np -= np.expand_dims(trend_fc.reshape(X_np.shape[0], X_np.shape[-2], X_np.shape[-1]), axis=1)
            y_np -= np.expand_dims(trend_an.reshape(y_np.shape[0], y_np.shape[-2], y_np.shape[-1]), axis=1)

        return X_np, y_np

    def _get_final_products_base_fn (self, model):
        return f"{model.__class__.__name__}_{self.supervised_str}_{self.var_forecast}-{self.var_analysis}-{self.levhpa}hPa_{self.lonini:+2.1f}-{self.latini:+2.1f}_{self.size_an}x{self.size_an}_{self.batch_size}bs-{self.learning_rate}lr-{self.epochs}epochs-{self.loss}_{self.norm_strategy}_{self.start_date.strftime(self.plot_date_strformat)}_{self.end_date.strftime(self.plot_date_strformat)}{self.suffix}"

    def _get_average_fn (self, model):
        return self._get_final_products_base_fn(model) + "_average" + self.data_extension

    def _get_trend_fn (self, model):
        return self._get_final_products_base_fn(model) + "_trend" + self.data_extension

    def get_weights_fn (self, model):
        return f"{self._get_final_products_base_fn(model)}.pth"

    def _get_pics_fn (self, model, date):
        return f"{self._get_final_products_base_fn(model)}_{date.strftime(self.plot_date_strformat)}.png"
        
    def save_weights (self, model):
        weights_fn = self.get_weights_fn(model)
        print(f"Save weights in file: {weights_fn}")
        torch.save(model.state_dict(), f"{self.weights_folder}{weights_fn}")

    def _create_cartopy_axis (
        self, fig, rows, cols, n, title, var, vmin_plt, vmax_plt, vcenter_plt, cmap,
        borders=False
    ):
        ax = fig.add_subplot(
            rows, cols, n,
            projection=ccrs.PlateCarree()
        )
        ax.set_title(title)
        im = ax.pcolormesh(self.lonan, self.latan,
                           var,
                           norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), cmap=cmap,
                           transform=ccrs.PlateCarree()
                          )
        ax.coastlines()
        if borders:
            ax.add_feature(cfeature.BORDERS)
        # divider1 = make_axes_locatable(ax1)
        # cax1 = divider1.append_axes('right', size="2%", pad=0.05)
        cb = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.047, pad=0.02)
        ax.set_aspect('equal', adjustable='box')

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color='gray',
            alpha=0.5,
            linestyle='--'
        )
        
        # Turn off labels on top and right side
        gl.top_labels = False
        gl.right_labels = False
        # Control which tick formatters are used
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        # Set spacing manually (e.g., every 10°)
        gl.xlocator = plt.MaxNLocator(5)  # or use FixedLocator
        gl.ylocator = plt.MaxNLocator(5)

        return ax

    def plot_figures (self, model, date, inputs, targets, outputs, X_max, X_min, y_max, y_min):
        average_data_fn = self._get_average_fn(model)
        average_data_path = f"{self.extra_data_folder}{average_data_fn}"
        average_d = self.get_averages_from_fn(average_data_path)
        # Remove channel dimension
        average_fc = np.squeeze(average_d["forecast"], axis=1)
        average_an = np.squeeze(average_d["analysis"], axis=1)
        # Remove channel dimension and denormalize, shape becomes (H, W)
        sample_forecast = Common.denormalize(inputs.cpu().squeeze().numpy(), X_max, X_min)
        sample_analysis = Common.denormalize(targets.cpu().squeeze().numpy(), y_max, y_min)
        prediction = Common.denormalize(outputs.cpu().squeeze().numpy(), y_max, y_min)
        
        # Plot forecast, analysis, and prediction error
        print(f"Plot one forecast {sample_forecast.shape}, analysis {sample_analysis.shape} and prediction {prediction.shape} in {date.strftime(self.plot_date_strformat)}.")
        fig = plt.figure(figsize=(12, 18)) # col, row
        
        if len(sample_forecast.shape) == 3:
            plot_sample_fc = sample_forecast[-1,:,:]
        elif len(sample_forecast.shape) == 2:
            plot_sample_fc = sample_forecast
        else:
            print(f"ERROR: unsupported dimensions {sample_forecast.shape} for sample_forecast")
        if len(sample_analysis.shape) == 3:
            plot_sample_an = sample_analysis[-1,:,:]
        elif len(sample_forecast.shape) == 2:
            plot_sample_an = sample_analysis
        else:
            print(f"ERROR: unsupported dimensions {sample_analysis.shape} for sample_analysis")
        if len(prediction.shape) == 3:
            plot_pred = prediction[-1,:,:]
        elif len(sample_forecast.shape) == 2:
            plot_pred = prediction
        else:
            print(f"ERROR: unsupported dimensions {prediction.shape} for prediction")
        if len(average_an.shape) == 3:
            plot_average_an = average_an[-1,:,:]
        elif len(average_an.shape) == 2:
            plot_average_an = average_an
        else:
            print(f"ERROR: unsupported dimensions {average_an.shape} for average analysis")
        # print(plot_average_an.shape)
        vmin_plt = np.min([np.min(plot_sample_fc), np.min(plot_sample_an), np.min(plot_pred)])
        vmax_plt = np.max([np.max(plot_sample_fc), np.max(plot_sample_an), np.min(plot_pred)])
        if self.anomaly:
            vcenter_plt = 0
            cmap = self.cmap_anomaly
        else:
            vcenter_plt = vmin_plt+(vmax_plt-vmin_plt)/2
            cmap = self.cmap
        title_details = f" {self.var_forecast}a" if self.anomaly else f" {self.var_forecast}"
        title_details += f" at {self.levhpa} hPa (" + date.strftime(self.plot_date_strformat) + ")"
        
        # Forecast
        ax1 = self._create_cartopy_axis (fig, 3, 2, 3, 'Forecast' + title_details, plot_sample_fc, vmin_plt, vmax_plt, vcenter_plt, cmap)
        
        # Analysis
        ax2 = self._create_cartopy_axis (fig, 3, 2, 1, 'Analysis' + title_details, plot_sample_an, vmin_plt, vmax_plt, vcenter_plt, cmap)

        # Prediction
        ax3 = self._create_cartopy_axis (fig, 3, 2, 5, 'Prediction' + title_details, plot_pred, vmin_plt, vmax_plt, vcenter_plt, cmap)

        # Average of analysis
        title_avg = f"Avg analysis {self.var_forecast} at {self.levhpa} hPa ({self.start_date.strftime(self.plot_date_strformat)} - {self.end_date.strftime(self.plot_date_strformat)})"
        if self.anomaly:
            vmin_plt = np.min(plot_average_an)
            vmax_plt = np.max(plot_average_an)
            vcenter_plt = vmin_plt+(vmax_plt-vmin_plt)/2
        ax6 = self._create_cartopy_axis (fig, 3, 2, 2, title_avg, plot_average_an, vmin_plt, vmax_plt, vcenter_plt, self.cmap) # always complete field cmap 
    
        # Error (Analysis - Forecast)
        error_fc = plot_sample_an - plot_sample_fc
        vmin_plt = np.min(error_fc)
        vmax_plt = np.max(error_fc)
        # vcenter_plt = vmin_plt+(vmax_plt-vmin_plt)/2
        vcenter_plt = 0
        # Error (Analysis - Prediction)
        error_pred = plot_sample_an - plot_pred
        # vmin_plt = np.min(error_pred)
        # vmax_plt = np.max(error_pred)
        # # vcenter_plt = vmin_plt+(vmax_plt-vmin_plt)/2
        vcenter_plt = 0

        # Pred error
        if len(error_pred.shape) == 3:
            plot_err = error_pred[-1,:,:]
        elif len(error_pred.shape) == 2:
            plot_err = error_pred
        else:
            print(f"ERROR: unsupported dimensions {error_pred.shape} for error")
        ax4 = self._create_cartopy_axis (fig, 3, 2, 6, 'Prediction Error' + title_details, plot_err, vmin_plt, vmax_plt, vcenter_plt, self.cmap_error)

        # Forecast error
        if len(error_fc.shape) == 3:
            plot_err = error_fc[-1,:,:]
        elif len(error_fc.shape) == 2:
            plot_err = error_fc
        else:
            print(f"ERROR: unsupported dimensions {error_fc.shape} for error")
        ax5 = self._create_cartopy_axis (fig, 3, 2, 4, 'Forecast Error' + title_details, plot_err, vmin_plt, vmax_plt, vcenter_plt, self.cmap_error)
        
        plt.tight_layout()
        plt.savefig(self.fig_folder + self._get_pics_fn(model, date))
        plt.close()


class WeatherDataset(Dataset):
    def __init__(self, forecasts, analyses):
        # Convert to PyTorch tensors (shape: [num_samples, 1, H, W])
        self.forecasts = torch.from_numpy(forecasts).float()
        self.analyses = torch.from_numpy(analyses).float()
        # Resize y to match model output
        if self.analyses.shape[-2] != self.forecasts.shape[-2] or self.analyses.shape[-1] != self.forecasts.shape[-1]:
            self.forecasts = torch.nn.functional.interpolate(
                self.forecasts, 
                size=(self.analyses.shape[-2], self.analyses.shape[-1]), 
                mode='bilinear'  # or 'bicubic' for higher precision
            )
    def __len__(self):
        return len(self.forecasts)
    def __getitem__(self, idx):
        return self.forecasts[idx], self.analyses[idx]

