import glob, os, sys, torch, datetime, cfgrib, time, toml, importlib, colorlog, logging, subprocess, psutil
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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import urllib.request, ssl, certifi
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
# from nets import WeatherCNN, MiniResNet18, ResNetUNet, UNetConvLSTM, SpatioTemporalForecastDataset
from smaatunet import SmaAt_UNet
from weatherunet import WeatherResNetUNet
from diffusion import DiffusionUNet, DiffusionNoiseScheduler, LitDiffusion
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    minmax_scale,
)
from itertools import product
from statsmodels.tsa.seasonal import seasonal_decompose

matplotlib.use('Agg')  # Use a non-interactive backend
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class Common:
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

class WeatherConfig:
    def __init__ (self, tomlfn):
        self.tomlfn = tomlfn
        # User configuration
        self.config = toml.load(tomlfn)

    def generate_hyper_combo (self):
        # Use itertools.product to get all combinations
        # The '*' unpacks the list of arrays into separate arguments for product
        hyper, keys = [], []
        for key in self.config["hyper"]:
            hyper.append(self.config["hyper"][key])
            keys.append(key)
        all_combinations = product(*hyper)
        run_configurations = {}
        for i, combo_tuple in enumerate(all_combinations):
            # self.logger.info(combo_tuple)
            # self.logger.info(keys)
            # Create a dictionary for the current combination
            # Use zip to map the parameter keys to their values in the current combination
            run_parameters = dict(zip(keys, combo_tuple))
            run_configurations[f"run_{i}"] = run_parameters
        return run_configurations


class WeatherRun:
    def __init__ (self, weather_config, hyper, run, dryrun=False):
        # User configuration
        self.config                  = weather_config.config
        # Hyperparameters
        self.hyper_dict              = hyper
        self.learning_rate           = self.hyper_dict["learning_rate"]
        self.batch_size              = self.hyper_dict["batch_size"]
        self.epochs                  = self.hyper_dict["epochs"]
        self.loss                    = self.hyper_dict["loss"]
        self.norm_strategy           = self.hyper_dict["norm_strategy"]
        self.supervised              = self.hyper_dict["supervised"]
        self.supervised_str = "supervised" if self.supervised else "unsupervised"
        criterion = getattr(nn, self.loss)
        self.criterion = criterion()
        self.norm = getattr(nn, self.norm_strategy)
        # Data
        self.var_forecast            = self.config["data"]["var_forecast"]
        self.var_analysis            = self.config["data"]["var_analysis"]
        self.error_limit             = self.config["data"]["error_limit"]
        self.levhpa                  = self.config["data"]["levhpa"]
        self.lonini, self.lonfin     = self.config["data"]["lonini"], self.config["data"]["lonfin"]
        self.latini, self.latfin     = self.config["data"]["latini"], self.config["data"]["latfin"]
        self.anomaly                 = self.config["data"]["anomaly"]
        self.deseason                = self.config["data"]["deseason"]
        self.detrend                 = self.config["data"]["detrend"]
        self.domain_size             = self.config["data"]["domain_size"] # set a square domain starting from lonini and latini, lonfin and latfin are ignored
        self.forecast_delta          = self.config["data"]["forecast_delta"]
        self.source                  = self.config["data"]["source"]
        self.scalername              = self.config["data"]["scaler_name"]
        self.Scaler                  = globals()[self.scalername]
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
        # Suffices for file names
        self.suffix = "_anomaly" if self.anomaly else ""
        self.suffix += "_detrend" if self.detrend else ""
        self.suffix += "_deseason" if self.deseason else ""
        # Formats, extensions and misc
        self.print_date_strformat    = self.config["global"]["print_date_strformat"]
        self.folder_date_strformat   = "%Y%m%d"
        self.forecast_date_strformat = "%m%d%H%M"
        self.new_date_strformat      = "%Y%m%d%H%M"
        self.plot_date_strformat     = "%Y%m%d-%H%M"
        self.data_extension          = ".npy.npz"
        self.non_sample_keys         = ['lon', 'lat', 'lev', 'samples', 'missed']
        # Run and paths
        self.netname                 = self.config["global"]["net"]
        self.Net                     = globals()[self.netname]
        self.run_base_name           = f"{self.var_forecast}-{self.var_analysis}{self.suffix}_{self.start_date.strftime(self.folder_date_strformat)}-{self.end_date.strftime(self.folder_date_strformat)}_{self.netname}_{self.source}_{self.scalername}_{self.loss}_{self.norm_strategy}_{self.config["global"]["run_name_suffix"]}_{self.epochs}epochs-{self.batch_size}bs-{self.learning_rate}lr"
        self.run_number              = run
        self.run_name                = self.run_base_name + "_" + self.run_number
        self.run_root_path           = self.config["global"]["run_root_path"]
        self.run_base_path           = self.run_root_path + self.run_base_name + "/"
        self.run_path                = self.run_base_path  + self.run_number + "/"
        self.work_root_path          = self.config["global"]["work_root_path"]
        self.download_path           = self.work_root_path + self.config["data"]["download_path"]
        self.save_data_folder        = self.work_root_path + self.config["global"]["save_data_path"]
        self.folder_freq             = "1d"
        # self.extra_data_folder       = self.run_path + self.config["global"]["extra_data_path"]
        self.fig_folder              = self.run_path + self.config["global"]["figure_path"]
        self.weights_folder          = self.run_path + self.config["global"]["weights_path"]
        # Log
        self.log_level_str           = self.config["global"]["log_level"]
        self.log_level               = getattr(logging, self.log_level_str.upper())
        self.log_format_string_color = (
                                        # '%(log_color)s%(asctime)s [%(levelname)s: %(name)s] %(message)s'
                                        '%(log_color)s[%(levelname)s]%(reset)s %(name)s: %(message)s'
                                        # '%(funcName)s:%(lineno)d - %(message)s'
        )
        self.log_format_string       = ('[%(asctime)s %(levelname)s] %(name)s: %(message)s')
        self.logger                  = logging.getLogger(__name__)
        if dryrun:
            self.log_folder          = self.run_base_path
            self.log_filename        = self.log_folder + "combo.log"
        else:
            self.log_folder          = self.run_path + "logs/"
            self.log_filename        = self.log_folder + self.run_name + ".log"
        self.tl_root_logdir          = self.run_root_path
        self.tl_logdir               = f"{self.tl_root_logdir}{self.run_base_name}/lightning_logs/"
        self.accumulate_grad_batches = self.config["global"]["accumulate_grad_batches"]
        # GPU
        self.device                  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Prepare
        if self.source == "ecmwf":
            en_datatype = self.config["ecmwf"]["ensemble_dataype"]
            self.head_forecast_fn = "forecast-"
            self.head_analysis_fn = "analysis-"
            self.forecast_root_path = self.download_path
            self.analysis_root_path = self.forecast_root_path
            self.orig_freq = self.config["ecmwf"]["origin_frequency"]
            self.grib_dict_fc = {'dataType': en_datatype, 'cfVarName': self.var_forecast}
            self.grib_dict_an = {'dataType': en_datatype, 'cfVarName': self.var_analysis}
            self.analysis_path = self.analysis_root_path
            self.forecast_path = self.forecast_root_path
        elif self.source == "cmcc":
            self.head_forecast_fn = "JLS"
            self.head_analysis_fn = "JLD"
            self.forecast_root_path = f"{self.config["cmcc"]["forecast_path"]}{self.config["cmcc"]["prod_freq_forecast"]}/{self.config["cmcc"]["file_format_forecast"]}/"
            # self.analysis_root_path = f"/data/inputs/METOCEAN/rolling/model/atmos/ECMWF/IFS_010/3.1analysis/{folder_acq_freq_analysis}/netcdf/"
            self.analysis_root_path = f"{self.config["cmcc"]["analysis_path"]}{self.config["cmcc"]["prod_freq_analysis"]}/{self.config["cmcc"]["file_format_analysis"]}/"
            self.orig_freq = self.config["cmcc"]["origin_frequency"]
            self.grib_dict_fc = {'cfVarName': self.var_forecast}
            self.grib_dict_an = {'cfVarName': self.var_analysis}
        else:
            self.logger.error("Unsupported source type. Aborting...")
            return
        # Make folders if necessary
        if not dryrun:
            os.makedirs(self.save_data_folder, exist_ok=True)
            # os.makedirs(self.extra_data_folder, exist_ok=True)
            os.makedirs(self.fig_folder, exist_ok=True)
            os.makedirs(self.weights_folder, exist_ok=True)
            os.makedirs(self.tl_logdir, exist_ok=True)
        os.makedirs(self.log_folder, exist_ok=True)
        # Model
        self.model = self.Net(
                        learning_rate=self.learning_rate,
                        loss=self.criterion,
                        norm=self.norm,
                        supervised=self.supervised,
                        extra_logger=self.logger
                    ).to(self.device)

    def log_global_setup (self):
        self.logger.info(f"GPU")
        self.logger.info("---------------------")
        self.logger.info(f" device             : {self.device}")
        self.logger.info("=====================")
        self.logger.info(f"Run")
        self.logger.info("---------------------")
        self.logger.info(f" run name           : {self.run_name}")
        self.logger.info(f" net name           : {self.netname}")
        self.logger.info(f" log level          : {self.log_level_str}")
        self.logger.info("=====================")
        self.logger.info(f"Paths")
        self.logger.info("---------------------")
        self.logger.info(f" work path          : {self.work_root_path}")
        self.logger.info(f" global data path   : {self.save_data_folder}")
        self.logger.info(f" run path           : {self.run_path}")
        self.logger.info(f" weights path       : {self.weights_folder}")
        # self.logger.info(f" extra data path    : {self.extra_data_folder}")
        self.logger.info(f" log path           : {self.log_folder}")
        self.logger.info(f" figure path        : {self.fig_folder}")
        self.logger.info(f" tensorboard log    : {self.tl_logdir}")
        self.logger.info("=====================")
        self.logger.info(f"Data")
        self.logger.info("---------------------")
        self.logger.info(f" forecast variable  : {self.var_forecast}{self.suffix}")
        self.logger.info(f" analysis variable  : {self.var_analysis}{self.suffix}")
        self.logger.info("=====================")
        self.logger.info(f"Hyperparameters")
        self.logger.info("---------------------")
        self.logger.info(f" learning rate      : {self.learning_rate}")
        self.logger.info(f" batch size         : {self.batch_size}")
        self.logger.info(f" epochs             : {self.epochs}")
        self.logger.info(f" loss               : {self.loss}")
        self.logger.info(f" norm               : {self.norm_strategy}")
        self.logger.info(f" learning strategy  : {self.supervised_str}")
        self.logger.info("=====================")

    def log_train_setup (self):
        self.logger.info(f"Training")
        self.logger.info("---------------------")
        self.logger.info(f" start date         : {self.start_date}")
        self.logger.info(f" end date           : {self.end_date}")
        self.logger.info("=====================")

    def log_test_setup (self):
        self.logger.info(f"Testing")
        self.logger.info("---------------------")
        self.logger.info(f" start date         : {self.test_start_date}")
        self.logger.info(f" end date           : {self.test_end_date}")
        self.logger.info("=====================")

    def setup_logger (self):
        self.logger.setLevel(self.log_level)
        self.logger.handlers = []  # Clear existing handlers
        formatter_color = colorlog.ColoredFormatter(
            self.log_format_string_color,
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            },
        )
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(formatter_color)
        self.logger.addHandler(console_handler)
        formatter = logging.Formatter(self.log_format_string)
        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _get_base_data_fn (self, start_date, end_date):
        return f"{self.save_data_folder}{self.source}_{self.var_forecast}-{self.var_analysis}_{self.forecast_delta}_{start_date.strftime(self.new_date_strformat)}_{end_date.strftime(self.new_date_strformat)}"

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
            self.logger.error(f"Unsupported type {type}")
            data_fn = None
        return data_fn

    def config_ssl_env (self, env_base_path):
        os.environ["SSL_CERT_FILE"] = str(env_base_path / 'ssl' / 'tls-ca-bundle.pem')
        os.environ["SSL_CERT_DIR"] = str(env_base_path / 'ssl')
        os.environ["REQUESTS_CA_BUNDLE"] = str(env_base_path / 'ssl' / 'tls-ca-bundle.pem')

    def _get_anfc_paths (self, d, special_glob=False):
        forecast_d = d - datetime.timedelta(days=self.forecast_delta)
        if self.source == "ecmwf":
            analysis_path = self.analysis_root_path
            forecast_path = self.forecast_root_path
            analysis_fn_glob = f"{self.head_analysis_fn}{self.var_analysis}-{d.strftime(self.new_date_strformat)}-{d.strftime(self.new_date_strformat)}*"
            forecast_fn_glob = f"{self.head_forecast_fn}{self.var_forecast}-{forecast_d.strftime(self.new_date_strformat)}-{d.strftime(self.new_date_strformat)}*"
        elif self.source == "cmcc":
            # analysis_path = f"{self.analysis_root_path}{d.strftime(folder_date_strformat)}/"
            analysis_path = f"{self.analysis_root_path}{d.strftime("%Y")}/{d.strftime("%m")}/"
            forecast_path = f"{self.forecast_root_path}{forecast_d.strftime(self.folder_date_strformat)}/"
            if special_glob:
                # Here we need to select the 00001 ending to avoid the 00011 which has got different variables. This is a workaround, not robust
                analysis_fn_glob = f"{self.head_analysis_fn}{d.strftime(self.forecast_date_strformat)}*01"
            else:
                # No need of *01 here because the time format 0000 is already excluding the 00011 ending. This is by chance, keep an eye
                analysis_fn_glob = f"{self.head_analysis_fn}{d.strftime(self.forecast_date_strformat)}{d.strftime(self.forecast_date_strformat)}*" 
            # analysis_fn_glob = f"{d.strftime(folder_date_strformat)}-ECMWF---AM0100-MEDATL-b{d.strftime(folder_date_strformat)}_an{h.strftime("%M")}-fv11.00.nc"
            forecast_fn_glob = f"{self.head_forecast_fn}{forecast_d.strftime(self.forecast_date_strformat)}{d.strftime(self.forecast_date_strformat)}*"
        return analysis_path, forecast_path, analysis_fn_glob, forecast_fn_glob

    def prepare_data (self, type):
        self.logger.info(f"Prepare {type} data")
        data_fn = self._get_data_fn_from_type(type)
        data_f = Path(data_fn)
        if data_f.is_file():
            self.logger.warning(f"File {data_fn} already exists.")
            return
        if type == 'train':
            start_date = self.start_date
            end_date = self.end_date
            coords_d = self.start_date
        elif type == 'test':
            start_date = self.test_start_date
            end_date = self.test_end_date
            coords_d = self.test_start_date
        else:
            self.logger.error(f"Unsupported type {type}")
            coords_d = None

        # Get the data
        date_range = pd.date_range(start=start_date, end=end_date, freq=self.orig_freq).to_pydatetime().tolist()
        self.logger.info(f"Get {self.var_forecast} (forecast), {self.var_analysis} (analysis) from {start_date.strftime(self.print_date_strformat)} to {end_date.strftime(self.print_date_strformat)}")
        var_d = {}
        missed_samples = []
        counter_samples = 0
        for d in date_range:
            forecast_d = d - datetime.timedelta(days=self.forecast_delta)
            self.logger.info(f"Analysis: {d}")
            self.logger.info(f"Forecast: {forecast_d}")
            # self.logger.debug(d)
            # self.logger.debug(forecast_d)
            # acq_range = pd.date_range(start=d, end=d+datetime.timedelta(days=1), freq=acq_freq).to_pydatetime().tolist()
            analysis_path, forecast_path, analysis_fn_glob, forecast_fn_glob = self._get_anfc_paths(d)
            # self.logger.debug(forecast_fn_glob)
            # self.logger.debug(analysis_fn_glob)
            errormsg = None
            nearest_minus1h_forecast_flag = False
            nearest_plus1h_forecast_flag = False
            try:
                forecast_fn = glob.glob(forecast_path + forecast_fn_glob)[0]
            except Exception as e:
                self.logger.error(f"Exception: {e}")
                errormsg = f"Couldn't find forecast file {forecast_path + forecast_fn_glob}"
                if self.source == "cmcc":
                    # If exact +deltaforecast is not present attempt to estimate with nearest +-1h forecasts
                    minus1h_d = d - datetime.timedelta(hours=1)
                    plus1h_d = d + datetime.timedelta(hours=1)
                    forecast_minus1h_fn_glob = f"{self.head_forecast_fn}{forecast_d.strftime(self.forecast_date_strformat)}{minus1h_d.strftime(self.forecast_date_strformat)}*"
                    forecast_plus1h_fn_glob = f"{self.head_forecast_fn}{forecast_d.strftime(self.forecast_date_strformat)}{plus1h_d.strftime(self.forecast_date_strformat)}*"
                    self.logger.info("Try nearest +-1h forecasts...")
                    try:
                        forecast_minus1h_fn = glob.glob(forecast_path + forecast_minus1h_fn_glob)[0]
                        nearest_minus1h_forecast_flag = True
                    except Exception as e:
                        self.logger.error(f"{errormsg}")
                        errormsg = f"Couldn't find forecast -1h file {forecast_path + forecast_minus1h_fn_glob}"
                    try:
                        forecast_plus1h_fn = glob.glob(forecast_path + forecast_plus1h_fn_glob)[0]
                        nearest_plus1h_forecast_flag = True
                    except Exception as e:
                        self.logger.error(f"{errormsg}")
                        errormsg = f"Couldn't find forecast +1h file {forecast_path + forecast_plus1h_fn_glob}"
                        self.logger.error(f"{errormsg}. Skipping...")
                        missed_samples.append(d)
                        continue
                # self.logger.error(f"{errormsg}. Skipping...")
            try:
                analysis_fn = glob.glob(analysis_path + analysis_fn_glob)[0]
            except Exception as e:
                self.logger.error(f"Exception: {e}")
                errormsg = f"Couldn't find analysis file {analysis_path + analysis_fn_glob}"
                self.logger.error(f"{errormsg}. Skipping...")
                missed_samples.append(d)
                continue
            # Get analysis dataset
            try:
                self.logger.info(analysis_fn)
                ds_analysis = xr.open_dataset(
                    analysis_fn, engine="cfgrib", indexpath="", decode_timedelta=True,
                    backend_kwargs={'filter_by_keys': self.grib_dict_an}
                )
                an_values = ds_analysis.variables[self.var_analysis].values
                # ds_analysis = nc.Dataset(analysis_fn)
            except Exception as e:
                self.logger.error(f"Exception: {e}")
                errormsg = f"Couldn't open analysis file {analysis_fn}"
                self.logger.error(f"{errormsg}. Skipping...")
                missed_samples.append(d)
                continue
            # Get forecast dataset
            try:
                self.logger.info(forecast_fn)
                if nearest_minus1h_forecast_flag and nearest_plus1h_forecast_flag:
                    ds_forecast_minus1h = xr.open_dataset(
                        forecast_minus1h_fn, engine="cfgrib", indexpath="", decode_timedelta=True,
                        backend_kwargs={'filter_by_keys': self.grib_dict_fc}
                    )
                    ds_forecast_plus1h = xr.open_dataset(
                        forecast_plus1h_fn, engine="cfgrib", indexpath="", decode_timedelta=True,
                        backend_kwargs={'filter_by_keys': self.grib_dict_fc}
                    )
                    self.logger.warning("Average +-1h forecasts...")
                    fc_values = np.mean( np.array([ ds_forecast_plus1h.variables[self.var_forecast].values, ds_forecast_minus1h.variables[self.var_forecast].values ]), axis=0 )
                else:
                    if nearest_minus1h_forecast_flag:
                        self.logger.warning("Use -1h forecast...")
                        ds_forecast = xr.open_dataset(
                            forecast_minus1h_fn, engine="cfgrib", indexpath="", decode_timedelta=True,
                            backend_kwargs={'filter_by_keys': self.grib_dict_fc}
                        )
                    elif nearest_plus1h_forecast_flag:
                        self.logger.warning("Use +1h forecast...")
                        ds_forecast = xr.open_dataset(
                            forecast_plus1h_fn, engine="cfgrib", indexpath="", decode_timedelta=True,
                            backend_kwargs={'filter_by_keys': self.grib_dict_fc}
                        )
                    else:
                        ds_forecast = xr.open_dataset(
                            forecast_fn, engine="cfgrib", indexpath="", decode_timedelta=True,
                            backend_kwargs={'filter_by_keys': self.grib_dict_fc}
                        )
                    fc_values = ds_forecast.variables[self.var_forecast].values
                # ds_forecast = xr.open_dataset(forecast_fn, engine="cfgrib", indexpath="")
                # ds_forecast = cfgrib.open_datasets(forecast_fn, indexpath="/tmp/tempgrib.{short_hash}.idx")
            except Exception as e:
                self.logger.error(f"Exception: {e}")
                errormsg = f"Couldn't open forecast file {forecast_fn}"
                self.logger.error(f"{errormsg}. Skipping...")
                missed_samples.append(d)
                continue
            self.logger.info(f"Forecast values shape: {fc_values.shape}")
            self.logger.info(f"Analysis values shape: {an_values.shape}")
            var_d[d] = {
                'forecast': fc_values,
                # 'analysis': ds_analysis.variables[self.var_analysis][:]
                'analysis': an_values
            }
            counter_samples += 1

        self.logger.info(f"Number of prepared samples: {counter_samples}")
        self.logger.info(f"Number of missed samples: {len(missed_samples)}")
        self.logger.debug(f"Missed samples: {missed_samples}")
        var_d['samples'] = counter_samples
        var_d['missed'] = missed_samples

        # Get coords data
        lon_fc_full, lon_an_full, lat_fc_full, lat_an_full, lev_fc_full, lev_an_full = self.get_coords(coords_d)
        var_d['lon'] = {'analysis': lon_an_full}
        var_d['lat'] = {'analysis': lat_an_full}
        var_d['lev'] = {'analysis': lev_an_full}
        var_d['lon']['forecast'] = lon_fc_full
        var_d['lat']['forecast'] = lat_fc_full
        var_d['lev']['forecast'] = lev_fc_full

        # Save data
        self.logger.info(f"Saving data in {data_fn}")
        with open(f"{data_fn}", 'wb') as f:
            np.savez(f"{data_fn}", var_d, allow_pickle=True)

    def get_coords (self, coords_d):
        analysis_coords_path, forecast_coords_path, analysis_fn_coords_glob, forecast_fn_coords_glob = self._get_anfc_paths(coords_d, special_glob=True)
        forecast_coords_fn = glob.glob(forecast_coords_path + forecast_fn_coords_glob)[0]
        analysis_coords_fn = glob.glob(analysis_coords_path + analysis_fn_coords_glob)[0]

        ds_forecast_coords = xr.open_dataset(
            forecast_coords_fn, engine="cfgrib", indexpath="", decode_timedelta=True,
            backend_kwargs={'filter_by_keys': self.grib_dict_fc}
        )
        ds_analysis_coords = xr.open_dataset(
            analysis_coords_fn, engine="cfgrib", indexpath="", decode_timedelta=True,
            backend_kwargs={'filter_by_keys': self.grib_dict_an}
        )

        coord_lon_names = ['longitude', 'lon']
        coord_lat_names = ['latitude', 'lat']
        for lon_name in coord_lon_names:
            try:
                lon_fc_full = ds_forecast_coords[lon_name].values
                break
            except Exception as e:
                self.logger.error(f"Exception: {e}")
                self.logger.error(f"Longitude coord name {lon_name} not present in {forecast_coords_fn}")
        for lon_name in coord_lon_names:
            try:
                lon_an_full = ds_analysis_coords[lon_name].values
                break
            except Exception as e:
                self.logger.error(f"Exception: {e}")
                self.logger.error(f"Longitude coord name {lon_name} not present in {analysis_coords_fn}")
        for lat_name in coord_lat_names:
            try:
                lat_fc_full = ds_forecast_coords[lat_name].values
                break
            except Exception as e:
                self.logger.error(f"Exception: {e}")
                self.logger.error(f"Latitude coord name {lat_name} not present in {forecast_coords_fn}")
        for lat_name in coord_lat_names:
            try:
                lat_an_full = ds_analysis_coords[lat_name].values
                break
            except Exception as e:
                self.logger.error(f"Exception: {e}")
                self.logger.error(f"Latitude coord name {lat_name} not present in {analysis_coords_fn}")
        lev_fc_full = ds_forecast_coords['isobaricInhPa'].values
        lev_an_full = ds_analysis_coords['isobaricInhPa'].values

        return lon_fc_full, lon_an_full, lat_fc_full, lat_an_full, lev_fc_full, lev_an_full

    def get_data_from_fn (self, data_fn):
        with open(f"{data_fn}", 'rb') as f:
            data_d = np.load(f, allow_pickle=True)['arr_0'].item()
        return data_d

    def load_data (self, type):
        data_fn = self._get_data_fn_from_type(type)
        average_data_path = self._get_average_fn()
        trend_data_path = self._get_trend_fn()
        season_data_path = self._get_season_fn()
        self.logger.info(f"Loading {type} data from {data_fn}")
        with open(f"{data_fn}", 'rb') as f:
            var_d = np.load(f, allow_pickle=True)['arr_0'].item()

        self.logger.info(f"Samples number: {var_d['samples']}")
        self.logger.info(f"Missed samples number: {len(var_d['missed'])}")
        self.logger.debug(f"Missed samples: {var_d['missed']}")
        lon_an_full = var_d['lon']['analysis']
        lat_an_full = var_d['lat']['analysis']
        lev_an_full = var_d['lev']['analysis']
        lon_fc_full = var_d['lon']['forecast']
        lat_fc_full = var_d['lat']['forecast']
        lev_fc_full = var_d['lev']['forecast']

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

        if self.domain_size:
            self.logger.info(f"Selected regular square size for analysis: {self.domain_size}x{self.domain_size}")
            self.logger.info(f"Ignoring latfin: {latfin} and lonfin: {lonfin}")
            self.logger.debug(f"Lonfin before adding size_an: {lonfin_an_idx}")
            self.logger.debug(f"Latfin before adding size_an: {latfin_an_idx}")
            lonfin_an_idx = lonini_an_idx + self.domain_size
            latfin_an_idx = latini_an_idx + self.domain_size
            self.logger.debug(f"Lonfin after adding size_an: {lonfin_an_idx}")
            self.logger.debug(f"Latfin after adding size_an: {latfin_an_idx}")
            lonfin = lon_an_full[lonfin_an_idx]
            latfin = lat_an_full[latfin_an_idx]
            lonfin_fc_idx = (np.abs(lon_fc_full - lonfin)).argmin()
            latfin_fc_idx = (np.abs(lat_fc_full - latfin)).argmin()

        lonfc = lon_fc_full[lonini_fc_idx:lonfin_fc_idx]
        latfc = lat_fc_full[latini_fc_idx:latfin_fc_idx]
        lonan = lon_an_full[lonini_an_idx:lonfin_an_idx]
        latan = lat_an_full[latini_an_idx:latfin_an_idx]

        first_step = next(iter(var_d))
        first_fc = var_d[first_step]['forecast']
        first_an = var_d[first_step]['analysis']
        first_fc_sel = first_fc[lev_forecast,latini_fc_idx:latfin_fc_idx,lonini_fc_idx:lonfin_fc_idx]
        first_an_sel = first_an[lev_forecast,latini_fc_idx:latfin_fc_idx,lonini_fc_idx:lonfin_fc_idx]
        # self.logger.debug(f"Selected region")
        # self.logger.debug(f"lonfc")
        # self.logger.debug(f"{lonfc}")
        # self.logger.debug(f"latfc")
        # self.logger.debug(f"{latfc}")
        # self.logger.debug(f"lonan")
        # self.logger.debug(f"{lonan}")
        # self.logger.debug(f"latan")
        # self.logger.debug(f"{latan}")

        self.logger.info("==================")
        self.logger.info(f"Full dataset")
        self.logger.info("==================")
        self.logger.info(f"Forecast ({self.var_forecast})")
        self.logger.info("------------------")
        self.logger.info(f" shape {first_step}: {first_fc.shape}")
        self.logger.info(f" lat[0], lat[{lat_fc_full.shape[0]-1}]: {lat_fc_full[0]:.2f}, {lat_fc_full[-1]:.2f}")
        self.logger.info(f" lon[0], lon[{lon_fc_full.shape[0]-1}]: {lon_fc_full[0]:.2f}, {lon_fc_full[-1]:.2f}")
        self.logger.info(f" lev[0], lev[{lev_fc_full.shape[0]-1}]  : {lev_fc_full[0]:.2f}, {lev_fc_full[-1]:.2f}")
        self.logger.info("==================")
        self.logger.info(f"Analysis ({self.var_analysis})")
        self.logger.info("------------------")
        self.logger.info(f" shape {first_step}: {first_an.shape}")
        self.logger.info(f" lat[0], lat[{lat_an_full.shape[0]-1}]: {lat_an_full[0]:.2f}, {lat_an_full[-1]:.2f}")
        self.logger.info(f" lon[0], lon[{lon_an_full.shape[0]-1}]: {lon_an_full[0]:.2f}, {lon_an_full[-1]:.2f}")
        self.logger.info(f" lev[0], lev[{lev_an_full.shape[0]-1}]  : {lev_an_full[0]:.2f}, {lev_an_full[-1]:.2f}")

        self.logger.info("====================")
        self.logger.info(f"Selected region")
        self.logger.info("====================")
        self.logger.info(f"Forecast ({self.var_forecast})")
        self.logger.info("--------------------")
        self.logger.info(f" shape var         : {first_fc_sel.shape}")
        self.logger.info(f" shape lat         : {latfc.shape}")
        self.logger.info(f" shape lon         : {lonfc.shape}")
        self.logger.info(f" shape lev         : {lev_fc_full.shape}")
        self.logger.info(f" lat[{latini_fc_idx}], lat[{latfin_fc_idx}]: {latfc[0]:.2f}, {latfc[-1]:.2f}")
        self.logger.info(f" lon[{lonini_fc_idx}], lon[{lonfin_fc_idx}]: {lonfc[0]:.2f}, {lonfc[-1]:.2f}")
        self.logger.info(f" lev[{lev_forecast}]            : {self.levhpa}")
        self.logger.info("====================")
        self.logger.info(f"Analysis ({self.var_analysis})")
        self.logger.info("--------------------")
        self.logger.info(f" shape var         : {first_an_sel.shape}")
        self.logger.info(f" shape lat         : {latan.shape}")
        self.logger.info(f" shape lon         : {lonan.shape}")
        self.logger.info(f" shape lev         : {lev_an_full.shape}")
        self.logger.info(f" lat[{latini_an_idx}], lat[{latfin_an_idx}]: {latan[0]:.2f}, {latan[-1]:.2f}")
        self.logger.info(f" lon[{lonini_an_idx}], lon[{lonfin_an_idx}]: {lonan[0]:.2f}, {lonan[-1]:.2f}")
        self.logger.info(f" lev[{lev_analysis}]            : {self.levhpa}")

        # Lists to hold all samples
        forecast_data = []
        analysis_data = []

        # Extract data from dictionary
        for key in var_d.keys():  # Loop through all timesteps
            if key not in self.non_sample_keys:
                # Get 2D fields
                forecast = var_d[key]["forecast"]  # Shape: (level, lat, lon)
                analysis = var_d[key]["analysis"]  # Shape: (level, lat, lon)

                if analysis.shape != first_an.shape:
                    if analysis.shape[1] == first_an.shape[1] and analysis.shape[2] == first_an.shape[2]:
                        self.logger.warn(f"Analysis level dimension mismatch. {first_step}: {first_an.shape}, {key}: {analysis.shape}")
                        _, _, _, _, _, lev_an_full = self.get_coords(key)
                        lev_analysis = (np.abs(lev_an_full - self.levhpa)).argmin()
                        first_an = var_d[key]['analysis']
                        self.logger.warn(f"New analysis level at {key}: lev[{lev_analysis}]: {self.levhpa}")
                    else:
                        self.logger.error(f"Analysis dimension mismatch. {first_step}: {first_an.shape}, {key}: {analysis.shape}")
                if forecast.shape != first_fc.shape:
                    if forecast.shape[1] == first_fc.shape[1] and forecast.shape[2] == first_fc.shape[2]:
                        self.logger.warn(f"Forecast level dimension mismatch. {first_step}: {first_fc.shape}, {key}: {forecast.shape}")
                        _, _, _, _, lev_fc_full, _ = self.get_coords(key)
                        lev_forecast = (np.abs(lev_fc_full - self.levhpa)).argmin()
                        first_fc = var_d[key]['forecast']
                        self.logger.warn(f"New forecast level at {key}: lev[{lev_analysis}]: {self.levhpa}")
                    else:
                        self.logger.error(f"Forecast dimension mismatch. {first_step}: {first_fc.shape}, {key}: {forecast.shape}")

                if len(analysis.shape) == 3:
                    analysis = analysis[lev_analysis,latini_an_idx:latfin_an_idx,lonini_an_idx:lonfin_an_idx]
                elif len(analysis.shape) == 2:
                    analysis = analysis[latini_an_idx:latfin_an_idx,lonini_an_idx:lonfin_an_idx]
                else:
                    self.logger.error(f"Unsupported dimensions {analysis.shape} for {self.var_analysis}")

                if len(forecast.shape) == 3:
                    forecast = forecast[lev_forecast,latini_fc_idx:latfin_fc_idx,lonini_fc_idx:lonfin_fc_idx]
                elif len(forecast.shape) == 2:
                    forecast = forecast[latini_fc_idx:latfin_fc_idx,lonini_fc_idx:lonfin_fc_idx]
                else:
                    self.logger.error(f"Unsupported dimensions {forecast.shape} for {self.var_forecast}")

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
        # Prune data of unwanted datapoints if required
        errs = y_np - X_np
        mean_errs = errs.mean(axis=(2,3))[:,0]
        self.logger.info(f"Average an-fc error: {errs.mean()}")
        self.logger.debug("Mean errors for each timestep:")
        self.logger.debug(mean_errs)
        dates = np.array([x for x in list(var_d.keys()) if x not in self.non_sample_keys])
        if self.error_limit:
            condition = (mean_errs > self.error_limit).astype(bool) | (mean_errs < -self.error_limit).astype(bool)
            full_dates = dates.copy()
            dates = np.delete(dates, condition)
            X_np  = np.delete(X_np, condition, axis=0)
            y_np  = np.delete(y_np, condition, axis=0)
            self.logger.debug(f"Full timesteps where an-fc >  {self.error_limit}:")
            self.logger.debug(f"{full_dates[(mean_errs > self.error_limit).astype(bool)]}")
            self.logger.debug(" max values:")
            self.logger.debug(f"{mean_errs[(mean_errs > self.error_limit).astype(bool)]}")
            self.logger.debug(f"Full timepsteps where an-fc < -{self.error_limit}:")
            self.logger.debug(f"{full_dates[(mean_errs < -self.error_limit).astype(bool)]}")
            self.logger.debug(" min values:")
            self.logger.debug(f"{mean_errs[(mean_errs < -self.error_limit).astype(bool)]}")
            pruned_mean_errs = np.delete(mean_errs, condition)
            self.logger.info(f"Average an-fc error (after pruning): {pruned_mean_errs.mean()}")
            self.logger.debug(f"Pruned timesteps where an-fc >  {self.error_limit}:")
            self.logger.debug(f"{dates[(pruned_mean_errs > self.error_limit).astype(bool)]}")
            self.logger.debug(f"Pruned timepsteps where an-fc < -{self.error_limit}:")
            self.logger.debug(f"{dates[(pruned_mean_errs < -self.error_limit).astype(bool)]}")

        if self.deseason:
            self.logger.info(f"Deseasonalize...")
            # decomposition_fc = seasonal_decompose(X_np, model='additive', period=12)
            # decomposition_an = seasonal_decompose(y_np, model='additive', period=12)
            if type == 'train':
                if self._check_seasons():
                    self.logger.info(f"Load training period seasonality from {season_data_path}")
                    season_d = self.get_data_from_fn(season_data_path)
                    season_fc = season_d["forecast"]
                    season_an = season_d["analysis"]
                    X_np, _ = self.deseasonalize(X_np, self.start_date, self.end_date, var_d['missed'], climatology_monthly=season_fc)
                    y_np, _ = self.deseasonalize(y_np, self.start_date, self.end_date, var_d['missed'], climatology_monthly=season_an)
                else:
                    self.logger.info(f"Save training period seasonality in {season_data_path}")
                    X_np, season_fc = self.deseasonalize(X_np, self.start_date, self.end_date, var_d['missed'])
                    y_np, season_an = self.deseasonalize(y_np, self.start_date, self.end_date, var_d['missed'])
                    # self.logger.debug(f"average_fc shape: {np.shape(average_fc)}")
                    season_d = {"forecast": season_fc, "analysis": season_an}
                    with open(season_data_path, 'wb') as f:
                        np.savez(season_data_path, season_d, allow_pickle=True)
            elif type == 'test':
                # Load average data
                self.logger.info(f"Load training period seasonality from {season_data_path}")
                season_d = self.get_data_from_fn(season_data_path)
                season_fc = season_d["forecast"]
                season_an = season_d["analysis"]
                X_np, _ = self.deseasonalize(X_np, self.test_start_date, self.test_end_date, var_d['missed'], climatology_monthly=season_fc)
                y_np, _ = self.deseasonalize(y_np, self.test_start_date, self.test_end_date, var_d['missed'], climatology_monthly=season_an)

        if self.anomaly:
            if type == 'train':
                if self._check_averages():
                    self.logger.info(f"Load training period average from {average_data_path}")
                    average_d = self.get_data_from_fn(average_data_path)
                    average_fc = average_d["forecast"]
                    average_an = average_d["analysis"]
                else:
                    self.logger.info(f"Save training period average in {average_data_path}")
                    average_fc = X_np.mean(axis=0, keepdims=True)
                    average_an = y_np.mean(axis=0, keepdims=True)
                    # self.logger.debug(f"average_fc shape: {np.shape(average_fc)}")
                    average_d = {"forecast": average_fc, "analysis": average_an}
                    with open(average_data_path, 'wb') as f:
                        np.savez(average_data_path, average_d, allow_pickle=True)
            elif type == 'test':
                # Load average data
                self.logger.info(f"Load training period average from {average_data_path}")
                average_d = self.get_data_from_fn(average_data_path)
                average_fc = average_d["forecast"]
                average_an = average_d["analysis"]
            X_np -= average_fc
            y_np -= average_an

        if self.detrend:
            # Load trend data
            self.logger.info(f"Loading trend from {trend_data_path}")
            with open(f"{trend_data_path}", 'rb') as f:
                trend_d = np.load(f, allow_pickle=True)['arr_0'].item()
                slopes_fc, intercepts_fc = trend_d['slopes']['forecast'], trend_d['intercepts']['forecast']
                slopes_an, intercepts_an = trend_d['slopes']['analysis'], trend_d['intercepts']['analysis']
                dim_an, dim_fc = trend_d['dims']['analysis'], trend_d['dims']['forecast']
            trend_fc = dim_fc*slopes_fc.flatten() + intercepts_fc.flatten()
            trend_an = dim_an*slopes_an.flatten() + intercepts_an.flatten()
            X_np -= np.expand_dims(trend_fc.reshape(X_np.shape[0], X_np.shape[-2], X_np.shape[-1]), axis=1)
            y_np -= np.expand_dims(trend_an.reshape(y_np.shape[0], y_np.shape[-2], y_np.shape[-1]), axis=1)

        return X_np, y_np, dates, lonfc, latfc, lonan, latan

    def deseasonalize (self, data, start_date, end_date, missing_dates, climatology_monthly=None):
        N, C, H, W = np.shape(data)
        data = np.squeeze(data, axis=1)
        # self.logger.info(len(missing_dates))
        date_range = pd.date_range(start=start_date, end=end_date, freq=self.config[self.source]["origin_frequency"]).difference(missing_dates)
        da = xr.DataArray(
            data,
            dims=['time', 'lat', 'lon'],
            coords={
                'time': date_range,
                'lat': range(H),
                'lon': range(W)
            }
        )
        if not isinstance(climatology_monthly, np.ndarray):
            climatology_monthly = da.groupby('time.month').mean('time')
            self.logger.debug(f"Monthly climatology shape: {climatology_monthly.shape}")
        deseasonalized_monthly = da.groupby('time.month') - climatology_monthly
        return deseasonalized_monthly.values.reshape(N, C, H, W), climatology_monthly

    def normalize (self, data):
        # Remove channel dimension (channel number C is always 1)
        N, C, H, W = np.shape(data)
        self.logger.debug(f"Data before normalization: {np.shape(data)}")
        data = np.squeeze(data, axis=1)
        # Reshape along samples N
        data = data.reshape(H, -1)
        self.logger.debug(f"Reshaped data before normalization: {np.shape(data)}")
        scaler = self.Scaler().fit(data)
        scaled_data = scaler.transform(data)
        self.logger.debug(f"Data after normalization: {np.shape(scaled_data)}")
        scaled_data = scaled_data.reshape(N, C, H, W)
        self.logger.debug(f"Reshaped data after normalization: {np.shape(scaled_data)}")
        return scaled_data, scaler

    def denormalize (self, scaled_data, scaler):
        self.logger.debug(f"Data before denormalization: {np.shape(scaled_data)}")
        N, C, H, W = np.shape(scaled_data)
        scaled_data = np.squeeze(scaled_data, axis=1)
        # Reshape along samples N
        scaled_data = scaled_data.reshape(H, -1)
        self.logger.debug(f"Reshaped data before denormalization: {np.shape(scaled_data)}")
        data = scaler.inverse_transform(scaled_data)
        data = data.reshape(N, C, H, W)
        return data

    def _get_final_products_base_fn (self):
        return f"{self.model.__class__.__name__}_{self.supervised_str}_{self.var_forecast}-{self.var_analysis}-{self.levhpa}hPa_{self.lonini:+2.1f}-{self.latini:+2.1f}_{self.domain_size}x{self.domain_size}_{self.batch_size}bs-{self.learning_rate}lr-{self.epochs}epochs-{self.loss}_{self.norm_strategy}_{self.start_date.strftime(self.plot_date_strformat)}_{self.end_date.strftime(self.plot_date_strformat)}{self.suffix}"

    def _check_averages (self):
        if Path(f"{self._get_average_fn()}").is_file():
            self.logger.warning(f"Averages file {self._get_average_fn()} already exists.")
            return True
        else:
            return False

    def _check_seasons (self):
        if Path(f"{self._get_season_fn()}").is_file():
            self.logger.warning(f"Seasonality file {self._get_season_fn()} already exists.")
            return True
        else:
            return False

    def _get_season_fn (self):
        return self._get_train_data_fn() + "_season" + self.data_extension

    def _get_average_fn (self):
        return self._get_train_data_fn() + "_average" + self.data_extension

    def _get_trend_fn (self):
        return self._get_train_data_fn() + "_trend" + self.data_extension

    def _get_weights_fn (self):
        return f"{self._get_final_products_base_fn()}.pth"

    def _get_pics_fn (self, date):
        return f"{date.strftime(self.plot_date_strformat)}.png"

    def _check_weights (self):
        if Path(f"{self.weights_folder}{self._get_weights_fn()}").is_file():
            self.logger.warning(f"Weights file {self._get_weights_fn()} already exists.")
            return True
        else:
            return False

    def save_weights (self):
        weights_fn = self._get_weights_fn()
        self.logger.info(f"Save weights in file: {weights_fn}")
        torch.save(self.model.state_dict(), f"{self.weights_folder}{weights_fn}")

    def train (self, X_np, y_np):
        if self._check_weights():
            return
        else:
            num_workers = min(os.cpu_count(), 8)  # safe default
            # Tensorboard
            tl_logger = TensorBoardLogger(self.tl_logdir, name=self.run_base_name, version=self.run_number)
            # Normalize data (example: min-max scaling to [0, 1])
            X_np, _ = self.normalize(X_np)
            y_np, _ = self.normalize(y_np)
            dataset = WeatherDataset(X_np, y_np)
            # Split into train (90%) and test (10%)
            num_samples = X_np.shape[0]
            test_size = int(0.1 * num_samples)
            train_size = num_samples - test_size
            train_dataset, validation_dataset = random_split(dataset, [train_size, test_size])
            # Create data loaders
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=num_workers)
            validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
            # DEBUG if crazy errors show up
            # x, y = next(iter(train_dataloader))
            # x, y = x.cuda(), y.cuda()  # if using CUDA
            # out = self.model(x)
            # loss = self.model.loss(out, y)
            # try:
            #     loss.backward()
            # except Exception as e:
            #     self.logger.info("Backward error:", e)
            #     self.logger.info("Loss value:", loss)
            #     self.logger.info("out stats:", out.min(), out.max(), out.mean())
            #     self.logger.info("target stats:", y.min(), y.max(), y.mean())
            #     raise
            # Log model info
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            # Train
            trainer = L.Trainer(
                max_epochs=self.epochs,
                precision="16-mixed",
                gradient_clip_val=1.0,           # Recommended starting value (e.g., 0.5, 1.0, 5.0)
                gradient_clip_algorithm="norm",  # "norm" for clipping by norm, "value" for clipping by value
                log_every_n_steps=1,
                logger=tl_logger,
                accumulate_grad_batches=self.accumulate_grad_batches
            )
            trainer.fit(self.model, train_dataloader, validation_dataloader)
            # Save model weights
            self.save_weights()

    def test (self, X_np_test, y_np_test, date_range):
        num_workers = min(os.cpu_count(), 8)  # safe default
        # Tensorboard
        tl_logger = TensorBoardLogger(self.tl_logdir, name=self.run_base_name, version=f"test_{self.run_number}")
        # Normalize data
        X_np_test, X_scaler = self.normalize(X_np_test)
        y_np_test, y_scaler = self.normalize(y_np_test)
        test_dataset = WeatherDataset(X_np_test, y_np_test)
        # Create data loaders
        test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=num_workers, shuffle=False)
        # Load weights
        self.logger.info(f"Load weights from file: {self._get_weights_fn()}")
        self.model.load_state_dict(torch.load(f"{self.weights_folder}{self._get_weights_fn()}", map_location=self.device))
        # Test
        trainer = L.Trainer(
            max_epochs=self.epochs,
            precision="16-mixed",
            gradient_clip_val=1.0,           # Recommended starting value (e.g., 0.5, 1.0, 5.0)
            gradient_clip_algorithm="norm",  # "norm" for clipping by norm, "value" for clipping by value
            log_every_n_steps=1,
            logger=tl_logger,
            accumulate_grad_batches=self.accumulate_grad_batches
        )
        output_d = trainer.test(self.model, dataloaders=test_dataloader)
        # all_outputs = self.model.test_step_outputs
        # self.logger.debug(f"Available keys in model output: {all_outputs[-1].keys()}")
        # Diffusion
        # all_outputs = torch.cat([r[0]["x0_recon"] for r in output_d], dim=0)
        # outputs = all_outputs[-1]['x0_recon']
        inputs, targets, outputs = [], [], []
        for idx, (input, target) in enumerate(test_dataloader): # input=forecast, target=analysis + x days
            inputs.append(input[0,:,:,:].cpu())
            targets.append(target[0,:,:,:].cpu())
            outputs.append(self.model.test_preds[idx,:,:,:])
        # Denormalize
        inputs = self.denormalize(np.array(inputs), X_scaler)
        targets = self.denormalize(np.array(targets), y_scaler)
        predictions = self.denormalize(np.array(outputs), X_scaler)
        return inputs, targets, predictions

    def _is_port_in_use(self, port, host='0.0.0.0'):
        """Checks if the given port is already in use."""
        for conn in psutil.net_connections(kind='inet'):
            if conn.status == psutil.CONN_LISTEN:
                if conn.laddr.port == port:
                    # Check if the host matches or if it's listening on all interfaces
                    if host == '0.0.0.0' or conn.laddr.ip == host:
                        try:
                            # Attempt to get process name to be more specific (optional)
                            process = psutil.Process(conn.pid)
                            if "tensorboard" in process.name().lower() or \
                               "python" in process.name().lower() and "tensorboard" in " ".join(process.cmdline()).lower():
                                self.logger.warning(f"TensorBoard (PID: {conn.pid}) is already listening on port {port}.")
                                return True
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            # Process might have ended, or permission denied to check its name/cmdline
                            self.logger.info(f"Port {port} is in use by a process (PID: {conn.pid}), potentially TensorBoard.")
                            return True
                        return True # Port is in use, regardless of process name check success
        return False

    def start_tensorboard (self):
        host = "0.0.0.0"
        port = "6007"
        command = [
            "tensorboard",
            f"--logdir={self.tl_root_logdir}",
            f"--host={host}",
            f"--port={port}"
        ]
        self.logger.info(f"Starting TensorBoard with command: {' '.join(command)}")
        if self._is_port_in_use(port, host):
            self.logger.info(f"TensorBoard is already running on http://jupyterhub.juno.cmcc.scc:8000/user/jd19424/proxy/{port}/. Skipping launch.")
        else:
            self.logger.info(f"Port {port} is free. Attempting to launch TensorBoard...")
            try:
                # Use Popen to run in the background (non-blocking)
                # If you want it to block until TensorBoard is closed, use subprocess.run() instead
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL, # Redirect stdout to devnull
                    stderr=subprocess.DEVNULL, # Redirect stderr to devnull
                    preexec_fn=os.setsid) # Detach from current process group
                self.logger.info(f"TensorBoard started. Access it at http://jupyterhub.juno.cmcc.scc:8000/user/jd19424/proxy/{port}/")
            except FileNotFoundError:
                self.logger.error("'tensorboard' command not found. Make sure TensorBoard is installed and in your PATH.")
            except Exception as e:
                self.logger.error(f"An error occurred: {e}")

    def _create_cartopy_axis (
        self, fig, rows, cols, n, title, var, lon, lat, vmin_plt, vmax_plt, vcenter_plt, cmap,
        borders=False
    ):
        ax = fig.add_subplot(
            rows, cols, n,
            projection=ccrs.PlateCarree()
        )
        ax.set_title(title)
        self.logger.debug(f"Ax ({rows}, {cols}, {n}) title: {title}")
        self.logger.debug(f" lat[0], lat[{lat.shape[0]-1}]: {lat[0]}, {lat[lat.shape[0]-1]}")
        self.logger.debug(f" lon[0], lon[{lon.shape[0]-1}]: {lon[0]}, {lon[lon.shape[0]-1]}")
        # self.logger.debug(f"Plot region")
        # self.logger.debug(f"lon")
        # self.logger.debug(f"{lon}")
        # self.logger.debug(f"lat")
        # self.logger.debug(f"{lat}")
        self.logger.debug(f" vmin: {vmin_plt}, vmax: {vmax_plt}, vcenter: {vcenter_plt}")
        im = ax.pcolormesh(lon, lat,
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

    def plot_figures (self, date, inputs, targets, outputs, lonfc, latfc, lonan, latan):
        average_data_path = self._get_average_fn()
        average_d = self.get_data_from_fn(average_data_path)
        # Remove channel dimension
        average_fc = np.squeeze(average_d["forecast"], axis=1)
        average_an = np.squeeze(average_d["analysis"], axis=1)

        sample_forecast = inputs
        sample_analysis = targets
        prediction = outputs

        # Plot forecast, analysis, and prediction error
        self.logger.info(f"Plot one forecast {sample_forecast.shape}, analysis {sample_analysis.shape} and prediction {prediction.shape} in {date.strftime(self.plot_date_strformat)}.")
        fig = plt.figure(figsize=(12, 18)) # col, row

        if len(sample_forecast.shape) == 3:
            plot_sample_fc = sample_forecast[-1,:,:]
        elif len(sample_forecast.shape) == 2:
            plot_sample_fc = sample_forecast
        else:
            self.logger.error(f"Unsupported dimensions {sample_forecast.shape} for sample_forecast")
        if len(sample_analysis.shape) == 3:
            plot_sample_an = sample_analysis[-1,:,:]
        elif len(sample_forecast.shape) == 2:
            plot_sample_an = sample_analysis
        else:
            self.logger.error(f"Unsupported dimensions {sample_analysis.shape} for sample_analysis")
        if len(prediction.shape) == 3:
            plot_pred = prediction[-1,:,:]
        elif len(sample_forecast.shape) == 2:
            plot_pred = prediction
        else:
            self.logger.error(f"Unsupported dimensions {prediction.shape} for prediction")
        if len(average_an.shape) == 3:
            plot_average_an = average_an[-1,:,:]
        elif len(average_an.shape) == 2:
            plot_average_an = average_an
        else:
            self.logger.error(f"Unsupported dimensions {average_an.shape} for average analysis")
        self.logger.debug("Plot shapes")
        self.logger.debug(f" sample forecast  : {plot_sample_fc.shape}")
        self.logger.debug(f" sample analysis  : {plot_sample_an.shape}")
        self.logger.debug(f" prediction       : {plot_pred.shape}")
        self.logger.debug(f" average analysis : {plot_average_an.shape}")
        # vmin_plt = np.min([np.min(plot_sample_fc), np.min(plot_sample_an), np.min(plot_pred)])
        # vmax_plt = np.max([np.max(plot_sample_fc), np.max(plot_sample_an), np.min(plot_pred)])
        vmin_plt = np.min([np.min(plot_sample_fc), np.min(plot_sample_an)])
        vmax_plt = np.max([np.max(plot_sample_fc), np.max(plot_sample_an)])
        if self.anomaly:
            vcenter_plt = 0 if vmin_plt < 0 and vmax_plt > 0 else vmin_plt+(vmax_plt-vmin_plt)/2
            cmap = self.cmap_anomaly
        else:
            vcenter_plt = vmin_plt+(vmax_plt-vmin_plt)/2
            cmap = self.cmap
        title_details = f" {self.var_forecast}a" if self.anomaly else f" {self.var_forecast}"
        title_details += " (deseasonalized)" if self.deseason else ""
        title_details += f" at {self.levhpa} hPa (" + date.strftime(self.plot_date_strformat) + ")"

        # Forecast
        ax1 = self._create_cartopy_axis (fig, 3, 2, 3, 'Forecast' + title_details, plot_sample_fc, lonfc, latfc, vmin_plt, vmax_plt, vcenter_plt, cmap)
        # Analysis
        ax2 = self._create_cartopy_axis (fig, 3, 2, 1, 'Analysis' + title_details, plot_sample_an, lonan, latan, vmin_plt, vmax_plt, vcenter_plt, cmap)
        # Prediction
        ax3 = self._create_cartopy_axis (fig, 3, 2, 5, 'Prediction' + title_details, plot_pred, lonfc, latfc, vmin_plt, vmax_plt, vcenter_plt, cmap)
        # Average of analysis
        title_avg = f"Avg analysis {self.var_forecast} at {self.levhpa} hPa ({self.start_date.strftime(self.plot_date_strformat)} - {self.end_date.strftime(self.plot_date_strformat)})"
        if self.anomaly:
            vmin_plt = np.min(plot_average_an)
            vmax_plt = np.max(plot_average_an)
            vcenter_plt = vmin_plt+(vmax_plt-vmin_plt)/2
        ax6 = self._create_cartopy_axis (fig, 3, 2, 2, title_avg, plot_average_an, lonan, latan, vmin_plt, vmax_plt, vcenter_plt, self.cmap) # always complete field cmap

        # Error (Analysis - Forecast)
        error_fc = plot_sample_an - plot_sample_fc
        vmin_plt = np.min(error_fc)
        vmax_plt = np.max(error_fc)
        vcenter_plt = 0 if vmin_plt < 0 and vmax_plt > 0 else vmin_plt+(vmax_plt-vmin_plt)/2
        # vcenter_plt = vmin_plt+(vmax_plt-vmin_plt)/2

        # Error (Analysis - Prediction)
        error_pred = plot_sample_an - plot_pred
        # vmin_plt = np.min(error_pred)
        # vmax_plt = np.max(error_pred)
        # # vcenter_plt = vmin_plt+(vmax_plt-vmin_plt)/2
        # vcenter_plt = 0

        # Pred error
        if len(error_pred.shape) == 3:
            plot_err = error_pred[-1,:,:]
        elif len(error_pred.shape) == 2:
            plot_err = error_pred
        else:
            self.logger.error(f"Unsupported dimensions {error_pred.shape} for error")
        ax4 = self._create_cartopy_axis (fig, 3, 2, 6, 'Prediction Error' + title_details, plot_err, lonan, latan, vmin_plt, vmax_plt, vcenter_plt, self.cmap_error)

        # Forecast error
        if len(error_fc.shape) == 3:
            plot_err = error_fc[-1,:,:]
        elif len(error_fc.shape) == 2:
            plot_err = error_fc
        else:
            self.logger.error(f"Unsupported dimensions {error_fc.shape} for error")
        ax5 = self._create_cartopy_axis (fig, 3, 2, 4, 'Forecast Error' + title_details, plot_err, lonan, latan, vmin_plt, vmax_plt, vcenter_plt, self.cmap_error)

        plt.tight_layout()
        plt.savefig(self.fig_folder + self._get_pics_fn(date))
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

