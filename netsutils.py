import glob, os, sys, torch, datetime, cfgrib, time, toml, importlib, colorlog, logging, subprocess, psutil, math
from torch import nn
from torch.utils.data import random_split, Dataset, DataLoader, SubsetRandomSampler
import torch.nn.functional as F
import numpy as np
from scipy.signal import detrend, welch
import scipy.stats as stats
import xarray as xr
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
from pathlib import Path
import urllib.request, ssl, certifi
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
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

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 

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
        self.train_percent           = self.hyper_dict["train_percent"]
        self.earlystopping_patience  = self.hyper_dict["earlystopping_patience"]
        self.supervised_str = "supervised" if self.supervised else "unsupervised"
        criterion = getattr(nn, self.loss)
        self.criterion = criterion()
        self.norm = getattr(nn, self.norm_strategy)
        # Data
        self.var_forecast            = self.config["data"]["var_forecast"]
        self.var_analysis            = self.config["data"]["var_analysis"]
        self.unit_forecast           = self.config["data"]["unit_forecast"]
        self.unit_analysis           = self.config["data"]["unit_analysis"]
        self.var3d                   = self.config["data"]["var3d"]
        self.error_limit             = self.config["data"]["error_limit"]
        self.levhpa                  = self.config["data"]["levhpa"]
        self.lonini, self.lonfin     = self.config["data"]["lonini"], self.config["data"]["lonfin"]
        self.latini, self.latfin     = self.config["data"]["latini"], self.config["data"]["latfin"]
        self.anomaly                 = self.config["data"]["anomaly"]
        self.deseason                = self.config["data"]["deseason"]
        self.detrend                 = self.config["data"]["detrend"]
        if "full_domain" in self.config["data"]:
            self.full_domain         = self.config["data"]["full_domain"]
        else:
            self.full_domain         = None
        if "domain_size" in self.config["data"]:
            self.domain_size         = self.config["data"]["domain_size"] # set a square domain starting from lonini and latini, lonfin and latfin are ignored
        else:
            self.domain_size         = None
        if "interpolation_size" in self.config["data"]:
            self.interpolation_size  = self.config["data"]["interpolation_size"]
        else:
            self.interpolation_size  = self.domain_size
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
        # Visualization and plotting
        self.cmap                    = self.config["viz"]["cmap"]
        self.cmap_anomaly            = self.config["viz"]["cmap_anomaly"]
        self.cmap_error              = self.config["viz"]["cmap_error"]
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
        self.combo_base_name         = f"{self.var_forecast}-{self.var_analysis}{self.suffix}_{self.start_date.strftime(self.folder_date_strformat)}-{self.end_date.strftime(self.folder_date_strformat)}_{self.netname}_{self.source}_{self.scalername}_{self.config["global"]["combo_name_suffix"]}"
        self.run_base_name           = f"{self.loss}_{self.norm_strategy}_{self.epochs}epochs-{self.batch_size}bs-{self.learning_rate}lr"
        self.run_number              = run
        self.run_name                = self.run_number + "-" + self.run_base_name
        self.combo_root_path         = self.config["global"]["combo_root_path"]
        self.combo_base_path         = self.combo_root_path + self.combo_base_name + "/"
        if Path(self.combo_base_path).is_dir():
            combo_base_path_glob_fn        = sorted(glob.glob(self.combo_root_path + self.combo_base_name + "_*"))
            if combo_base_path_glob_fn:
                if dryrun:
                    combo_suffix = int(combo_base_path_glob_fn[-1].split('_')[-1]) + 1
                else:
                    combo_suffix = int(combo_base_path_glob_fn[-1].split('_')[-1])
                self.combo_base_path       = f"{self.combo_root_path}{self.combo_base_name}_{combo_suffix:05d}/"
            else:
                if dryrun:
                    combo_suffix = 1
                    self.combo_base_path   = f"{self.combo_root_path}{self.combo_base_name}_{combo_suffix:05d}/"
                else:
                    self.combo_base_path   = self.combo_root_path + self.combo_base_name + "/"
        self.run_path                = self.combo_base_path + self.run_name + "/"
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
            self.log_folder          = self.combo_base_path
            self.log_filename        = self.log_folder + "combo.log"
        else:
            self.log_folder          = self.run_path + "logs/"
            self.log_filename        = self.log_folder + self.run_name + ".log"
        self.tl_root_logdir          = self.combo_root_path
        self.tl_logdir               = f"{self.combo_base_path}/lightning_logs/"
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
                # ds_all_an = xr.open_dataset(
                #     analysis_fn, engine="cfgrib", indexpath="", decode_timedelta=True
                # )
                # self.logger.info("Analysis variables:")
                # self.logger.info(list(ds_all_an.data_vars))
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
                    # ds_all_fc = xr.open_dataset(
                    #     forecast_minus1h_fn, engine="cfgrib", indexpath="", decode_timedelta=True
                    # )
                    # self.logger.info("Forecast variables:")
                    # self.logger.info(list(ds_all_fc.data_vars))
                    # self.logger.warning("Average +-1h forecasts...")
                    fc_values = np.mean( np.array([ds_forecast_plus1h.variables[self.var_forecast].values, ds_forecast_minus1h.variables[self.var_forecast].values ]), axis=0 )
                else:
                    if nearest_minus1h_forecast_flag:
                        self.logger.warning("Use -1h forecast...")
                        actual_forecast_fn = forecast_minus1h_fn
                    elif nearest_plus1h_forecast_flag:
                        self.logger.warning("Use +1h forecast...")
                        actual_forecast_fn = forecast_plus1h_fn
                    else:
                        actual_forecast_fn = forecast_fn
                    ds_forecast = xr.open_dataset(
                        actual_forecast_fn, engine="cfgrib", indexpath="", decode_timedelta=True,
                        backend_kwargs={'filter_by_keys': self.grib_dict_fc}
                    )
                    # ds_all_fc = xr.open_dataset(
                    #     actual_forecast_fn, engine="cfgrib", indexpath="", decode_timedelta=True
                    # )
                    # self.logger.info("Forecast variables:")
                    # self.logger.info(list(ds_all_fc.data_vars))
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
        try:
            lev_fc_full = ds_forecast_coords['isobaricInhPa'].values
            lev_an_full = ds_analysis_coords['isobaricInhPa'].values
        except Exception as e:
            self.logger.warning(f"Exception: {e}")
            self.logger.warning(f"Level coordinates not found")
            lev_fc_full = None
            lev_an_full = None

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

        if self.full_domain:
            lonini_fc = lon_fc_full[0]
            lonini_an = lon_an_full[0]
            latini_fc = lat_fc_full[0]
            latini_an = lat_an_full[0]
            lonfin_fc = lon_fc_full[-1]
            lonfin_an = lon_an_full[-1]
            latfin_fc = lat_fc_full[-1]
            latfin_an = lat_an_full[-1]
        else:
            lonini_fc = self.lonini
            lonini_an = self.lonini
            latini_fc = self.latini
            latini_an = self.latini
            lonfin_fc = self.lonfin
            lonfin_an = self.lonfin
            latfin_fc = self.latfin
            latfin_an = self.latfin
        lonini_fc_idx = (np.abs(lon_fc_full - lonini_fc)).argmin()
        lonfin_fc_idx = (np.abs(lon_fc_full - lonfin_fc)).argmin()
        lonini_an_idx = (np.abs(lon_an_full - lonini_an)).argmin()
        lonfin_an_idx = (np.abs(lon_an_full - lonfin_an)).argmin()
        latini_fc_idx = (np.abs(lat_fc_full - latini_fc)).argmin()
        latfin_fc_idx = (np.abs(lat_fc_full - latfin_fc)).argmin()
        latini_an_idx = (np.abs(lat_an_full - latini_an)).argmin()
        latfin_an_idx = (np.abs(lat_an_full - latfin_an)).argmin()
        lev_analysis = (np.abs(lev_an_full - self.levhpa)).argmin() if lev_an_full is not None else None
        lev_forecast = (np.abs(lev_fc_full - self.levhpa)).argmin() if lev_fc_full is not None else None

        if self.domain_size:
            self.logger.info(f"Selected regular square size for analysis: {self.domain_size}x{self.domain_size}")
            self.logger.info(f"Ignoring latfin: {self.latfin} and lonfin: {self.lonfin}")
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

        lonfc = lon_fc_full[lonini_fc_idx:lonfin_fc_idx+1]
        latfc = lat_fc_full[latini_fc_idx:latfin_fc_idx+1]
        lonan = lon_an_full[lonini_an_idx:lonfin_an_idx+1]
        latan = lat_an_full[latini_an_idx:latfin_an_idx+1]

        first_step = next(iter(var_d))
        first_fc = var_d[first_step]['forecast']
        first_an = var_d[first_step]['analysis']
        first_fc_sel = first_fc[lev_forecast,latini_fc_idx:latfin_fc_idx+1,lonini_fc_idx:lonfin_fc_idx+1] if lev_forecast is not None else first_fc[latini_fc_idx:latfin_fc_idx+1,lonini_fc_idx:lonfin_fc_idx+1]
        first_an_sel = first_an[lev_analysis,latini_fc_idx:latfin_fc_idx+1,lonini_fc_idx:lonfin_fc_idx+1] if lev_analysis is not None else first_an[latini_fc_idx:latfin_fc_idx+1,lonini_fc_idx:lonfin_fc_idx+1]
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
        self.logger.info(f" lev[0], lev[{lev_fc_full.shape[0]-1}]  : {lev_fc_full[0]:.2f}, {lev_fc_full[-1]:.2f}") if lev_fc_full is not None else self.logger.info(" No levels")
        self.logger.info("==================")
        self.logger.info(f"Analysis ({self.var_analysis})")
        self.logger.info("------------------")
        self.logger.info(f" shape {first_step}: {first_an.shape}")
        self.logger.info(f" lat[0], lat[{lat_an_full.shape[0]-1}]: {lat_an_full[0]:.2f}, {lat_an_full[-1]:.2f}")
        self.logger.info(f" lon[0], lon[{lon_an_full.shape[0]-1}]: {lon_an_full[0]:.2f}, {lon_an_full[-1]:.2f}")
        self.logger.info(f" lev[0], lev[{lev_an_full.shape[0]-1}]  : {lev_an_full[0]:.2f}, {lev_an_full[-1]:.2f}") if lev_an_full is not None else self.logger.info(" No levels")

        self.logger.info("====================")
        self.logger.info(f"Selected region")
        self.logger.info("====================")
        self.logger.info(f"Forecast ({self.var_forecast})")
        self.logger.info("--------------------")
        self.logger.info(f" shape var         : {first_fc_sel.shape}")
        self.logger.info(f" shape lat         : {latfc.shape}")
        self.logger.info(f" shape lon         : {lonfc.shape}")
        self.logger.info(f" shape lev         : {lev_fc_full.shape}") if lev_fc_full is not None else self.logger.info(" No levels")
        self.logger.info(f" lat[{latini_fc_idx}], lat[{latfin_fc_idx}]: {latfc[0]:.2f}, {latfc[-1]:.2f}")
        self.logger.info(f" lon[{lonini_fc_idx}], lon[{lonfin_fc_idx}]: {lonfc[0]:.2f}, {lonfc[-1]:.2f}")
        self.logger.info(f" lev[{lev_forecast}]            : {self.levhpa}") if lev_forecast is not None else self.logger.info(" No levels")
        self.logger.info("====================")
        self.logger.info(f"Analysis ({self.var_analysis})")
        self.logger.info("--------------------")
        self.logger.info(f" shape var         : {first_an_sel.shape}")
        self.logger.info(f" shape lat         : {latan.shape}")
        self.logger.info(f" shape lon         : {lonan.shape}")
        self.logger.info(f" shape lev         : {lev_an_full.shape}") if lev_an_full is not None else self.logger.info(" No levels")
        self.logger.info(f" lat[{latini_an_idx}], lat[{latfin_an_idx}]: {latan[0]:.2f}, {latan[-1]:.2f}")
        self.logger.info(f" lon[{lonini_an_idx}], lon[{lonfin_an_idx}]: {lonan[0]:.2f}, {lonan[-1]:.2f}")
        self.logger.info(f" lev[{lev_analysis}]            : {self.levhpa}") if lev_analysis is not None else self.logger.info(" No levels")

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
                    analysis = analysis[lev_analysis,latini_an_idx:latfin_an_idx+1,lonini_an_idx:lonfin_an_idx+1]
                elif len(analysis.shape) == 2:
                    analysis = analysis[latini_an_idx:latfin_an_idx+1,lonini_an_idx:lonfin_an_idx+1]
                else:
                    self.logger.error(f"Unsupported dimensions {analysis.shape} for {self.var_analysis}")

                if len(forecast.shape) == 3:
                    forecast = forecast[lev_forecast,latini_fc_idx:latfin_fc_idx+1,lonini_fc_idx:lonfin_fc_idx+1]
                elif len(forecast.shape) == 2:
                    forecast = forecast[latini_fc_idx:latfin_fc_idx+1,lonini_fc_idx:lonfin_fc_idx+1]
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
            self.logger.info(f"Pruned timesteps where an-fc >  {self.error_limit}: {len(full_dates[(mean_errs > self.error_limit).astype(bool)])}")
            self.logger.debug(f"{full_dates[(mean_errs > self.error_limit).astype(bool)]}")
            self.logger.info(f"Pruned timesteps where an-fc < -{self.error_limit}: {len(full_dates[(mean_errs < -self.error_limit).astype(bool)])}")
            self.logger.debug(f"{full_dates[(mean_errs < -self.error_limit).astype(bool)]}")

        if type == 'train':
            if not self._check_averages():
                self.logger.info(f"Save training period average in {average_data_path}")
                average_fc = X_np.mean(axis=0, keepdims=True)
                average_an = y_np.mean(axis=0, keepdims=True)
                # self.logger.debug(f"average_fc shape: {np.shape(average_fc)}")
                average_d = {"forecast": average_fc, "analysis": average_an}
                with open(average_data_path, 'wb') as f:
                    np.savez(average_data_path, average_d, allow_pickle=True)

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
                # if self._check_averages():
                self.logger.info(f"Load training period average from {average_data_path}")
                average_d = self.get_data_from_fn(average_data_path)
                average_fc = average_d["forecast"]
                average_an = average_d["analysis"]
                # else:
                #     self.logger.info(f"Save training period average in {average_data_path}")
                #     average_fc = X_np.mean(axis=0, keepdims=True)
                #     average_an = y_np.mean(axis=0, keepdims=True)
                #     # self.logger.debug(f"average_fc shape: {np.shape(average_fc)}")
                #     average_d = {"forecast": average_fc, "analysis": average_an}
                #     with open(average_data_path, 'wb') as f:
                #         np.savez(average_data_path, average_d, allow_pickle=True)
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
            # Set seed for reproducibility
            seed = 42
            L.seed_everything(seed)
            num_workers = min(os.cpu_count(), 8)  # safe default
            self.logger.info(f"Train step workers (CPUs) in use: {num_workers}")
            # Tensorboard
            tl_logger = TensorBoardLogger(self.tl_logdir, name=self.run_base_name, version=self.run_number)
            dataset = WeatherDataset(X_np, y_np, self.Scaler, self.interpolation_size, self.logger)
            # Split into train and test based on self.train_percent
            datamodule = WeatherDataModule(dataset, train_fraction=self.train_percent, batch_size=self.batch_size, seed=seed, num_workers=num_workers)
            # num_samples = X_np.shape[0]
            # train_size = int(self.train_percent * num_samples)
            # valid_size = num_samples - train_size
            # train_dataset, validation_dataset = random_split(dataset, [train_size, valid_size])
            # Create data loaders
            # train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
            # validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
            # Initialize Callbacks
            callbacks = []
            # Model Checkpointing
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.weights_folder,
                monitor="val_loss",
                save_top_k=1,
                mode="min",
                filename=self._get_weights_fn()
            )
            callbacks.append(checkpoint_callback)
            # Early Stopping
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                patience=self.earlystopping_patience,
                verbose=True,
                mode="min"
            )
            callbacks.append(early_stop_callback)
            # Checkpointing every N epochs
            periodic_checkpoint_callback = ModelCheckpoint(
                dirpath=self.weights_folder,
                every_n_epochs=1,
                filename="checkpoint_{epoch:02d}-{val_loss:.2f}"
            )
            callbacks.append(periodic_checkpoint_callback)
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
                # precision="16-mixed",
                # gradient_clip_val=1.0,           # Recommended starting value (e.g., 0.5, 1.0, 5.0)
                # gradient_clip_algorithm="norm",  # "norm" for clipping by norm, "value" for clipping by value
                log_every_n_steps=1,
                logger=tl_logger,
                # accumulate_grad_batches=self.accumulate_grad_batches,
                callbacks=callbacks,
                # deterministic=True
            )
            # trainer.fit(self.model, train_dataloader, validation_dataloader)
            trainer.fit(self.model, datamodule=datamodule)
            # Save model weights
            # self.save_weights()

    def test (self, X_np_test, y_np_test, date_range):
        num_workers = min(os.cpu_count(), 8)  # safe default
        # Tensorboard
        # tl_logger = TensorBoardLogger(self.tl_logdir, name=self.run_base_name, version=f"test_{self.run_number}")
        self.logger.info(f"Input  mean before test: {X_np_test.mean()}, max: {X_np_test.max()}, min: {X_np_test.min()}")
        self.logger.info(f"Target mean before test: {y_np_test.mean()}, max: {y_np_test.max()}, min: {y_np_test.min()}")
        test_dataset = WeatherDataset(X_np_test, y_np_test, self.Scaler, self.interpolation_size, self.logger)
        # Create data loaders
        test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=num_workers, shuffle=False)
        # Load weights
        self.logger.info(f"Load weights from file: {self._get_weights_fn()}.ckpt")
        checkpoint = torch.load(f"{self.weights_folder}{self._get_weights_fn()}.ckpt", map_location=self.device)
        model_weights = checkpoint["state_dict"]
        self.model.load_state_dict(model_weights)
        # Test
        trainer = L.Trainer(
            max_epochs=self.epochs,
            # precision="16-mixed",
            # gradient_clip_val=1.0,           # Recommended starting value (e.g., 0.5, 1.0, 5.0)
            # gradient_clip_algorithm="norm",  # "norm" for clipping by norm, "value" for clipping by value
            log_every_n_steps=1,
            # logger=tl_logger,
            # accumulate_grad_batches=self.accumulate_grad_batches
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
        inputs = test_dataset.denormalize_forecast(np.array(inputs))
        targets = test_dataset.denormalize_analysis(np.array(targets))
        predictions = test_dataset.denormalize_analysis(np.array(outputs))
        # return inputs, targets, predictions
        return X_np_test, y_np_test, predictions

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
        self, ax, title, var, lon, lat, vmin_plt, vmax_plt, vcenter_plt, cmap, cbar_label,
        borders=False
    ):
        ax.set_title(title)
        self.logger.debug(f"Ax title: {title}")
        # self.logger.debug(f"Ax ({rows}, {cols}, {n}) title: {title}")
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
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
        if borders:
            ax.add_feature(cfeature.BORDERS)

        # cbar_orientation = "horizontal" if lat.shape[0] >= lon.shape[0] else "vertical"
        cbar_orientation = "vertical"
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05, axes_class=plt.Axes)
        cb = ax.figure.colorbar(im, cax=cax, orientation=cbar_orientation, label=cbar_label)
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

    def interpolate_coords (self, vector):
        return np.interp(
                np.linspace(0, len(vector) - 1, self.interpolation_size), # New x-coordinates (indices) spanning original range
                np.arange(len(vector)),                                   # Original x-coordinates (indices)
                vector                                                    # Original y-values (the vector itself)
            )

    @staticmethod
    def _squeeze_and_average (data):
        """Squeeze channel dimension and calculare temporal average """
        return data.squeeze(axis=1).mean(axis=0)

    @staticmethod
    def _calculate_rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2, axis=0))

    @staticmethod
    def _calculate_bias(y_true, y_pred):
        return np.mean(y_pred - y_true, axis=0)

    def _bootstrap_stats (self, inputs, targets, outputs, lon, lat):
        # Bootstrap to get distribution of RMSE and bias
        n_bootstrap = 1000
        rmse_bootstrap_pred, rmse_bootstrap_input = [], []
        bias_bootstrap_pred, bias_bootstrap_input = [], []
        rmse_bootstrap_pred_distro, rmse_bootstrap_input_distro = [], []
        bias_bootstrap_pred_distro, bias_bootstrap_input_distro = [], []
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(targets), len(targets), replace=True)
            y_true_sample = targets[indices]
            y_pred_sample = outputs[indices]
            y_input_sample = inputs[indices]
            rmse_pred = self._calculate_rmse(y_true_sample, y_pred_sample)
            bias_pred = self._calculate_bias(y_true_sample, y_pred_sample)
            rmse_input = self._calculate_rmse(y_true_sample, y_input_sample)
            bias_input = self._calculate_bias(y_true_sample, y_input_sample)
            rmse_bootstrap_pred.append(rmse_pred)
            bias_bootstrap_pred.append(bias_pred)
            rmse_bootstrap_input.append(rmse_input)
            bias_bootstrap_input.append(bias_input)
            # average on spatial dimensions
            rmse_bootstrap_pred_distro.append(self.spatial_stats_weighted(rmse_pred.squeeze(axis=0), lon, lat)[0])
            bias_bootstrap_pred_distro.append(self.spatial_stats_weighted(bias_pred.squeeze(axis=0), lon, lat)[0])
            rmse_bootstrap_input_distro.append(self.spatial_stats_weighted(rmse_input.squeeze(axis=0), lon, lat)[0])
            bias_bootstrap_input_distro.append(self.spatial_stats_weighted(bias_input.squeeze(axis=0), lon, lat)[0])
        # distributions (keep all dimensions)
        rmse_bootstrap_pred = np.array(rmse_bootstrap_pred)
        bias_bootstrap_pred = np.array(bias_bootstrap_pred)
        rmse_bootstrap_input = np.array(rmse_bootstrap_input)
        bias_bootstrap_input = np.array(bias_bootstrap_input)
        # distributions (only bootstrap n)
        rmse_bootstrap_pred_distro = np.array(rmse_bootstrap_pred_distro)
        bias_bootstrap_pred_distro = np.array(bias_bootstrap_pred_distro)
        rmse_bootstrap_input_distro = np.array(rmse_bootstrap_input_distro)
        bias_bootstrap_input_distro = np.array(bias_bootstrap_input_distro)
        # rmse_bootstrap_pred_distro = rmse_bootstrap_pred.squeeze(axis=1).mean(axis=(1,2))
        # bias_bootstrap_pred_distro = bias_bootstrap_pred.squeeze(axis=1).mean(axis=(1,2))
        # rmse_bootstrap_input_distro = rmse_bootstrap_input.squeeze(axis=1).mean(axis=(1,2))
        # bias_bootstrap_input_distro = bias_bootstrap_input.squeeze(axis=1).mean(axis=(1,2))
        # confidence intervals for scalars (average spatial dimensions)
        percentile_intervals = [2.5, 50, 97.5]
        rmse_pred_percentiles = np.percentile(rmse_bootstrap_pred_distro, percentile_intervals)
        bias_pred_percentiles = np.percentile(bias_bootstrap_pred_distro, percentile_intervals)
        rmse_input_percentiles = np.percentile(rmse_bootstrap_input_distro, percentile_intervals)
        bias_input_percentiles = np.percentile(bias_bootstrap_input_distro, percentile_intervals)
        # time averages
        rmse_bootstrap_pred_avg = self._squeeze_and_average(rmse_bootstrap_pred)
        bias_bootstrap_pred_avg = self._squeeze_and_average(bias_bootstrap_pred)
        rmse_bootstrap_input_avg = self._squeeze_and_average(rmse_bootstrap_input)
        bias_bootstrap_input_avg = self._squeeze_and_average(bias_bootstrap_input)
        # max min of time average
        rmse_bootstrap_pred_maxmin = self._maxmin_index_values(rmse_bootstrap_pred_avg)
        bias_bootstrap_pred_maxmin = self._maxmin_index_values(bias_bootstrap_pred_avg)
        rmse_bootstrap_input_maxmin = self._maxmin_index_values(rmse_bootstrap_input_avg)
        bias_bootstrap_input_maxmin = self._maxmin_index_values(bias_bootstrap_input_avg)
        self.logger.info(f"Bootstrap stats full distro: rmse in {rmse_bootstrap_input.shape}, bias in {bias_bootstrap_input.shape}, rmse out {rmse_bootstrap_pred.shape}, bias out {bias_bootstrap_pred.shape}")
        self.logger.info(f"Bootstrap stats distro: rmse in {rmse_bootstrap_input_distro.shape}, bias in {bias_bootstrap_input_distro.shape}, rmse out {rmse_bootstrap_pred_distro.shape}, bias out {bias_bootstrap_pred_distro.shape}")
        return {
            'pred':  {'rmse': {'average': rmse_bootstrap_pred_avg, 'distro': rmse_bootstrap_pred_distro, 'percentiles': rmse_pred_percentiles}  | rmse_bootstrap_pred_maxmin,
                      'bias': {'average': bias_bootstrap_pred_avg, 'distro': bias_bootstrap_pred_distro, 'percentiles': bias_pred_percentiles}  | bias_bootstrap_pred_maxmin},
            'input': {'rmse': {'average': rmse_bootstrap_input_avg, 'distro': rmse_bootstrap_input_distro, 'percentiles': rmse_input_percentiles} | rmse_bootstrap_input_maxmin,
                      'bias': {'average': bias_bootstrap_input_avg, 'distro': bias_bootstrap_input_distro, 'percentiles': bias_input_percentiles} | bias_bootstrap_input_maxmin}
        }

    def _regular_stats (self, inputs, targets, outputs):
        bias_in_full = inputs-targets
        rmse_in_regular = np.sqrt(self._squeeze_and_average(bias_in_full**2))
        bias_in_regular = self._squeeze_and_average(bias_in_full) # time average
        rmse_in_reg_maxmin = self._maxmin_index_values(rmse_in_regular)
        bias_in_reg_maxmin = self._maxmin_index_values(bias_in_regular)
        bias_out_full = outputs-targets
        rmse_out_regular = np.sqrt(self._squeeze_and_average(bias_out_full**2))
        bias_out_regular = self._squeeze_and_average(bias_out_full) # time average
        rmse_out_reg_maxmin = self._maxmin_index_values(rmse_out_regular)
        bias_out_reg_maxmin = self._maxmin_index_values(bias_out_regular)
        self.logger.info(f"Regular stats shapes: rmse in {rmse_in_regular.shape}, bias in {bias_in_regular.shape}, rmse out {rmse_out_regular.shape}, bias out {bias_out_regular.shape}")
        return {
            'pred':  {'rmse': {'average': rmse_out_regular} | rmse_out_reg_maxmin,
                      'bias': {'average': bias_out_regular} | bias_out_reg_maxmin},
            'input': {'rmse': {'average': rmse_in_regular} | rmse_in_reg_maxmin,
                      'bias': {'average': bias_in_regular} | bias_in_reg_maxmin}
        }

    @staticmethod
    def spatial_stats_weighted(data, lon, lat, mask=None):
        """
        Calculate area-weighted spatial mean and std for geophysical data.
        Parameters:
        - data: 2D array of values (lat, lon)
        - lon: 1D array of longitude values
        - lat: 1D array of latitude values  
        - mask: optional boolean mask for valid data points
        """
        # Create 2D coordinate grids
        LON, LAT = np.meshgrid(lon, lat)
        # Calculate area weights (cosine of latitude)
        weights = np.cos(np.radians(LAT)) # normalized
        # Apply mask if provided
        if mask is not None:
            data = np.ma.masked_array(data, mask=~mask)
            weights = np.ma.masked_array(weights, mask=~mask)
        # Calculate weighted mean, np.average normalize weights to 1 internally
        weighted_mean = np.average(data, weights=weights)
        # Calculate weighted standard deviation
        variance = np.average((data - weighted_mean)**2, weights=weights)
        weighted_std = np.sqrt(variance)
        return weighted_mean, weighted_std

    def _mean_spacing_m(lon, lat):
        """
        Compute mean grid spacing dx (east-west) and dy (north-south) in meters
        from 1D lon, lat vectors in degrees.
        Assumes lon,lat are 1D and correspond to array axes: lon -> x, lat -> y.
        """
        R_earth_m = 6371000.0  # Earth radius in meters
        # ensure arrays
        lon = np.asarray(lon)
        lat = np.asarray(lat)
        # degree -> rad
        deg2rad = np.pi/180.0
    
        # dy: mean lat spacing (use mean absolute diff)
        dlat_deg = np.mean(np.abs(np.diff(lat)))
        dy = R_earth_m * (dlat_deg * deg2rad)       # meters per lat-step
    
        # dx: depends on latitude (use cos(mean_lat))
        mean_lat_rad = np.deg2rad(np.mean(lat))
        dlon_deg = np.mean(np.abs(np.diff(lon)))
        dx = R_earth_m * np.cos(mean_lat_rad) * (dlon_deg * deg2rad)  # meters per lon-step
    
        return dx, dy
    
    def power_spectrum_physical(data, lon, lat, return_wavelength_km=True):
        """
        Radially averaged 2D PSD with k converted to physical units.
        - data: 2D array [ny, nx]
        - lon: 1D lon vector in degrees length nx
        - lat: 1D lat vector in degrees length ny
        Returns:
          - k_vals_cpkm: radial wavenumbers in cycles per km (1/km)
          - Pk: PSD (units: data**2; see notes)
          - lambda_km (optional): corresponding wavelength in km (1 / k)
        """
        if data.ndim > 2:
            data = data.squeeze()
        ny, nx = data.shape
        assert nx == len(lon) and ny == len(lat), "lon/lat lengths must match data shape"
    
        # physical spacings (meters)
        dx, dy = _mean_spacing_m(lon, lat)   # dx: east-west per grid cell, dy: north-south per grid cell
    
        # fftfreq with physical spacing: returns cycles per meter
        kx = np.fft.fftfreq(nx, d=dx)  # cycles / meter along x
        ky = np.fft.fftfreq(ny, d=dy)  # cycles / meter along y
        kxg, kyg = np.meshgrid(kx, ky)
        kgrid = np.sqrt(kxg**2 + kyg**2).ravel()  # cycles / meter
    
        # Fourier power (normalized by number of points to give PSD comparable across sizes)
        F = np.fft.fft2(data)
        power = (np.abs(F)**2) / (nx * ny)**2
        power = power.ravel()
    
        # radial binning up to Nyquist; convert Nyquist to cycles/m -> then to cycles/km for bins
        # compute maximum k index (in cycles/m) from sampling: nyquist_x = 1/(2*dx), nyquist_y = 1/(2*dy)
        kmax_m = min(1.0/(2*dx), 1.0/(2*dy))
        # define bins in cycles per meter; choose bin width = 1/(domain length) ~ 1/(N*dx)
        # but simpler: create bins as multiples of fundamental radial step:
        dk_m = 1.0 / (max(nx*dx, ny*dy))   # fundamental radial resolution in cycles/m
        kbins_m = np.arange(dk_m/2, kmax_m + dk_m, dk_m)
    
        # bin and compute mean PSD in each annulus
        Abins, _, _ = stats.binned_statistic(kgrid, power, statistic="mean", bins=kbins_m)
        kvals_m = 0.5 * (kbins_m[1:] + kbins_m[:-1])  # cycles / meter (for each bin)
    
        # convert to cycles / km and wavelengths in km
        kvals_cpkm = kvals_m * 1000.0  # cycles per km
        with np.errstate(divide='ignore', invalid='ignore'):
            lambda_km = 1.0 / (kvals_cpkm)   # km per cycle (wavelength)
            # where k=0 will produce inf -> leave as inf or mask later
    
        if return_wavelength_km:
            return kvals_cpkm, Abins, lambda_km
        else:
            return kvals_cpkm, Abins
    
    def spectral_scale_metrics(k_cpkm, P, return_all=True):
        """
        Compute metrics describing prevalent spatial scale(s) from a radially-averaged PSD.
        Inputs:
          - k_cpkm: 1D array of radial wavenumbers (cycles per km), shape (M,)
                    NOTE: k=0 bin (if present) should be included but handled (it yields infinite wavelength).
          - P:      1D array of PSD values for each bin (same length M). Can be raw or normalized.
                    If P are mean powers per bin, we'll treat bin widths explicitly.
        Returns:
          dict with:
            - 'k_peak' : wavenumber of maximum PSD (cycles/km)
            - 'lambda_peak_km': wavelength at peak (km) = 1/k_peak (inf if k_peak==0)
            - 'k_centroid' : spectral centroid (cycles/km) = sum(k * P * dk) / sum(P * dk)
            - 'lambda_centroid_km' : 1 / k_centroid (km) (inf if centroid==0)
            - 'lambda_median_km' : wavelength at which cumulative power reaches 50% (km)
            - 'lambda_geometric_km': geometric mean of wavelength (km)
            - 'bandwidth_k' : standard deviation of k around centroid (cycles/km)
            - 'bandwidth_lambda_km' : approximate stddev of lambda (km) computed on lambda values
            - 'total_power' : sum(P * dk)  (should equal variance if PSD normalized appropriately)
        Notes:
          - k_cpkm should be strictly non-negative and monotonic (increasing).
          - We construct dk from midpoints of k bins; if k are bin centers, ensure spacing is correct.
        """
        k = np.asarray(k_cpkm)
        P = np.asarray(P)
        if k.shape != P.shape:
            raise ValueError("k_cpkm and P must have same shape")
    
        # Build dk: approximate bin widths using midpoints
        # If k has length 1, handle trivially
        if k.size == 1:
            dk = np.array([k[0]])  # degenerate
        else:
            # assume k are bin centers: boundaries halfway between centers
            kb = np.zeros(k.size + 1)
            kb[1:-1] = 0.5 * (k[:-1] + k[1:])
            # leftmost/rightmost edges: mirror spacing
            kb[0]  = k[0] - (kb[1] - k[0])
            kb[-1] = k[-1] + (k[-1] - kb[-2])
            dk = kb[1:] - kb[:-1]
            # ensure non-negative
            dk = np.maximum(dk, 1e-16)
    
        # total power (discrete integral approximation)
        total = np.sum(P * dk)
    
        # avoid division by zero
        if total == 0:
            raise ValueError("Total spectral power is zero. Check PSD input.")
    
        # Peak
        imax = np.nanargmax(P)
        k_peak = k[imax]
        lambda_peak = np.inf if k_peak == 0 else 1.0 / k_peak
    
        # Centroid (mean k)
        k_centroid = np.sum(k * P * dk) / total
        lambda_centroid = np.inf if k_centroid == 0 else 1.0 / k_centroid
    
        # Median wavelength: find k_med such that cumulative power up to that k is 0.5
        # compute cumulative in increasing k (small scale -> large k). But median wavelength we often want
        # the wavelength where cumulative power from small k (large scales) reaches 0.5.
        # We'll compute cumulative from low-k to high-k, then find k where cumulative=0.5
        cumsum = np.cumsum(P * dk)
        frac = cumsum / total
        # ensure monotonicity/nan handling
        frac = np.nan_to_num(frac)
        idx = np.searchsorted(frac, 0.5)
        if idx == 0:
            k_median = k[0]
        elif idx >= len(k):
            k_median = k[-1]
        else:
            # linear interpolation between k[idx-1] and k[idx]
            f1, f2 = frac[idx-1], frac[idx]
            k1, k2 = k[idx-1], k[idx]
            if f2 == f1:
                k_median = k1
            else:
                t = (0.5 - f1) / (f2 - f1)
                k_median = k1 + t * (k2 - k1)
        lambda_median = np.inf if k_median == 0 else 1.0 / k_median
    
        # Geometric mean wavelength: integrate ln(lambda) weighted by P
        # ln(lambda) = -ln(k); careful with k==0 (exclude k==0 bin)
        positive = k > 0
        if np.any(positive):
            ln_lambda = -np.log(k[positive])
            geom_ln = np.sum(ln_lambda * P[positive] * dk[positive]) / np.sum(P[positive] * dk[positive])
            lambda_geom = np.exp(geom_ln)
        else:
            lambda_geom = np.inf
    
        # Spectral variance (bandwidth) in k
        k_var = np.sum(((k - k_centroid)**2) * P * dk) / total
        k_sigma = np.sqrt(k_var)
    
        # approximate lambda statistics (not exact because lambda = 1/k)
        lambda_vals = np.zeros_like(k)
        with np.errstate(divide='ignore', invalid='ignore'):
            lambda_vals = 1.0 / k
        # mask infinite (k==0)
        finite_mask = np.isfinite(lambda_vals)
        if np.any(finite_mask):
            lambda_mean = np.sum(lambda_vals[finite_mask] * P[finite_mask] * dk[finite_mask]) / np.sum(P[finite_mask] * dk[finite_mask])
            lambda_var = np.sum(( (lambda_vals[finite_mask] - lambda_mean)**2 ) * P[finite_mask] * dk[finite_mask]) / np.sum(P[finite_mask] * dk[finite_mask])
            lambda_sigma = np.sqrt(lambda_var)
        else:
            lambda_mean = np.inf
            lambda_sigma = np.inf
    
        out = {
            'k_peak': float(k_peak),
            'lambda_peak_km': float(lambda_peak),
            'k_centroid': float(k_centroid),
            'lambda_centroid_km': float(lambda_centroid),
            'lambda_median_km': float(lambda_median),
            'lambda_geometric_km': float(lambda_geom),
            'bandwidth_k': float(k_sigma),
            'bandwidth_lambda_km': float(lambda_sigma),
            'total_power': float(total),
        }
        if return_all:
            # optionally include arrays used
            out['k'] = k
            out['P'] = P
            out['dk'] = dk
            out['cumulative_fraction'] = cumsum / total
        return out


    def power_spectrum(self, data):
        """Calculate radially averaged 2D FFT power spectrum"""
        self.logger.info(f"Calculating power spectrum on data of shape {data.shape}")
        if len(data.shape) > 2:
            self.logger.warning(f"Data shape greater than 2, attempt squeezing...")
            data = data.squeeze()
        ny, nx = data.shape
        kfreqx = np.fft.fftfreq(nx) * nx
        kfreqy = np.fft.fftfreq(ny) * ny
        kx, ky = np.meshgrid(kfreqx, kfreqy)
        knrm = np.sqrt(kx**2 + ky**2).flatten()
        self.logger.debug(f"knrm: {knrm.shape}")
        fourier_amplitudes = np.abs(np.fft.fftn(data))**2
        fourier_amplitudes = fourier_amplitudes.flatten()
        self.logger.debug(f"fourier_amplitudes: {fourier_amplitudes.shape}")
        # Bin edges up to Nyquist (half the min dimension)
        kmax = min(nx, ny) // 2
        kbins = np.arange(0.5, kmax + 1, 1.)
        self.logger.debug(f"kbins: {kbins.shape}")
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                     statistic="mean", bins=kbins)
        Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
        return kvals, Abins

    def plot_spatial_ps (self, inputs):
        kvals, Abins = self.power_spectrum(inputs[:,0,:,:].mean(axis=0))
        plt.loglog(kvals, Abins)
        plt.xlabel('k')
        plt.ylabel('P(k)')
        plt.savefig(self.fig_folder + f"power_spectrum_{self.test_start_date.strftime(self.plot_date_strformat)}-{self.test_end_date.strftime(self.plot_date_strformat)}.png")

    @staticmethod
    def gc_distance(loc1, loc2, R=6378.1):
        """
        Return the great-circle distance between two points on a sphere of radius
        R, provided as (latitude, longitude) pairs, loc=(phi, lamdbda) in degrees.
        """
        haversin = lambda alpha: math.sin(alpha/2)**2
        (phi1, lambda1), (phi2, lambda2) = loc1, loc2
        d = 2 * R * math.asin(math.sqrt(haversin(math.radians(phi2-phi1)) + math.cos(math.radians(phi1))*math.cos(math.radians(phi2))*haversin(math.radians(lambda2-lambda1))))
        return d

    def spatial_powerspectrum_welch (self, data, fs):
        """
        Compute power spectra using Welch's method.
        Parameters:
        - data: 2D array of values (lat, lon)
        - fs: sampling frequency in degrees
        Returns:
        - fxx, fyy: wavenumbers
        - Pxx, Pyy: power spectral densities
        """
        from scipy.signal import welch
        # Welch spectral analysis params, noverlap < nperseg
        nperseg, noverlap, nfft = 50, 30, 100
        Pxy, Pyx = [], []
        # Compute Welch's power spectra
        for i in range(data.shape[0]):
            # all longitudes at fixed latitude for all latitudes
            fyy, Py = welch(data[i, :], fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            Pyx.append(Py[1:])
        for j in range(data.shape[1]):
            # all latitudes at fixed longitude for all longitudes
            fxx, Px = welch(data[:, j], fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            Pxy.append(Px[1:])
        self.logger.debug(f"Welch lon idx {j}: fxx {fxx}")
        self.logger.debug(f"Welch lat idx {i}: fyy {fyy}")
        return fxx[1:], fyy[1:], np.array(Pxy), np.array(Pyx)

    def _create_ps_axes (
        self, data, fs, lon, lat, axs, title,
        normalize=True, distance='km' , vmin_plt=None, vmax_plt=None, cmap='jet',
        cbar_orientation = 'vertical', cbar_label='log P',
    ):
        """Create zonal and meridional power spectrum plots using Welch's method on given axes"""
        fxx, fyy, Pxy, Pyx = self.spatial_powerspectrum_welch(data, fs)
        if normalize:
            total_px = np.sum(Pxy*fxx)
            total_py = np.sum(Pyx*fyy)
            self.logger.debug(f"PS Welch {title}: total_px: {total_px}, total_py: {total_py}")
            Pxy /= total_px
            Pyx /= total_py
        Pxy, Pyx = np.log(Pxy), np.log(Pyx)
        vmin_plt = np.min([np.min(Pxy), np.min(Pyx)]) if vmin_plt is None else vmin_plt
        vmax_plt = np.max([np.max(Pxy), np.max(Pyx)]) if vmax_plt is None else vmax_plt
        vcenter_plt = 0 if vmin_plt < 0 and vmax_plt > 0 else vmin_plt+(vmax_plt-vmin_plt)/2
        self.logger.info(f"PS Welch {title}: vmin: {vmin_plt}, vmax: {vmax_plt}, vcenter: {vcenter_plt}")
        if distance == 'km':
            kxx_km, kyy_km = np.zeros_like(fxx), np.zeros_like(fyy)
            for idx_kxx in np.arange(len(fxx)):
                for idx_kyy in np.arange(len(fyy)):
                    kxx_km[idx_kxx] = self.gc_distance((0, 1/fyy[idx_kyy]), (1/fxx[idx_kxx], 1/fyy[idx_kyy]))
                    kyy_km[idx_kyy] = self.gc_distance((1/fxx[idx_kxx], 0), (1/fxx[idx_kxx], 1/fyy[idx_kyy]))
            kxx, kyy = kxx_km, kyy_km
        else:
            kxx, kyy = 1/fxx, 1/fyy

        P, K, L = [Pxy, Pyx], [kxx, kyy], [lon, lat]
        labels = ['Meridional', 'Zonal']
        xlabels = ['Longitude', 'Latitude']
        invert_yaxis_bool = [True, False]
        ims, cbs = [], []
        for ax, label, xlabel, inverty, p, l, k in zip(axs, labels, xlabels, invert_yaxis_bool, P, L, K):
            im = ax.pcolormesh(
                k, l, p, label=label, cmap=cmap,
                norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt)
            )
            ims.append(im)
            ax.set_title(f'{label} {title}')
            ax.set_xlabel(f'Wavelength ({distance})')
            ax.set_ylabel(f'{xlabel} (degrees)')
            ax.set_xscale('log')
            ax.invert_yaxis() if inverty else None
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05, axes_class=plt.Axes)
            cb = ax.figure.colorbar(im, cax=cax, orientation=cbar_orientation, label=f"{cbar_label} ({self.unit_forecast}$^2 \\times$ degrees)")
            cbs.append(cb)
        return ims, cbs

    def _create_twin_axis (self, ax, data, coord, title, axis='y', logaxis=False, invert=True, color='k', linestyle='-'):
        ax_sec = ax.twiny() if axis == 'y' else ax.twinx()
        ax_sec.plot(data, coord, color=color, linestyle=linestyle)
        ax_sec.set_xlabel(title, color=color) if axis == 'y' else ax_sec.set_ylabel(title, color=color)
        ax_sec.tick_params(axis=axis, labelcolor=color)
        if logaxis:
            ax_sec.set_xscale('log') if axis == 'y' else ax_sec.set_yscale('log')
        if invert:
            ax_sec.invert_xaxis() if axis == 'y' else ax_sec.invert_yaxis()
        return ax_sec

    def plot_ps_welch (
        self, data, ground_truth, fs, lon, lat, corrected=None, normalize=True,
        vmin_power_mean=None, vmax_power_mean=None,
        vmin_power_var=None, vmax_power_var=None,
        vmin_power_rmse=None, vmax_power_rmse=None,
        trim=5, cmap='jet'
    ):
        data2d = data[:,0,:,:].mean(axis=0)
        var2d = data[:,0,:,:].var(axis=0)
        rmse_spatial = np.sqrt(np.mean((data - ground_truth)**2, axis=0))[0,:,:]
        if corrected is not None:
            corrected2d = corrected[:,0,:,:].mean(axis=0)
            corrected_var2d = corrected[:,0,:,:].var(axis=0)
            rmse_corrected_spatial = np.sqrt(np.mean((corrected - ground_truth)**2, axis=0))[0,:,:]
            rmse_diff_lon = rmse_spatial.mean(axis=0) - rmse_corrected_spatial.mean(axis=0)
            rmse_diff_lat = rmse_spatial.mean(axis=1) - rmse_corrected_spatial.mean(axis=1)
        date_string = f"({self.test_start_date.strftime(self.folder_date_strformat)} - {self.test_end_date.strftime(self.folder_date_strformat)})"
        nrows, ncols = 6 if corrected is None else 9, 1
        # H, W = data2d.shape[-2], data2d.shape[-1]
        ax_height = 9  # H inches per subplot
        # ax_width = ax_height * W / H // 2
        ax_width = 24  # W inches per subplot
        self.logger.info(f"Creating figure with size: {ncols * ax_width} x {nrows * ax_height}")
        plt.rcParams.update({'font.size': 22})
        fig = plt.figure(figsize=(ncols * ax_width, nrows * ax_height))
        ax1 = fig.add_subplot(nrows,ncols*2,1)
        ax2 = fig.add_subplot(nrows,ncols*2,2)
        ax3 = fig.add_subplot(nrows,ncols*2,3)
        ax4 = fig.add_subplot(nrows,ncols*2,4)
        ax5 = fig.add_subplot(nrows,ncols,3, projection=ccrs.PlateCarree())
        ax6 = fig.add_subplot(nrows,ncols,4, projection=ccrs.PlateCarree())
        if corrected is not None:
            ax7 = fig.add_subplot(nrows,ncols,5, projection=ccrs.PlateCarree())
            ax8 = fig.add_subplot(nrows,ncols,6, projection=ccrs.PlateCarree())
            ax9 = fig.add_subplot(nrows,ncols*2,13)
            ax10 = fig.add_subplot(nrows,ncols*2,14)
            ax11 = fig.add_subplot(nrows,ncols,8, projection=ccrs.PlateCarree())
            ax12 = fig.add_subplot(nrows,ncols,9, projection=ccrs.PlateCarree())
        else:
            ax9 = fig.add_subplot(nrows,ncols*2,9)
            ax10 = fig.add_subplot(nrows,ncols*2,10)
            ax11 = fig.add_subplot(nrows,ncols,6, projection=ccrs.PlateCarree())
        # Mean and variance power spectra
        self._create_ps_axes(
            data2d, fs, lon, lat, [ax1, ax2], normalize=normalize, vmin_plt=vmin_power_mean, vmax_plt=vmax_power_mean,
            title=f"Power Spectrum mean original {self.var_forecast}"
        )
        self._create_ps_axes(
            var2d, fs, lon, lat, [ax3, ax4], normalize=normalize, vmin_plt=vmin_power_var, vmax_plt=vmax_power_var,
            title=f"Power Spectrum variance original {self.var_forecast}"
        )
        if corrected is not None:
            title_plot_rmse = f"RMSE original - corrected ({self.unit_forecast})"
            ax1_sec = self._create_twin_axis(ax1, rmse_diff_lon, lon, title_plot_rmse)
            ax2_sec = self._create_twin_axis(ax2, rmse_diff_lat, lat, title_plot_rmse)
            ax3_sec = self._create_twin_axis(ax3, rmse_diff_lon, lon, title_plot_rmse)
            ax4_sec = self._create_twin_axis(ax4, rmse_diff_lat, lat, title_plot_rmse)
        vmin_data, vmax_data = (np.min([data2d.min(), corrected2d[trim:-trim,trim:-trim].min()]), np.max([data2d.max(), corrected2d[trim:-trim,trim:-trim].max()])) if corrected is not None else (data2d.min(), data2d.max())
        vcenter_data = 0 if vmin_data < 0 and vmax_data > 0 else vmin_data+(vmax_data-vmin_data)/2
        self._create_cartopy_axis(
            ax5, f"Mean original {self.var_forecast} {date_string}", data2d, lon, lat,
            vmin_data, vmax_data, vcenter_data, self.cmap, self.unit_forecast, borders=True
        )
        if corrected is not None:
            self._create_cartopy_axis(
                ax7, f"Mean corrected {self.var_forecast} {date_string}", corrected2d, lon, lat,
                vmin_data, vmax_data, vcenter_data, self.cmap, self.unit_forecast, borders=True
            )
        vmin_data, vmax_data = (np.min([var2d.min(), corrected_var2d[trim:-trim,trim:-trim].min()]), np.max([var2d.max(), corrected_var2d[trim:-trim,trim:-trim].max()])) if corrected is not None else (var2d.min(), var2d.max())
        vcenter_data = 0 if vmin_data < 0 and vmax_data > 0 else vmin_data+(vmax_data-vmin_data)/2
        self._create_cartopy_axis(
            ax6, f"Variance original {self.var_forecast} {date_string}", var2d, lon, lat,
            vmin_data, vmax_data, vcenter_data, self.cmap, self.unit_forecast, borders=True
        )
        if corrected is not None:
            self._create_cartopy_axis(
                ax8, f"Variance corrected {self.var_forecast} {date_string}", corrected_var2d, lon, lat,
                vmin_data, vmax_data, vcenter_data, self.cmap, self.unit_forecast, borders=True
            )
        # RMSE power spectrum
        self._create_ps_axes(
            rmse_spatial, fs, lon, lat, [ax9, ax10], normalize=normalize, vmin_plt=vmin_power_rmse, vmax_plt=vmax_power_rmse,
            title=f"Power Spectrum original RMSE {self.var_forecast}"
        )
        if corrected is not None:
            ax9_sec = self._create_twin_axis(ax9, rmse_diff_lon, lon, title_plot_rmse)
            ax10_sec = self._create_twin_axis(ax10, rmse_diff_lat, lat, title_plot_rmse)
        vmin_data, vmax_data = (np.min([rmse_spatial.min(), rmse_corrected_spatial[trim:-trim,trim:-trim].min()]), np.max([rmse_spatial.max(), rmse_corrected_spatial[trim:-trim,trim:-trim].max()])) if corrected is not None else (rmse_spatial.min(), rmse_spatial.max())
        vcenter_data = 0 if vmin_data < 0 and vmax_data > 0 else vmin_data+(vmax_data-vmin_data)/2
        self._create_cartopy_axis(
            ax11, f"RMSE original {self.var_forecast} {date_string}", rmse_spatial, lon, lat,
            vmin_data, vmax_data, vcenter_data, self.cmap, self.unit_forecast, borders=True
        )
        if corrected is not None:
            self._create_cartopy_axis(
                ax12, f"RMSE corrected {self.var_forecast} {date_string}", rmse_corrected_spatial, lon, lat,
                vmin_data, vmax_data, vcenter_data, self.cmap, self.unit_forecast, borders=True
            )
        plt.tight_layout()
        plt.savefig(f"{self.fig_folder}/power_spectrum_{self.var_forecast}.png")
        plt.close()

    @staticmethod
    def _maxmin_index_values (data):
        max_arg, min_arg = data.argmax(), data.argmin()
        max_coords = np.unravel_index(max_arg, data.shape)
        max_coords = tuple(coord.item() for coord in max_coords)
        min_coords = np.unravel_index(min_arg, data.shape)
        min_coords = tuple(coord.item() for coord in min_coords)
        return {
            'max': {'coords': max_coords, 'value': data.max()},
            'min': {'coords': min_coords, 'value': data.min()}
        }

    def _print_mean_std_maxmin_str (self, data, technique, data_type, metric, sigdig=2):        
        self.logger.info(f"{(metric.upper())} ({technique:>10}) {data_type:>5} mean: {data[data_type][metric]['average'].mean():.{sigdig}f} +- {data[data_type][metric]['average'].std():.{sigdig}f}, max: {data[data_type][metric]['max']['coords']}->{data[data_type][metric]['max']['value']:.{sigdig}f}, min: {data[data_type][metric]['min']['coords']}->{data[data_type][metric]['min']['value']:.{sigdig}f}")

    def calculate_error_metrics(predictions, ground_truth, lon, lat, mask=None, percentiles=[5, 50, 95]):
        """
        Calculate various error metrics and bias
        """
        predictions = np.asarray(predictions, dtype=float)
        ground_truth = np.asarray(ground_truth, dtype=float)
        if predictions.shape != ground_truth.shape:
            raise ValueError("predictions and ground_truth must have same shape")
        # Raw errors (signed differences)
        raw_errors = predictions - ground_truth
        # Absolute errors
        absolute_errors = np.abs(raw_errors)
        # Squared errors  
        squared_errors = raw_errors ** 2
    
        # Create pixelwise bias, MAE, MSE and RMSE mean maps
        bias_map = np.mean(raw_errors, axis=0)
        mae_map = np.mean(absolute_errors, axis=0)
        mse_map = np.mean(squared_errors, axis=0)
        rmse_map = np.sqrt(mse_map)
    
        # # Scalars, spatial averages of mean maps
        # bias_map_mean = spatial_stats_weighted(bias_map, lon, lat, mask=mask, use_cell_area=True)
        # mae_map_mean = spatial_stats_weighted(mae_map, lon, lat, mask=mask, use_cell_area=True)
        # mse_map_mean = spatial_stats_weighted(mse_map, lon, lat, mask=mask, use_cell_area=True)
        # rmse_map_mean = spatial_stats_weighted(rmse_map, lon, lat, mask=mask, use_cell_area=True)
        
        # Distributions, spatial averages for each sample
        bias_distro, mae_distro, mse_distro, rmse_distro = [], [], [], []
        for n in range(raw_errors.shape[0]):
            bias_spatial_avg, _ = spatial_stats_weighted(raw_errors[n,:,:], lon, lat, mask=mask, use_cell_area=True)
            mae_spatial_avg, _ = spatial_stats_weighted(absolute_errors[n,:,:], lon, lat, mask=mask, use_cell_area=True)
            mse_spatial_avg, _ = spatial_stats_weighted(squared_errors[n,:,:], lon, lat, mask=mask, use_cell_area=True)
            bias_distro.append(bias_spatial_avg)
            mae_distro.append(mae_spatial_avg)
            mse_distro.append(mse_spatial_avg)
            rmse_distro.append(np.sqrt(mse_spatial_avg))
        bias_distro = np.array(bias_distro)
        mae_distro = np.array(mae_distro)
        mse_distro = np.array(mse_distro)
        rmse_distro = np.array(rmse_distro)
        # # Means, stds, percentiles of distributions
        # bias_distro_mean, bias_distro_std = bias_distro.mean(), bias_distro.std()
        # bias_distro_percentiles = np.percentile(bias_distro, percentiles)
        # mae_distro_mean, mae_distro_std = mae_distro.mean(), mae_distro.std()
        # mae_distro_percentiles = np.percentile(mae_distro, percentiles)
        # mse_distro_mean, mse_distro_std = mse_distro.mean(), mse_distro.std()
        # mse_distro_percentiles = np.percentile(mse_distro, percentiles)
        # rmse_distro_mean, rmse_distro_std = rmse_distro.mean(), rmse_distro.std()
        # rmse_distro_percentiles = np.percentile(rmse_distro, percentiles)
        
        # Scalar RMSE
        rmse = math.sqrt(mse_distro.mean())
    
        # Distributions, bootstrap
        n_bootstrap = 1000
        bias_mean_bootstrap, mae_mean_bootstrap, mse_mean_bootstrap, rmse_mean_bootstrap = [], [], [], []
        rmse_bootstrap = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(rmse_distro), len(rmse_distro), replace=True)
            # Resample
            raw_errors_res = predictions[indices,:,:] - ground_truth[indices,:,:]
            absolute_errors_res = np.abs(raw_errors_res)
            squared_errors_res = raw_errors_res ** 2
            bias_distro_res, mae_distro_res, mse_distro_res, rmse_distro_res = [], [], [], []
            for n in range(raw_errors.shape[0]):
                bias_spatial_res_avg, _ = spatial_stats_weighted(raw_errors_res[n,:,:], lon, lat, mask=mask, use_cell_area=True)
                mae_spatial_res_avg, _ = spatial_stats_weighted(absolute_errors_res[n,:,:], lon, lat, mask=mask, use_cell_area=True)
                mse_spatial_res_avg, _ = spatial_stats_weighted(squared_errors_res[n,:,:], lon, lat, mask=mask, use_cell_area=True)
                bias_distro_res.append(bias_spatial_res_avg)
                mae_distro_res.append(mae_spatial_res_avg)
                mse_distro_res.append(mse_spatial_res_avg)
                rmse_distro_res.append(np.sqrt(mse_spatial_res_avg))
            mse_distro_res_mean = np.array(mse_distro_res).mean() # save value so we calculate only once
            bias_mean_bootstrap.append(np.array(bias_distro_res).mean())
            mae_mean_bootstrap.append(np.array(mae_distro_res).mean())
            mse_mean_bootstrap.append(mse_distro_res_mean)
            rmse_mean_bootstrap.append(np.array(rmse_distro_res).mean()) # mean of resampled 1D RMSE distribution
            rmse_bootstrap.append(math.sqrt(mse_distro_res_mean)) # scalar RMSE of resampled data
        bias_mean_bootstrap = np.array(bias_mean_bootstrap)
        mae_mean_bootstrap = np.array(mae_mean_bootstrap)
        mse_mean_bootstrap = np.array(mse_mean_bootstrap)
        rmse_mean_bootstrap = np.array(rmse_mean_bootstrap)
        rmse_bootstrap = np.array(rmse_bootstrap)
    
        return {
            'raw_errors': raw_errors, 'absolute_errors': absolute_errors, 'squared_errors': squared_errors, # 3D
            'bias_map': bias_map, 'mae_map': mae_map, 'mse_map': mse_map, 'rmse_map': rmse_map, # maps, sample mean only (spatial dimensions preserved)
            'bias_distro': bias_distro, 'mae_distro': mae_distro, 'mse_distro': mse_distro, 'rmse_distro': rmse_distro,  # distributions of spatially weighted averages (arrays of N dimension)
            'rmse': rmse, # scalar RMSE (sqrt after spatial average and sample mean)
            'bias_mean_bootstrap': bias_mean_bootstrap, 'mae_mean_bootstrap': mae_mean_bootstrap, # bootstrap distributions of spatial snd sample means
            'mse_mean_bootstrap': mse_mean_bootstrap, 'rmse_mean_bootstrap': rmse_mean_bootstrap,
            'rmse_bootstrap': rmse_bootstrap # bootstrap distribution of scalar RMSE (sqrt after spatial average and sample mean)
        }

    def pretty_print_metrics(metrics_dict, metric_str, lon, lat, mask=None, percentiles=[5, 50, 95], confidence=0.95):
        """
        Nicely print a dictionary of metrics with scalars and arrays.
        For large arrays, prints shape and summary stats (min, max, mean, percentiles).
        """
        k_max_len = len(max(metrics_dict, key=len))
        header = (
            f"{'Key':{k_max_len}s} | {'Shape':16s} | {'Mean':>10s} | {'Std':>10s} | "
            f"{'p5':>10s} | {'p50':>10s} | {'p95':>10s} | {'ci95':>20s}"
        )
        print(f"{metric_str} Comparison:")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
    
        for key, value in metrics_dict.items():
            if np.isscalar(value):
                print(f"{key:{k_max_len}s} | {'-':16s} | {value:10.4f} | {'-':10s} | {'-':10s} | {'-':10s} | {'-':10s} | {'-':20s}")
            elif isinstance(value, np.ndarray):
                if value.ndim == 0:  # scalar wrapped in array
                    v = value.item()
                    print(f"{key:20s} | {'-':16s} | {v:10.4f} | {'-':10s} | {'-':10s} | {'-':10s} | {'-':10s} | {'-':20s}")
                elif value.ndim == 1:
                    mean = value.mean()
                    p5, p50, p95 = np.percentile(value, percentiles)
                    ci = scipy_confidence_interval(value, confidence)
                    print(
                        f"{key:{k_max_len}s} | {str(value.shape):16s} | {mean:10.4f} | {value.std():10.4f} | "
                        f"{p5:10.4f} | {p50:10.4f} | {p95:10.4f} | [{ci[0]:8.4f}, {ci[1]:8.4f}]"
                    )
                elif value.ndim == 2:
                    map_mean, map_std = spatial_stats_weighted(value, lon, lat, mask=mask, use_cell_area=True)
                    print(
                        f"{key:{k_max_len}s} | {str(value.shape):16s} | {map_mean:10.4f} | {map_std:10.4f} | "
                        f"{'-':10s} | {'-':10s} | {'-':10s} | {'-':20s}"
                    )
                elif value.ndim == 3:
                    distro = []
                    for n in range(value.shape[0]):
                        spatial_avg, _ = spatial_stats_weighted(value[n,:,:], lon, lat, mask=mask, use_cell_area=True)
                        distro.append(spatial_avg)
                    distro = np.array(distro)
                    p5, p50, p95 = np.percentile(distro, percentiles)
                    print(
                        f"{key:{k_max_len}s} | {str(value.shape):16s} | {distro.mean():10.4f} | {distro.std():10.4f} | "
                        f"{p5:10.4f} | {p50:10.4f} | {p95:10.4f} | {'-':20s}"
                    )
            else:
                print(f"{key:{k_max_len}s} | {'?':16s} | {str(value):>10s}")
    
    def rel_improvement (baseline, improved, metric_name):
        improvement = baseline - improved
        improvement_mean = improvement
        baseline_mean = baseline
        if baseline.ndim >= 1:
            improvement_mean = improvement.mean()
            baseline_mean = baseline.mean()
        return {f'{metric_name}_rel_improv': improvement_mean / baseline_mean * 100, f'{metric_name}_abs_improv': improvement_mean, f'{metric_name}_improv_distro': improvement}
    
    def scipy_confidence_interval(data, confidence=0.95):
        return stats.t.interval(confidence, len(data)-1, loc=np.mean(data), scale=stats.sem(data))
        
    def print_stats (self, inputs, targets, outputs, bootstrap, regular, lonfc, latfc, lonan, latan, sigdig=2):
        self.logger.info(f"----------------------------------------")
        self.logger.info(f"Statistics (time-average):")
        self.logger.info(f"Input  mean: {inputs.mean():.{sigdig}f} +- {inputs.std():.{sigdig}f}, max: {inputs.mean(axis=0).max():.{sigdig}f}, min: {inputs.mean(axis=0).min():.{sigdig}f}")
        self.logger.info(f"Target mean: {targets.mean():.{sigdig}f} +- {targets.std():.{sigdig}f}, max: {targets.mean(axis=0).max():.{sigdig}f}, min: {targets.mean(axis=0).min():.{sigdig}f}")
        self.logger.info(f"Output mean: {outputs.mean():.{sigdig}f} +- {outputs.std():.{sigdig}f}, max: {outputs.mean(axis=0).max():.{sigdig}f}, min: {outputs.mean(axis=0).min():.{sigdig}f}")
        self._print_mean_std_maxmin_str(bootstrap, 'bootstrap', 'input', 'rmse', sigdig=sigdig)
        self._print_mean_std_maxmin_str(regular, 'regular', 'input', 'rmse', sigdig=sigdig)
        self._print_mean_std_maxmin_str(bootstrap, 'bootstrap', 'input', 'bias', sigdig=sigdig)
        self._print_mean_std_maxmin_str(regular, 'regular', 'input', 'bias', sigdig=sigdig)
        self._print_mean_std_maxmin_str(bootstrap, 'bootstrap', 'pred', 'rmse', sigdig=sigdig)
        self._print_mean_std_maxmin_str(regular, 'regular', 'pred', 'rmse', sigdig=sigdig)
        self._print_mean_std_maxmin_str(bootstrap, 'bootstrap', 'pred', 'bias', sigdig=sigdig)
        self._print_mean_std_maxmin_str(regular, 'regular', 'pred', 'bias', sigdig=sigdig)

        inputs_wmean, inputs_wstd = self.spatial_stats_weighted(self._squeeze_and_average(inputs), lonfc, latfc)
        targets_wmean, targets_wstd = self.spatial_stats_weighted(self._squeeze_and_average(targets), lonan, latan)
        outputs_wmean, outputs_wstd = self.spatial_stats_weighted(self._squeeze_and_average(outputs), lonfc, latfc)
        inputs_bs_rmse_wmean, inputs_bs_rmse_wstd = self.spatial_stats_weighted(bootstrap['input']['rmse']['average'], lonfc, latfc)
        inputs_bs_bias_wmean, inputs_bs_bias_wstd = self.spatial_stats_weighted(bootstrap['input']['bias']['average'], lonfc, latfc)
        outputs_bs_rmse_wmean, outputs_bs_rmse_wstd = self.spatial_stats_weighted(bootstrap['pred']['rmse']['average'], lonfc, latfc)
        outputs_bs_bias_wmean, outputs_bs_bias_wstd = self.spatial_stats_weighted(bootstrap['pred']['bias']['average'], lonfc, latfc)
        inputs_reg_rmse_wmean, inputs_reg_rmse_wstd = self.spatial_stats_weighted(regular['input']['rmse']['average'], lonfc, latfc)
        inputs_reg_bias_wmean, inputs_reg_bias_wstd = self.spatial_stats_weighted(regular['input']['bias']['average'], lonfc, latfc)
        outputs_reg_rmse_wmean, outputs_reg_rmse_wstd = self.spatial_stats_weighted(regular['pred']['rmse']['average'], lonfc, latfc)
        outputs_reg_bias_wmean, outputs_reg_bias_wstd = self.spatial_stats_weighted(regular['pred']['bias']['average'], lonfc, latfc)
        self.logger.info(f"----------------------------------------")
        self.logger.info(f"Area-weighted statistics (time-average):")
        self.logger.info(f"Input  weighted mean: {inputs_wmean:.{sigdig}f} +- {inputs_wstd:.{sigdig}f}")
        self.logger.info(f"Target weighted mean: {targets_wmean:.{sigdig}f} +- {targets_wstd:.{sigdig}f}")
        self.logger.info(f"Output weighted mean: {outputs_wmean:.{sigdig}f} +- {outputs_wstd:.{sigdig}f}")
        self.logger.info(f"RMSE ( bootstrap) input weighted mean: {inputs_bs_rmse_wmean:.{sigdig}f} +- {inputs_bs_rmse_wstd:.{sigdig}f}")
        self.logger.info(f"RMSE (   regular) input weighted mean: {inputs_reg_rmse_wmean:.{sigdig}f} +- {inputs_reg_rmse_wstd:.{sigdig}f}")
        self.logger.info(f"Bias ( bootstrap) input weighted mean: {inputs_bs_bias_wmean:.{sigdig}f} +- {inputs_bs_bias_wstd:.{sigdig}f}")
        self.logger.info(f"Bias (   regular) input weighted mean: {inputs_reg_bias_wmean:.{sigdig}f} +- {inputs_reg_bias_wstd:.{sigdig}f}")
        self.logger.info(f"RMSE ( bootstrap)  pred weighted mean: {outputs_bs_rmse_wmean:.{sigdig}f} +- {outputs_bs_rmse_wstd:.{sigdig}f}")
        self.logger.info(f"RMSE (   regular)  pred weighted mean: {outputs_reg_rmse_wmean:.{sigdig}f} +- {outputs_reg_rmse_wstd:.{sigdig}f}")
        self.logger.info(f"Bias ( bootstrap)  pred weighted mean: {outputs_bs_bias_wmean:.{sigdig}f} +- {outputs_bs_bias_wstd:.{sigdig}f}")
        self.logger.info(f"Bias (   regular)  pred weighted mean: {outputs_reg_bias_wmean:.{sigdig}f} +- {outputs_reg_bias_wstd:.{sigdig}f}")
        self.logger.info(f"----------------------------------------")

    def plot_averages (self, inputs, targets, outputs, lonfc, latfc, lonan, latan, vmin_plt_rmse=None, vmax_plt_rmse=None, vmin_plt_bias=None, vmax_plt_bias=None):
        self.logger.info(f"Data shapes: input {inputs.shape}, target {targets.shape}, prediction {outputs.shape}")
        bootstrap = self._bootstrap_stats(inputs, targets, outputs, lonfc, latfc)
        regular = self._regular_stats(inputs, targets, outputs)
        self.print_stats(inputs, targets, outputs, bootstrap, regular, lonfc, latfc, lonan, latan)

        plt_stats = regular
        rmse_in_avg_plot = plt_stats['input']['rmse']['average']
        rmse_out_avg_plot = plt_stats['pred']['rmse']['average']
        bias_in_avg_plot = plt_stats['input']['bias']['average']
        bias_out_avg_plot = plt_stats['pred']['bias']['average']

        nrows, ncols = 2, 2
        H, W = inputs.shape[-2], inputs.shape[-1]
        ax_height = 5  # H inches per subplot
        ax_width = ax_height * W / H
        if ax_width < 7:
            ax_width = 7
            ax_height = ax_width * H / W
        fig, axs = plt.subplots(
            nrows, ncols,
            subplot_kw={'projection': ccrs.PlateCarree()},
            figsize=(ncols * ax_width, nrows * ax_height)
        )
        axs = axs.ravel()
        # vmin_plt_rmse = np.min([np.min(rmse_in_avg_plot), np.min(rmse_out_avg_plot)])
        # vmax_plt_rmse = np.max([np.max(rmse_in_avg_plot), np.max(rmse_out_avg_plot)])
        if vmin_plt_rmse == None:
            vmin_plt_rmse = rmse_in_avg_plot.min()
        if vmax_plt_rmse == None:
            vmax_plt_rmse = rmse_in_avg_plot.max()
        vcenter_plt_rmse = 0 if vmin_plt_rmse < 0 and vmax_plt_rmse > 0 else vmin_plt_rmse+(vmax_plt_rmse-vmin_plt_rmse)/2
        # vmin_plt_bias = np.min([np.min(bias_in_avg_plot), np.min(bias_out_avg_plot)])
        # vmax_plt_bias = np.max([np.max(bias_in_avg_plot), np.max(bias_out_avg_plot)])
        if vmin_plt_bias == None:
            vmin_plt_bias = np.min(bias_in_avg_plot)
        if vmax_plt_bias == None:
            vmax_plt_bias = np.max(bias_in_avg_plot)
        vcenter_plt_bias = 0 if vmin_plt_bias < 0 and vmax_plt_bias > 0 else vmin_plt_bias+(vmax_plt_bias-vmin_plt_bias)/2

        title_details = f" {self.var_forecast}a" if self.anomaly else f" {self.var_forecast}"
        title_details += " (deseasonalized)" if self.deseason else ""
        title_details += f" at {self.levhpa} hPa" if self.var3d else ""
        title_details += f" ({self.test_start_date.strftime(self.plot_date_strformat)} - {self.test_end_date.strftime(self.plot_date_strformat)})"

        # RMSE
        self._create_cartopy_axis(axs[0], 'RMSE' + title_details, rmse_in_avg_plot, lonfc, latfc, vmin_plt_rmse, vmax_plt_rmse, vcenter_plt_rmse, self.cmap, self.unit_forecast, borders=True)
        self._create_cartopy_axis(axs[1], 'RMSE corrected' + title_details, rmse_out_avg_plot, lonfc, latfc, vmin_plt_rmse, vmax_plt_rmse, vcenter_plt_rmse, self.cmap, self.unit_forecast, borders=True)
        # bias
        self._create_cartopy_axis(axs[2], 'bias' + title_details, bias_in_avg_plot, lonfc, latfc, vmin_plt_bias, vmax_plt_bias, vcenter_plt_bias, self.cmap_error, self.unit_forecast, borders=True)
        self._create_cartopy_axis(axs[3], 'bias corrected' + title_details, bias_out_avg_plot, lonfc, latfc, vmin_plt_bias, vmax_plt_bias, vcenter_plt_bias, self.cmap_error, self.unit_forecast, borders=True)

        plt.tight_layout()
        plt.savefig(self.fig_folder + f"rmse-bias_{self.test_start_date.strftime(self.plot_date_strformat)}-{self.test_end_date.strftime(self.plot_date_strformat)}.png")
        plt.close()

        results_df = pd.DataFrame({'input': bootstrap['input']['rmse']['percentiles'], 'pred': bootstrap['pred']['rmse']['percentiles']}, index=["2.5% RMSE", "Median RMSE", "97.5% RMSE"]).T
        results_df = results_df.sort_values(by="Median RMSE")
        self.logger.info(f"{self.var_forecast} bootstrap-based RMSE uncertainty (95% CI):")
        print(results_df)
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=pd.DataFrame({'input': bootstrap['input']['rmse']['distro'], 'pred': bootstrap['pred']['rmse']['distro']}), orient="h")
        plt.title("Bootstrap RMSE Distributions")
        plt.xlabel("RMSE")
        plt.xlim(vmin_plt_rmse, vmax_plt_rmse)
        plt.tight_layout()
        plt.grid(True, axis='x')
        plt.savefig(self.fig_folder + f"boxplot_{self.test_start_date.strftime(self.plot_date_strformat)}-{self.test_end_date.strftime(self.plot_date_strformat)}.png")
        plt.close()

    def plot_figures (self, date, inputs, targets, outputs, lonfc, latfc, lonan, latan):
        average_data_path = self._get_average_fn()
        average_d = self.get_data_from_fn(average_data_path)
        average_fc = average_d["forecast"]
        average_an = average_d["analysis"]
        if self.domain_size != self.interpolation_size and not self.full_domain:
            lonfc = self.interpolate_coords(lonfc)
            latfc = self.interpolate_coords(latfc)
            lonan = self.interpolate_coords(lonan)
            latan = self.interpolate_coords(latan)
            average_fc = torch.from_numpy(average_fc).float()
            average_an = torch.from_numpy(average_an).float()
            average_fc = F.interpolate(average_fc, size=(self.interpolation_size, self.interpolation_size), mode='bilinear', align_corners=False)
            average_an = F.interpolate(average_an, size=(self.interpolation_size, self.interpolation_size), mode='bilinear', align_corners=False)
            average_fc = np.squeeze(average_fc.numpy(), axis=1)
            average_an = np.squeeze(average_an.numpy(), axis=1)
        else:
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
        title_details += f" at {self.levhpa} hPa" if self.var3d else ""
        title_details += f" (" + date.strftime(self.plot_date_strformat) + ")"

        # Forecast
        ax1 = self._create_cartopy_axis (fig, 3, 2, 3, 'Forecast' + title_details, plot_sample_fc, lonfc, latfc, vmin_plt, vmax_plt, vcenter_plt, cmap)
        # Analysis
        ax2 = self._create_cartopy_axis (fig, 3, 2, 1, 'Analysis' + title_details, plot_sample_an, lonan, latan, vmin_plt, vmax_plt, vcenter_plt, cmap)
        # Prediction
        ax3 = self._create_cartopy_axis (fig, 3, 2, 5, 'Prediction' + title_details, plot_pred, lonfc, latfc, vmin_plt, vmax_plt, vcenter_plt, cmap)
        # Average of analysis
        title_avg = f"Avg analysis {self.var_forecast}"
        title_avg += f" at {self.levhpa} hPa" if self.var3d else ""
        title_avg += f" ({self.start_date.strftime(self.plot_date_strformat)} - {self.end_date.strftime(self.plot_date_strformat)})"
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
    def __init__(self, forecasts, analyses, Scaler, size, logger):
        super().__init__()
        self.logger = logger
        self.Scaler = Scaler
        self.forecasts = forecasts
        self.analyses = analyses
        # Interpolate if provided size is different from original size
        if size:
            if size != self.forecasts.shape[-2] or size != self.forecasts.shape[-1]:
                # Convert to PyTorch tensors (shape: [num_samples, 1, H, W])
                self.forecasts = torch.from_numpy(self.forecasts).float()
                self.logger.info(f"Interpolate forecasts to {size}x{size}") 
                self.forecasts = F.interpolate(self.forecasts, size=(size, size), mode='bilinear', align_corners=False)
                self.forecasts = self.forecasts.numpy()
            if size != self.analyses.shape[-2] or size != self.analyses.shape[-1]:
                # Convert to PyTorch tensors (shape: [num_samples, 1, H, W])
                self.analyses = torch.from_numpy(self.analyses).float()
                self.logger.info(f"Interpolate analyses to {size}x{size}")
                self.analyses = F.interpolate(self.analyses, size=(size, size), mode='bilinear', align_corners=False)
                self.analyses = self.analyses.numpy()
            # Resize y to match model output
            if self.analyses.shape[-2] != self.forecasts.shape[-2] or self.analyses.shape[-1] != self.forecasts.shape[-1]:
                # Convert to PyTorch tensors (shape: [num_samples, 1, H, W])
                self.forecasts = torch.from_numpy(self.forecasts).float()
                self.analyses = torch.from_numpy(self.analyses).float()
                self.forecasts = F.interpolate(
                    self.forecasts, 
                    size=(self.analyses.shape[-2], self.analyses.shape[-1]), 
                    mode='bilinear'  # or 'bicubic' for higher precision
                )
                self.forecasts = self.forecasts.numpy()
                self.analyses = self.analyses.numpy()
        # Normalize data
        self.forecasts, self.fc_scaler = self._normalize(self.forecasts)
        self.analyses, self.an_scaler = self._normalize(self.analyses)
        self.forecasts = torch.from_numpy(self.forecasts).float()
        self.analyses = torch.from_numpy(self.analyses).float()
        self.logger.info(f"Forecast tensor shape: {self.forecasts.shape}")
        self.logger.info(f"Analysis tensor shape: {self.analyses.shape}")

    def _normalize (self, data):
        # Remove channel dimension (channel number C is always 1)
        N, C, H, W = np.shape(data)
        self.logger.debug(f"Data before normalization: {np.shape(data)}")
        data = np.squeeze(data, axis=1)
        # Reshape along samples N
        data = data.reshape(N, -1)
        self.logger.debug(f"Reshaped data before normalization: {np.shape(data)}")
        scaler = self.Scaler().fit(data)
        scaled_data = scaler.transform(data)
        self.logger.debug(f"Data after normalization: {np.shape(scaled_data)}")
        scaled_data = scaled_data.reshape(N, C, H, W)
        self.logger.debug(f"Reshaped data after normalization: {np.shape(scaled_data)}")
        return scaled_data, scaler

    def _denormalize (self, scaled_data, scaler):
        self.logger.debug(f"Data before denormalization: {np.shape(scaled_data)}")
        N, C, H, W = np.shape(scaled_data)
        scaled_data = np.squeeze(scaled_data, axis=1)
        # Reshape along samples N
        scaled_data = scaled_data.reshape(N, -1)
        self.logger.debug(f"Reshaped data before denormalization: {np.shape(scaled_data)}")
        data = scaler.inverse_transform(scaled_data)
        data = data.reshape(N, C, H, W)
        return data

    def denormalize_forecast (self, fc_scaled_data):
        return self._denormalize(fc_scaled_data, self.fc_scaler)

    def denormalize_analysis (self, an_scaled_data):
        return self._denormalize(an_scaled_data, self.an_scaler)

    def __len__(self):
        return len(self.forecasts) # assume forecast and analysis have the same N

    def __getitem__(self, idx):
        return self.forecasts[idx], self.analyses[idx]


class WeatherDataModule(L.LightningDataModule):
    def __init__(self, dataset, train_fraction=0.9, batch_size=32, seed=42, num_workers=0):
        super().__init__()
        self.dataset = dataset
        self.train_fraction = train_fraction
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

    def setup(self, stage=None):
        # initial split
        self._resplit()

    def _resplit(self):
        torch.manual_seed(self.seed)

        num_samples = len(self.dataset)
        indices = torch.randperm(num_samples)
        subset_size = int(num_samples * self.train_fraction)

        train_indices = indices[:subset_size]
        valid_indices = indices[subset_size:]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        self._train_dl = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            sampler=train_sampler,  
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self._val_dl = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            sampler=valid_sampler,  
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._train_dl

    def val_dataloader(self):
        return self._val_dl

    def on_train_epoch_start(self):
        # re-split at every epoch
        self._resplit()
