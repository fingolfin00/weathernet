import sys
import pandas as pd
import numpy as np
from pathlib import Path
from netsutils import WeatherConfig, WeatherRun

def train_weathernet():
    # Init data object
    wc = WeatherConfig("weather.toml")
    # Combinations of hyperparams
    combo = wc.generate_hyper_combo()
    run0 = next(iter(combo))
    wr0 = WeatherRun(wc, combo[run0], run0, dryrun=True)
    wr0.setup_logger()
    # Prepare training data if necessary
    wr0.prepare_data("train")
    # Load data for training
    X_np, y_np, _, _, _, _, _ = wr0.load_data('train')
    # Prepare test data if necessary
    wr0.prepare_data("test")
    # Load data for testing
    X_np_test, y_np_test, date_range, test_lonfc, test_latfc, test_lonan, test_latan = wr0.load_data('test')
    for run, hyper in combo.items():
        wr = WeatherRun(wc, hyper, run)
        # Set plot limits
        rmse_spatial_plot_type = 'percent'
        if wr.var_forecast == 'msl':
            vmin_plt_rmse, vmax_plt_rmse, vmin_plt_bias, vmax_plt_bias = 0, 300, -60, 60
            vmin_plt_rmse_diff, vmax_plt_rmse_diff, vmin_plt_bias_diff, vmax_plt_bias_diff = -50, 50, -60, 60
        elif wr.var_forecast == 't2m':
            vmin_plt_rmse, vmax_plt_rmse, vmin_plt_bias, vmax_plt_bias = 0, 4, -3, 3
            vmin_plt_rmse_diff, vmax_plt_rmse_diff, vmin_plt_bias_diff, vmax_plt_bias_diff = -2, 2, -2, 2
        elif wr.var_forecast == 'u10' or wr.var_forecast == 'v10':
            vmin_plt_rmse, vmax_plt_rmse, vmin_plt_bias, vmax_plt_bias = 0, 4, -1, 1
            vmin_plt_rmse_diff, vmax_plt_rmse_diff, vmin_plt_bias_diff, vmax_plt_bias_diff = -0.45, 0.45, -0.6, 0.6
            rmse_spatial_plot_type = 'original'
        else:
            vmin_plt_rmse, vmax_plt_rmse, vmin_plt_bias, vmax_plt_bias = None, None, None, None
            vmin_plt_rmse_diff, vmax_plt_rmse_diff, vmin_plt_bias_diff, vmax_plt_bias_diff = None, None, None, None
        # Logging
        wr.setup_logger()
        wr.log_global_setup()
        wr.log_train_setup()
        # Train
        # wr.start_tensorboard()
        model = wr.train(X_np, y_np)
        wr.log_test_setup()
        # SSL stuff for cartopy
        wr.config_ssl_env(Path(sys.executable).parents[1])
        # Test
        inputs, targets, predictions_denorm_an, predictions_denorm_fc, norm_data = wr.test(X_np_test, y_np_test, date_range)
        norm_inputs, norm_targets, norm_preds = norm_data
        # Plot
        date_range = np.array(pd.date_range(start=wr.test_start_date, end=wr.test_end_date, freq=wr.config[wr.source]["origin_frequency"]).to_pydatetime().tolist())
        test_time = np.arange(inputs.shape[0])
        wr.plot_ps_welch( # lon, lat normalized
            norm_inputs, norm_targets, (1/0.1, 1/0.1), test_lonfc, test_latfc, axis_mean=0,
            ps_along_axes=['Meridional', 'Zonal'], ylabels=['Longitude', 'Latitude'], invert_yaxis_bool=[True, False], projection='platecarree',
            rmse_spatial_plot_type='regular', units_ps=('km','km'),
            corrected=norm_preds, normalize_power_spectrum=True, extra_info="normalized_lonlat"
        )
        wr.plot_ps_welch( # time, lon normalized
            norm_inputs, norm_targets, (12, 1/0.1), test_lonfc, date_range, axis_mean=2,
            ps_along_axes=['Zonal', 'Time'], ylabels=['Longitude', 'Time'], invert_yaxis_bool=[False, False], projection=None,
            rmse_spatial_plot_type='regular', units_ps=('hours','degrees'),
            corrected=norm_preds, normalize_power_spectrum=True, extra_info="normalized_timelon"
        )
        wr.plot_ps_welch( # time, lat normalized
            norm_inputs, norm_targets, (12, 1/0.1), test_latfc, date_range, axis_mean=3,
            ps_along_axes=['Meridional', 'Time'], ylabels=['Latitude', 'Time'], invert_yaxis_bool=[False, False], projection=None,
            rmse_spatial_plot_type='regular', units_ps=('hours','degrees'),
            corrected=norm_preds, normalize_power_spectrum=True, extra_info="normalized_timelat"
        )
        for predictions, extra_info in zip([predictions_denorm_an, predictions_denorm_fc], ['analysis_scaler', 'forecast_scaler']):
            wr.plot_averages(
                inputs, targets, predictions, test_lonfc, test_latfc, test_lonan, test_latan, extra_info=extra_info,
                vmin_plt_rmse=vmin_plt_rmse, vmax_plt_rmse=vmax_plt_rmse, vmin_plt_bias=vmin_plt_bias, vmax_plt_bias=vmax_plt_bias,
                vmin_plt_rmse_diff=vmin_plt_rmse_diff, vmax_plt_rmse_diff=vmax_plt_rmse_diff, vmin_plt_bias_diff=vmin_plt_bias_diff, vmax_plt_bias_diff=vmax_plt_bias_diff
            )
            # Plot power spectrum
            wr.plot_ps_welch( # lon, lat
                inputs, targets, (1/0.1, 1/0.1), test_lonfc, test_latfc, axis_mean=0,
                ps_along_axes=['Meridional', 'Zonal'], ylabels=['Longitude', 'Latitude'], invert_yaxis_bool=[True, False], projection='platecarree',
                rmse_spatial_plot_type=rmse_spatial_plot_type, units_ps=('km','km'),
                corrected=predictions, normalize_power_spectrum=True, extra_info=extra_info+"_lonlat",
                vmin_power_mean=-23, vmax_power_mean=-4,
                vmin_power_var=-25, vmax_power_var=-4,
                vmin_power_rmse=-20, vmax_power_rmse=-4,
            )
            wr.plot_ps_welch( # time, lon
                inputs, targets, (12, 1/0.1), test_lonfc, date_range, axis_mean=2,
                ps_along_axes=['Zonal', 'Time'], ylabels=['Longitude', 'Time'], invert_yaxis_bool=[False, False], projection=None,
                rmse_spatial_plot_type=rmse_spatial_plot_type, units_ps=('hours','degrees'),
                corrected=predictions, normalize_power_spectrum=True, extra_info=extra_info+"_timelon",
                vmin_power_mean=-23, vmax_power_mean=-4,
                vmin_power_var=-25, vmax_power_var=-4,
                vmin_power_rmse=-20, vmax_power_rmse=-4,
            )
            wr.plot_ps_welch( # time, lat
                inputs, targets, (12, 1/0.1), test_latfc, date_range, axis_mean=3,
                ps_along_axes=['Meridional', 'Time'], ylabels=['Latitude', 'Time'], invert_yaxis_bool=[False, False], projection=None,
                rmse_spatial_plot_type=rmse_spatial_plot_type, units_ps=('hours','degrees'),
                corrected=predictions, normalize_power_spectrum=True, extra_info=extra_info+"_timelat",
                vmin_power_mean=-23, vmax_power_mean=-4,
                vmin_power_var=-25, vmax_power_var=-4,
                vmin_power_rmse=-20, vmax_power_rmse=-4,
            )
            # Plot gradient ps
            gradx_inputs, grady_inputs = np.gradient(inputs, axis=-1), np.gradient(inputs, axis=-2)
            gradx_targets, grady_targets = np.gradient(targets, axis=-1), np.gradient(targets, axis=-2)
            gradx_predictions, grady_predictions = np.gradient(predictions, axis=-1), np.gradient(predictions, axis=-2)
            wr.plot_ps_welch(
                gradx_inputs, gradx_targets, (1/0.1, 1/0.1), test_lonfc, test_latfc, axis_mean=0,
                ps_along_axes=['Meridional', 'Zonal'], ylabels=['Longitude', 'Latitude'], invert_yaxis_bool=[True, False], projection='platecarree',
                rmse_spatial_plot_type=rmse_spatial_plot_type, units_ps=('km','km'),
                corrected=gradx_predictions, normalize_power_spectrum=True, extra_info=extra_info+"_gradx_lonlat",
                vmin_power_mean=-23, vmax_power_mean=-4,
                vmin_power_var=-25, vmax_power_var=-4,
                vmin_power_rmse=-20, vmax_power_rmse=-4,
            )
            wr.plot_ps_welch(
                grady_inputs, grady_targets, (1/0.1, 1/0.1), test_lonfc, test_latfc, axis_mean=0,
                ps_along_axes=['Meridional', 'Zonal'], ylabels=['Longitude', 'Latitude'], invert_yaxis_bool=[True, False], projection='platecarree',
                rmse_spatial_plot_type=rmse_spatial_plot_type, units_ps=('km', 'km'),
                corrected=grady_predictions, normalize_power_spectrum=True, extra_info=extra_info+"_grady_lonlat",
                vmin_power_mean=-23, vmax_power_mean=-4,
                vmin_power_var=-25, vmax_power_var=-4,
                vmin_power_rmse=-20, vmax_power_rmse=-4,
            )
        # for idx, (date, i, t, o) in enumerate(zip(date_range, inputs, targets, predictions)):
        #     wr.plot_figures(date, i, t, o, test_lonfc, test_latfc, test_lonan, test_latan)

if __name__ == "__main__":
    train_weathernet()
