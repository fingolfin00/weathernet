import sys, re
import pandas as pd
import numpy as np
from pathlib import Path
from netsutils import WeatherConfig, WeatherRun

def test_weathernet():
    # Init data object
    wc = WeatherConfig("weather.toml")
    # Combinations of hyperparams
    combo = wc.generate_hyper_combo()
    run0 = next(iter(combo))
    wr = WeatherRun(wc, combo[run0], run0)
    # Logging
    wr.setup_logger()
    wr.log_global_setup()
    wr.log_test_setup()
    # SSL stuff for cartopy
    wr.config_ssl_env(Path(sys.executable).parents[1])
    # Plotting settings
    rmse_spatial_plot_type = 'percent'
    plot_gradients = False
    vmin_plt_rmse_norm, vmax_plt_rmse_norm, vmin_plt_bias_norm, vmax_plt_bias_norm = 0, 1, -1, 1
    vmin_plt_rmse_norm_diff, vmax_plt_rmse_norm_diff, vmin_plt_bias_norm_diff, vmax_plt_bias_norm_diff = -0.5, 0.5, -1, 1
    if wr.var_forecast == 'msl':
        vmin_plt_rmse, vmax_plt_rmse, vmin_plt_bias, vmax_plt_bias = 0, 200, -60, 60
        vmin_plt_rmse_diff, vmax_plt_rmse_diff, vmin_plt_bias_diff, vmax_plt_bias_diff = -50, 50, -60, 60
        vmin_plt_rmse_norm, vmax_plt_rmse_norm, vmin_plt_bias_norm, vmax_plt_bias_norm = 0, 0.25, -0.1, 0.1
        vmin_plt_rmse_norm_diff, vmax_plt_rmse_norm_diff, vmin_plt_bias_norm_diff, vmax_plt_bias_norm_diff = -0.1, 0.1, -0.1, 0.1
    elif wr.var_forecast == 't2m':
        vmin_plt_rmse, vmax_plt_rmse, vmin_plt_bias, vmax_plt_bias = 0, 4, -3, 3
        vmin_plt_rmse_diff, vmax_plt_rmse_diff, vmin_plt_bias_diff, vmax_plt_bias_diff = -2, 2, -2, 2
        vmin_plt_rmse_norm, vmax_plt_rmse_norm, vmin_plt_bias_norm, vmax_plt_bias_norm = 0, 0.4, -0.25, 0.25
        vmin_plt_rmse_norm_diff, vmax_plt_rmse_norm_diff, vmin_plt_bias_norm_diff, vmax_plt_bias_norm_diff = -0.1, 0.1, -0.1, 0.1
    elif wr.var_forecast == 'u10' or wr.var_forecast == 'v10':
        vmin_plt_rmse, vmax_plt_rmse, vmin_plt_bias, vmax_plt_bias = 0, 4, -1, 1
        vmin_plt_rmse_diff, vmax_plt_rmse_diff, vmin_plt_bias_diff, vmax_plt_bias_diff = -0.45, 0.45, -0.6, 0.6
        vmin_plt_rmse_norm, vmax_plt_rmse_norm, vmin_plt_bias_norm, vmax_plt_bias_norm = 0, 1, -0.5, 0.5
        vmin_plt_rmse_norm_diff, vmax_plt_rmse_norm_diff, vmin_plt_bias_norm_diff, vmax_plt_bias_norm_diff = -0.4, 0.4, -0.3, 0.3
        rmse_spatial_plot_type = 'original'
    elif wr.var_forecast == 'gh':
        vmin_plt_rmse, vmax_plt_rmse, vmin_plt_bias, vmax_plt_bias = 0, 12, -4, 4
        vmin_plt_rmse_diff, vmax_plt_rmse_diff, vmin_plt_bias_diff, vmax_plt_bias_diff = -3, 3, -3, 3
        vmin_plt_rmse_norm, vmax_plt_rmse_norm, vmin_plt_bias_norm, vmax_plt_bias_norm = 0, 0.4, -0.1, 0.1
        vmin_plt_rmse_norm_diff, vmax_plt_rmse_norm_diff, vmin_plt_bias_norm_diff, vmax_plt_bias_norm_diff = -0.05, 0.05, -0.1, 0.1
    else:
        vmin_plt_rmse, vmax_plt_rmse, vmin_plt_bias, vmax_plt_bias = None, None, None, None
        vmin_plt_rmse_diff, vmax_plt_rmse_diff, vmin_plt_bias_diff, vmax_plt_bias_diff = None, None, None, None
    # for lan/lot plots only
    vmin_ps_mean, vmax_ps_mean = -23, -4
    vmin_ps_var, vmax_ps_var = -25, -4
    vmin_ps_rmse, vmax_ps_rmse = -20, -4
    # Prepare test data if necessary
    wr.prepare_data("test")
    # Load data for testing
    X_np_test, y_np_test, date_range, test_lonfc, test_latfc, test_lonan, test_latan = wr.load_data('test')
    # Test
    inputs, targets, predictions_denorm_an, predictions_denorm_fc, norm_data = wr.test(X_np_test, y_np_test, date_range)
    norm_inputs, norm_targets, norm_preds = norm_data
    wr.log_metrics(
        inputs, targets, [predictions_denorm_an, predictions_denorm_fc],
        ["analysis scaler", "forecast scaler"], test_lonfc, test_latfc
    )
    # Plot
    max_std_norm = np.max([norm_inputs.std(), norm_targets.std(), norm_preds.std()])
    wr.create_plots(
        norm_inputs, norm_targets, [norm_preds],
        ["normalized"], test_lonfc, test_latfc, test_lonan, test_latan, date_range,
        vmin_plt_rmse_norm, vmax_plt_rmse_norm, vmin_plt_bias_norm, vmax_plt_bias_norm,
        vmin_plt_rmse_norm_diff, vmax_plt_rmse_norm_diff, vmin_plt_bias_norm_diff, vmax_plt_bias_norm_diff,
        vmin_ps_mean, vmax_ps_mean, vmin_ps_var, vmax_ps_var, vmin_ps_rmse, vmax_ps_rmse,
        gradients=False, rmse_spatial_plot_type='original'
    )
    wr.create_plots(
        inputs, targets, [predictions_denorm_an, predictions_denorm_fc],
        ["analysis_scaler", "forecast_scaler"], test_lonfc, test_latfc, test_lonan, test_latan, date_range,
        vmin_plt_rmse, vmax_plt_rmse, vmin_plt_bias, vmax_plt_bias,
        vmin_plt_rmse_diff, vmax_plt_rmse_diff, vmin_plt_bias_diff, vmax_plt_bias_diff,
        vmin_ps_mean, vmax_ps_mean, vmin_ps_var, vmax_ps_var, vmin_ps_rmse, vmax_ps_rmse,
        gradients=plot_gradients, rmse_spatial_plot_type=rmse_spatial_plot_type
    )
    # for idx, (date, i, t, o) in enumerate(zip(date_range, inputs, targets, predictions_denorm_an)):
    #     wr.plot_daily_figures(date, i, t, o, test_lonfc, test_latfc, test_lonan, test_latan)

if __name__ == "__main__":
    test_weathernet()
