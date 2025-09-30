import sys
import pandas as pd
from pathlib import Path
from netsutils import WeatherConfig, WeatherRun

def test_weathernet():
    # Init data object
    wc = WeatherConfig("weather.toml")
    # Combinations of hyperparams
    combo = wc.generate_hyper_combo()
    run0 = next(iter(combo))
    wr = WeatherRun(wc, combo[run0], run0)
    wr.setup_logger()
    # Prepare test data if necessary
    wr.prepare_data("test")
    # Load data for testing
    X_np_test, y_np_test, date_range, test_lonfc, test_latfc, test_lonan, test_latan = wr.load_data('test')
    # Logging
    wr.setup_logger()
    wr.log_global_setup()
    wr.log_test_setup()
    # SSL stuff for cartopy
    wr.config_ssl_env(Path(sys.executable).parents[1])
    # Test
    inputs, targets, predictions = wr.test(X_np_test, y_np_test, date_range)
    # Plot
    date_range = pd.date_range(start=wr.test_start_date, end=wr.test_end_date, freq=wr.config[wr.source]["origin_frequency"]).to_pydatetime().tolist()
    wr.plot_spatial_ps(inputs)
    # wr.plot_averages(inputs, targets, predictions, test_lonfc, test_latfc, test_lonan, test_latan, vmin_plt_rmse=0, vmax_plt_rmse=300, vmin_plt_bias=-60, vmax_plt_bias=60) # msl
    wr.plot_averages(inputs, targets, predictions, test_lonfc, test_latfc, test_lonan, test_latan, vmin_plt_rmse=0, vmax_plt_rmse=4, vmin_plt_bias=-3, vmax_plt_bias=3) # t2m
    # Plotting the power spectrum
    wr.plot_ps_welch(
        X_np_test, y_np_test, 1/0.1, test_lonfc, test_latfc, corrected=predictions, normalize=True,
        vmin_power_mean=-23, vmax_power_mean=-4,
        vmin_power_var=-25, vmax_power_var=-4,
        vmin_power_rmse=-20, vmax_power_rmse=-4,
    )
    # wr.plot_averages(inputs, targets, predictions, test_lonfc, test_latfc, test_lonan, test_latan, "m", vmin_plt_rmse=30, vmax_plt_rmse=80, vmin_plt_bias=-20, vmax_plt_bias=20) # gh
    # for idx, (date, i, t, o) in enumerate(zip(date_range, inputs, targets, predictions)):
    #     wr.plot_figures(date, i, t, o, test_lonfc, test_latfc, test_lonan, test_latan)

if __name__ == "__main__":
    test_weathernet()
