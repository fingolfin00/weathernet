import sys
import pandas as pd
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
        inputs, targets, predictions = wr.test(X_np_test, y_np_test, date_range)
        # Plot
        date_range = pd.date_range(start=wr.test_start_date, end=wr.test_end_date, freq=wr.config[wr.source]["origin_frequency"]).to_pydatetime().tolist()
        # wr.plot_averages(inputs, targets, predictions, test_lonfc, test_latfc, test_lonan, test_latan, "°C", vmin_plt_rmse=0, vmax_plt_rmse=4, vmin_plt_bias=-3, vmax_plt_bias=3)
        # wr.plot_averages(inputs, targets, predictions, test_lonfc, test_latfc, test_lonan, test_latan, "Pa", vmin_plt_rmse=0, vmax_plt_rmse=200, vmin_plt_bias=-60, vmax_plt_bias=60)
        wr.plot_averages(inputs, targets, predictions, test_lonfc, test_latfc, test_lonan, test_latan, "m", vmin_plt_rmse=30, vmax_plt_rmse=80, vmin_plt_bias=-20, vmax_plt_bias=20)
        # for idx, (date, i, t, o) in enumerate(zip(date_range, inputs, targets, predictions)):
        #     wr.plot_figures(date, i, t, o, test_lonfc, test_latfc, test_lonan, test_latan)

if __name__ == "__main__":
    train_weathernet()
