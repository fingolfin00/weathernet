from netsutils import WeatherConfig, WeatherRun

def train_weathernet():
    # Init data object
    wc = WeatherConfig("weather.toml")
    # Combinations of hyperparams
    combo = wc.generate_hyper_combo()
    for run, hyper in combo.items():
        wr = WeatherRun(wc, hyper, run)
        # Logging
        wr.setup_logger()
        wr.log_global_setup()
        wr.log_train_setup()
        # Train
        # wr.start_tensorboard()
        model = wr.train()
        wr.log_train_setup()
        # SSL stuff for cartopy
        wr.config_ssl_env(Path(sys.executable).parents[1])
        # Test
        model, inputs, targets, predictions = wr.test()
        # Plot
        date_range = pd.date_range(start=wr.test_start_date, end=wr.test_end_date, freq=wr.config[wr.source]["origin_frequency"]).to_pydatetime().tolist()
        for idx, (date, i, t, o) in enumerate(zip(date_range, inputs, targets, predictions)):
            wr.plot_figures(model, date, i, t, o)

if __name__ == "__main__":
    train_weathernet()
