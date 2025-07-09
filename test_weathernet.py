import sys
import pandas as pd
from pathlib import Path
from netsutils import WeatherUtils

def test_weathernet():
    # Init data object
    wd = WeatherUtils("weather.toml")
    # Log
    wd.setup_logger()
    wd.log_global_setup()
    wd.log_train_setup()
    # SSL stuff for cartopy
    wd.config_ssl_env(Path(sys.executable).parents[1])
    # Test
    model, inputs, targets, predictions = wd.test()
    # Plot
    date_range = pd.date_range(start=wd.test_start_date, end=wd.test_end_date, freq=wd.config[wd.source]["origin_frequency"]).to_pydatetime().tolist()
    for idx, (date, i, t, o) in enumerate(zip(date_range, inputs, targets, predictions)):
        wd.plot_figures(model, date, i, t, o)

if __name__ == "__main__":
    test_weathernet()
