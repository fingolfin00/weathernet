import sys
import pandas as pd
from pathlib import Path
from netsutils import WeatherConfig, WeatherRun

def prepare_data():
    # Init data object
    wc = WeatherConfig("weather.toml")
    # Combinations of hyperparams
    combo = wc.generate_hyper_combo()
    run0 = next(iter(combo))
    wr0 = WeatherRun(wc, combo[run0], run0, dryrun=True)
    wr0.setup_logger()
    # Prepare training data
    wr0.prepare_data("train")
    # Prepare test data
    wr0.prepare_data("test")

if __name__ == "__main__":
    prepare_data()
