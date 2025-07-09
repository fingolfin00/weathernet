from netsutils import WeatherUtils
from itertools import product

def train_weathernet():
    # Init data object
    wd = WeatherUtils("weather.toml")
    # Logging
    wd.setup_logger()
    wd.log_global_setup()
    wd.log_train_setup()
    # Train
    model = wd.train()

if __name__ == "__main__":
    train_weathernet()
