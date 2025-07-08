from netsutils import WeatherDataset, Common, WeatherUtils

def prepare_weather_data():
    # Init data object
    wd = WeatherUtils("weather.toml")
    # wd.prepare_data('train')
    wd.prepare_data('test')
    
if __name__ == "__main__":
    prepare_weather_data()
