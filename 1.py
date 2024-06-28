import pandas as pd
from sklearn.linear_model import LinearRegression

file_path = 'data/502acf0c3bfbe29dd8496a42634e85c7.csv'
output_file_path = 'data/processed_rainfall_data.csv'


def read_rainfall_data(file_path):
    try:
        data = pd.read_csv(file_path, usecols=['dt', 'lat', 'lon', 'rain_1h', 'city_name', 'weather_main'])
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def fill_missing_values(data):
    rain_weather_main = ['Rain', 'Drizzle', 'Thunderstorm']  # Weather indicating rain

    models = {}
    for weather in rain_weather_main:
        train_data = data.dropna(subset=['rain_1h'])
        weather_data = train_data[train_data['weather_main'] == weather]

        X_train = weather_data[['lat', 'lon']]
        y_train = weather_data['rain_1h']

        model = LinearRegression()
        model.fit(X_train, y_train)
        models[weather] = model

    predict_data = data[data['rain_1h'].isnull()]

    for weather in rain_weather_main:
        rain_predict_data = predict_data[predict_data['weather_main'] == weather]
        if not rain_predict_data.empty:
            model = models[weather]
            rain_predicted_values = model.predict(rain_predict_data[['lat', 'lon']])
            data.loc[rain_predict_data.index, 'rain_1h'] = rain_predicted_values

    no_rain_predict_data = predict_data[~predict_data['weather_main'].isin(rain_weather_main)]
    data.loc[no_rain_predict_data.index, 'rain_1h'] = 0

    return data


def validate_data(data):
    required_columns = ['dt', 'lat', 'lon', 'rain_1h', 'city_name', 'weather_main']
    if not all(col in data.columns for col in required_columns):
        print("Data is missing required columns.")
        return False
    return True


def process_data(data):
    data['timestamp'] = pd.to_datetime(data['dt'], unit='s')
    data.set_index('timestamp', inplace=True)
    return data


def save_processed_data(data, output_file_path):
    try:
        data.to_csv(output_file_path)
        print(f"Processed data saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving processed data: {e}")


def main(file_path, output_file_path):
    data = read_rainfall_data(file_path)
    if data is not None:
        if validate_data(data):
            print("Initial dataset:")
            print(data)
            data = fill_missing_values(data)
            print("Dataset after filling missing values:")
            print(data)
            processed_data = process_data(data)
            save_processed_data(processed_data, output_file_path)
        else:
            print("Data validation failed.")
    else:
        print("Failed to read data.")


main(file_path, output_file_path)