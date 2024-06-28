import pandas as pd
from sklearn.linear_model import LinearRegression

file_path = 'data/502acf0c3bfbe29dd8496a42634e85c7.csv'

def read_rainfall_data(file_path):
    try:
        data = pd.read_csv(file_path, usecols=['dt', 'lat', 'lon', 'rain_1h', 'city_name'])
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def fill_missing_values(data):
    train_data = data.dropna(subset=['rain_1h'])
    predict_data = data[data['rain_1h'].isnull()]

    X_train = train_data[['lat', 'lon']]
    y_train = train_data['rain_1h']
    X_predict = predict_data[['lat', 'lon']]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predicted_values = model.predict(X_predict)
    data.loc[data['rain_1h'].isnull(), 'rain_1h'] = predicted_values

    return data

def validate_data(data):
    required_columns = ['dt', 'lat', 'lon', 'rain_1h', 'city_name']
    if not all(col in data.columns for col in required_columns):
        print("Data is missing required columns.")
        return False
    return True

def process_data(data):
    data['timestamp'] = pd.to_datetime(data['dt'], unit='s')
    data.set_index('timestamp', inplace=True)
    average_rainfall_by_city = data.groupby('city_name')['rain_1h'].mean()
    return average_rainfall_by_city

def main(file_path):
    data = read_rainfall_data(file_path)
    if data is not None:
        if validate_data(data):
            print("Initial dataset:")
            print(data)
            data = fill_missing_values(data)
            print("Dataset after filling missing values:")
            print(data)
            processed_data = process_data(data)
            print("Average rainfall by city:\n", processed_data)
        else:
            print("Data validation failed.")
    else:
        print("Failed to read data.")


main(file_path)