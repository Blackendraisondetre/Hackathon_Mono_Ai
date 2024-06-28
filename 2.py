import pandas as pd

processed_file_path = 'data/processed_rainfall_data.csv'

def read_processed_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def process_daily_rainfall(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)

    daily_rainfall = data.groupby(['city_name', 'lat', 'lon']).resample('D')['rain_1h'].sum().reset_index()
    average_daily_rainfall = daily_rainfall.groupby(['city_name', 'lat', 'lon'])['rain_1h'].mean()

    return average_daily_rainfall

def predict_flood_zones(average_daily_rainfall, threshold=5):
    flood_zones = average_daily_rainfall[average_daily_rainfall > threshold]
    return flood_zones

def main(processed_file_path):
    data = read_processed_data(processed_file_path)
    if data is not None:
        average_daily_rainfall_by_location = process_daily_rainfall(data)
        print("Average daily rainfall by location:\n", average_daily_rainfall_by_location)
        flood_zones = predict_flood_zones(average_daily_rainfall_by_location)
        print("Potential flood zones:\n", flood_zones)

        total_zones = len(average_daily_rainfall_by_location)
        flood_zone_count = len(flood_zones)
        print(f"\nTotal zones: {total_zones}")
        print(f"Potential flood zones: {flood_zone_count}")
        print(f"Percentage of flood zones: {flood_zone_count / total_zones * 100:.2f}%")
    else:
        print("Failed to read processed data.")

main(processed_file_path)