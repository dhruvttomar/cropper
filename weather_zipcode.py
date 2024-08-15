import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def get_coordinates(zip_code, geocode_api_key):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={zip_code}&key={geocode_api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        latitude = data['results'][0]['geometry']['lat']
        longitude = data['results'][0]['geometry']['lng']
        return latitude, longitude
    else:
        return None, None

def get_weather_data(latitude, longitude, weather_api_key):
    url = f"https://api.tomorrow.io/v4/timelines?location={latitude},{longitude}&fields=temperature,humidity,precipitationIntensity&units=metric&apikey={weather_api_key}&timesteps=current"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temperature = data['data']['timelines'][0]['intervals'][0]['values']['temperature']
        humidity = data['data']['timelines'][0]['intervals'][0]['values']['humidity']
        precipitation = data['data']['timelines'][0]['intervals'][0]['values']['precipitationIntensity']
        return temperature, humidity, precipitation
    else:
        return "Failed to retrieve data", "Failed to retrieve data", "Failed to retrieve data"

# Example usage
geocode_api_key = '553a89a6d5444e8fb1749bca7de0e8cc'
weather_api_key = 'e3GvuosMyCWmdXrc0hOaOKyCXKHnEaj1'
zip_code = input("What is your zip code?")  # Example ZIP code


latitude, longitude = get_coordinates(zip_code, geocode_api_key)
if latitude and longitude:
    temperature, humidity, precipitation = get_weather_data(latitude, longitude, weather_api_key)
    print(f"The current temperature in ZIP code {zip_code} is {temperature}°C")
    print(f"The current humidity is {humidity}%")
    print(f"The current precipitation intensity is {precipitation} mm/hr")
else:
    print("Failed to geocode the ZIP code")

model = tf.keras.models.load_model('crop_detection.keras')

