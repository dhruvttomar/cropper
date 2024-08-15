from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import requests
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the pre-trained soil classification model
soil_model = tf.keras.models.load_model('soil_classification_model.keras')

# Load the pre-trained crop detection model
crop_model = tf.keras.models.load_model('crop_detection.keras')

# Properly formatted list of crop names
crop_names = [
    'Apple', 'Banana', 'Blackgram', 'Chickpea', 'Coconut', 'Coffee',
    'Cotton', 'Grapes', 'Jute', 'Kidneybeans', 'Lentil', 'Maize',
    'Mango', 'Mothbeans', 'Mungbean', 'Muskmelon', 'Orange', 'Papaya',
    'Pigeonpeas', 'Pomegranate', 'Rice', 'Watermelon'
]


# Function to predict soil type from image
def predict_soil(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image

    prediction = soil_model.predict(img_array)
    predicted_class = np.argmax(prediction[0])

    # Return the name of the predicted soil type
    soil_types = ['Alluvial', 'Black', 'Clay', 'Red']  # Replace with actual soil types
    return soil_types[predicted_class]


# Function to get weather data using zip code
def get_weather_data(zip_code, geocode_api_key, weather_api_key):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={zip_code}&key={geocode_api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        latitude = data['results'][0]['geometry']['lat']
        longitude = data['results'][0]['geometry']['lng']

        # Weather API to get temperature, humidity, and precipitation
        weather_url = f"https://api.tomorrow.io/v4/timelines?location={latitude},{longitude}&fields=temperature,humidity,precipitationIntensity&units=metric&apikey={weather_api_key}&timesteps=current"
        weather_response = requests.get(weather_url)
        if weather_response.status_code == 200:
            weather_data = weather_response.json()
            temperature = weather_data['data']['timelines'][0]['intervals'][0]['values']['temperature']
            humidity = weather_data['data']['timelines'][0]['intervals'][0]['values']['humidity']
            precipitation = weather_data['data']['timelines'][0]['intervals'][0]['values']['precipitationIntensity']
            return temperature, humidity, precipitation
    return None, None, None


# Function to predict crop using the updated model with temperature, humidity, and precipitation
def predict_crop(temperature, humidity, precipitation):
    input_features = pd.DataFrame({
        'temperature': [temperature],
        'humidity': [humidity],
        'precipitation': [precipitation]
    })

    print( temperature,
        humidity,
        precipitation)

    print(f"Input Features before scaling: {input_features}")

    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_features)

    print(f"Scaled Input Features: {input_scaled}")

    prediction = crop_model.predict(input_scaled)
    predicted_index = np.argmax(prediction[0])

    print(f"Prediction array: {prediction[0]}")
    print(f"Predicted index: {predicted_index}")
    print(f"Available crop names: {crop_names}")

    if predicted_index < len(crop_names):
        return crop_names[predicted_index]
    else:
        return "Unknown Crop"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['soilImage']
        zip_code = request.form['zipCode']

        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)

            # Predict soil type (for display purposes only)
            predicted_soil_type = predict_soil(filepath)
            os.remove(filepath)

            # Get weather data
            geocode_api_key = '553a89a6d5444e8fb1749bca7de0e8cc'
            weather_api_key = 'e3GvuosMyCWmdXrc0hOaOKyCXKHnEaj1'
            temperature, humidity, precipitation = get_weather_data(zip_code, geocode_api_key, weather_api_key)

            if temperature is not None and humidity is not None and precipitation is not None:
                # Predict crop recommendation
                recommended_crop = predict_crop(
                    temperature,
                    humidity,
                    precipitation
                )
            else:
                recommended_crop = "Weather data not available"

            return render_template(
                'result.html',
                soil_type=predicted_soil_type,  # Display the predicted soil type
                zip_code=zip_code,
                recommended_crop=recommended_crop
            )

    return render_template('index.html')


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=False)
