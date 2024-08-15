import requests

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

geocode_api_key = '553a89a6d5444e8fb1749bca7de0e8cc'
weather_api_key = 'e3GvuosMyCWmdXrc0hOaOKyCXKHnEaj1'

while True:
    zip_code = input("What is your zip code?")

    temperature, humidity, precipitation = get_weather_data(zip_code, geocode_api_key, weather_api_key)

    print(temperature, humidity, precipitation)
