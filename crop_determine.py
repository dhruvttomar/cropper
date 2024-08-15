import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the data
data = pd.read_csv(f'C:\\Users\\dhruvto\\Downloads\\archive (1)\\Crop_recommendation.csv')

# Select only the columns we need: temperature, humidity, rainfall, and label
data = data[['temperature', 'humidity', 'rainfall', 'label']]

# Splitting data into features and target
X = data[['temperature', 'humidity', 'rainfall']]
y = data['label']

# Encode categorical target variable
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Building the model
model1 = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(len(np.unique(y_encoded)), activation='softmax')  # Output layer: one neuron per class
])

# Compile the model
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model1.fit(X_train, y_train, epochs=50, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model1.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

model1.save("crop_detection.keras")

"""
def make_prediction(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data], columns=X.columns)

    # Standardize the input data
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model1.predict(input_scaled)
    predicted_class = encoder.inverse_transform([np.argmax(prediction)])

    return predicted_class[0]


# Example of new data (make sure to exclude the pH value)
new_data = {
    'N': 90,
    'P': 42,
    'K': 43,
    'temperature': 20.879744,
    'humidity': 82.002744,
    'rainfall': 202.935536
}

# Predict the crop type
predicted_crop = make_prediction(new_data)
print(f"Predicted Crop: {predicted_crop}")

"""
