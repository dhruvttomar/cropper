import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('soil_classification_model.keras')

# Dictionary mapping the predicted class index to the soil type
class_labels = {
    0: 'Alluvial',
    1: 'Black',
    2: 'Clay',
    3: 'Red'
}

def predict_soil(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])

    # Return the predicted class
    return class_labels[predicted_class]


if __name__ == "__main__":
    # Prompt the user for the image file path
    image_path = input("Please enter the path to the soil image file: ")

    # Ensure the image path is not empty
    if not image_path:
        print("No image path provided. Exiting.")
        exit(1)

    # Predict and display the soil type
    predicted_soil_type = predict_soil(image_path)
    print(f"Predicted soil type: {predicted_soil_type}")
