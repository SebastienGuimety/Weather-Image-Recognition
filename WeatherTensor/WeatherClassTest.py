from PIL import Image
import numpy as np
from keras.models import load_model
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from fonctions import load_data, imgIsJpeg, preprocess_image

class WeatherClassifier:
    def __init__(self, model_path='model_final', dataset_dir='../weather/dataset'):
        self.model_path = model_path
        self.dataset_dir = dataset_dir
        self.class_labels = os.listdir(dataset_dir)
        self.label_encoder = None
        self.loaded_model = None

    def load_and_preprocess_image(self, image_path):
        # Function defined in fonctions.py, which tests if the image is in JPEG format
        # and returns a new image_path in JPEG format if it is not the case
        image_path = imgIsJpeg(image_path)

        # Preprocess the image from the function defined in fonctions.py
        image = preprocess_image(image_path)
        return image

    def load_model(self):
        # Load the pre-trained model
        self.loaded_model = load_model(self.model_path)

    def load_data_and_labels(self):
        # Load dataset and labels
        data, labels = load_data(self.dataset_dir)

        # Split the data into training, validation, and test sets
        train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
        val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)

        print("Number of training images: {}".format(len(train_data)))
        print("Test labels: {}".format(test_labels))

        # Use LabelEncoder to encode the labels into numerical values
        self.label_encoder = LabelEncoder()
        train_labels = self.label_encoder.fit_transform(train_labels)

    def predict_weather(self, image_path):
        if not self.loaded_model:
            self.load_model()

        if not self.label_encoder:
            self.load_data_and_labels()

        # Load and preprocess the image
        image = self.load_and_preprocess_image(image_path)

        # Make predictions
        predictions = self.loaded_model.predict(np.expand_dims(image, axis=0))

        # Decode the predictions (if using label encoder)
        decoded_predictions = self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))

        # Get the name of the predicted class using the index as the label
        predicted_class_index = decoded_predictions[0]

        # Get the name of the predicted class
        predicted_class_name = self.class_labels[predicted_class_index]

        return predicted_class_name

if __name__ == "__main__":
    weather_classifier = WeatherClassifier()

    # Replace 'test_img/neige2.jpg' with the path to your image
    image_path = 'test_img/neige2.jpg'

    prediction = weather_classifier.predict_weather(image_path)
    print("Predicted weather class:", prediction)
