from PIL import Image
import numpy as np
from keras.models import Sequential, load_model

import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from keras import layers

from fonctions import load_data, imgIsJpeg, preprocess_image

def predictions(loaded_model, image, label_encoder, class_labels):
    # Make predictions
    predictions = loaded_model.predict(np.expand_dims(image, axis=0))

    # Decode the predictions (if using label encoder)
    decoded_predictions = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    # Get the name of the predicted class using the index as the label
    predicted_class_index = decoded_predictions[0]

    # Get the name of the predicted class
    predicted_class_name = class_labels[predicted_class_index]

    print("Predicted class name:", predicted_class_name)


dir = '../weather/dataset'

class_labels = os.listdir(dir)
data, labels = load_data(dir)

# Divisez les données en ensembles d'entraînement, de validation et de test
train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)


print("Nombre d'images d'entraînement: {}".format(len(train_data)))
print("label test: {}".format(test_labels))

# Utilisez LabelEncoder pour encoder les étiquettes en valeurs numériques
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)

label_encoder.fit(train_labels)
loaded_model = load_model("model_final")

# Load and preprocess the image
image_path = 'test_img/neige2.jpg'  # Replace with the path to your image


# Fonction defined in fonctions.py, whiich test if the image is in JPEG format
# and return a new image_path in JPEG format if it is not the case
image_path = imgIsJpeg(image_path)

# Preprocess the image from the fonction defined in fonctions.py
image = preprocess_image(image_path)

predictions(loaded_model, image, label_encoder, class_labels)


# Pour des evolutions future, on pourrait penser a faire un dispositif sur une maison d'un particulier
# Imaginons que des que le temps change, des capteurs envoient des images a un serveur qui les analyses
# et qui envoie un message a l'utilisateur pour lui dire de fermer les volets, de rentrer les plantes, etc...
# Ou bien de fermer automatiquement les volets, etc...