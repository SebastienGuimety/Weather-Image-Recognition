import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras

from keras import layers
from keras.models import Sequential, load_model

from PIL import Image
import numpy as np

# Vérifier si une image est un GIF animé
def is_animated_gif(image_path):
    try:
        with Image.open(image_path) as img:
            return img.is_animated
    except:
        return False

# Vérifier si une image est au format BMP
def is_bmp_image(image_path):
    try:
        with Image.open(image_path) as img:
            return img.format == 'BMP'
    except:
        return False

# Définition d'une fonction pour charger les données d'un répertoire donné
def load_data(data_dir):
    class_labels = os.listdir(data_dir)
    print("label test: {}".format(class_labels))
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    data = []
    labels = []

    for label in class_labels:
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            image_files = os.listdir(label_dir)
            for image_file in image_files:
                image_path = os.path.join(label_dir, image_file)
                data.append(image_path)
                labels.append(label_to_index[label])

    return data, labels



def imgIsJpeg(image_path):
    try:
        # Attempt to open the image
        with Image.open(image_path) as img:
            # Check if the image format is JPEG
            if img.format == 'JPEG':
                print("The image is already in JPEG format.")
            else:
                # Convert the image to 'RGB' mode (or 'L' mode for grayscale)
                img = img.convert('RGB')
                
                # Save the converted image as JPEG
                image_path = 'test_img/pluie_converted.jpg'
                img.save(image_path, 'JPEG')
                print(f"The image has been converted to JPEG and saved as {image_path}")
        # Proceed with image processing or prediction here
    except Exception as e:
        print(f"Error: {e}")
    
    return image_path


def preprocess_image(image_path):
    img = tf.io.read_file(image_path)

    # Après avoir lu le contenu du fichier, cette ligne décode l'image en utilisant tf.image.decode_jpeg. 
    # L'argument channels=3 spécifie que l'image doit être décodée avec 3 canaux de couleur (RVB). 
    image = tf.image.decode_jpeg(img, channels=3)
        
     # redimensionne l'image à une taille de 224 pixels de largeur et 224 pixels de hauteur.    
    # Toutes les images ont la même taille avant de les utiliser dans des modèles d'apprentissage automatique
    image = tf.image.resize(image, (224, 224))  # Redimensionnez les images à la taille souhaitée
    image = image / 255.0  # Normalisation

    # normalise l'intensité des pixels de l'image en divisant chaque valeur de pixel par 255.0. 
    # Cela réduit les valeurs de pixel de l'intervalle [0, 255] à l'intervalle [0, 1]

    # Par exemple, un pixel ayant une valeur de 127 serait normalisé en 0,5, 
    # ce qui signifie que sa couleur serait équidistante entre le noir et le blanc



    #image = Image.open(image_path)
    #image = image.resize((224, 224))  # Resize to the same dimensions as your training images
    #image = np.array(image) / 255.0  # Normalize the image
    return image