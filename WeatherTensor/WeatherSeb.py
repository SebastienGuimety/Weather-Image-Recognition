import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from keras import layers
import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint

from fonctions import load_data, imgIsJpeg, is_animated_gif, is_bmp_image
#Chemin dossier

dir = '../weather/dataset'

data, labels = load_data(dir)

#Log_dir variable for tensorboard
log_dir = "logsTensor" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

# Divisez les données en ensembles d'entraînement, de validation et de test
# On divise les données en 80% pour l'entraînement, 10% pour la validation et 10% pour le test
# random_state = 42 pour que les résultats soient reproductibles ( This helps in verifying the output)
train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)

num_train_images = len(train_data)
num_val_images = len(val_data)
num_test_images = len(test_data)

print("Nombre d'images d'entraînement:", num_train_images)
print("Nombre d'images de validation:", num_val_images)
print("Nombre d'images de test:", num_test_images)


print("Nombre d'images d'entraînement: {}".format(len(train_data)))
print("label test: {}".format(test_labels))


# Divisez les données en ensembles d'entraînement, de validation et de test
#train_data, temp_data, train_labels, temp_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
#val_data, test_data, val_labels, test_labels = train_test_split(temp_data, test_labels, test_size=0.5, random_state=42)

# Utilisez LabelEncoder pour encoder les étiquettes en valeurs numériques
# Convert categorical variables into numerical format
# LabelEncoder encode les étiquettes en valeurs numériques de 0 à n_classes-1
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
val_labels = label_encoder.transform(val_labels)
test_labels = label_encoder.transform(test_labels)

# Exemple de prétraitement d'images avec TensorFlow (redimensionnement et normalisation)
def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)

    # Après avoir lu le contenu du fichier, cette ligne décode l'image en utilisant tf.image.decode_jpeg. 
    # L'argument channels=3 spécifie que l'image doit être décodée avec 3 canaux de couleur (RVB). 
    image = tf.image.decode_jpeg(image, channels=3)
    
    # redimensionne l'image à une taille de 224 pixels de largeur et 224 pixels de hauteur.    
    # Toutes les images ont la même taille avant de les utiliser dans des modèles d'apprentissage automatique
    image = tf.image.resize(image, (224, 224))  # Redimensionnez les images à la taille souhaitée
    image = image / 255.0  # Normalisation
    # normalise l'intensité des pixels de l'image en divisant chaque valeur de pixel par 255.0. 
    # Cela réduit les valeurs de pixel de l'intervalle [0, 255] à l'intervalle [0, 1]

    # Par exemple, un pixel ayant une valeur de 127 serait normalisé en 0,5, 
    # ce qui signifie que sa couleur serait équidistante entre le noir et le blanc
    return image, label

from PIL import Image



# Filter out animated GIF and BMP images
filtered_train_data = [image_path for image_path in train_data if not (is_animated_gif(image_path) or is_bmp_image(image_path))]
filtered_train_labels = [label for i, label in enumerate(train_labels) if not (is_animated_gif(train_data[i]) or is_bmp_image(train_data[i]))]


# on rassemble les données d'entraînement et les étiquettes en un seul ensemble de données
# Les données filtrées sont utilisées pour l'entraînement
train_dataset = tf.data.Dataset.from_tensor_slices((filtered_train_data, filtered_train_labels))
train_dataset = train_dataset.map(preprocess_image)

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.map(preprocess_image)

test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
test_dataset = test_dataset.map(preprocess_image)

# Mélangez les données et divisez-les en lots de 32 images
batch_size = 32

train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)


# Assurez-vous que vos données sont prêtes à être utilisées pour l'entraînement
for image, label in train_dataset.take(1):
    print("Image shape:", image.shape)
    print("Label:", label)



model = keras.Sequential([
    # Note the input shape is the desired size of the image 224x224 with 3 bytes color
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    # réduire la taille spatiale des cartes de caractéristiques. 
    # La taille de la fenêtre de max pooling est (2, 2), la taille de chaque carte de caractéristiques est réduite de moitié
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2), 
    layers.Conv2D(32, (3,3), activation='relu'), 
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2), 
    layers.Conv2D(64, (3,3), activation='relu'), 
    layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    layers.Flatten(), 
    # 512 neuron hidden layer
    layers.Dense(712, activation='relu'),
    layers.Dense(11, activation='softmax')
])

# Ajuster le taux d'apprentissage
optimizer = keras.optimizers.legacy.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_dataset, validation_data=val_dataset, epochs=5,callbacks=[checkpoint])

loss, accuracy = model.evaluate(val_dataset)

model.save('model_final')

print("Accuracy on validation data: {:.2f}%".format(accuracy * 100))
