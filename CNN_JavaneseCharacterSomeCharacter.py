from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pathlib
import os

## Preparing dataset
data_dir = pathlib.Path("dataset_ready")
image_count = len(list(data_dir.glob('*/*')))
data_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
## Shuffle the glob of directory. This will also shuffle the position of dataset that created later
data_ds = data_ds.shuffle(image_count, reshuffle_each_iteration=False)
## This is the Label of javanese character but sorted alphabetically
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
## Size of single image to fit in the model later
img_height = 32
img_width = 32
## batch size for training
batch_size = tf.data.experimental.cardinality(data_ds).numpy()

## Dividing dataset into training data and validation data
## in this case val_data also being used as testing dataset
val_size = int(image_count * 0.2)
train_ds = data_ds.skip(val_size)
val_ds = data_ds.take(val_size)

## This function to get the label of the data in the directory
def get_label(file_path):
  # Convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)
   
## This function read the image of the determined directory
def decode_img(img):
  # Convert the compressed string to a 3D uint8 tensor with rgb color
  img = tf.io.decode_jpeg(img, channels=1)
  # Resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])
  
## This function call 2 function before and return data with the image and the label
def process_path(file_path):
  label = get_label(file_path)
  # Load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  img = tf.convert_to_tensor(img, dtype=tf.float32)
  # This is the input image color, you can change it to white black by switching the 0 and 1
  img = tf.where(img > 127, 0, 1)
  return img, label
  
## Call the function process path to get img and label from the glob of directory
## Autotune = based on your cpu or gpu performance
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

## This function created for increasing the performance when call the img and label for futher operation
def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

## Call the performance function and separate image and label on both training dataset and validation dataset
train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
train_images, train_labels = next(iter(train_ds))
val_images, val_labels = next(iter(val_ds))

## Creating Convolutional Neural Network Model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(20, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


## Training The Model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images,val_labels))

## Evaluating The Model
model_loss, model_acc = model.evaluate(train_ds, verbose=2)

## Predicting the number of images from the validation dataset 
numberPred = 20
predictions = model.predict(val_images[:numberPred])

## Sketch the graph on accuracy of the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

## This function creating the first plot to show the image after going through the model
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)

## This function will show the graph of percentage of all 20 label to respond the predicted/tested images
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(20))
    plt.yticks([])
    thisplot = plt.bar(class_names, predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    
## Calling those function to show 1 by 1 graph of the predicted result
i = 1
for i in range(len(predictions)):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions[i], val_labels, val_images)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i],  val_labels)
    
plt.show()
