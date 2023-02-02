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
## You can change the ha to another label in your preference
image_count = len(list(data_dir.glob('ha/*')))
data_ds = tf.data.Dataset.list_files(str(data_dir/'ha/*'), shuffle=False)
data_ds = data_ds.shuffle(image_count, reshuffle_each_iteration=False)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
img_height = 32
img_width = 32
batch_size = tf.data.experimental.cardinality(data_ds).numpy()

## Dividing the dataset directory for training and validating
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
  # Convert the compressed string to a 3D uint8 tensor with gray color
  img = tf.io.decode_jpeg(img, channels=1)
  # Resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])
  
## This function call 2 function before and return data with the image and the label
def process_path(file_path):
  label = get_label(file_path)
  # Load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
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

## Sketch the graph on accuracy of the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

## Video capture System
import cv2
cap = cv2.VideoCapture(0)
width = 512
height = 512
while True:
 #Read every frame while ret mean return to get the boolean of True while it works
 ret, frame = cap.read()  
 #Resizing the frame to desired size
 inp = cv2.resize(frame, (width , height))
 #Creating 2 frame for rgb color and gray color
 rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
 gray = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
 #Converting to black and white color
 (thresh, bwimg) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
 try:
  #Removing the whitespace
  bwimg = bwimg[170:342,170:342]
  bwimg_inv = np.invert(bwimg)
  coord = cv2.findNonZero(bwimg_inv)
  x, y, w, h = cv2.boundingRect(coord)
  trgt = bwimg[y:y+h, x:x+w]
  #Is optional but recommended (float convertion and convert img to tensor image)
  trgt_tensor = tf.convert_to_tensor(trgt.reshape(trgt.shape+(1,)), dtype=tf.float32)
  #Add dims to binary_tensor and resize it for fitting to the model
  trgt_tensor = tf.expand_dims(trgt_tensor , 0)
  trgt_tensor = tf.image.resize(trgt_tensor, [img_height, img_width])
  #Predict the model which return array of percentage on 20 label javanese character
  pred = model.predict(trgt_tensor)
  #Taking the biggest percentage to take conclusion if the predicted image is the said label
  pred_labels = np.argmax(pred)
  pred_acc = np.max(pred)*100
  #Box for system writing result and box for static bounding box
  img_boxes = cv2.rectangle(rgb,(0, height),(width, 0),(255,255,255),1)   
  box_target = cv2.rectangle(img_boxes,(170, 342),(342, 170),(255,0,0),5)   
  font = cv2.FONT_HERSHEY_SIMPLEX
  #cv2.putText will write text to the boxes which will shown in the windows on execution
  #This will show in the windows prediction label and accuracy if above 50% and Not found if fail
  if pred_acc > 50:      
   cv2.putText(img_boxes,class_names[pred_labels],(20, height-20), font, 2, (255,255,0), 1, cv2.LINE_AA)
   cv2.putText(img_boxes,str(pred_acc),(width-180, height-20), font, 2, (255,255,0), 1, cv2.LINE_AA)
  else:
   cv2.putText(img_boxes,'Not Found',(20, height-20), font, 2, (255,255,0), 1, cv2.LINE_AA)
  #Display the resulting frame
  cv2.imshow('Camera',img_boxes)
  #Display the selected bounding box
  cv2.imshow('Bounding Box',bwimg)
  #Display the actual image to be fed to the model
  cv2.imshow('Target',trgt)
 except:
  continue
 #For break the continously frame or to stop the system 
 if cv2.waitKey(1) & 0xFF == ord('q'):
  break

## When everything done, release the capture and clean the stopped system 
cap.release()
cv2.destroyAllWindows()
