from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pathlib
import os
import time	

## Preparing dataset
data_dir = pathlib.Path("dataset_ready")
image_count = len(list(data_dir.glob('*/*.png')))
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
## This function also convert input image to binary image
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
train_ds = data_ds.map(process_path, num_parallel_calls=AUTOTUNE)

## This function created for increasing the performance when call the img and label for futher operation
def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

## Call the performance function and separate image and label on training dataset
train_ds = configure_for_performance(train_ds)
train_images, train_labels = next(iter(train_ds))

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

## Training The Model with timer
start = time.time()
history = model.fit(train_images, train_labels, epochs=7)
end = time.time()
print(end-start)

## Evaluating The Model
model_loss, model_acc = model.evaluate(train_ds, verbose=2)

## Sketch the graph on accuracy of the model
dim_acc = np.arange(1,len(history.history['accuracy'])+1)
plt.plot(dim_acc, history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.4, 1])
plt.legend(loc='lower right')
plt.show()

## Sketch the graph on loss of the model
dim_loss = np.arange(1,len(history.history['loss'])+1)
plt.plot(dim_loss, history.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.show()

## Video capture System 
## Variable to prepare for testing
timer = []
trigger_test = 0
character = 'ha na ca ra ka da ta sa wa la pa dha ja ya nya ma ga ba tha nga'
character = character.split()
character_indexSorted = []
for i in range(len(character)):
 character_indexSorted.append(np.where(class_names == character[i])[0][0])
result_perdataset = [[] for i in range(len(character))]
result = []
labeliter = 0
truelabel = 0

import cv2
cap = cv2.VideoCapture(0)
width = 512
height = 512
while True:
 #For testing the time needed to predict one frame of image, this will take 500 data as sample
 if len(timer) <= 499:
  start = time.time()
 #Iteration of label when testing the image
 #This is easier my work because my testing data have form of fullset javanese character on sheet
 if labeliter >= len(character_indexSorted):
  labeliter = 0 
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
  #Convert b&w color image to binary image
  #You can change it to white & black by switching the 0 and 1
  trgt_tensor = tf.where(trgt_tensor > 127, 0, 1)
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
  if pred_acc > 50:      
   cv2.putText(img_boxes,class_names[pred_labels],(20, height-20), font, 2, (255,255,0), 1, cv2.LINE_AA)
   cv2.putText(img_boxes,str(pred_acc),(width-180, height-20), font, 2, (255,255,0), 1, cv2.LINE_AA)
  else:
   cv2.putText(img_boxes,'Not Found',(20, height-20), font, 2, (255,255,0), 1, cv2.LINE_AA)
  #Display the text label and true label accuracy for testing purpose
  cv2.putText(img_boxes,class_names[truelabel],(width-180, 100), font, 2, (255,255,0), 1, cv2.LINE_AA)
  cv2.putText(img_boxes,str(pred[0][truelabel]*100),(width-180, 150), font, 2, (255,255,0), 1, cv2.LINE_AA)
  #This taking exactly 10 frame for testing purpose
  if trigger_test > 0 :
   trigger_test = trigger_test - 1
   #Add the result to variable with list datatype for testing purpose
   result.append([pred_labels,pred_acc,truelabel,pred[0][truelabel]])
   if trigger_test == 0:  
    #Add to the list actual list that will be the outcome for the testing
    result_perdataset[labeliter] = result_perdataset[labeliter] + result
    #Resetting the 10 frame consist of one label of javanese character result
    result = []
    #Switch to next label of javanese character
    labeliter = labeliter + 1
   #If the 10 frame is not finish yet show text "In Test"
   cv2.putText(img_boxes,"In Test",(width-180, 50), font, 2, (255,255,0), 1, cv2.LINE_AA)
  else:
   #If the 10 frame is finish then show "Ready"
   cv2.putText(img_boxes,"Ready",(width-180, 50), font, 2, (255,255,0), 1, cv2.LINE_AA)
  #Display the resulting frame  
  cv2.imshow('Camera',img_boxes)
  #Display the selected bounding box
  cv2.imshow('Green Box',bwimg)
  #Display the actual image to be fed to the model
  cv2.imshow('Target',trgt)
  #Add the timer needed to predict 1 javanese character to the variable timer
  if len(timer) <= 499 :
  	end = time.time()
  	timer.append(end-start)
 except:
  continue
 #If key pressed it will saved to the variable for further use
 pressedKey = cv2.waitKey(1) & 0xFF
 #Pressing Q key when the system run will stop the system
 if pressedKey == ord('q'):
  break
 #Pressing P key when the system run will test the 10 frame to get 10 predicted label and the true label accuracy
 #The system will give true label from ha to nga character in this case
 #Will start to beginning again it will be ha after nga character
 elif pressedKey == ord('p') and labeliter <= 19:
  trigger_test = 10
  truelabel = character_indexSorted[labeliter]
  time.sleep(1)
  continue
 #Will undo the iteration of test
 #This is useful when hovering on webcam got trouble
 elif pressedKey == ord('u'):
  labeliter = labeliter-1
  for i in range(10):
   result_perdataset[labeliter].pop(len(result_perdataset[labeliter])-1)
  truelabel = character_indexSorted[labeliter]
  print('deleted')
  

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

#This result in average of time needed to process 1 javanese character
print(len(timer))
print(sum(timer)/len(timer))

#This use for the 10 frame per character testing
Success=[0 for i in range(len(character))]
Fail=[0 for i in range(len(character))]
Percentage=[0 for i in range(len(character))]
iter=0
for i in result_perdataset:
 for j in i:
  if j[0] == j[2]:
   Success[iter]=Success[iter]+1
   Percentage[iter]=Percentage[iter]+j[3]
  else:
   Fail[iter]=Fail[iter]+1
   Percentage[iter]=Percentage[iter]+j[3]
 iter = iter + 1 

#The result will be list of number on variable success, Fail and the average of accuracy in variable Percentage 
print(Success)
print(Fail)
for i in range(len(Percentage)):
 if Percentage[i] != 0:
  Percentage[i]=round((Percentage[i]*100)/(Success[i]+Fail[i]),2)
 else:
  continue
print(Percentage)
