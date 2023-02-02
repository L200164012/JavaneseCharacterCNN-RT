# JavaneseCharacterCNN-RT
According to the File on this directory:
1. These code will run supposedly on Python3.10 with all stated in requirement.txt
2. All of the code are independent one to another. The code start with CNN have many similar code but i dont separate it because try code by code purpose. In other word the code isn't really optimized.
3. This work used the dataset from creating myself and from kaggle.com from Hannan Hunafa and Phiard. The model itself are based on tensorflow tutorial with understanding from deeplearning.org online book and some research paper with the same topic
4. The a.png and c.png is test file for ConvertColor.py
5. dataset folder is directory of the image cropped from the fullset 20 javanese characters or by drawing the image of 20 javanese characters
6. dataset_ready is the preprocessed result from dataset using Extract_Image.py 
7. CNN_OneCharacterRT.py is used for test the model to recognize only 1 character in real time, it will result on the 1 character only no matter of input image. The only thing change it just the black pixel over the input data
8. CNN_JavaneseCharacterRT.py is used for being the system which can recognize 20 javanese character in real time
9. CNN_JavaneseCharacterRTandTesting.py is used for testing purpose based on my publication paper but with inverted color
10. CNN_JavaneseCharacterSomeCharacter.py is used for recognize some of label from 20 javanese character but with graph output per tested image with data that not used directly to training the model
11. CNN_JavaneseCharacterSummary.py is used for recognize the javanese character with graph output summarying the ability of the model to recognize the data that not used directly to training the model
12. CNN_JavaneseCharacterRTandTesting_withoutInvertColor.py is used for testing purpose based on my publication paper

Note:
1. The input image color can be inverted and it might give different result
2. The addition of dataset might give the model more valid to predict or recognize the javanese character
3. The noise on webcam can be configured by tweaking the captured frame

My Publication Title: JAVANESE CHARACTER RECOGNITION WITH REAL-TIME DETECTION USING CONVOLUTIONAL NEURAL NETWORK
