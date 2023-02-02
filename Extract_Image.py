import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

## This python code is for preprocessing each javanese character image support png and jpg only
dirData="dataset"
dirSave="dataset_ready"
## This x is label of character
x = 'ha na ca ra ka da ta sa wa la pa dha ja ya nya ma ga ba tha nga'
x = x.split()
totalData = 162
for i in x:
    for j in range(totalData):
        img_name = i + str(j+1)
        try:
        	img = Image.open(dirData+'/'+ x[x.index(i)] +'/'+ img_name +'.png')
        except:
        	img = Image.open(dirData+'/'+ x[x.index(i)] +'/'+ img_name +'.jpg')
        img.load()
        if img.mode == 'RGBA':
            r,g,b,a = img.split()
            img = Image.merge('RGB', (r,g,b))
        img_data = np.asarray(img)
        img_data_bw = img_data.max(axis=2)
        non_empty_columns = np.where(img_data_bw.min(axis=0)<200)[0]
        non_empty_rows = np.where(img_data_bw.min(axis=1)<200)[0]
        x1 = min(non_empty_columns)
        x2 = max(non_empty_columns)
        y1 = min(non_empty_rows)
        y2 = max(non_empty_rows)
        cropBox = (y1, y2, x1, x2)
        try:
        	img_data_new = img_data[cropBox[0]+2:cropBox[1]-2, cropBox[2]+2:cropBox[3]-2 , :]
        except:
        	img_data_new = img_data[cropBox[0]:cropBox[1], cropBox[2]:cropBox[3] , :]
        img = Image.fromarray(img_data_new)
        img.thumbnail((32, 32), Image.Resampling.LANCZOS)
        img.save(dirSave+'/'+ x[x.index(i)] +'/'+ img_name +'.png')

