import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter
import os
import shutil
from sklearn.model_selection import train_test_split
#data = pd.read_csv('train.csv', sep=",", header=None)
data = pd.read_csv('test.csv', sep=",", header=None)
#data = pd.read_csv('other.csv', sep=",", header=None)
data.columns = ["img", "x1", "y1", "x2", "y2", "id"]

tmp = data.copy()
print("Data has image files with traffic signs numbers:", len(data['img'].unique()))
print("Data has traffic signs class numbers:", len(data['id'].unique()))
print("Data has traffic signs instance numbers:", data['id'].count())

red_round_labels =  ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16','p17', 'p18',
                 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29', 'pa10', 'pb', 'pc', 'pd',
                 'pe', 'ph3.5', 'pl40', 'pm10', 'pn', 'pne', 'pnl', 'pw3']
data['id'] = np.where(tmp['id'].isin(red_round_labels), 1, 0)
labels = data['id'].values
full_data = data.drop('id', 1)

x_train, x_test, y_train, y_test = train_test_split(full_data, labels, test_size=0.3, random_state=42)

train_data = []
test_data = []

#data_dir = '../data/train'
data_dir = '../data/test'
#data_dir = '../data/other'
for i in range(x_train.shape[0]):
    temp = str(x_train['img'].iloc[i])+'.jpg'
    curr_im = os.path.join(data_dir,temp )
    img = Image.open(curr_im)
    cropped_rect = (x_train['x1'].iloc[i], x_train['y1'].iloc[i], x_train['x2'].iloc[i], x_train['y2'].iloc[i])
    crop_im = img.crop(cropped_rect)
    crop_im = crop_im.resize((48, 48), Image.ANTIALIAS)
    crop_im = ImageOps.autocontrast(crop_im)
    train_data.append(np.array(crop_im))

for i in range(x_test.shape[0]):
    temp = str(x_test['img'].iloc[i]) + '.jpg'
    curr_im = os.path.join(data_dir, temp )
    img = Image.open(curr_im)
    cropped_rect = (x_test['x1'].iloc[i], x_test['y1'].iloc[i], x_test['x2'].iloc[i], x_test['y2'].iloc[i])
    crop_im = img.crop(cropped_rect)
    crop_im = crop_im.resize((48, 48), Image.ANTIALIAS)
    crop_im = ImageOps.autocontrast(crop_im)
    test_data.append(np.array(crop_im))

train_data = np.array(train_data)
test_data = np.array(test_data)


num_train = train_data.shape[0]
num_test = test_data.shape[0]
num_classes = 2

train_label = np.zeros((num_train, num_classes), dtype=np.int8)
test_label = np.zeros((num_test, num_classes), dtype=np.int8)

for i in range(len(y_train)):
    if y_train[i] == 1:
        train_label[i][1] = 1
    else:
        train_label[i][0] = 1

for i in range(len(y_test)):
    if y_test[i] == 1:
        test_label[i][1] = 1
    else:
        test_label[i][0] = 1

count = 0
for i in range(len(y_train)):
    if y_train[i] == 1:
        count += 1

print("Number of red round signs in training data: ", count)
print("Number of negatives in training data: ", num_train - count)

np.save('train_data.npy', train_data)
np.save('test_data.npy', test_data)
np.save('train_labels.npy', train_label)
np.save('test_labels.npy', test_label)

