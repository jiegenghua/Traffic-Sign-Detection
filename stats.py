import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_train = pd.read_csv('train.csv', sep=",", header=None)
data_test = pd.read_csv('test.csv', sep=",", header=None)
data_other = pd.read_csv('other.csv', sep=",", header=None)

data_train.columns = ["img", "x1", "y1", "x2", "y2", "label"]
data_test.columns = ["img", "x1", "y1", "x2", "y2", "label"]
data_other.columns = ["img", "x1", "y1", "x2", "y2", "label"]

temp1 = data_train.copy()
temp2 = data_test.copy()
temp3 = data_other.copy()

numOfImg_train = len(data_train['img'].unique())
numOfCategory_train = len(data_train['label'].unique())
numOfObj_train = data_train['label'].count()

numOfImg_test = len(data_test['img'].unique())
numOfCategory_test = len(data_test['label'].unique())
numOfObj_test = data_test['label'].count()

numOfImg_other = len(data_other['img'].unique())
numOfCategory_other = len(data_other['label'].unique())
numOfObj_other = data_other['label'].count()

# extract the red round sign and give them label 1, the remaining images have label 0
red_round_labels =  ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16','p17', 'p18',
                 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29', 'pa10', 'pb', 'pc', 'pd',
                 'pe', 'ph3.5', 'pl40', 'pm10', 'pn', 'pne', 'pnl', 'pw3']

data_train['label'] = np.where(temp1['label'].isin(red_round_labels), 1, 0)
labels_train = data_train['label'].values

data_test['label'] = np.where(temp2['label'].isin(red_round_labels), 1, 0)
labels_test = data_test['label'].values

data_other['label'] = np.where(temp3['label'].isin(red_round_labels), 1, 0)
labels_other = data_other['label'].values

# RRTS: Red Round Traffic Sign
numOfRRTS_train = len(np.where(labels_train==1)[0])
numOfRRTS_test = len(np.where(labels_test==1)[0])
numOfRRTS_other = len(np.where(labels_other==1)[0])

print("Number of images in train folder:", numOfImg_train)
print("Number of total categories in train folder:", numOfCategory_train)
print("Number of objects in train folder:", numOfObj_train)
print("Number of Red Round Traffic Sign in train folder:",numOfRRTS_train)
print('---------------------------------------------------------------')
print("Number of images in test folder:", numOfImg_test)
print("Number of total categories in test folder:", numOfCategory_test)
print("Number of objects in test folder:", numOfObj_test)
print("Number of Red Round Traffic Sign in test folder:",numOfRRTS_test)
print('---------------------------------------------------------------')
print("Number of images in other folder:", numOfImg_other)
print("Number of total categories in other folder:", numOfCategory_other)
print("Number of objects in other folder:", numOfObj_other)
print("Number of Red Round Traffic Sign in other folder:",numOfRRTS_other)

x=[]
x.append(labels_train)
x.append(labels_test)
x.append(labels_other)

fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()
colors = ['red', 'tan', 'lime']
la = ['train folder','test folder','other folder']
ax0.hist(labels_train)
ax0.set_title('data distribution in the train folder')

ax1.hist(labels_test)
ax1.set_title('data distribution in the test folder')

ax2.hist(labels_other)
ax2.set_title('data distribution in the other folder')

ax3.hist(x, histtype='bar', color=colors,label=la)
ax3.legend()
ax3.set_title('data distributions')

fig.tight_layout()
plt.savefig('statisticAnalysis')
plt.show()
