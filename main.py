import pandas as pd
import os
from sklearn import svm
import numpy as np
from skimage.feature import hog
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ******************************************************************************************************* #

data = pd.read_csv('sampleSubmission.csv')
data = data[:1100]
dataFrame = pd.DataFrame(data)
dataFrame2 = pd.DataFrame(data)
features = []
features2 = []
direction = r'train'
count = 0

# ******************************************************************************************************* #

for index, image in enumerate(os.listdir(direction)):
    if count < 1000:
      img_direction = os.path.join(direction, image)
      Im = Image.open(img_direction)
      resizedImage = Im.resize((128, 64))
      fd, hogImage = hog(resizedImage, orientations=9, pixels_per_cell=(8, 8)
                         , cells_per_block=(2, 2), visualize=True, multichannel=True)
      features.append(fd)
      if image[0] == 'd':
          row = {'id': index+1, 'label': 1}
          dataFrame = dataFrame.append(row, ignore_index=True)
    elif 1000 <= count < 1100:
        img_direction = os.path.join(direction, image)
        Im = Image.open(img_direction)
        resizedImage = Im.resize((128, 64))
        fd, hogImage = hog(resizedImage, orientations=9, pixels_per_cell=(8, 8)
                           , cells_per_block=(2, 2), visualize=True, multichannel=True)
        features2.append(fd)
        if image[0] == 'd':
            row = {'id': index + 1, 'label': 1}
            dataFrame2 = dataFrame2.append(row, ignore_index=True)
    elif 23900 <= count < 24000:
        img_direction = os.path.join(direction, image)
        Im = Image.open(img_direction)
        resizedImage = Im.resize((128, 64))
        fd, hogImage = hog(resizedImage, orientations=9, pixels_per_cell=(8, 8)
                           , cells_per_block=(2, 2), visualize=True, multichannel=True)
        features2.append(fd)
        if image[0] == 'd':
            row = {'id': index + 1, 'label': 1}
            dataFrame2 = dataFrame2.append(row, ignore_index=True)

    elif count >= 24000:
      img_direction = os.path.join(direction, image)
      Im = Image.open(img_direction)
      resizedImage = Im.resize((128, 64))
      fd, hogImage = hog(resizedImage, orientations=9, pixels_per_cell=(8, 8)
                         , cells_per_block=(2, 2), visualize=True, multichannel=True)
      features.append(fd)
      if image[0] == 'd':
          row = {'id': index+1, 'label': 1}
          dataFrame = dataFrame.append(row, ignore_index=True)
    count = count+1

# ******************************************************************************************************* #

dataFrame = dataFrame[:2000]
dataFrame = dataFrame.assign(features=features)
dataFrame2 = dataFrame2[:200]
dataFrame2 = dataFrame2.assign(features=features2)
X_train = dataFrame['features']
y_train = dataFrame['label']
X_test = dataFrame2['features']
y_test = dataFrame2['label']

# ******************************************************************************************************* #

svc = svm.SVC(kernel='linear', C=0.1).fit(list(X_train), list(y_train))
rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=0.1).fit(list(X_train), list(y_train))
poly_svc = svm.SVC(kernel='poly', degree=2, C=0.1).fit(list(X_train), list(y_train))

# ******************************************************************************************************* #

kernel_list = ['SVC(linear)', 'SVC(rbf)', 'SVC(poly)']
for i, clf in enumerate((svc, rbf_svc, poly_svc)):
    predictions = clf.predict(list(X_test))
    accuracy = np.mean(predictions == list(y_test))
    print('the accuracy of ', kernel_list[i], 'is : ', accuracy)
df = pd.DataFrame(predictions, columns=['Animal_label'])
for i in range(len(df)):
    if df.at[i, 'Animal_label'] == 0:
        df.at[i, 'Animal_label'] = 'cat'
    else:
        df.at[i, 'Animal_label'] = 'dog'
print('\n', df)
print('\n', 'the length of test data is:', len(df))




