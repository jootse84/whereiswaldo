from __future__ import print_function
# import argparse
import imghdr
import cv2
import os, os.path
import numpy as np
import random

'''
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
'''

positives = []
negatives = []
path = './training/grayscale/positives/'
for name in os.listdir(path):
    image = cv2.imread(path + name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    positives.append({
        'image': np.reshape(image, -1),
        'label': 1
    })

path = './training/grayscale/negatives/'
for name in os.listdir(path):
    image = cv2.imread(path + name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    negatives.append({
        'image': np.reshape(image, -1),
        'label': 0
    })

data = np.concatenate((positives, negatives[:126]), axis=0)
random.shuffle(data)

test = data[:10]
data = data[10:]

img_data = []
lab_data = []
for image in data:
    img_data.append(image['image'])
    lab_data.append(image['label'])

img_test = []
lab_test = []
for image in test:
    img_test.append(image['image'])
    lab_test.append(image['label'])

from sklearn import svm, metrics

classifier = svm.SVC(gamma=0.001)
classifier.fit(img_data, lab_data)

predicted = classifier.predict(img_test)

print(metrics.classification_report(lab_test, predicted))

'''
image = cv2.imread('./training/grayscale/positives/flipped_solo_waldo1.png')
print(image.shape)
print("width: {} pixels".format(image.shape[1]))
print("height: {} pixels".format(image.shape[0]))
print("channels: {}".format(image.shape[2]))
cv2.imshow("Image", image)
cv2.imwrite("newimage.jpg", image)
'''

cv2.waitKey(0)
