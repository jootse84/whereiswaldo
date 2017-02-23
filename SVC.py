from __future__ import print_function
from sklearn import svm, metrics
import argparse
import imghdr
import cv2
import os, os.path
import numpy as np
import random
import helper

def grayscaleSVC(pos_files_path, neg_files_path):
    positives = []
    negatives = []
    for name in os.listdir(pos_files_path):
        image = cv2.imread(pos_files_path + name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        positives.append({
            'image': np.reshape(image, -1),
            'label': 1
        })
    for name in os.listdir(neg_files_path):
        image = cv2.imread(neg_files_path + name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        negatives.append({
            'image': np.reshape(image, -1),
            'label': 0
        })
    predict(positives, negatives)

def histogramSVC(pos_files_path, neg_files_path):
    positives = []
    negatives = []
    for name in os.listdir(pos_files_path):
        image = cv2.imread(pos_files_path + name)
        positives.append({
            'image': helper.color_histogram(image),
            'label': 1
        })
    for name in os.listdir(neg_files_path):
        image = cv2.imread(neg_files_path + name)
        negatives.append({
            'image': helper.color_histogram(image),
            'label': 0
        })
    predict(positives, negatives)


def predict(positives, negatives):
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
    classifier = svm.SVC(gamma=0.001)
    classifier.fit(img_data, lab_data)
    predicted = classifier.predict(img_test)
    print(metrics.classification_report(lab_test, predicted))

def SVC(type):
    if type == "grayscale":
        grayscaleSVC('./training/grayscale/positives/', './training/grayscale/negatives/')
    else:
        histogramSVC('./training/positives/', './training/negatives/')


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", required = True, help = "Type of the SVC approach - 'histogram' or 'grayscale'")
args = vars(ap.parse_args())
SVC(args['type'])

cv2.waitKey(0)
