from __future__ import print_function
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from imagetools import ImageTools, Puzzle
import numpy as np

class SVC():
    gamma = 0.001

    def __init__(self, img_type):
        self.type = img_type
        self.imtools = ImageTools(img_type)
        self.classifier = svm.SVC(gamma=self.gamma)
        self.dataset = self.imtools.dataset
        self.X = [image['image'] for image in self.dataset]
        self.y = [image['label'] for image in self.dataset]
        self.classifier.fit(self.X, self.y)

    def test_classifier(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, stratify=self.y, random_state=42)
        self.classifier.fit(X_train, y_train)
        y_predicted = self.classifier.predict(X_test)
        print('Mean: %f' % np.mean(y_test - y_predicted))
        print(metrics.classification_report(y_test, y_predicted))

    def predict(self, img):
        if type(img) is str:
            img = self.imtools.read(img)
        [result] = self.classifier.predict([img])
        return result

    def find_waldo(self):
        images = []
        for img in Puzzle(self.type):
            pred = self.predict(img)
            # print(pred)
        # now get the best prediction and save it
        # self.imtools.write("is_waldo.png", img)

