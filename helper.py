import numpy as np
import cv2
from matplotlib import pyplot as plt

def create_mirrors():
    # expand our dataset by creating mirrors of our images
    for folder in ['positives', 'negatives']:
        path = "./training/%s/" % folder
        for name in os.listdir(path):
            if imghdr.what(path + name) in ['jpg', 'jpeg', 'png']:
                image = cv2.imread(path + name)
                cv2.imwrite(path + 'flipped_' + name, cv2.flip(image, 1))

def create_grayscales():
    # from our RGB images to a grayscale
    for folder in ['positives', 'negatives']:
        path = "./training/%s/" % folder
        for name in os.listdir(path):
            if imghdr.what(path + name) in ['jpg', 'jpeg', 'png']:
                image = cv2.imread(path + name)
                dest_path = "./training/grayscale/%s/" % folder
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(dest_path + name, gray_image)

# http://www.ippatsuman.com/2014/08/13/day-and-night-an-image-classifier-with-scikit-learn/
def show_color_histogram(image, title):
    # create as a feature vector the histogram including the RGB colors of an image
    chanels = cv2.split(image) # list of 3 arrays, each one representing a color
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("Histogram %s" % title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (chan, color) in zip(chanels, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color = color)
        plt.xlim([0, 256])

    plt.show()

def color_histogram(image):
    result = []
    for chan in cv2.split(image):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        result.append(np.reshape(hist, -1))
    return np.reshape(result, -1).astype(int)

#image = cv2.imread("./training/positives/flipped_solo_waldo3.png")
#show_color_histogram(image, "solo_waldo3")
image = cv2.imread("./training/negatives/no35.png")
cv2.waitKey(0)
