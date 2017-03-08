import cv2
import imghdr
import os, os.path
import numpy as np

class ImageTools():
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        msg (str): Human readable string describing the exception.
        code (:obj:`int`, optional): Error code.

    Attributes:
        msg (str): Human readable string describing the exception.
        code (int): Exception error code.

    """
    _pos_path = './training/positives/'
    _neg_path = './training/negatives/'
    GREYSCALE = 'greyscale'
    HISTOGRAM = 'histogram'

    def __init__(self, img_type):
        if img_type == self.GREYSCALE:
            self.imread_fn = lambda x: np.reshape(x, -1)
            self._read = lambda x: cv2.imread(x, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        elif img_type == self.HISTOGRAM:
            self.imread_fn = lambda x: self._color_histogram(x)
            self._read = lambda path: cv2.imread(path)
        else:
            err = ('{} is not an ImageTools type').format(img_type)
            raise ValueError(err)

        pos_images = [self._read(self._pos_path + name)
            for name in os.listdir(self._pos_path)]
        positives = map(lambda x: {'image': self.imread_fn(x), 'label': 1}, pos_images)

        neg_images = [self._read(self._neg_path + name)
            for name in os.listdir(self._neg_path)]
        negatives = map(lambda x: {'image': self.imread_fn(x), 'label': 0}, neg_images)

        self.dataset = np.concatenate((positives, negatives), axis=0)

    def _color_histogram(self, image):
        fn = lambda x: np.reshape(cv2.calcHist([x], [0], None, [256], [0, 256]), -1)
        rgbs = [fn(channel) for channel in cv2.split(image)]
        return np.reshape(rgbs, -1).astype(int)

    def _extract_red(self, image):
        # convert colors from BGR to HSV
        img_hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # lower mask (0-10)
        lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([170,50,50])
        upper_red = np.array([180,255,255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

        # join my masks
        mask = mask0+mask1

        # set my output img to zero everywhere except my mask
        output_img = image.copy()
        output_img[np.where(mask==0)] = 255

        return output_img

    def write(self, name, img):
        cv2.imwrite(name, img)

    def read(self, path):
        try:
            is_image = lambda x: imghdr.what(x) in ['jpg', 'jpeg', 'png']
            if not is_image(path):
                raise IOError()
        except IOError:
            raise IOError("'{}' is not a path to an image".format(path))
        return self.imread_fn(self._read(path))


class Puzzle:
    _img_ref = './training/positives/solo_waldo2.png'

    def __init__(self, img_type):
        self.imtools = ImageTools(img_type)
        self.puzzle = cv2.imread("nobodys.jpg")
        (p_height, p_width, p_channels) = self.puzzle.shape

        ref = cv2.imread(self._img_ref)
        (height, width, channels) = ref.shape
        self.w = width
        self.h = height

        self.decr_top = width / 2
        self.decr_left = height / 2
        self.top = p_height - self.h
        self.reset_left = p_width - self.w
        self.left = self.reset_left + self.decr_left

    def __iter__(self):
        return self

    def _crop(self):
        # Crop from img[y: y + height, x: x + width]
        x = max(0, self.left)
        y = max(0, self.top)
        return self.puzzle[y : y + self.h, x : x + self.w]

    def next(self):
        if self.left < 0:
            self.top -= self.decr_left
            self.left = self.reset_left
        else:
            self.left -= self.decr_left

        if (self.top < 0 and self.left < 0):
            self.imtools.write("img.png", self._crop())
            raise StopIteration

        return self.imtools.imread_fn(self._crop())
