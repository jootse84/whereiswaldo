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
    _waldo_img = './training/positives/solo_waldo2.png'
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

    def read(self, path):
        try:
            is_image = lambda x: imghdr.what(x) in ['jpg', 'jpeg', 'png']
            if not is_image(path):
                raise IOError()
        except IOError:
            raise IOError("'{}' is not a path to an image".format(path))
        return self.imread_fn(self._read(path))

