import pytest
from imagetools import ImageTools

_waldo_img = './training/positives/solo_waldo2.png'

def test_class():
    with pytest.raises(ValueError): 
        imgtools = ImageTools("")

def test_read():
    imgtools = ImageTools("greyscale")
    with pytest.raises(IOError):
        imgtools.read('./')
    assert type(imgtools.read(_waldo_img)).__name__ == "ndarray", "Not an numpy array"

def test_dataset():
    grey_ds = ImageTools("greyscale").dataset
    histo_ds = ImageTools("histogram").dataset
    is_nparray = lambda x: type(x).__name__ == "ndarray"
    ok_size = lambda x: x.shape[0] == 308

    assert is_nparray(grey_ds), "Not an numpy array"
    assert ok_size(histo_ds), "Dataset size not expected"
    assert is_nparray(grey_ds), "Not an numpy array"
    assert ok_size(histo_ds), "Dataset size not expected"

def test_compare():
    grey_ds_1 = ImageTools("greyscale").dataset
    histo_ds_1 = ImageTools("histogram").dataset
    grey_ds_2 = ImageTools("greyscale").dataset
    histo_ds_2 = ImageTools("histogram").dataset
    '''
    assert grey_ds_1 == grey_ds_2, "Dataset from same type is different"
    assert histo_ds_1 == histo_ds_2, "Dataset from same type is different"
    assert grey_ds_1 != histo_ds_1, "Dataset from different type is the same"
    '''

