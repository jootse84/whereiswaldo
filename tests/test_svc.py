import pytest
from svc import SVC

_waldo_img = './training/positives/solo_waldo2.png'

def test_class():
    with pytest.raises(ValueError):
        SVM = SVC("")
    SVM = SVC("greyscale")

def _predict(type_svc):
    SVM = SVC(type_svc)
    with pytest.raises(IOError):
        SVM.predict("")
    prediction = SVM.predict(_waldo_img)
    prediction2 = SVM.predict(_waldo_img)
    assert prediction == prediction2, "Different prediction on same image"

def test_predict():
    _predict("histogram")
    _predict("greyscale")
    
