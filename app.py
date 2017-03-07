import argparse
from svc import SVC

ap = argparse.ArgumentParser()
help_info = "Type of the SVC approach - 'histogram' or 'greyscale'"
ap.add_argument("-t", "--type", required = True, help=help_info)
args = vars(ap.parse_args())

SVM = SVC(args['type'])
#SVM.test_classifier()
SVM.find_waldo()
