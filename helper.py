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
