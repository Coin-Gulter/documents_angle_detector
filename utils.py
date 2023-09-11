import keras.backend as K
import keras
import glob
import time
import cv2 as cv

def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.
    """
    return 180 - abs(abs(x - y) - 180)

@keras.saving.register_keras_serializable('angle_error')
def angle_error(y_true, y_pred):
    """
    Calculate the mean diference between the true angles
    and the predicted angles. Each angle is represented
    as a binary vector.
    """
    diff = angle_difference(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(K.cast(K.abs(diff), K.floatx()))

def read_images_from_list(folder_pathes):
    images = []
    percentage = 0
    prev_time = time.time()

    length = len(folder_pathes)

    for count, file_path in enumerate(folder_pathes):

        if int((count*100)/length) != percentage:

            percentage = int((count*100)/length)
            eta = time.strftime("%H:%M:%S", time.gmtime((100 - percentage) * (time.time() - prev_time)))

            print(f'Percentage of loading images for train = {percentage}% -- ETA = {eta}')

            prev_time = time.time()

        images.append(cv.imread(file_path))

    return images
