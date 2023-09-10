import random
import glob
import cv2 as cv
import time
import json
import numpy as np
import time
import augmentation


class preprocesing():
    """
    Class for preprocessing the data.

    Args:
        h_parameters (dict): Hyperparameters of the model.
    """

    def __init__(self, h_parameters):
        self.h = h_parameters

        # Get the paths to the training and test images.
        self.train_pathes = glob.glob(self.h.train_path + "*" + self.h.train_img_format)
        self.train_pathes = self.train_pathes[:self.h.train_amount_of_images]
        random.shuffle(self.train_pathes)

        if self.h.with_test:
            self.test_pathes = glob.glob(self.h.test_path + "*" + self.h.test_img_format)
            self.test_pathes = self.test_pathes[:self.h.test_amount_of_images]
            random.shuffle(self.test_pathes)

        # Load the labels dictionary, which maps image paths to their corresponding board states.
        self.labels_dict = None


    def read_angle_from_file(self, image_path, label_folder):
        """
        Read the angle from the file name.

        Args:
            image_path (str): Path to the image file.
            label_folder (str): Path to the folder containing the labels file.

        Returns:
            int: The angle.
        """

        # If the labels dictionary is not yet loaded, load it.
        if self.labels_dict == None:
            with open(label_folder + self.h.labels_file_name, 'r') as file:
                self.labels_dict = json.load(file)

        # Get the angle from the labels dictionary.
        angle_label = self.labels_dict[image_path]

        return angle_label


    def image_read_and_augmenting(self, file, augmenting:bool = True):
        """
        Read and augment the image.

        Args:
            file (str): Path to the image file.
            augmenting (bool): Whether to augment the image.

        Returns:
            np.array: The image.
            int: The angle.
        """

        # Read the image.
        image = cv.imread(file)

        # If augmentation is enabled, augment the image.
        if augmenting:
            augmented_image, angle = augmentation.augmenting_photo(image, self.h)
        else:
            augmented_image =  image
            angle = None

        # Downsize the image.
        img_size = (self.h.image_size, self.h.image_size)
        resized_image = cv.resize(augmented_image, img_size)
        resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)

        # Prepare the image for training.
        resized_image = augmentation.prep_img(resized_image, self.h)

        return resized_image, angle


def prepare_data_train(h_parameters):
    """
    Prepare the training data.

    Args:
        h_parameters (dict): Hyperparameters of the model.

    Returns:
        X_train, y_train: Training data.
    """

    preprocess = preprocesing(h_parameters)

    X_train = []
    y_train = []
    length = preprocess.h.train_amount_of_images
    percentage = 0
    prev_time = time.time()

    # If image generator is enabled, load the images one by one and augment them.

    if preprocess.h.image_generator:
        for count in range(length):

            if int((count*100)/length) != percentage:

                percentage = int((count*100)/length)
                eta = time.strftime("%H:%M:%S", time.gmtime((100 - percentage) * (time.time() - prev_time)))

                print(f'Percentage of loaded train images = {percentage}% -- ETA = {eta}')

                prev_time = time.time()

            img_path = random.choice(preprocess.train_pathes)

            # Read the image and augment it.
            pre_img, angle_label = preprocess.image_read_and_augmenting(img_path)
            X_train.append(pre_img)

            # Read the angle from the file name if the angle label is not available.
            if angle_label == None:
                angle_label = preprocess.read_angle_from_file(img_path)

            y_train.append(angle_label)

    # Otherwise, load the images without augmentation.

    else:
        for count, img_path in enumerate(preprocess.train_pathes):

            if int((count*100)/length) != percentage:

                percentage = int((count*100)/length)
                eta = time.strftime("%H:%M:%S", time.gmtime((100 - percentage) * (time.time() - prev_time)))

                print(f'Percentage of loaded train images = {percentage}% -- ETA = {eta}')

                prev_time = time.time()

            # Read the image and do not augment it.
            pre_img, angle_label = preprocess.image_read_and_augmenting(img_path, augmenting=False)
            X_train.append(pre_img)

            # Read the angle from the file name if the angle label is not available.
            if angle_label == None:
                angle_label = preprocess.read_angle_from_file(img_path)

            y_train.append(angle_label)

    # Transform the data to NumPy arrays for model use.

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train


def prepare_data_test(h_parameters):
    """
    Prepare the test data.

    Args:
        h_parameters (dict): Hyperparameters of the model.

    Returns:
        X_test, y_test: Test data.
    """

    preprocess = preprocesing(h_parameters)

    if preprocess.h.with_test:

        # If test data is available, prepare it.
        X_test = []
        y_test = []
        length = preprocess.h.test_amount_of_images
        percentage = 0
        prev_time = time.time()

        for count, img_path in enumerate(preprocess.test_pathes):

            if int((count*100)/length) != percentage:

                percentage = int((count*100)/length)
                eta = time.strftime("%H:%M:%S", time.gmtime((100 - percentage) * (time.time() - prev_time)))
            
                print(f'Percentage of loaded test images = {percentage}% -- ETA = {eta}')

                prev_time = time.time()

            # Read the image and do not augment it.
            pre_img, angle_label = preprocess.image_read_and_augmenting(img_path, augmenting=False)
            X_test.append(pre_img)

            # Read the angle from the file name if the angle label is not available.
            if angle_label == None:
                angle_label = preprocess.read_angle_from_file(img_path, preprocess.h.test_path)      

            y_test.append(angle_label)

        # Transform the data to NumPy arrays for model use.
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        return X_test, y_test

    else:
        # If test data is not available, return None.
        return None, None

