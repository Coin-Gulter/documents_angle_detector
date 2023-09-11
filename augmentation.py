import numpy as np
import time
import cv2 as cv
import random
from scipy import ndimage

class image_augmentation():

    def __init__(self, h_parameters):
        self.h = h_parameters

    """
    Class for image augmentation.

    Args:
        h_parameters (dict): Hyperparameters of the model.
    """


    def image_invert(self, image):
        """
        Invert the image.

        Args:
            image (np.array): The image.

        Returns:
            np.array: The inverted image.
        """

        return (255 - image)
    

    def image_random_crop(self, image):
        if image.shape[0] > self.h.max_image_size:
            max_y = image.shape[0] - self.h.max_image_size
            y = np.random.randint(0, max_y)
        else:
            y = 0
        
        if image.shape[1] > self.h.max_image_size:
            max_x = image.shape[1] - self.h.max_image_size
            x = np.random.randint(0, max_x)
        else:
            x = 0

        crop = image[y: y + self.h.max_image_size, x: x + self.h.max_image_size]

        return crop

    def image_rotate(self, image):
        """
        Rotate the image.

        Args:
            image (np.array): The image.

        Returns:
            np.array, int: The rotated image and the angle.
        """
        if self.h.rotation_preprocess:
            angle_label = random.randint(0, self.h.max_angle*2)
            angle = angle_label - self.h.max_angle
            rotated_img = ndimage.rotate(image, angle)

            return rotated_img, angle_label

        else:
            # Do not rotate the image.
            return image, None


def prep_img(image, h_parameters, with_angle=True, invert=True, random_crop=True, to_gray=True):

    """
    Prepare the image for the model.

    Args:
        image (np.array): The image.v
        h_parameters (dict): Hyperparameters of the model.

    Returns:
        np.array: The prepared image.
    """

    augment = image_augmentation(h_parameters)
    angle = None

    if random_crop:
        image = augment.image_random_crop(image)

    if to_gray:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    if invert:
        # Invert the image.
        image = augment.image_invert(image)

    if with_angle:
        image, angle = augment.image_rotate(image)
        
        
    _, binar_img = cv.threshold(image, 128, 255, cv.THRESH_BINARY)


    return binar_img, angle

