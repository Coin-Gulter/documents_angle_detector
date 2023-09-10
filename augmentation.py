import numpy as np
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

    def image_contrast(self, image):
        """
        Apply contrast to the image.

        Args:
            image (np.array): The image.

        Returns:
            np.array: The image with contrast applied.
        """

        # Apply contrast with probability h.contrast_prob.

        if np.random.rand() < self.h.contrast_prob:
            brightness = 10
            contrast = random.randint(40, 100)

            # Apply contrast to the image.
            dummy = np.int16(image)
            dummy = dummy * (contrast / 127 + 1) - contrast + brightness
            dummy = np.clip(dummy, 0, 255)
            img = np.uint8(dummy)

            return img

        else:
            # Do not apply contrast.
            return image

    def image_blur(self, image):
        """
        Apply blur to the image.

        Args:
            image (np.array): The image.

        Returns:
            np.array: The image with blur applied.
        """

        # Apply blur with probability h.blur_prob.

        if np.random.rand() < self.h.blur_prob:
            img = image.copy()
            fsize = 3

            # Apply blur to the image.
            return cv.GaussianBlur(img, (fsize, fsize), 0)

        else:
            # Do not apply blur.
            return image

    def image_rotate(self, image):
        """
        Rotate the image.

        Args:
            image (np.array): The image.

        Returns:
            np.array, int: The rotated image and the angle.
        """

        # Rotate the image with probability h.rotation_preprocess.

        if self.h.rotation_preprocess:
            angle_label = random.randint(0, self.h.max_angle)
            rotated_img = ndimage.rotate(image, angle_label)

            return rotated_img, angle_label

        else:
            # Do not rotate the image.
            return image, None
        
    def image_sharpering(self, image):

        """
        Sharpen the image.

        Args:
            image (np.array): The image.

        Returns:
            np.array: The sharpened image.
        """

        # Create a sharpening kernel.
        sharpening = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])

        # Apply the sharpening kernel to the image.
        sharpened = cv.filter2D(image, -1, sharpening)

        return sharpened



def augmenting_photo(image, h_parameters):

    """
    Augment the image.

    Args:
        image (np.array): The image.
        h_parameters (dict): Hyperparameters of the model.

    Returns:
        np.array, int: The augmented image and the angle.
    """

    augment = image_augmentation(h_parameters)

    augment_methods = {'sharpening': augment.image_sharpering,
                       'blur': augment.image_blur,
                       'rotate': augment.image_rotate,
                       'contrast': augment.image_contrast,
                       'invert': augment.image_invert}

    mix_prob = h_parameters['mixing_prob']
    augment_types = h_parameters['augmenting_types']
    angle = None

    if np.random.rand() < mix_prob:
        # Choose two augmentation methods randomly.
        augment1 = random.choice(augment_types)
        augment2 = random.choice(augment_types)

        # If rotation is enabled, apply it first.
        if h_parameters['rotation_preprocess']:
            image, angle = augment_methods['rotate'](image)

        # Apply the first augmentation method.
        augmented_image = augment_methods[augment1](image)

        # Apply the second augmentation method.
        augmented_image = augment_methods[augment2](augmented_image)

    else:
        # Choose one augmentation method randomly.
        augment1 = random.choice(augment_types)

        # If rotation is enabled, apply it first.
        if h_parameters['rotation_preprocess']:
            image, angle = augment_methods['rotate'](image)

        # Apply the chosen augmentation method.
        augmented_image = augment_methods[augment1](image)

    return augmented_image, angle


def prep_img(image, h_parameters):

    """
    Prepare the image for the model.

    Args:
        image (np.array): The image.
        h_parameters (dict): Hyperparameters of the model.

    Returns:
        np.array: The prepared image.
    """

    augment = image_augmentation(h_parameters)

    # Invert the image.
    inverted_image = augment.image_invert(image)

    # Sharpen the image.
    sharpened_image = augment.image_sharpering(inverted_image)

    return sharpened_image

