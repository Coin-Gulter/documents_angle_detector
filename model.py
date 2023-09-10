import numpy as np
import cv2 as cv
import os
import data_loader
import augmentation
import tensorflow as tf
from scipy import ndimage
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dropout, Dense, Flatten, MaxPooling2D

class page_ai():
    """
    Class for the page_ai model.

    Args:
        h_parameters (dict): Hyperparameters for the model.
    """

    def __init__(self, h_parameters):
        """
        Constructor for the page_ai class.

        Args:
            h_parameters (dict): Hyperparameters for the model.
        """

        self.h = h_parameters

        # Check if the pretrained model is used.
        if self.h.pretrained:
            print('Loading pretrained model...')
            self.model = tf.keras.models.load_model(self.h.save_model_name)
            print('Model loaded')
        else:
            # Create the model.
            print('Creating model...')
            self.model = Sequential()

            # Add a convolutional layer with 16 filters, 3x3 kernel size and ReLU activation.
            self.model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(self.h.image_size, self.h.image_size, 1)))

            # Add a 2D max pooling layer after a conv. layer to reduce the dimensionality of the feature maps and allow for better model generalisiation.
            self.model.add(MaxPooling2D((2, 2)))

            # Add another conv. layer with 48 filters, allowing the model to learn more feature maps / patterns.
            self.model.add(Conv2D(48, (3, 3), activation='relu'))

            # Add a 2D max pooling layer after a conv. layer to reduce the dimensionality of the feature maps and allow for better model generalisiation.
            self.model.add(MaxPooling2D((2, 2)))

            # Flatten the output from the previous layer into a 1D vector.
            self.model.add(Flatten())

            # Add another dense layer to add more capacity to the model and to learn more complex features from the flattened output of the previous layer.
            self.model.add(Dense(512, activation='relu'))

            # Apply a dropout regularization with a rate of 0.5 to reduce overfitting.
            self.model.add(Dropout(self.h.lr_dropout))

            # Use a Dense Layer to convert previous output into a suitable form (for 8 * 8 fields * 13 categories = 832). Using 'softmax' to create a probability distribution over the possible classes.
            self.model.add(Dense(self.h.num_classes, activation=self.h.act_type))
            print('Model created')

        # Summarize the model.
        self.model.summary()


    def train(self):
        """
        Function to train the page_ai model.

        Args:
            None.

        Returns:
            None.
        """

        # Load the training and test data.
        X_train, y_train, X_test, y_test = data_loader.loading_data(self.h)

        # Compile the model.
        self.model.compile(optimizer=self.h.optimizer, loss=self.h.loss, metrics=['accuracy'])

        # Train the model.
        self.model.fit(np.array(X_train), 
                        np.array(y_train), 
                        epochs=self.h.num_epochs, 
                        batch_size=self.h.batch_size, 
                        validation_split=self.h.validation_split)

        # Evaluate the model on the test set.
        if X_test.any() != None:
            test_loss, test_acc = self.model.evaluate(np.array(X_test),
                                                        np.array(y_test))
            print('Tested lost:', test_loss)
            print('Tested accuracy:', test_acc)

        # Save the model.
        self.model.save(os.path.join(self.h.save_model_folder, self.h.save_model_name))
    

    def predict(self, img:np.ndarray):
        """
        Function to predict the rotation of an image.

        Args:
            img (np.ndarray): Image to be predicted.

        Returns:
            int: Predicted rotation angle.
        """

        # Check if the image is grayscale or RGB.
        if len(img.shape) < 3:
            pass
        elif len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            raise Exception(f'Given image not in correct format - {img.shape} , \
                                try using image of shape (x, y, 1) for grayscale or (x, y, 3) if image colorful')

        # Preprocess the image.
        img_size = (self.h.image_size, self.h.image_size)
        img = augmentation.prep_img(img, self.h)

        # Resize the image.
        resized_image = cv.resize(img, img_size)

        # Normalize the image.
        resized_image = resized_image.astype('float32') / 255

        # Reshape the image.
        resized_image = resized_image.reshape((1, self.h.image_size, self.h.image_size, 1))

        # Make the prediction.
        result = self.model.predict(resized_image)
        angle = tf.argmax(result[0])

        return angle.numpy()


    def rotate_image(self, img:np.ndarray):
        """
        Function to rotate an image.

        Args:
            img (np.ndarray): Image to be rotated.

        Returns:
            np.ndarray: Rotated image.
        """

        # Predict the rotation angle of the image.
        result = self.predict(img)
        print('result - ', result)

        # Rotate the image by the predicted angle.
        img = ndimage.rotate(img, angle=-result)

        return img
