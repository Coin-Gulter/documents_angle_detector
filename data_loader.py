import preprocesing
from tensorflow.keras.utils import to_categorical


def loading_data(h_parameters):
    """
    Load the data from the disk and prepare it for training and testing.

    Args:
        h_parameters (dict): Hyperparameters of the model.

    Returns:
        X_train, y_train, X_test, y_test: Training and test data.
    """

    print('Start data loader...')

    # Load the training data.
    X_train, y_train = preprocesing.prepare_data_train(h_parameters)

    # Load the test data.
    X_test, y_test = preprocesing.prepare_data_test(h_parameters)

    print('create train/test dataset')

    # Reshape the training data to have a shape of (number of images, image size, image size, 1).
    X_train = X_train.reshape((h_parameters.train_amount_of_images, h_parameters.image_size, h_parameters.image_size, 1))

    # Normalize the training data by dividing each pixel value by 255.
    X_train = X_train.astype('float32') / 255

    # Convert the labels to one-hot encoded vectors.
    y_train = to_categorical(y_train, num_classes=h_parameters.num_classes)

    # If the test data exists, reshape it and normalize it.
    if X_test.any() != None:
        X_test = X_test.reshape((h_parameters.test_amount_of_images, h_parameters.image_size, h_parameters.image_size, 1))
        X_test = X_test.astype('float32') / 255

        y_test = to_categorical(y_test, num_classes=h_parameters.num_classes)

    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    if X_test.any() != None:
        print("y_test: ", y_test.shape)
        print("X_test: ", X_test.shape)

    return X_train, y_train, X_test, y_test

