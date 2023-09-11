# Import necessary libraries
import argparse
import cv2 as cv
import parameters
import os
import glob
import model

def main():
    """
    Main function of the program.

    Args: 
        None.

    Returns:
        None.
    """

    parser = argparse.ArgumentParser(description="Page rotation AI", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-S", "--image_size", default=430, type=int, help="input image size for model")
    parser.add_argument("-E", "--num_epochs", default=10, type=int, help="number of epochs to train")
    parser.add_argument("-p", "--pretrained", action="store_true", help="if model pretrained")
    parser.add_argument("-LR", "--learning_rate", default=0.006, type=float, help="learning rate for training")
    parser.add_argument("-ts", "--with_test", action="store_true", help="testing after training or not")
    parser.add_argument("-VS", "--validation_split", default=0.1, type=float, help="how much validation example \
                        from train dataset geting (in percentage)")
    parser.add_argument("-TI", "--train_amount_of_images", default=1000, type=int, help="number of images for training from train dataset")
    parser.add_argument("-TSI", "--test_amount_of_images", default=17, type=int, help="number of images for testing from test dataset")
    parser.add_argument("-TF", "--train_img_format", default='.jpg', type=str, help="format of images for training")
    parser.add_argument("-TSF", "--test_img_format", default='.jpg', type=str, help="format of images for testing")
    parser.add_argument("-L", "--labels_file_name", default='labels.json', type=str, help="name of file with labels (.json)")
    parser.add_argument("-r", "--rotation_preprocess", action="store_true", help="always using rotate augmentation for foto and angle for foto")
    parser.add_argument("-B", "--batch_size", default=128, type=int, help="batch size for training")
    parser.add_argument("-t", "--training", action="store_true", help="train model using dataset from src or predict and rotate photo of test_image_format from src to dst")
    parser.add_argument("-g", "--image_generator", action="store_true", help="generating image for training for dataset")
    parser.add_argument("-m", "--on_mnist", action="store_true", help="Train and test program on mnist dataset")
    parser.add_argument("-MXS", "--max_image_size", default=860, type=int, help="max size of image that input all, if bigger than croped")
    parser.add_argument("src", default='input/', help="dataset folder")
    parser.add_argument("dst", default='./', help="folder to save file")
    args = parser.parse_args()
    config = vars(args)

    # Get the training and testing flags.
    training = config['training']
    source_folder = config['src']
    destination_folder = config['dst']

    # Remove the training and testing flags from the config dictionary.
    del config['training']
    del config['src']
    del config['dst']

    # Set the train and test paths.
    config['train_path'] = os.path.join(source_folder, 'train/')
    config['test_path'] = os.path.join(source_folder, 'test/')

    # Get the parameters for the page_ai model.
    parameters.page_model_param_dict['angle_detector'] = config
    h_parameters = parameters.get_config()

    # Create the page_ai model.
    page_ai = model.page_ai(h_parameters)

    # If training, then train the model.
    if training:
        print('start training')
        page_ai.train()

    # Otherwise, predict the rotation of the images in the test folder and save the results to the destination folder.
    else:
        print('Start predicting')
        test_pathes = glob.glob(os.path.join(source_folder,'*.jpg')) 
        test_pathes.extend(glob.glob(os.path.join(source_folder,'*.png')))

        for path in test_pathes:
            print(f'now predict - {path}')
            img = cv.imread(path)
            new_img_name = f'rotated_{os.path.basename(path)}'

            new_img = page_ai.rotate_image(img)
            new_path = os.path.join(destination_folder, new_img_name)
            cv.imwrite(new_path, new_img)
            print(f'predicted - {new_path}')
    

if __name__=="__main__":
    main()
