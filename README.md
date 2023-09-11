# Page angle detection model

This repository contains program for detecting angle of object on images from 0 to 360 degree. The code is written in Python3.

## Getting Started

To get started, you will need to download project and install dependencies. You can clone the repository and run the following command to install the project dependencies:

pip install -r requirements.txt


## Usage

You can use this program as import to your own code or as undependent program in CMD to rotate document photos. To use the code as CMD (comand line program) you need to run "main.py" file with approprite flags. For example to train:

    python main.py -TI 100 -S 480 -E 7 -VS 0.2 -ts -r -t -g input/ ./

Where "input/" it's folder dataset folder that must have two subfolders "train" and "test". "./" its path where program put its saved model file "angle_detect_ai.keras" after training.
You can see all flags if you run:

    python main.py --help

and see:

    usage: main.py [-h] [-S IMAGE_SIZE] [-E NUM_EPOCHS] [-p] [-LR LEARNING_RATE] [-ts] [-VS VALIDATION_SPLIT]
               [-TI TRAIN_AMOUNT_OF_IMAGES] [-TSI TEST_AMOUNT_OF_IMAGES] [-TF TRAIN_IMG_FORMAT] [-TSF TEST_IMG_FORMAT]
               [-L LABELS_FILE_NAME] [-r] [-B BATCH_SIZE] [-t] [-g]
               src dst

    Page rotation AI

    positional arguments:
    src                   dataset folder
    dst                   folder to save file

    options:
    -h, --help            show this help message and exit
    -S IMAGE_SIZE, --image_size IMAGE_SIZE
                            input image size for model (default: 600)
    -E NUM_EPOCHS, --num_epochs NUM_EPOCHS
                            number of epochs to train (default: 8)
    -p, --pretrained      if model pretrained (default: False)
    -LR LEARNING_RATE, --learning_rate LEARNING_RATE
                            learning rate for training (default: 0.006)
    -ts, --with_test      testing after training or not (default: False)
    -VS VALIDATION_SPLIT, --validation_split VALIDATION_SPLIT
                            how much validation example from train dataset geting (in percentage) (default: 0.1)
    -TI TRAIN_AMOUNT_OF_IMAGES, --train_amount_of_images TRAIN_AMOUNT_OF_IMAGES
                            number of images for training from train dataset (default: 1000)
    -TSI TEST_AMOUNT_OF_IMAGES, --test_amount_of_images TEST_AMOUNT_OF_IMAGES
                            number of images for testing from test dataset (default: 17)
    -TF TRAIN_IMG_FORMAT, --train_img_format TRAIN_IMG_FORMAT
                            format of images for training (default: .jpg)
    -TSF TEST_IMG_FORMAT, --test_img_format TEST_IMG_FORMAT
                            format of images for testing (default: .jpg)
    -L LABELS_FILE_NAME, --labels_file_name LABELS_FILE_NAME
                            name of file with labels (.json) (default: labels.json)
    -r, --rotation_preprocess
                            always using rotate augmentation for foto and angle for foto (default: False)
    -B BATCH_SIZE, --batch_size BATCH_SIZE
                            batch size for training (default: 64)
    -t, --training        train model using dataset from src or predict and rotate photo of test_image_format from src
                            to dst (default: False)
    -g, --image_generator
                            generating image for training for dataset (default: False)

If you want to use program for image rotation you don't need "-t" flags instead better use "-p", for example:

    python main.py -S 240 -p input/using/ input/using/test

Where "input/using/" it's folder with images to rotate and "input/using/test" path to save rotated images. 

In repository six main files:
    augmentation.py - for images augmentation methods and image changing
    data_loader.py - for data loading to train and test model
    main.py - main function of program to start everything
    model.py - machine learning model to train, test, predict result in the end
    parameters.py - hyper parameters for program, every setting saved here, you can change setting if you need
    preprocesing.py - main processing with images dataset

## Parameters

In parameters.py there are "default_configs()" function with default parameters and "page_model_param_dict" dict that override some parameters from default. You can change both if you need. To override you can change "page_model_param_dict" and after that use function "get_config()". You need to know that when you run "main.py" default setting automatically override with default setting of ArgumentParser so you need to change it too.

Contributions are welcome! Please open a pull request if you have any improvements or bug fixes.

## Pretrained
Repository on github doesn't have pretrained model because of its big size around 2GB. So if you want to download it you can do it from google drive using link:

    https://drive.google.com/drive/folders/196l4I5IJmFuuoIanXSEW9cEOkKBXUJae?usp=sharing

## License

This project is licensed under the MIT License.
```

I hope this is helpful!