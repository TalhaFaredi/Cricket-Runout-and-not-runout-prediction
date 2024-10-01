

# CNN for Run Out Prediction

This project implements a Convolutional Neural Network (CNN) to predict whether a player is run out or not based on image input. The model is built using TensorFlow and Keras and trained on a dataset of labeled images. It uses data augmentation to improve generalization and is capable of predicting "out" or "not out" for unseen images.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Introduction

This project aims to automate the prediction of run out scenarios in cricket using deep learning. We employ a Convolutional Neural Network (CNN) to classify images as either "out" or "not out." The model is trained on a set of images from cricket matches that are pre-labeled for the presence of a run out.

## Dataset

The dataset is divided into two directories:
- `train/`: Contains training images labeled as "out" or "not out".
- `validation/`: Contains validation images used for evaluating the model’s performance.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/run-out-prediction.git
   cd run-out-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` should include the following dependencies:
   - `tensorflow`
   - `numpy`
   - `Pillow`

3. Organize your dataset into `dataset/train` and `dataset/validation` directories, with subdirectories for each class (`out/` and `not_out/`).

## Training the Model

To train the CNN model, run the following script:
```bash
python train.py
```

The script performs the following steps:
1. Data augmentation on the training set.
2. Model initialization and compilation.
3. Training the CNN for 20 epochs.
4. Evaluation of the model on the validation dataset.

## Model Architecture

The CNN model consists of the following layers:
- **Conv2D**: Extracts features from the input images.
- **MaxPooling2D**: Reduces the spatial dimensions of the feature maps.
- **Flatten**: Converts the 2D feature maps into a 1D vector.
- **Dense**: Fully connected layers for prediction.
- **Sigmoid Activation**: Final output layer for binary classification.

### Model Summary:
- Conv2D (32 filters, 3x3 kernel)
- MaxPooling2D (2x2 pool size)
- Conv2D (64 filters, 3x3 kernel)
- MaxPooling2D (2x2 pool size)
- Conv2D (128 filters, 3x3 kernel)
- Conv2D (64 filters, 3x3 kernel)
- MaxPooling2D (2x2 pool size)
- Flatten
- Dense (512 units, ReLU activation)
- Dense (1 unit, Sigmoid activation for binary output)

## Evaluation

The model's performance is evaluated using the validation set after training. The script outputs the validation loss and accuracy:

```bash
Validation loss: 0.7577
Validation accuracy: 0.6346
```

To evaluate the model:
```bash
python evaluate.py
```

## Testing

To test the model on a new image:
```bash
python test.py --img_path "path_to_image.jpg"
```

This will output a prediction for the image as either "out" or "not out" based on the model’s output.

Example:
```bash
Predicted Label: out
```

## Future Enhancements

- **Expand the dataset**: Increase the number of labeled images to improve model performance.
- **Optimize model architecture**: Experiment with different architectures and hyperparameters to boost accuracy.
- **Real-time prediction**: Implement the model in a live video stream for real-time run-out detection.
- **Improve validation accuracy**: Perform hyperparameter tuning and implement regularization techniques.

## License

This project is licensed under the MIT License.
```

### Additional Notes:
1. **`train.py`**: This file contains the training code, including data augmentation, model building, and training logic.
2. **`evaluate.py`**: This file handles the evaluation of the model on the validation set.
3. **`test.py`**: This file is used to make predictions on individual images.
4. Ensure your dataset is structured as:
   ```
   dataset/
     ├── train/
     │   ├── out/
     │   └── not_out/
     └── validation/
         ├── out/
