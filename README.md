# MNIST Digit Classification with CNN

This repository contains a Jupyter Notebook implementation of a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset using TensorFlow and Keras.

## Project Overview

The notebook (`main.ipynb`) demonstrates the process of building, training, and evaluating a CNN model to classify handwritten digits (0-9) from the MNIST dataset. The model achieves high accuracy through convolutional layers, max pooling, and fully connected layers.

## Requirements

To run the notebook, ensure you have the following Python libraries installed:

- TensorFlow
- NumPy
- Matplotlib

You can install them using pip:

```bash
pip install tensorflow numpy matplotlib
```

## Notebook Structure

1. **Import Libraries**: Loads necessary libraries including TensorFlow, Keras, NumPy, and Matplotlib.
2. **Data Preparation**:
   - Loads the MNIST dataset using `tf.keras.datasets.mnist`.
   - Normalizes pixel values to the range [0, 1].
   - Reshapes images to include a channel dimension for CNN input.
3. **Model Creation**:
   - Builds a Sequential CNN model with:
     - Three `Conv2D` layers with ReLU activation (32, 32, and 64 filters).
     - Three `MaxPooling2D` layers for dimensionality reduction.
     - A `Flatten` layer to transition to fully connected layers.
     - Two `Dense` layers (64 units with ReLU, 10 units with softmax).
   - Compiles the model with `sparse_categorical_crossentropy` loss, `adam` optimizer, and `accuracy` metric.
4. **Model Training**:
   - Trains the model on the training dataset for 3 epochs.
5. **Model Evaluation**:
   - Evaluates the model on the test dataset to report accuracy and loss.
6. **Prediction Example**:
   - Visualizes a test image and predicts its digit using the trained model.

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/mnist-cnn-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd mnist-cnn-classification
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook main.ipynb
   ```
4. Run the cells in the notebook to train the model and test predictions.

## Results

- The model achieves approximately 98% accuracy on the test dataset after 3 epochs.
- A sample prediction is demonstrated by visualizing a test image and predicting its digit.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
