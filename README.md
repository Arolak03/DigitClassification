## README for Image Classifier using PyTorch

This code implements a basic image classifier for the MNIST handwritten digit dataset using PyTorch.

**Functionality:**

* **Data Loading:**
    * Downloads the MNIST dataset (if not already downloaded) and applies a `ToTensor` transformation to convert images to tensors suitable for neural network training.
    * Creates a `DataLoader` to iterate over the training data in batches.
* **Model Architecture:**
    * Defines an `ImageClassifier` class inheriting from `nn.Module`.
    * The classifier uses convolutional layers for feature extraction followed by fully connected layers for classification.
* **Training:**
    * Trains the model for 10 epochs using the Adam optimizer and cross-entropy loss function.
    * Prints the loss after each epoch.
* **Saving and Loading:**
    * Saves the trained model's state dictionary to a file (`model_state.pt`).
    * Loads the saved model when needed for inference.
* **Inference:**
    * Opens an image (assumed to be located at `/content/image.jpg`).
    * Applies the same transformation used for training data.
    * Makes a prediction using the loaded model and prints the predicted label.

**Requirements:**

* Python 3.x
* PyTorch
* Pillow (PIL Fork)
* torchvision

**Running the Script:**

1. Save the code as a Jupyter Notebook file (e.g., `Untitled1.ipynb`).
2. Open the notebook in Google Colab or a local Jupyter Notebook environment.
3. Run the notebook cells. It will download the MNIST dataset (if needed), train the model, save it, and perform inference on an image.

**Notes:**

* This is a basic example and can be extended with more complex architectures, hyperparameter tuning, and data augmentation techniques.
* The code assumes the image for inference is located at `/content/image.jpg`. Modify the path if the image is in a different location.
