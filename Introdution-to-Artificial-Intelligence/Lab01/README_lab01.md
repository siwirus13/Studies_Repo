# 3 Classificators

This project includes three tasks focused on handwritten digit classification using the MNIST dataset. Each task employs a different approach for training, evaluating, and visualizing model performance.

## Task 1: Logistic Regression on MNIST

- Uses scikit-learn's `LogisticRegression` to classify digits from the MNIST dataset.
- The dataset is scaled using `StandardScaler`.
- Evaluation includes accuracy, precision, recall, F1-score, and confusion matrix (both raw and logarithmic).
- Visualization:
  - Bar plots of precision and recall per class (log scale)
  - Heatmaps of the confusion matrix

## Task 2: Keras Model Evaluation on PNG Images

- Loads a pre-trained Keras model (`model.h5`) and tests it on PNG images stored in a specified folder.
- Images are preprocessed (grayscale, resized to 28x28, normalized).
- Evaluation includes accuracy, classification report, and confusion matrix (standard and log scale).
- Displays file names for misclassified images.

## Task 3: Random Forest Classification on MNIST

- Trains a `RandomForestClassifier` on the MNIST dataset.
- Evaluates the model using accuracy, precision, recall, and a detailed classification report.
- Visualizes:
  - Sample digit images with labels
  - Bar plots for precision and recall per class (log scale)
  - Confusion matrix visualizations (raw and log scale)

## Citations

Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373
"""
