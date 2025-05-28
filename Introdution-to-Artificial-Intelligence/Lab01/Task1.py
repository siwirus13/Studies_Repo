## TASK 1 


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

matplotlib.use("TkAgg")

"""

Citation:

Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373

"""

# Fetch EMNIST 
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]

y = y.astype(np.int8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42, stratify=y)


# Data preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Model training
model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto', n_jobs=-1)
model.fit(X_train, y_train)


# Prediction and accuracy
y_pred = model.predict(X_test)

test_acc = accuracy_score(y_test, y_pred)
print(f'Accuracy on test set: {test_acc:.4f}')

report = classification_report(y_test, y_pred, digits=4)
print(report)


# Visualisation
classes = list(range(10))

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=classes)

plt.figure(figsize=(10, 5))
sns.barplot(x=classes, y=precision, label='Precision', color='blue', alpha=0.6)
sns.barplot(x=classes, y=recall, label='Recall', color='red', alpha=0.6)
plt.yscale('log')
plt.xlabel('Number')
plt.ylabel('Value')
plt.title('Precision and Recall for each class (log scale)')
plt.legend()
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred)

conf_matrix_log = np.where(conf_matrix > 0, np.log(conf_matrix), 0)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_log, annot=True, fmt='.2f', cmap='YlGn', xticklabels=classes, yticklabels=classes)
plt.xlabel('Prediction')
plt.ylabel('True Class')
plt.title('Confusion Matrix (log scale)')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='RdPu', xticklabels=classes, yticklabels=classes)
plt.xlabel('Prediction')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()
