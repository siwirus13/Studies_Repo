## TASK 3

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(np.uint8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Prediction
y_pred = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Dokładność modelu: {accuracy:.4f}')

report = classification_report(y_test, y_pred, digits=4)
print(report)

# Visualisation
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Label: {y_test[i]}')
    ax.axis('off')
plt.show()

# Math
precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
classes = np.arange(10)

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(x=classes, y=precision, label='Precision', color='blue', alpha=0.6)
sns.barplot(x=classes, y=recall, label='Recall', color='red', alpha=0.6)
plt.yscale('log')
plt.xlabel('Number')
plt.ylabel('Value')
plt.title('Precision and Recall for Each Class')
plt.legend()
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred)

# Logarithmic scale transition
conf_matrix_log = np.where(conf_matrix > 0, np.log(conf_matrix), 0)

# Plot confusion matrix with logarithmic scale
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_log, annot=True, fmt='.2f', cmap='YlGn', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Logarithmic Scale)')
plt.show()

conf_matrix2 = confusion_matrix(y_test, y_pred)

# Plot confusion matrix with logarithmic scale
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix2, annot=True, fmt='.2f', cmap='RdPu', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
