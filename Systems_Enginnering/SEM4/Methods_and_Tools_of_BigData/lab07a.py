import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import sys


def zadanie2():
    # Iris Dataset
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    print("Iris dataset sample:")
    print(X.head())
    print("\nTarget names:", iris.target_names)
    print("\nFeature names:", iris.feature_names)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    # KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title("Confusion Matrix - Iris")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def zadanie3():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    print("Breast Cancer dataset sample:")
    print(X.head())

    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.3, random_state=42)

    # Logistic Regression
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=data.target_names, yticklabels=data.target_names)
    plt.title("Confusion Matrix - Breast Cancer")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def zadanie4():
    digits = load_digits()
    X = digits.data
    y = digits.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.3, random_state=42)

    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Digits classification accuracy:", accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title("Confusion Matrix - Digits")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print("\nMost confused digits:")
    most_confused = np.unravel_index(np.argmax(cm - np.diag(np.diag(cm))), cm.shape)
    print(f"Digit {most_confused[0]} confused with {most_confused[1]}")


def zadanie5(titanic_path):
    df = pd.read_csv(titanic_path)
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    X = df.drop(columns="Survived")
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nTitanic Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
    plt.title("Confusion Matrix - Titanic")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def zadanie6(heart_path):
    df = pd.read_csv(heart_path)
    X = df.drop(columns="DEATH_EVENT")
    y = df["DEATH_EVENT"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.3, random_state=42)

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nHeart Disease Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title("Confusion Matrix - Heart Disease")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


if __name__ == "__main__":
    zadanie2()
    zadanie3()
    zadanie4()
    zadanie5("/home/siwirus/Workspace/Studies_Repo/Systems_Enginnering/SEM4/Methods_and_Tools_of_BigData/data/Titanic-Dataset.csv")
    zadanie6("/home/siwirus/Workspace/Studies_Repo/Systems_Enginnering/SEM4/Methods_and_Tools_of_BigData/data/heart_failure_clinical_records_dataset.csv")


