
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def zadanie1():
    # Wczytanie zbioru danych Iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Eksploracja danych
    print("Pierwsze 5 wierszy danych:")
    print(pd.DataFrame(X, columns=iris.feature_names).head())
    
    # Przygotowanie danych do budowy modelu KNN
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Budowa klasyfikatora KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    # Ocena jakości klasyfikatora
    print("\nMetryki klasyfikatora KNN:")
    print(f"Dokładność: {accuracy_score(y_test, y_pred)}")
    print(f"Precyzja: {precision_score(y_test, y_pred, average='weighted')}")
    print(f"Czułość: {recall_score(y_test, y_pred, average='weighted')}")
    print(f"Specyficzność: {recall_score(y_test, y_pred, average='weighted', pos_label=0)}")

def zadanie2():
    # Wczytanie zbioru danych Breast Cancer
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    
    # Czyszczenie danych
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Budowa modelu regresji logistycznej
    log_reg = LogisticRegression(max_iter=10000)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    
    # Ocena jakości klasyfikatora
    print("\nMetryki klasyfikatora regresji logistycznej:")
    print(f"Dokładność: {accuracy_score(y_test, y_pred)}")
    print(f"Precyzja: {precision_score(y_test, y_pred)}")
    print(f"Czułość: {recall_score(y_test, y_pred)}")
    print(f"Specyficzność: {recall_score(y_test, y_pred, pos_label=0)}")

def zadanie3():
    # Wczytanie zbioru danych Digits (MNIST)
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Przygotowanie danych
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Zastosowanie SVM do klasyfikacji cyfr
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    
    # Ocena jakości klasyfikatora
    print("\nMetryki klasyfikatora SVM:")
    print(f"Dokładność: {accuracy_score(y_test, y_pred)}")
    print(f"Macierz pomyłek:\n{confusion_matrix(y_test, y_pred)}")

def zadanie4():
    # Wczytanie zbioru danych Titanic
    titanic = fetch_openml('titanic', version=1)
    df = pd.DataFrame(titanic.data, columns=titanic.feature_names)
    df['survived'] = titanic.target
    
    # Eksploracja danych
    print("\nPierwsze 5 wierszy danych Titanic:")
    print(df.head())
    
    # Przygotowanie danych
    df = df.dropna()  # Usuwanie brakujących wartości
    df = pd.get_dummies(df, drop_first=True)  # Kodowanie zmiennych kategorycznych
    X = df.drop('survived', axis=1)
    y = df['survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Budowa klasyfikatora regresji logistycznej
    log_reg = LogisticRegression(max_iter=10000)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    
    # Ocena jakości klasyfikatora
    print("\nMetryki klasyfikatora regresji logistycznej Titanic:")
    print(f"Dokładność: {accuracy_score(y_test, y_pred)}")
    print(f"Krzywa ROC-AUC: {roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])}")

def zadanie5():
    # Wczytanie zbioru danych Heart Disease
    heart = fetch_openml('heart-disease', version=1)
    df = pd.DataFrame(heart.data, columns=heart.feature_names)
    df['target'] = heart.target
    
    # Eksploracja danych
    print("\nPierwsze 5 wierszy danych Heart Disease:")
    print(df.head())
    
    # Przygotowanie danych
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Budowa klasyfikatora SVM
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    
    # Ocena jakości klasyfikatora
    print("\nMetryki klasyfikatora SVM Heart Disease:")
    print(f"Precyzja: {precision_score(y_test, y_pred)}")
    print(f"Czułość: {recall_score(y_test, y_pred)}")
    print(f"Specyficzność: {recall_score(y_test, y_pred, pos_label=0)}")

if __name__ == "__main__":
    zadanie1()
    zadanie2()
    zadanie3()
    zadanie4()
    zadanie5()
