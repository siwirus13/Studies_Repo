import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_digits, fetch_lfw_people, load_wine, fetch_20newsgroups
from sklearn.decomposition import PCA, NMF, TruncatedSVD, LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def zadanie2():
    """Redukcja PCA na danych breast cancer i wizualizacja."""
    data = load_breast_cancer()
    X = data.data
    y = data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.7)
    plt.title("PCA – Breast Cancer Dataset")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def zadanie3():
    """Redukcja t-SNE na danych digits i wizualizacja 2D."""
    digits = load_digits()
    X = digits.data
    y = digits.target

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.title("t-SNE – Digits Dataset")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter, label="Digit Label")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def zadanie4():
    """Redukcja NMF na danych LFW i wizualizacja komponentów jako obrazów."""
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X = lfw_people.data
    n_components = 6

    nmf = NMF(n_components=n_components, init='nndsvda', random_state=0, max_iter=200)
    W = nmf.fit_transform(X)
    H = nmf.components_

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(H[i].reshape(lfw_people.images.shape[1:]), cmap='gray')
        ax.set_title(f"Component {i + 1}")
        ax.axis("off")
    plt.suptitle("NMF Components – LFW Faces")
    plt.tight_layout()
    plt.show()


def zadanie5():
    """Redukcja SVD na danych Wine z analizą wariancji i wizualizacją."""
    data = load_wine()
    X = data.data
    y = data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svd = TruncatedSVD(n_components=X.shape[1] - 1)
    X_svd = svd.fit_transform(X_scaled)
    explained = np.cumsum(svd.explained_variance_ratio_)

    n_components = np.argmax(explained >= 0.95) + 1

    print(f"Optymalna liczba komponentów (>=95% wariancji): {n_components}")

    svd = TruncatedSVD(n_components=n_components)
    X_reduced = svd.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='Set1', alpha=0.7)
    plt.title("SVD – Wine Dataset (pierwsze 2 komponenty)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(*scatter.legend_elements(), title="Wine Class")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def zadanie6():
    """Przetwarzanie tekstu i LDA na danych 20 Newsgroups."""
    newsgroups = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        stop_words='english'
    )
    X_counts = vectorizer.fit_transform(newsgroups.data)

    lda = LatentDirichletAllocation(n_components=10, max_iter=10, learning_method='online', random_state=0)
    X_lda = lda.fit_transform(X_counts)

    dominant_topics = np.argmax(X_lda, axis=1)

    plt.figure(figsize=(10, 6))
    plt.hist(dominant_topics, bins=np.arange(11) - 0.5, rwidth=0.8, color='skyblue', edgecolor='black')
    plt.title("LDA – Dominujące tematy w 20 Newsgroups")
    plt.xlabel("Temat")
    plt.ylabel("Liczba dokumentów")
    plt.xticks(range(10))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    zadanie2()
    zadanie3()
    zadanie4()
    zadanie5()
    zadanie6()
 



