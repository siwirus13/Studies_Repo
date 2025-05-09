
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.datasets import load_files



def zadanie1():
    # Przykładowy zbiór danych psychologicznych - symulowany dla celów demonstracyjnych
    np.random.seed(0)
    data = pd.DataFrame(np.random.rand(200, 10))

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)

    nmf = NMF(n_components=2, init='random', random_state=0)
    nmf_result = nmf.fit_transform(np.abs(data_scaled))

    tsne = TSNE(n_components=2, random_state=0)
    tsne_result = tsne.fit_transform(data_scaled)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(pca_result[:, 0], pca_result[:, 1])
    axes[0].set_title("PCA")

    axes[1].scatter(nmf_result[:, 0], nmf_result[:, 1])
    axes[1].set_title("NMF")

    axes[2].scatter(tsne_result[:, 0], tsne_result[:, 1])
    axes[2].set_title("t-SNE")

    plt.tight_layout()
    plt.show()


def zadanie2():
    from sklearn.datasets import load_files
    import os
    # Użycie domyślnego datasetu dla uproszczenia - można zastąpić rzeczywistym Amazon Reviews
    reviews = fetch_20newsgroups(subset='train', categories=['rec.autos', 'rec.sport.baseball'])
    texts = reviews.data[:500]
    labels = [0 if 'autos' in reviews.target_names[i] else 1 for i in reviews.target[:500]]

    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts).toarray()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.title("PCA na danych tekstowych (symulacja sentymentu)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label='Sentyment')
    plt.show()


def zadanie3():
    categories = ['talk.politics.misc', 'rec.autos', 'sci.space']
    newsgroups = fetch_20newsgroups(subset='train', categories=categories)

    tfidf = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf.fit_transform(newsgroups.data)

    svd = TruncatedSVD(n_components=2)
    X_svd = svd.fit_transform(X_tfidf)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_svd[:, 0], X_svd[:, 1], c=newsgroups.target, cmap='tab10', alpha=0.7)
    plt.title("SVD redukcja wymiarowości (TF-IDF)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(label='Kategoria')
    plt.show()

    print("Przykładowe dokumenty:")
    for i in range(3):
        print(f"Dokument {i+1}: {newsgroups.data[i][:200]}...")


def zadanie4():
    categories = ['talk.politics.misc', 'rec.sport.baseball', 'sci.space']
    newsgroups = fetch_20newsgroups(subset='train', categories=categories)

    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X_tfidf = tfidf.fit_transform(newsgroups.data)

    lda = LDA(n_components=3, random_state=0)
    X_lda = lda.fit_transform(X_tfidf)

    feature_names = tfidf.get_feature_names_out()
    for idx, topic in enumerate(lda.components_):
        print(f"Temat {idx+1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-11:-1]]))

    plt.figure(figsize=(8, 6))
    plt.scatter(X_lda[:, 0], X_lda[:, 1], c=newsgroups.target, cmap='tab10', alpha=0.7)
    plt.title("LDA - przypisanie tematów")
    plt.xlabel("Topic 1")
    plt.ylabel("Topic 2")
    plt.colorbar(label='Kategoria')
    plt.show()


if __name__ == "__main__":
    zadanie1()
    zadanie2()
    zadanie3()
    zadanie4()
