import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# =====================================================
# === NLTK RESOURCES ==================================
# =====================================================
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("vader_lexicon")

# =====================================================
# === LOAD AND CLEAN DATA ==============================
# =====================================================
def load_reviews(filename):
    df = pd.read_csv(filename)

    df = df.rename(columns={"Review Text": "review_text"})
    df = df.dropna(subset=["review_text"])
    df = df[df["review_text"].str.strip() != ""]

    return df


# =====================================================
# === SENTIMENT ANALYSIS ===============================
# =====================================================
def analyze_sentiment(df):
    sia = SentimentIntensityAnalyzer()

    sentiments = []
    scores = []

    for text in df["review_text"]:
        score = sia.polarity_scores(text)["compound"]
        scores.append(score)

        if score >= 0.05:
            sentiments.append("positive")
        elif score <= -0.05:
            sentiments.append("negative")
        else:
            sentiments.append("neutral")

    df["sentiment"] = sentiments
    df["sentiment_score"] = scores

    return df


# =====================================================
# === WORD EXTRACTION =================================
# =====================================================
def tokenize_reviews(reviews):
    stop_words = set(stopwords.words("english"))
    tokenized = []

    for text in reviews:
        tokens = word_tokenize(text.lower())
        tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
        tokenized.append(set(tokens))  # SET -> jedno wystąpienie na opinię

    return tokenized


# =====================================================
# === TOP WORDS (UNIQUE + % REVIEWS) ==================
# =====================================================
def top_words_unique_with_percentage(df, top_n=10):
    results = {}

    for sentiment in ["positive", "negative"]:
        reviews = df[df["sentiment"] == sentiment]["review_text"]
        tokenized_reviews = tokenize_reviews(reviews)

        all_words = Counter()
        for tokens in tokenized_reviews:
            all_words.update(tokens)

        results[sentiment] = {
            "counter": all_words,
            "tokenized_reviews": tokenized_reviews,
            "total_reviews": len(tokenized_reviews)
        }

    pos_words = []
    neg_words = []

    pos_iter = results["positive"]["counter"].most_common()
    neg_iter = results["negative"]["counter"].most_common()

    pos_idx = neg_idx = 0

    while len(pos_words) < top_n and pos_idx < len(pos_iter):
        word, count = pos_iter[pos_idx]
        if word not in [w for w, _, _ in neg_words]:
            percentage = (count / results["positive"]["total_reviews"]) * 100
            pos_words.append((word, count, round(percentage, 2)))
        pos_idx += 1

    while len(neg_words) < top_n and neg_idx < len(neg_iter):
        word, count = neg_iter[neg_idx]
        if word not in [w for w, _, _ in pos_words]:
            percentage = (count / results["negative"]["total_reviews"]) * 100
            neg_words.append((word, count, round(percentage, 2)))
        neg_idx += 1

    return pos_words, neg_words


# =====================================================
# === VISUALIZATION ===================================
# =====================================================
def plot_sentiment_distribution(df):
    df["sentiment"].value_counts().plot(
        kind="bar",
        title="Sentiment Distribution (NLTK VADER)",
        ylabel="Number of Reviews"
    )
    plt.tight_layout()
    plt.savefig("sentiment_distribution.png")
    plt.close()


# =====================================================
# === MAIN ============================================
# =====================================================
def main():
    print("Loading dataset...")
    df = load_reviews("reviews.csv")

    print("Performing sentiment analysis...")
    df = analyze_sentiment(df)

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    print("\nTop 10 POSITIVE words (unique + % reviews):")
    positive_words, negative_words = top_words_unique_with_percentage(df)

    for word, count, pct in positive_words:
        print(f"{word}: {count} reviews ({pct}%)")

    print("\nTop 10 NEGATIVE words (unique + % reviews):")
    for word, count, pct in negative_words:
        print(f"{word}: {count} reviews ({pct}%)")

    plot_sentiment_distribution(df)

    df.to_csv("reviews_with_sentiment.csv", index=False)
    print("\nSaved file: reviews_with_sentiment.csv")
    print("Saved plot: sentiment_distribution.png")


# =====================================================
# === RUN =============================================
# =====================================================
if __name__ == "__main__":
    main()
