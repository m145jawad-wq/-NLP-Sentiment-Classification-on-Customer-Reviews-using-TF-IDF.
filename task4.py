
import re
import zipfile
import urllib.request
import argparse
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from nltk.stem import WordNetLemmatizer

# ==============================
# CONFIGURATION
# ==============================

DATA_DIR = Path("sentiment_data")
MODEL_DIR = Path("saved_models")

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_sentiment_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
METADATA_PATH = MODEL_DIR / "metadata.joblib"

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/331/"
    "sentiment+labelled+sentences.zip"
)

lemmatizer = WordNetLemmatizer()


# ==============================
# TEXT PREPROCESSING
# ==============================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    words = re.findall(r"[a-z]+", text)
    cleaned_words = []

    for word in words:
        if word in ENGLISH_STOP_WORDS:
            continue

        try:
            word = lemmatizer.lemmatize(word)
        except LookupError:
            pass

        cleaned_words.append(word)

    return " ".join(cleaned_words)


# ==============================
# LOAD DATASET
# ==============================

def load_dataset():
    zip_path = DATA_DIR / "sentiment_dataset.zip"

    if not zip_path.exists():
        print("Downloading dataset...")

        request = urllib.request.Request(
            DATA_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        with urllib.request.urlopen(
            request, timeout=60
        ) as response:
            zip_path.write_bytes(response.read())

    records = []

    required_files = {
        "amazon_cells_labelled.txt",
        "imdb_labelled.txt",
        "yelp_labelled.txt",
    }

    with zipfile.ZipFile(zip_path) as archive:
        for filename in archive.namelist():
            if Path(filename).name not in required_files:
                continue

            with archive.open(filename) as file:
                for line in file:
                    parts = line.decode(
                        "utf-8", errors="replace"
                    ).strip().rsplit("\t", 1)

                    if len(parts) != 2:
                        continue

                    review, label = parts

                    if label in ("0", "1"):
                        records.append({
                            "review": review,
                            "sentiment": int(label),
                        })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("Dataset is empty or invalid.")

    df = df.drop_duplicates(
        subset=["review", "sentiment"]
    ).dropna()

    df = df[df["review"].str.strip().ne("")]

    print(f"Total reviews: {len(df)}")
    return df.reset_index(drop=True)


# ==============================
# TRAIN AND EVALUATE
# ==============================

def train_and_save():
    df = load_dataset()

    print("\nCleaning review text...")
    df["cleaned_review"] = df["review"].apply(clean_text)

    df = df[df["cleaned_review"].str.strip().ne("")]

    X = df["cleaned_review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nCreating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
    }

    results = []

    print("\nTraining models...")

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        predictions = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, predictions)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
        })

        print(f"\n{name}")
        print(f"Accuracy: {accuracy:.2%}")

        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Negative", "Positive"],
            zero_division=0,
        ))

        cm = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        )

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )
        plt.title(name)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    results_df = pd.DataFrame(results).sort_values(
        "Accuracy", ascending=False
    )

    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_model = models[best_name]

    # Save the best classifier and matching vectorizer.
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    joblib.dump({
        "model_name": best_name,
        "max_features": 5000,
        "ngram_range": (1, 2),
    }, METADATA_PATH)

    results_df.to_csv(
        DATA_DIR / "model_comparison.csv",
        index=False,
    )

    df.to_csv(
        DATA_DIR / "processed_reviews.csv",
        index=False,
    )

    print("\nTraining completed!")
    print(f"Best model: {best_name}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved vectorizer: {VECTORIZER_PATH}")


# ==============================
# LOAD SAVED MODEL
# ==============================

def load_saved_model():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Saved model not found. Run training first:\n"
            "python sentiment_classification.py --train"
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("Saved model loaded successfully.")
    return model, vectorizer


# ==============================
# PREDICT SENTIMENT
# ==============================

def predict_sentiment(text, model, vectorizer):
    if not isinstance(text, str) or not text.strip():
        print("Please enter a valid review.")
        return

    cleaned = clean_text(text)

    if not cleaned:
        print("Review contains no usable words.")
        return

    features = vectorizer.transform([cleaned])
    prediction = int(model.predict(features)[0])

    sentiment = (
        "Positive" if prediction == 1 else "Negative"
    )

    print(f"\nReview: {text}")
    print(f"Predicted sentiment: {sentiment}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = probabilities[class_index]

        print(f"Model confidence: {confidence:.2%}")


# ==============================
# INTERACTIVE PREDICTIONS
# ==============================

def interactive_mode():
    model, vectorizer = load_saved_model()

    print("\nSentiment Prediction System")
    print("Enter a review or type 'exit' to stop.")

    while True:
        try:
            text = input("\nEnter review: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if text.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        predict_sentiment(text, model, vectorizer)


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NLP Sentiment Classification"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train and save the models",
    )

    parser.add_argument(
        "--predict",
        action="store_true",
        help="Load saved model for predictions",
    )

    args = parser.parse_args()

    if args.train:
        train_and_save()
    elif args.predict:
        interactive_mode()
    else:
        parser.print_help()
        print(
            "\nFirst run:  python sentiment_classification.py --train"
            "\nThen run:   python sentiment_classification.py --predict"
        )
