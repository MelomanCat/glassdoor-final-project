import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === Setting up MLflow Tracking URI ===
mlflow.set_tracking_uri("https://jedha0padavan-mlflow-server-final-project.hf.space")

with mlflow.start_run(run_name="TFIDF + LogisticRegression (final)"):

    # === Load and prepare data ===
    df = pd.read_csv("reviews_processed_with_spacy_md.csv") 
    # Remove neutral rating
    df = df[df['overall_rating'] != 3].copy()

    # Combine texts
    df["text"] = (
        df["headline_clean"].fillna('') + ' ' +
        df["pros_clean"].fillna('') + ' ' +
        df["cons_clean"].fillna('')
    )

    # Create encoded labels
    df['label'] = df['overall_rating'].apply(lambda x: 1 if x > 3 else 0)

    
    df = df[["text", "label"]].dropna()

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # === TF-IDF Vectorizer with our parameters from notebook ===
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.8,
        stop_words='english'
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # === Train LogisticRegression with best parameter found C=10 ===
    model = LogisticRegression(C=10, max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # === Metrics
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")


    mlflow.log_param("C", 10)
    mlflow.log_param("vectorizer_max_features", 5000)
    mlflow.log_param("vectorizer_min_df", 5)
    mlflow.log_param("vectorizer_max_df", 0.8)
    mlflow.log_param("vectorizer_stop_words", "english")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_macro", f1)

    # === Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png")
    mlflow.log_artifact("outputs/confusion_matrix.png")

    # === Save model and vectorizer
    joblib.dump(model, "outputs/model.pkl")
    joblib.dump(vectorizer, "outputs/tfidf_vectorizer.pkl")
    mlflow.log_artifact("outputs/model.pkl")
    mlflow.log_artifact("outputs/tfidf_vectorizer.pkl")

    mlflow.sklearn.log_model(model, "model")  # log as MLflow-model as well

    print("Model trained and logged to MLflow.")
