# src/train.py

import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True, help="Commit SHA")
    args = parser.parse_args()
    commit_id = args.commit

    # Load cleaned dataset
    df = pd.read_csv("src/dataset/tweet_sentiment_clean.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_tweet"], df["sentiment"], test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=200))
    ])

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("tweet_sentiment_experiment")

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(pipeline, "model")

        # Save metrics to output and run_id for workflow
        os.makedirs("output", exist_ok=True)
        with open("output/accuracy.txt", "w") as f:
            f.write(str(accuracy))
        with open("output/run_id.txt", "w") as f:
            f.write(run.info.run_id)

        print_top_features(pipeline, class_names=["negative", "neutral", "positive"])

def print_top_features(model, class_names):
    tfidf = model.named_steps['tfidf']
    classifier = model.named_steps['classifier']
    feature_names = tfidf.get_feature_names_out()
    for i, class_name in enumerate(class_names):
        top_features = sorted(zip(classifier.coef_[i], feature_names), reverse=True)[:10]
        print(f"\nTop features for {class_name}:")
        for coef, feature in top_features:
            print(f"{feature}: {coef:.4f}")

if __name__ == "__main__":
    main()