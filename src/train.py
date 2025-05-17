import pandas as pd
import nltk
import mlflow
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import os
import argparse
import pickle

MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI")
CURRENT_DATASET_NAME = "tweet_sentiment"
CURRENT_DATASET = f"src/dataset/{CURRENT_DATASET_NAME}_clean.csv"
data = pd.read_csv(CURRENT_DATASET)

print(f"CURRENT_DATASET: {CURRENT_DATASET}")

parser = argparse.ArgumentParser()
parser.add_argument("--commit", required=True, help="Commit hash or unique identifier for the model name")
args = parser.parse_args()
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
current_mode_name = f"model"
print("Start training ...")
mlflow.set_tracking_uri(MLFLOW_URL)
print(f"MLFLOW_TRACKING_URI: {MLFLOW_URL}")

# Check for duplicates
print(f"Number of duplicate tweets: {data.duplicated().sum()}")

# Prepare features and labels
X = data['clean_tweet']
y = data['sentiment']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a pipeline with TF-IDF and LinearSVC
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words=stop_words)),
    ('classifier', LinearSVC(max_iter=5000))  # Increase max_iter
])

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'tfidf__max_df': [0.8, 0.9, 1.0],
    'tfidf__min_df': [1, 2, 5],
    'classifier__C': [0.1, 1.0, 10.0]
}


# Start MLflow run
with mlflow.start_run(run_name=args.commit) as run:
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.2f}")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))

    # Log parameters
    mlflow.log_params(grid_search.best_params_)
    
    # Log metrics
    mlflow.log_metric("cross_validation_accuracy", grid_search.best_score_)
    mlflow.log_metric("test_accuracy", test_accuracy)
    os.makedirs("output", exist_ok=True)
    with open("output/accuracy.txt", "w") as f:
        f.write(str(test_accuracy))
    # Log model type
    mlflow.log_param("model_type", "LinearSVC")

    # Save and log the model as an artifact
    model_path = f"{current_mode_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    mlflow.log_artifact(model_path)

    # Log the model to MLflow model registry
    mlflow.sklearn.log_model(best_model, current_mode_name)

    # Print TF-IDF vocabulary (sample)
    print("\nSample TF-IDF Vocabulary:")
    print(best_model.named_steps['tfidf'].get_feature_names_out()[:20])

    # Feature importance (top words per class)
    def print_top_features(model, class_names):
        tfidf = model.named_steps['tfidf']
        classifier = model.named_steps['classifier']
        feature_names = tfidf.get_feature_names_out()
        for i, class_name in enumerate(class_names):
            top_features = sorted(zip(classifier.coef_[i], feature_names), reverse=True)[:10]
            print(f"\nTop features for {class_name}:")
            for coef, feature in top_features:
                print(f"{feature}: {coef:.4f}")

    print_top_features(best_model, ['negative', 'neutral', 'positive'])
    with open("output/run_id.txt", "w") as f:
        f.write(run.info.run_id)
