import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import os
import argparse
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Environment variables and dataset
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI")
CURRENT_DATASET_NAME = "tweet_sentiment"
CURRENT_DATASET = f"src/dataset/{CURRENT_DATASET_NAME}_clean.csv"
data = pd.read_csv(CURRENT_DATASET)

print(f"CURRENT_DATASET: {CURRENT_DATASET}")

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--commit", required=True, help="Commit hash or unique identifier for the model name")
args = parser.parse_args()

print("Start training ...")
mlflow.set_tracking_uri(MLFLOW_URL)
print(f"MLFLOW_TRACKING_URI: {MLFLOW_URL}")

# Check for duplicates
print(f"Number of duplicate tweets: {data.duplicated().sum()}")

# Enhanced text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Tokenize
        tokens = word_tokenize(text.lower())
        # Lemmatize and remove stopwords
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalnum()]
        return ' '.join(tokens)
    return ''

# Apply preprocessing
data['clean_tweet'] = data['clean_tweet'].apply(preprocess_text)

# Prepare features and labels
X = data['clean_tweet']
y = data['sentiment']

# Check class distribution
print("\nClass distribution:")
print(y.value_counts(normalize=True))

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Compute class weights for imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Create pipelines for LinearSVC and XGBoost
svc_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1, 3))),
    ('classifier', LinearSVC(max_iter=5000, class_weight='balanced'))
])

xgb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1, 3))),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
])

# Hyperparameter tuning for both models
svc_param_grid = {
    'tfidf__max_df': [0.8, 0.9],
    'tfidf__min_df': [1, 2],
    'tfidf__max_features': [10000, 15000],
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__loss': ['hinge', 'squared_hinge']
}

xgb_param_grid = {
    'tfidf__max_df': [0.8, 0.9],
    'tfidf__min_df': [1, 2],
    'tfidf__max_features': [10000, 15000],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__n_estimators': [100, 200]
}

# Start MLflow run
with mlflow.start_run(run_name=args.commit) as run:
    # Grid search for LinearSVC
    svc_grid_search = GridSearchCV(svc_pipeline, svc_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    svc_grid_search.fit(X_train, y_train)

    # Grid search for XGBoost
    xgb_grid_search = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    xgb_grid_search.fit(X_train, y_train)

    # Select best model
    if svc_grid_search.best_score_ > xgb_grid_search.best_score_:
        best_model = svc_grid_search.best_estimator_
        best_params = svc_grid_search.best_params_
        best_score = svc_grid_search.best_score_
        model_type = "LinearSVC"
    else:
        best_model = xgb_grid_search.best_estimator_
        best_params = xgb_grid_search.best_params_
        best_score = xgb_grid_search.best_score_
        model_type = "XGBoost"

    print(f"Best model: {model_type}")
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation accuracy: {best_score:.2f}")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))

    # Log parameters and metrics
    mlflow.log_params(best_params)
    mlflow.log_metric("cross_validation_accuracy", best_score)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_param("model_type", model_type)

    # Save accuracy
    os.makedirs("output", exist_ok=True)
    with open("output/accuracy.txt", "w") as f:
        f.write(str(test_accuracy))

    # Save and log the model
    model_path = f"model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    mlflow.log_artifact(model_path)
    mlflow.sklearn.log_model(best_model, "model")

    # Log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = "output/confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

    # Print TF-IDF vocabulary (sample)
    print("\nSample TF-IDF Vocabulary:")
    print(best_model.named_steps['tfidf'].get_feature_names_out()[:20])

    # Feature importance
    def print_top_features(model, class_names):
        tfidf = model.named_steps['tfidf']
        feature_names = tfidf.get_feature_names_out()
        if model_type == "LinearSVC":
            classifier = model.named_steps['classifier']
            for i, class_name in enumerate(class_names):
                top_features = sorted(zip(classifier.coef_[i], feature_names), reverse=True)[:10]
                print(f"\nTop features for {class_name}:")
                for coef, feature in top_features:
                    print(f"{feature}: {coef:.4f}")
        else:
            classifier = model.named_steps['classifier']
            importance = classifier.feature_importances_
            top_indices = importance.argsort()[-10:][::-1]
            print("\nTop features for XGBoost:")
            for idx in top_indices:
                print(f"{feature_names[idx]}: {importance[idx]:.4f}")

    print_top_features(best_model, ['negative', 'neutral', 'positive'])

    # Save run ID
    with open("output/run_id.txt", "w") as f:
        f.write(run.info.run_id)