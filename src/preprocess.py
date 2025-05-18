import pandas as pd
import string
import re

# Download stopwords if not already downloaded
DATASET = "tweet_sentiment"
DATASET_dir = "dataset/"
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    return ' '.join(tokens)

def main():
    input_csv = f"{DATASET_dir}{DATASET}.csv"
    output_csv = f"{DATASET_dir}{DATASET}_clean.csv"

    df = pd.read_csv(input_csv)
    print(f"Original lines: {len(df)}")  # Debug: original number of lines

    df['clean_tweet'] = df['tweet'].astype(str).apply(clean_text)
    print(f"After cleaning (same number of lines): {len(df)}")  # Debug: lines after cleaning (should be same)

    #Remove duplicates based on cleaned tweets
    df = df.drop_duplicates(subset=['clean_tweet'])
    print(f"After removing duplicates: {len(df)}")  # Debug: lines after removing duplicates

    df[['clean_tweet', 'sentiment']].to_csv(output_csv, index=False)
    print(f"Saved cleaned data to {output_csv}")

main()