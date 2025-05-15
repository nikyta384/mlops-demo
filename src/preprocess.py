import pandas as pd
import string
import re

# Download stopwords if not already downloaded
DATASET = "tweet_sentiment"

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
    input_csv = f"src/dataset/{DATASET}.csv"
    output_csv = f"src/dataset/{DATASET}_clean.csv"

    df = pd.read_csv(input_csv)
    df['clean_tweet'] = df['tweet'].astype(str).apply(clean_text)
    
    # Remove duplicates based on cleaned tweets
    df = df.drop_duplicates(subset=['clean_tweet'])

    # Save only cleaned tweets and label, or both original and cleaned
    df[['clean_tweet', 'sentiment']].to_csv(output_csv, index=False)
    print(f"Saved cleaned data to {output_csv}")

main()