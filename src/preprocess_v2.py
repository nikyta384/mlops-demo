import pandas as pd
import string
import re
import glob

DATASET = "tweet_sentiment"
DATASET_PATH = "src/dataset/"

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    return ' '.join(tokens)

def clean_file(filepath):
    try:
        df = pd.read_csv(
            filepath, 
            engine='python', 
            on_bad_lines='skip',  # skips malformed lines instead of erroring out
            sep=',',
            quotechar='"'
        )
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return pd.DataFrame(columns=['clean_tweet', 'sentiment'])

    if 'tweet' not in df.columns or 'sentiment' not in df.columns:
        print(f"Skipping {filepath} due to missing columns")
        return pd.DataFrame(columns=['clean_tweet', 'sentiment'])

    df['clean_tweet'] = df['tweet'].astype(str).apply(clean_text)
    df = df[df['clean_tweet'].str.strip() != '']
    df = df.drop_duplicates(subset=['clean_tweet'])
    return df[['clean_tweet', 'sentiment']]

def main():
    input_pattern = f"{DATASET_PATH}{DATASET}_v*.csv"
    input_files = sorted(glob.glob(input_pattern))
    original_file = f"{DATASET_PATH}{DATASET}.csv"
    if original_file not in input_files:
        input_files.append(original_file)

    print(f"Files to process: {input_files}")

    cleaned_frames = []
    for f in input_files:
        print(f"Processing {f}")
        cleaned_df = clean_file(f)
        print(f"Read and cleaned {len(cleaned_df)} rows from {f}")
        cleaned_frames.append(cleaned_df)

    df_all = pd.concat(cleaned_frames, ignore_index=True)
    print(f"Total rows before final deduplication: {len(df_all)}")

    df_all = df_all.drop_duplicates(subset=['clean_tweet'])
    print(f"Rows after final deduplication: {len(df_all)}")

    output_csv = f"{DATASET_PATH}{DATASET}_clean.csv"
    df_all.to_csv(output_csv, index=False)
    print(f"Saved cleaned data to {output_csv}")

if __name__ == "__main__":
    main()