import re
import pandas as pd
from datasets import load_dataset

def load_math_dataset():
    """Load the MATH dataset from Hugging Face."""
    return load_dataset("Maxwell-Jia/MATH")

def clean_text(text):
    """Clean LaTeX and extra whitespace from text."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"\s+", " ", text)  # Remove excessive whitespace
    text = text.replace("\n", " ")    # Remove newlines
    text = text.strip()
    return text

def clean_dataframe(df):
    """Apply cleaning to problem and solution columns."""
    df["problem"] = df["problem"].apply(clean_text)
    df["solution"] = df["solution"].apply(clean_text)
    return df

def save_to_csv(df, filename):
    """Save a DataFrame to CSV."""
    df.to_csv(filename, index=False)
    print(f"✅ Saved {filename}")

def process_and_save_dataset(combine=False):
    """Main function to process the dataset and save as CSV."""
    dataset = load_math_dataset()
    
    # Convert splits to DataFrames
    df_train = pd.DataFrame(dataset["train"])
    df_test = pd.DataFrame(dataset["test"])

    # Clean text fields
    df_train = clean_dataframe(df_train)
    df_test = clean_dataframe(df_test)

    if combine:
        # Add split column and combine
        df_train["split"] = "train"
        df_test["split"] = "test"
        df_combined = pd.concat([df_train, df_test], ignore_index=True)
        save_to_csv(df_combined, "math_dataset_combined.csv")
    else:
        save_to_csv(df_train, "math_train.csv")
        save_to_csv(df_test, "math_test.csv")

if __name__ == "__main__":
    process_and_save_dataset(combine=False)  # Set to True if you want a single CSV
