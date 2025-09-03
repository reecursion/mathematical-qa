#!/usr/bin/env python3
"""
Script to download the deepmind/math_dataset (algebra__linear_1d only) 
from Hugging Face, randomly sample 7,500 questions in the same ratio 
as easy/medium/hard, and save as CSV.
"""

import os
import pandas as pd
from datasets import load_dataset

def download_linear_1d(num_samples=7500, output_file="algebra_linear_1d.csv"):
    """
    Download algebra__linear_1d subset of math_dataset, sample random questions 
    from test split, and save as CSV.
    """
    print("Starting download of deepmind/math_dataset (algebra__linear_1d only)")

    try:
        # Load only the linear_1d module
        dataset = load_dataset("deepmind/math_dataset", "algebra__linear_1d")
        print("Dataset loaded successfully!")

        # Get test split info
        test_size = len(dataset['test'])
        print(f"Test split size: {test_size} examples")
        
        # Check if we have enough samples
        if test_size < num_samples:
            print(f"Warning: Only {test_size} examples available, sampling all of them")
            num_samples = test_size

        # Convert test split to DataFrame
        df = dataset['test'].to_pandas()
        print(f"Columns available: {df.columns.tolist()}")

        # Sample random questions from test split
        sampled_df = df.sample(n=num_samples, random_state=42)
        print(f"Sampled {len(sampled_df)} questions from test split")

        # Clean the data - remove byte string prefixes and newlines
        print("Cleaning data...")
        def clean_byte_string(text):
            if isinstance(text, str) and text.startswith("b'") and text.endswith("'"):
                # Remove b' and ' and decode
                clean_text = text[2:-1]
                # Replace \n with actual newlines and strip
                clean_text = clean_text.replace('\\n', '\n').strip()
                return clean_text
            return text
        
        sampled_df['question'] = sampled_df['question'].apply(clean_byte_string)
        sampled_df['answer'] = sampled_df['answer'].apply(clean_byte_string)

        # Save to CSV
        sampled_df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"Saved {len(sampled_df)} questions to {output_file}")

        # Print sample data
        print("\nSample data:")
        sample = sampled_df.iloc[0]
        for key, value in sample.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print(f"Working directory: {os.getcwd()}")
    print("=" * 50)

    success = download_linear_1d()

    if success:
        print("\n" + "=" * 50)
        print("Dataset sampling completed successfully!")
        print("File saved in:", os.getcwd())
    else:
        print("\n" + "=" * 50)
        print("Dataset sampling failed!")
