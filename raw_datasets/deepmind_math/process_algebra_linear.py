#!/usr/bin/env python3
"""
Script to process the algebra_linear_1d.csv file and format it for training.
Creates the same format as the existing test.csv file.
"""

import pandas as pd
import os

def process_algebra_data():
    """
    Process the algebra_linear_1d.csv file and create formatted dataset
    """
    # Input and output paths
    input_file = "/home/gganeshl/mathematical-qa/raw_datasets/deepmind_math/algebra_linear_1d.csv"
    output_dir = "/home/gganeshl/mathematical-qa/processed_dataset/deepmind_math"
    output_file = os.path.join(output_dir, "test.csv")
    
    print("Processing algebra_linear_1d data...")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Read the input CSV
        print(f"Reading data from {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} examples")
        print(f"Columns: {df.columns.tolist()}")
        
        # Create the processed dataframe
        processed_data = []
        
        for idx, row in df.iterrows():
            question = str(row['question']).strip()
            answer = str(row['answer']).strip()
            
            # Remove "Solve" 
            if question.startswith('Solve '):
                question = question[6:]  # Remove "Solve "

            instruction_input = f'Solve {question}'

            expected_output = f'{answer}'
            
            # Create the processed row with your specified columns
            processed_row = {
                'question': question,
                'instruction_input': instruction_input,
                'answer': answer,
                'expected_output': expected_output
            }
            
            processed_data.append(processed_row)
        
        # Create DataFrame from processed data
        processed_df = pd.DataFrame(processed_data)
        
        # Save to CSV
        processed_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Saved processed data to {output_file}")
        print(f"Processed {len(processed_df)} examples")
        
        # Show sample data
        print("\nSample of processed data:")
        for i in range(2):
            print(f"Example {i+1}:")
            print(f"  Question: {processed_df.iloc[i]['question']}")
            print(f"  Instruction Input: {processed_df.iloc[i]['instruction_input']}")
            print(f"  Answer: {processed_df.iloc[i]['answer']}")
            print(f"  Expected Output: {processed_df.iloc[i]['expected_output']}")
            print()
        
        return True
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return False

if __name__ == "__main__":
    success = process_algebra_data()
    
    if success:
        print("Data processing completed successfully!")
    else:
        print("Data processing failed!")