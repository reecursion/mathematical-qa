import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from datasets import load_dataset

class MathProcessor:
    """
    A class for processing math problem datasets from Hugging Face.
    
    This class takes a Hugging Face dataset containing mathematical problems with instructions and answers,
    and processes it to extract:
    1. Original instruction (with the "Please solve..." prefix)
    2. Clean instruction (without the "Please solve..." prefix)
    3. Full answer (including solution steps)
    4. Final answer (just the numerical or algebraic result)
    """
    
    def __init__(self, dataset_name: str = "reecursion/mmiqc-subset"):
        """
        Initialize the processor with the Hugging Face dataset name.
        
        Args:
            dataset_name: Name of the Hugging Face dataset
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self.processed_dataset = None
        
    def load_dataset(self) -> None:
        """Load the dataset from Hugging Face."""
        self.dataset = load_dataset(self.dataset_name)
    
    def _extract_clean_instruction(self, instruction: str) -> str:
        """
        Extract the clean instruction by removing the standard prefix.
        
        Args:
            instruction: The original instruction with prefix
            
        Returns:
            The instruction without the standard prefix
        """
        # Fixed the prefix to match exactly what's in the dataset
        prefix = 'Please solve the following problem and put your answer at the end with "The answer is: ".\n'
        
        if instruction.startswith(prefix):
            return instruction[len(prefix):].strip()
        
        # Try alternative prefix formats if the main one doesn't match
        alt_prefix = 'Please solve the following problem and put your answer at the end with "The answer is: ". '
        if instruction.startswith(alt_prefix):
            return instruction[len(alt_prefix):].strip()
            
        # If no prefix matches, return the original instruction
        return instruction
    
    def _extract_final_answer(self, answer: str) -> str:
        """
        Extract just the final answer from the full answer.
        
        Args:
            answer: The full answer including steps
            
        Returns:
            Just the final numerical or algebraic result
        """
        final_answer = ""
        
        lines = answer.split('\n')
        for line in reversed(lines):
            if "The answer is:" in line:
                final_answer = line.split("The answer is:")[-1].strip()
                break
        
        return final_answer
    
    def process_dataset(self) -> Dict:
        """
        Process the dataset to extract clean instructions and final answers.
        
        Returns:
            The processed dataset with added columns
        """
        if not self.dataset:
            self.load_dataset()
        
        processed_splits = {}
        
        for split_name, split_data in self.dataset.items():
            # Create a new DataFrame from the split
            df = pd.DataFrame(split_data)
            
            # Add the clean instruction column
            df['clean_instruction'] = df['instruction'].apply(self._extract_clean_instruction)
            
            # Add the final answer column
            df['final_answer'] = df['output'].apply(self._extract_final_answer)
            
            # Store the processed split
            processed_splits[split_name] = df
        
        self.processed_dataset = processed_splits
        return self.processed_dataset
    
    def save_processed_dataset(self, output_dir: str) -> None:
        """
        Save the processed dataset splits to CSV files.
        
        Args:
            output_dir: Directory where the processed data will be saved
        """
        import os
        
        if self.processed_dataset is None:
            self.process_dataset()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each split to a separate CSV file
        for split_name, split_data in self.processed_dataset.items():
            output_path = os.path.join(output_dir, f"{split_name}.csv")
            split_data.to_csv(output_path, index=False)
            print(f"Processed {split_name} split saved to {output_path}")
    
    def get_sample(self, split: str = "train", n: int = 5) -> pd.DataFrame:
        """
        Get a sample of n processed entries from a specific split.
        
        Args:
            split: The dataset split to sample from
            n: Number of samples to return
            
        Returns:
            A pandas DataFrame with n samples
        """
        if self.processed_dataset is None:
            self.process_dataset()
        
        return self.processed_dataset[split].head(n)
    
    def to_huggingface_dataset(self) -> Any:
        """
        Convert the processed dataframes back to a Hugging Face Dataset object.
        
        Returns:
            A Hugging Face Dataset object with the additional columns
        """
        from datasets import Dataset, DatasetDict
        
        if self.processed_dataset is None:
            self.process_dataset()
        
        hf_datasets = {}
        for split_name, split_data in self.processed_dataset.items():
            hf_datasets[split_name] = Dataset.from_pandas(split_data)
        
        return DatasetDict(hf_datasets)

if __name__ == "__main__":
    processor = MathProcessor()
    processed_dataset = processor.process_dataset()
    
    print("Dataset splits:", list(processed_dataset.keys()))
    
    print("\nSample from train split:")
    sample = processor.get_sample("train", 3)
    for i, row in sample.iterrows():
        print(f"\nExample {i+1}:")
        print(f"Original instruction: {row['instruction'][:100]}...")
        print(f"Clean instruction: {row['clean_instruction'][:100]}...")
        print(f"Full answer: {row['output'][:100]}...")
        print(f"Final answer: {row['final_answer']}")
    
    processor.save_processed_dataset("processed_dataset")
    
    hf_dataset = processor.to_huggingface_dataset()
    print("\nConverted back to Hugging Face dataset format")
    print("Available splits:", list(hf_dataset.keys()))
    print("Columns in train split:", hf_dataset["train"].column_names)