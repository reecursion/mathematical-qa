import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
import argparse
from tqdm import tqdm
from typing import List, Dict, Tuple, Any, Optional
# import matplotlib.pyplot as plt
import os
from huggingface_hub import login



class MathProcessor:
    def __init__(self, dataset_name: str = "Rith335/MATH_filtered_math_equation_problems", debug=True):
        self.dataset_name = dataset_name
        self.dataset = None
        self.processed_dataset = None
        self.debug = debug

    def load_dataset(self) -> None:
        self.dataset = load_dataset(self.dataset_name)
        if self.debug:
            print(f"Dataset '{self.dataset_name}' loaded with splits: {list(self.dataset.keys())}")

    def _extract_clean_instruction(self, instruction: str) -> str:
        prefix = 'Please solve the following problem and put your answer at the end with "The answer is: ".\n'
        alt_prefix = 'Please solve the following problem and put your answer at the end with "The answer is: ". '
        if instruction.startswith(prefix):
            return instruction[len(prefix):].strip()
        elif instruction.startswith(alt_prefix):
            return instruction[len(alt_prefix):].strip()
        return instruction

    def _extract_final_answer(self, answer: str) -> str:
        final_answer = ""
        lines = answer.split('\n')
        for line in reversed(lines):
            if "The answer is:" in line:
                final_answer = line.split("The answer is:")[-1].strip()
                break
        return final_answer

    def process_dataset(self) -> Dict:
        if not self.dataset:
            self.load_dataset()
        processed_splits = {}
        for split_name, split_data in self.dataset.items():
            df = pd.DataFrame(split_data)
            df['clean_instruction'] = df['instruction'].apply(self._extract_clean_instruction)
            df['final_answer'] = df['output'].apply(self._extract_final_answer)
            processed_splits[split_name] = df
            if self.debug:
                print(f"Processed split '{split_name}' with {len(df)} examples.")
        self.processed_dataset = processed_splits
        return self.processed_dataset

    def save_processed_dataset(self, output_dir: str) -> None:
        import os
        if self.processed_dataset is None:
            self.process_dataset()
        os.makedirs(output_dir, exist_ok=True)
        for split_name, split_data in self.processed_dataset.items():
            output_path = os.path.join(output_dir, f"{split_name}.csv")
            split_data.to_csv(output_path, index=False)
            print(f"Processed {split_name} split saved to {output_path}")

    def get_sample(self, split: str = "train", n: int = 5) -> pd.DataFrame:
        if self.processed_dataset is None:
            self.process_dataset()
        return self.processed_dataset[split].head(n)

    def to_huggingface_dataset(self) -> Any:
        from datasets import Dataset, DatasetDict
        if self.processed_dataset is None:
            self.process_dataset()
        hf_datasets = {}
        for split_name, split_data in self.processed_dataset.items():
            hf_datasets[split_name] = Dataset.from_pandas(split_data)
        return DatasetDict(hf_datasets)


class CustomizedFlanT5Inference:
    def __init__(self, model_name="google/flan-t5-base", prompt = "", device=None, debug=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='auto')
        if self.debug:
            print("Model and tokenizer loaded.")
        self.original_state_dict = {
            k: v.clone() for k, v in self.model.state_dict().items()
            if 'encoder' in k and 'attention' in k and 'relative_attention_bias' not in k
        }
        self.prompt = prompt
        if self.debug:
            print(f"Prompt being used is {self.prompt}")
            print(f"Original attention weights snapshot stored: {len(self.original_state_dict)} tensors.")
        
        # Prepare word lists for word-based detection
        self._prepare_math_word_lists()
        # Initialize token mappings from fixed lists
        self._prepare_token_mappings()

    def _prepare_math_word_lists(self):
        """
        Prepare lists of mathematical words and their variations for enhanced detection.
        """
        # Basic number words and their variants
        number_words = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
            "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
            "hundred", "thousand", "million", "billion", "half", "quarter", "third", "fourth", "fifth"
        ]
        
        self.number_word_stems = set(number_words)
        self.number_word_variants = set(number_words)
        
        # Add variations of number words
        for word in number_words:
            self.number_word_variants.add(word.capitalize())
            if word.endswith("y"):
                self.number_word_variants.add(word[:-1] + "ies")  # twenty -> twenties
            else:
                self.number_word_variants.add(word + "s")  # thousand -> thousands
        
        # Mathematical operation words and their variants
        operation_words = [
            "plus", "minus", "times", "divided", "equals", "equal", "multiply", "divide", "add", "subtract",
            "sum", "difference", "product", "quotient", "remainder", "total", "average", "mean",
            "percent", "percentage", "ratio", "proportion", "fraction", "decimal",
            "greater", "less", "than", "same", "different", "increase", "decrease",
            "square", "cube", "root", "power", "exponent", "logarithm", "factorial",
            "sine", "cosine", "tangent", "derivative", "integral", "limit", "amount", "many"
        ]
        
        self.operation_word_stems = set(operation_words)
        self.operation_word_variants = set(operation_words)
        
        # Add variations of operation words
        for word in operation_words:
            self.operation_word_variants.add(word.capitalize())
            if word.endswith("e"):
                self.operation_word_variants.add(word + "d")  # divide -> divided
                self.operation_word_variants.add(word + "s")  # divide -> divides
                self.operation_word_variants.add(word[:-1] + "ing")  # divide -> dividing
            elif word.endswith("y"):
                self.operation_word_variants.add(word[:-1] + "ies")  # multiply -> multiplies
                self.operation_word_variants.add(word + "ing")  # multiply -> multiplying
            else:
                self.operation_word_variants.add(word + "ed")  # add -> added
                self.operation_word_variants.add(word + "ing")  # add -> adding
                self.operation_word_variants.add(word + "s")  # add -> adds

    def _prepare_token_mappings(self):
        # Store tokens as lists of lists to handle subword tokenization
        self.number_token_lists = []
        self.operator_token_lists = []
        
        # Process digit tokens (full tokens and subword pieces)
        digits = "0123456789"
        for num in digits:
            self.number_token_lists.append(self.tokenizer.encode(num, add_special_tokens=False))
        
        # Process number words with their subword pieces
        number_words = list(self.number_word_variants)
        
        for word in number_words:
            self.number_token_lists.append(self.tokenizer.encode(word, add_special_tokens=False))
        
        # Process symbol operators
        symbol_operators = ["+", "-", "*", "/", "=", "<", ">", "(", ")", "^", "×", "÷", "±", "≠", "≤", "≥", "%"]
        for op in symbol_operators:
            self.operator_token_lists.append(self.tokenizer.encode(op, add_special_tokens=False))
        
        # Process word operators with their potential subword pieces
        word_operators = list(self.operation_word_variants)
        
        # Add common mathematical operation words and their derivatives
        additional_op_words = [
            "addition", "subtraction", "multiplication", "division", 
            "equation", "inequality", "formula", "calculate", "computation"
        ]
        word_operators.extend(additional_op_words)
        
        for word in word_operators:
            self.operator_token_lists.append(self.tokenizer.encode(word, add_special_tokens=False))
        
        # Flatten lists for traditional token-by-token operations while keeping structure for subword aware operations
        self.number_tokens = set(token for sublist in self.number_token_lists for token in sublist)
        self.operator_tokens = set(token for sublist in self.operator_token_lists for token in sublist)
        
        if self.debug:
            print(f"Number token lists identified: {len(self.number_token_lists)}")
            print(f"Operator token lists identified: {len(self.operator_token_lists)}")
            print(f"Unique number tokens: {len(self.number_tokens)}")
            print(f"Unique operator tokens: {len(self.operator_tokens)}")
            
            # Debug: sample some tokenizations to verify
            debug_samples = ["addition", "subtraction", "twenty-five", "45/3", "multiply"]
            print("\nSample tokenizations:")
            for sample in debug_samples:
                tokens = self.tokenizer.encode(sample, add_special_tokens=False)
                token_strs = self.tokenizer.convert_ids_to_tokens(tokens)
                print(f"  '{sample}' → {tokens} → {token_strs}")

    def modify_attention_for_question(self, question, num_scaling=1.5, op_scaling=2.0):
        """
        Combined approach: Analyze the tokenization of a question and modify attention weights
        using both pattern matching and word-based detection.
        """
        self.reset_weights()
        
        # Skip if no question is provided
        if not question:
            return
        
        # Tokenize the question
        tokens = self.tokenizer.encode(question, add_special_tokens=False)
        token_strs = self.tokenizer.convert_ids_to_tokens(tokens)
        
        # Define patterns for identifying math content in tokens
        num_pattern = re.compile(r'[0-9]')  # Contains digits
        op_pattern = re.compile(r'[+\-*/=<>^%]')  # Contains operators
        
        # Classify each token
        number_tokens = []
        operator_tokens = []
        mixed_tokens = []
        token_classifications = []  # Store classification details for each token
        
        # First pass: Pattern-based detection
        for token_id, token_str in zip(tokens, token_strs):
            has_number = bool(num_pattern.search(token_str))
            if '<unk>' not in token_str:
                has_operator = bool(op_pattern.search(token_str))
            
            classification = {"token_id": token_id, "token_str": token_str, "classification": "unmodified", "reason": "", "scaling": None}
            
            if has_number and has_operator:
                mixed_tokens.append(token_id)
                classification["classification"] = "mixed"
                classification["reason"] = "Contains both numbers and operators"
                classification["scaling"] = f"{(num_scaling + op_scaling * 1.2) / 2.2:.2f}"
            elif has_number:
                number_tokens.append(token_id)
                classification["classification"] = "number"
                classification["reason"] = "Contains digits"
                classification["scaling"] = f"{num_scaling:.2f}"
            elif has_operator:
                operator_tokens.append(token_id)
                classification["classification"] = "operator"
                classification["reason"] = "Contains operator symbols"
                classification["scaling"] = f"{op_scaling:.2f}"
                
            token_classifications.append(classification)
        
        # Second pass: Word-based detection
        for i, (token_id, token_str) in enumerate(zip(tokens, token_strs)):
            # Skip tokens already classified
            if token_classifications[i]["classification"] != "unmodified":
                continue
                
            # Remove leading space marker and lowercase
            clean_token = token_str.lower().strip('▁')
            
            # Check for number words
            for word in self.number_word_variants:
                if word.lower() == clean_token or word.lower() in clean_token:
                    number_tokens.append(token_id)
                    token_classifications[i]["classification"] = "number"
                    token_classifications[i]["reason"] = f"Contains number word '{word}'"
                    token_classifications[i]["scaling"] = f"{num_scaling:.2f}"
                    break
            
            # If not a number word, check for operation words
            if token_classifications[i]["classification"] == "unmodified":
                for word in self.operation_word_variants:
                    if word.lower() == clean_token or word.lower() in clean_token:
                        operator_tokens.append(token_id)
                        token_classifications[i]["classification"] = "operator"
                        token_classifications[i]["reason"] = f"Contains operator word '{word}'"
                        token_classifications[i]["scaling"] = f"{op_scaling:.2f}"
                        break
        
        # Apply scaling to identified tokens
        modified_token_count = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'encoder' in name and 'SelfAttention' in name and 'relative_attention_bias' not in name:
                    if 'q.weight' in name or 'k.weight' in name:
                        modified_param = param.clone()
                        
                        # Apply number scaling
                        for token_id in number_tokens:
                            if token_id < param.size(1):
                                modified_param[:, token_id] *= num_scaling
                                modified_token_count += 1
                        
                        # Apply operator scaling
                        for token_id in operator_tokens:
                            if token_id < param.size(1):
                                modified_param[:, token_id] *= op_scaling
                                modified_token_count += 1
                        
                        # Apply mixed scaling (average of both, weighted toward operators)
                        mixed_scaling = (num_scaling + op_scaling * 1.2) / 2.2
                        for token_id in mixed_tokens:
                            if token_id < param.size(1):
                                modified_param[:, token_id] *= mixed_scaling
                                modified_token_count += 1
                        
                        param.copy_(modified_param)
        
        if self.debug:
            print(f"\n{'='*80}")
            print(f"Question: '{question}'")
            print(f"Tokenization: {token_strs}")
            print(f"\nDetailed Token Analysis:")
            print(f"{'Token ID':<10} {'Token':<15} {'Classification':<15} {'Scaling':<10} {'Reason'}")
            print(f"{'-'*70}")
            
            for tc in token_classifications:
                token_id = tc["token_id"]
                token_str = tc["token_str"]
                classification = tc["classification"]
                scaling = tc["scaling"] if tc["scaling"] else "None"
                reason = tc["reason"]
                
                print(f"{token_id:<10} {token_str:<15} {classification:<15} {scaling:<10} {reason}")
            
            print(f"\nSummary:")
            print(f"- Number tokens: {len(number_tokens)}")
            print(f"- Operator tokens: {len(operator_tokens)}")
            print(f"- Mixed tokens: {len(mixed_tokens)}")
            print(f"- Unmodified tokens: {len(tokens) - len(number_tokens) - len(operator_tokens) - len(mixed_tokens)}")
            print(f"- Total modified tokens: {modified_token_count}")
            print(f"{'='*80}")

    def modify_attention_for_type(self, flag, num_scaling=1.5, op_scaling=2.0, question=None):
        """
        Support for the legacy flag-based approach while offering question-specific analysis
        """
        self.reset_weights()
        
        if flag is None:
            return
            
        if question is not None and (flag == "both" or self.debug):
            # Use the more sophisticated question-specific approach when possible
            current_num_scaling = num_scaling if flag in ["numbers", "both"] else 1.0
            current_op_scaling = op_scaling if flag in ["operators", "both"] else 1.0
            
            # Adjust scaling based on question type
            word_prob_score = self.analyze_question_type(question)
            if word_prob_score < 0.4:  # More like an equation
                current_num_scaling = current_num_scaling * 0.9
                current_op_scaling = current_op_scaling * 1.2
            elif word_prob_score > 0.7:  # More like a word problem
                current_num_scaling = current_num_scaling * 1.2
                current_op_scaling = current_op_scaling * 1.1
                
            if self.debug:
                print(f"[ADAPTIVE SCALING] Score: {word_prob_score:.2f} → Num: {current_num_scaling:.2f}, Op: {current_op_scaling:.2f}")
                
            # Use the question-specific attention modification
            self.modify_attention_for_question(question, current_num_scaling, current_op_scaling)
            return
        
        # Otherwise, fallback to the original approach
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'encoder' in name and 'SelfAttention' in name and 'relative_attention_bias' not in name:
                    if 'q.weight' in name or 'k.weight' in name:
                        modified_param = param.clone()
                    
                        # Apply scaling to both individual tokens and consider token sequences
                        if flag in ["numbers", "both"]:
                            # Scale individual number tokens
                            for token_id in self.number_tokens:
                                if token_id < param.size(1):
                                    modified_param[:, token_id] *= num_scaling
                        
                        if flag in ["operators", "both"]:
                            # Scale individual operator tokens
                            for token_id in self.operator_tokens:
                                if token_id < param.size(1):
                                    modified_param[:, token_id] *= op_scaling
                        
                        param.copy_(modified_param)
            
            if self.debug:
                print(f"Attention weights modified for flag: {flag}")

    def reset_weights(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.original_state_dict:
                    param.copy_(self.original_state_dict[name])
        if self.debug:
            print("Model attention weights reset to original.")

    def analyze_question_type(self, question):
        equation_indicators = ["+", "-", "*", "/", "=", "<", ">", "^", "×", "÷"]
        word_problem_indicators = ["how many", "total", "each", "per", "if", "when"]
        eq_count = sum(1 for ind in equation_indicators if ind in question)
        wp_count = sum(1 for ind in word_problem_indicators if ind.lower() in question.lower())
        has_equation_pattern = bool(re.search(r"\d[\s]*[+\-*/=][\s]*\d", question))
        words = len(re.findall(r"\b\w+\b", question))
        numbers = len(re.findall(r"\d+", question))
        if has_equation_pattern and words < 15:
            return 0.1
        elif eq_count > wp_count and numbers > 3:
            return 0.3
        elif wp_count > eq_count and words > 20:
            return 0.9
        else:
            return min(0.7, max(0.4, words / (numbers * 10 + 1)))

    def process_dataset(self, dataset_name="Rith335/MATH_filtered_math_equation_problems", split="test"):
        processor = MathProcessor(dataset_name, debug=self.debug)
        processed_dataset = processor.process_dataset()
        df = pd.DataFrame({
            "question": processed_dataset[split]["clean_instruction"],
            "ground_truth_full": processed_dataset[split]["output"],
            "answer": processed_dataset[split]["final_answer"]
        })
        return df

    def prepare_prompt(self, question):

        # return f"Please solve the following problem and only output the answer at the end with \"The answer is: \".{question}"
        return f"{self.prompt}{question}"

    def extract_final_answer(self, text):
        return text.strip()

    def run_inference(self, df, batch_size=8, flag=None, num_scaling=1.5, op_scaling=2.0):
        """
        Modified inference function that uses both approaches:
        - Traditional flag-based approach for batch processing
        - Token-analysis approach for question-by-question processing
        """
        results = []
        if self.debug:
            print(f"Running inference with flag: {flag}, num_scaling: {num_scaling}, op_scaling: {op_scaling}")
        
        # Use question-specific approach if flag is "both"
        if flag == "both":
            print("Adaptive scaling will be applied for each question.")
            total_examples = len(df)
            
            for i in tqdm(range(0, total_examples, batch_size)):
                batch = df.iloc[i:i+batch_size]
                prompts = [self.prepare_prompt(q) for q in batch["question"]]
                batch_results = []
                
                if self.debug:
                    print(f"\nProcessing batch {i//batch_size + 1}/{(total_examples+batch_size-1)//batch_size}")
                
                for j, (idx, row) in enumerate(batch.iterrows()):
                    question = row["question"]
                    if self.debug:
                        print(f"\nProcessing question {idx} ({j+1}/{len(batch)}):")
                        print(f"QUESTION: {question}")
                    
                    # Determine question type before modifying attention
                    question_type = "word_problem" if self.analyze_question_type(question) > 0.6 else "equation"
                    
                    # Use our new question-specific analysis
                    self.modify_attention_for_question(
                        question=question,
                        num_scaling=num_scaling,
                        op_scaling=op_scaling
                    )
                    
                    # Generate answer
                    single_input = self.tokenizer([prompts[j]], return_tensors="pt", padding=True, truncation=True).to(self.device)
                    with torch.no_grad():
                        output = self.model.generate(
                            input_ids=single_input.input_ids,
                            attention_mask=single_input.attention_mask,
                            max_length=512,
                            num_beams=4,
                            early_stopping=True
                        )
                    prediction = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    final_answer = self.extract_final_answer(prediction)
                    
                    if self.debug:
                        print(f"Question type: {question_type}")
                        print(f"Ground truth: {row['answer']}")
                        print(f"Predicted: {final_answer}")
                        print(f"Correct: {row['answer'].strip().lower() == final_answer.strip().lower()}")
                    
                    batch_results.append({
                        "idx": idx,
                        "question": row["question"], 
                        "prompt": prompts[j],
                        "ground_truth_full": row["ground_truth_full"],
                        "ground_truth": row["answer"],
                        "model_response": prediction,
                        "predicted": final_answer,
                        "question_type": question_type
                    })
                    
                    # Reset weights for next question
                    self.reset_weights()
                
                results.extend(batch_results)
        else:
            # Use traditional approach for other flags
            if flag == "numbers":
                self.modify_attention_for_type("numbers", num_scaling=num_scaling)
                if self.debug:
                    print(f"Applied 'numbers' scaling with factor {num_scaling}")
            elif flag == "operators":
                self.modify_attention_for_type("operators", op_scaling=op_scaling)
                if self.debug:
                    print(f"Applied 'operators' scaling with factor {op_scaling}")
                
            for i in tqdm(range(0, len(df), batch_size)):
                batch = df.iloc[i:i+batch_size]
                prompts = [self.prepare_prompt(q) for q in batch["question"]]
                
                if self.debug and i == 0:
                    print(f"\nProcessing first batch of {len(batch)} questions with batch approach...")
                
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=150,
                        num_beams=4,
                        early_stopping=True
                    )
                predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                final_answers = [self.extract_final_answer(pred) for pred in predictions]
                
                for j, (idx, row) in enumerate(batch.iterrows()):
                    question_type = "word_problem" if self.analyze_question_type(row["question"]) > 0.6 else "equation"
                    results.append({
                        "idx": idx,
                        "question": row["question"], 
                        "prompt": prompts[j],
                        "ground_truth_full": row["ground_truth_full"],
                        "ground_truth": row["answer"],
                        "model_response": predictions[j],
                        "predicted": final_answers[j],
                        "question_type": question_type
                    })
                    
                    if self.debug and i == 0 and j < 3:  # Show debug for first few examples
                        print(f"\nSample {j+1}:")
                        print(f"Question: {row['question'][:100]}...")
                        print(f"Type: {question_type}")
                        print(f"Ground truth: {row['answer']}")
                        print(f"Predicted: {final_answers[j]}")
                        print(f"Correct: {row['answer'].strip().lower() == final_answers[j].strip().lower()}")
            
            # Reset weights after batch processing
            self.reset_weights()
            
        return pd.DataFrame(results)

    def save_results(self, results_df, output_path="inference_results.csv"):
        results_df["correct"] = results_df.apply(
            lambda x: x["ground_truth"].strip().lower() == x["predicted"].strip().lower(), axis=1
        )
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        accuracy = results_df["correct"].mean()
        print(f"Overall Accuracy: {accuracy:.4f}")
        
        if "question_type" in results_df.columns:
            eq_df = results_df[results_df["question_type"] == "equation"]
            wp_df = results_df[results_df["question_type"] == "word_problem"]
            
            if not eq_df.empty:
                eq_acc = eq_df["correct"].mean()
                print(f"Equation Accuracy: {eq_acc:.4f} (n={len(eq_df)})")
            
            if not wp_df.empty:
                wp_acc = wp_df["correct"].mean()
                print(f"Word Problem Accuracy: {wp_acc:.4f} (n={len(wp_df)})")
                
        if self.debug:
            # Print some analysis of the results
            correct_df = results_df[results_df["correct"] == True]
            incorrect_df = results_df[results_df["correct"] == False]
            
            print("\n--- Debug Analysis ---")
            print(f"Total examples: {len(results_df)}")
            print(f"Correct examples: {len(correct_df)} ({len(correct_df)/len(results_df)*100:.1f}%)")
            print(f"Incorrect examples: {len(incorrect_df)} ({len(incorrect_df)/len(results_df)*100:.1f}%)")
            
            if len(incorrect_df) > 0:
                print("\nSample of incorrect predictions:")
                for i, (_, row) in enumerate(incorrect_df.head(3).iterrows()):
                    print(f"\nExample {i+1} (Type: {row['question_type']}):")
                    print(f"Question: {row['prompt'][:100]}...")
                    print(f"Ground truth: {row['ground_truth']}")
                    print(f"Predicted: {row['predicted']}")


def main():
    parser = argparse.ArgumentParser(description="Run customized Flan-T5 inference on MMIQC dataset")
    parser.add_argument("--model", type=str, default="google/flan-t5-base", help="Model name or path")
    parser.add_argument("--output", type=str, default="results/attention_experiments/inference_results.csv", help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--flag", type=str, choices=["none", "numbers", "operators", "both"], default="none", help="Modification flag")
    parser.add_argument("--num_scaling", type=float, default=1.5, help="Number token scaling")
    parser.add_argument("--op_scaling", type=float, default=2.0, help="Operator token scaling")
    parser.add_argument("--debug", action="store_true", help="Enable debug/verbose output")
    parser.add_argument("--modification", type=str, default="Please solve the following problem and only output the answer at the end with \"The answer is: \". ", help="Modifications to the prompt")
    args = parser.parse_args()

    flag = None if args.flag == "none" else args.flag

    inference = CustomizedFlanT5Inference(model_name=args.model, debug=args.debug, prompt=args.modification)
    df = inference.process_dataset()
    results = inference.run_inference(
        df,
        batch_size=args.batch_size,
        flag=flag,
        num_scaling=args.num_scaling,
        op_scaling=args.op_scaling
    )
    inference.save_results(results, output_path=args.output)


if __name__ == "__main__":
    main()

# python src/flan_t5_attention_mod.py --model google/flan-t5-xxl --output results/attention_experiments-equations/without/xxl/inference_baseline.csv --flag none --modification "Please solve the following problem and only output the answer at the end with \"The answer is: \". "

# python src/flan_t5_attention_mod.py --model google/flan-t5-xl --output results/attention_experiments-equations/without/xl/inference_both.csv --flag both --modification "Please solve the following problem and only output the answer at the end with \"The answer is: \". "