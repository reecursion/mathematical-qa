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
import time


class MathProcessor:
    def __init__(self, dataset_name: str = "reecursion/mmiqc-subset", debug=True):
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
    def __init__(self, model_name="google/flan-t5-base", prompt="", device=None, debug=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,device_map="cuda")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda")
        if self.debug:
            print("Model and tokenizer loaded.")
        
        # Store original weights for both encoder and decoder
        self.original_state_dict = {
            k: v.clone() for k, v in self.model.state_dict().items()
            if ('encoder' in k or 'decoder' in k) and 'attention' in k and 'relative_attention_bias' not in k
        }
        
        self.prompt = prompt
        if self.debug:
            print(f"Prompt being used is {self.prompt}")
            encoder_weights = sum(1 for k in self.original_state_dict if 'encoder' in k)
            decoder_weights = sum(1 for k in self.original_state_dict if 'decoder' in k)
            print(f"Original attention weights snapshot stored: {len(self.original_state_dict)} tensors "
                  f"({encoder_weights} encoder, {decoder_weights} decoder).")
        
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
            "sine", "cosine", "tangent", "derivative", "integral", "limit", "amount", "many", "frac", "\\frac", 
            "\frac", "\\sqrt", "sqrt",  "\\sum", "\\prod", "\\int", "\\lim", "\\sin", "\\cos", "\\tan", "\\log", "\\ln",
            "\\cdot", "\\times", "\\div", "\\leq", "\\geq", "\\neq", "\\approx", "\\mod",
            "\\over", "\\binom", "prod", "int", "lim", "sin", "cos", "tan", "log", "ln",
            "cdot", "times", "div", "leq", "geq", "neq", "approx", "mod", "over", "binom"
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

    def modify_attention(self, question=None, num_scaling=1.5, op_scaling=2.0, model_part="both"):
        """
        Unified approach to modify attention weights based on question content.
        
        Args:
            question (str, optional): The question text to analyze for mathematical tokens
            num_scaling (float): Scaling factor for number tokens
            op_scaling (float): Scaling factor for operator tokens
            model_part (str): Which part of the model to modify ("encoder", "decoder", or "both")
        """
        # Always reset weights before any modification
        self.reset_weights()
        
        # If no question provided, no modifications needed
        if not question:
            return
        
        tokenize_start_time = time.time()
        # Tokenize the question
        tokens = self.tokenizer.encode(question, add_special_tokens=False)
        token_strs = self.tokenizer.convert_ids_to_tokens(tokens)

        tokenize_end_time = time.time()

        tokenize_time = tokenize_end_time - tokenize_start_time
        
        regex_start_time = time.time()

        # Define patterns for identifying math content in tokens
        num_pattern = re.compile(r'[0-9]')  # Contains digits
        op_pattern = re.compile(r'[+\-*/=<>^%]')  # Contains operators
        
        # Classify each token
        number_tokens = []
        operator_tokens = []
        mixed_tokens = []
        token_classifications = []  # For debugging
        
        # First pass: Pattern-based detection
        for token_id, token_str in zip(tokens, token_strs):
            has_number = bool(num_pattern.search(token_str))
            has_operator = bool(op_pattern.search(token_str)) if '<unk>' not in token_str else False
            
            classification = {
                "token_id": token_id, 
                "token_str": token_str, 
                "classification": "unmodified", 
                "reason": "", 
                "scaling": None
            }
            
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

        regex_end_time = time.time()
        regex_time = regex_end_time - regex_start_time

        attention_start_time = time.time()
        
        # Apply scaling to identified tokens
        modified_token_count = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Check if we should modify this parameter based on model_part
                should_modify = False
                
                if model_part == "encoder" and 'encoder' in name:
                    should_modify = True
                elif model_part == "decoder" and 'decoder' in name:
                    should_modify = True
                elif model_part == "both" and ('encoder' in name or 'decoder' in name):
                    should_modify = True
                
                if should_modify and 'SelfAttention' in name and 'relative_attention_bias' not in name:
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

        attention_end_time = time.time()
        attention_time = attention_end_time - attention_start_time

        if self.debug:
            print(f"\n{'='*80}")
            print(f"Question: '{question}'")
            print(f"Tokenization: {token_strs}")
            print(f"Model part being modified: {model_part}")
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
        return modified_token_count

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

    def process_dataset(self, dataset_name="reecursion/mmiqc-subset", split="test"):
        processor = MathProcessor(dataset_name, debug=self.debug)
        processed_dataset = processor.process_dataset()
        df = pd.DataFrame({
            "question": processed_dataset[split]["clean_instruction"],
            "ground_truth_full": processed_dataset[split]["output"],
            "answer": processed_dataset[split]["final_answer"]
        })
        return df

    def prepare_prompt(self, question):
        return f"{self.prompt}{question}"

    def extract_final_answer(self, text):
        match = re.search(r"The answer is:\s*(.*)", text)
        if match:
            return match.group(1).strip()
        return ""

    def run_inference(self, df, batch_size=8, num_scaling=1.5, op_scaling=2.0, model_part="both", baseline=False):
        """
        Run inference on a dataset with the option to modify attention weights.
        Logs time taken per example and total time.
        Also logs attention sparsity (percentage of near-zero values).
        """
        results = []
        timing_data = []
        total_time = 0

        if self.debug:
            print(f"Running inference with num_scaling: {num_scaling}, op_scaling: {op_scaling}, model_part: {model_part}")

        total_examples = len(df)

        for i in tqdm(range(0, total_examples, batch_size)):
            batch = df.iloc[i:i + batch_size]
            prompts = [self.prepare_prompt(q) for q in batch["question"]]
            batch_results = []

            for j, (idx, row) in enumerate(batch.iterrows()):
                start_inference_time = time.time()
                question = row["question"]

                if self.debug:
                    print(f"\nProcessing question {idx} ({j + 1}/{len(batch)}):")
                    print(f"QUESTION: {question}")

                # Modify attention if not in baseline mode
                start_attention_mod_time = time.time()
                if not baseline:
                    modified_tokens = self.modify_attention(
                        question=question,
                        num_scaling=num_scaling,
                        op_scaling=op_scaling,
                        model_part=model_part
                    )
                end_attention_mod_time = time.time()
                attention_mod_time = end_attention_mod_time - start_attention_mod_time

                # Tokenize input
                attention_tokenize_start_time = time.time()
                single_input = self.tokenizer([prompts[j]], return_tensors="pt", padding=True, truncation=True).to(self.device)
                attention_tokenize_end_time = time.time() - attention_tokenize_start_time

                # Run model with output_attentions
                forward_start_time = time.time()
                self.model.config.output_attentions = True  # ensure attentions are returned
                with torch.no_grad():
                    output = self.model.generate(
                        input_ids=single_input.input_ids,
                        attention_mask=single_input.attention_mask,
                        max_length=512,
                        num_beams=4,
                        early_stopping=True,
                        output_attentions=True,
                        return_dict_in_generate=True
                    )
                forward_end_time = time.time()
                overall_inference_time = forward_end_time - forward_start_time

                # Decode and extract final answer
                decode_start_time = time.time()
                prediction = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
                final_answer = self.extract_final_answer(prediction)
                decode_end_time = time.time()
                decode_time = decode_end_time - decode_start_time

                # Compute attention sparsity
                sparse_count = 0
                total_count = 0
                threshold = 1e-3

                if output.decoder_attentions is not None:
                    for layer_attention_group in output.decoder_attentions:
                        if isinstance(layer_attention_group, tuple):
                            for attn_tensor in layer_attention_group:
                                sparse_count += (attn_tensor < threshold).sum().item()
                                total_count += attn_tensor.numel()
                        else:
                            sparse_count += (layer_attention_group < threshold).sum().item()
                            total_count += layer_attention_group.numel()

                    sparsity_percent = 100 * sparse_count / total_count if total_count else 0
                else:
                    sparsity_percent = None

                # Store result
                batch_results.append({
                    "idx": idx,
                    "question": row["question"],
                    "prompt": prompts[j],
                    "ground_truth_full": row["ground_truth_full"],
                    "ground_truth": row["answer"],
                    "model_response": prediction,
                    "predicted": final_answer
                })

                if not baseline:
                    self.reset_weights()

                end_inference_time = time.time()
                inference_time = end_inference_time - start_inference_time
                total_time += inference_time

                timing_data.append({
                    "idx": idx,
                    "question": question,
                    "total inference": round(inference_time, 4),
                    "tokenize_time_sec": round(attention_tokenize_end_time, 4),
                    "tokenize+inference": round(overall_inference_time, 4),
                    "attention_mod_time_sec": round(attention_mod_time, 4),
                    "decode time": round(decode_time, 4),
                    "sparse attention count": sparse_count,
                    "total attention values": total_count,
                    "sparsity %": round(sparsity_percent, 4) if sparsity_percent is not None else None,
                })

                torch.cuda.empty_cache()

            results.extend(batch_results)

        # total_end_time = time.time()
        # total_inference_time = total_end_time - total_start_time
        print(f"\n✅ Total inference time for {total_examples} examples: {total_time:.2f} seconds")

        # Save inference timing data to CSV
        timing_df = pd.DataFrame(timing_data)
        timing_df.to_csv("inference_times.csv", index=False)

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
    parser.add_argument("--num_scaling", type=float, default=1.5, help="Number token scaling")
    parser.add_argument("--op_scaling", type=float, default=2.0, help="Operator token scaling")
    parser.add_argument("--model_part", type=str, choices=["encoder", "decoder", "both"], default="both", 
                        help="Which part of the model to apply attention modifications to")
    parser.add_argument("--debug", action="store_true", help="Enable debug/verbose output")
    parser.add_argument("--modification", type=str, default="Please solve the following problem and only output the answer at the end with \"The answer is: \". ", help="Modifications to the prompt")
    parser.add_argument("--baseline", type=bool, default=False)

    args = parser.parse_args()

    torch.cuda.empty_cache()

    inference = CustomizedFlanT5Inference(model_name=args.model, debug=args.debug, prompt=args.modification)
    df = inference.process_dataset()
    results = inference.run_inference(
        df,
        batch_size=args.batch_size,
        num_scaling=args.num_scaling,
        op_scaling=args.op_scaling,
        model_part=args.model_part,
        baseline=args.baseline
    )
    inference.save_results(results, output_path=args.output)


if __name__ == "__main__":
    main()

# Example usage commands:
# Basic (no attention modification):
# python src/flan_t5_attention_mod.py --model google/flan-t5-xxl --output results/attention_experiments-equations/without/xxl/inference_baseline.csv --num_scaling 1.0 --op_scaling 1.0 --modification "Please solve the following problem and only output the answer at the end with \"The answer is: \". "

# With number and operator attention modification (both encoder and decoder):
# python src/flan_t5_attention_mod.py --model google/flan-t5-xl --output results/attention_experiments-equations/without/xl/inference_both.csv --num_scaling 1.5 --op_scaling 2.0 --modification "Please solve the following problem and only output the answer at the end with \"The answer is: \". " --model_part both

# Only modify encoder attention:
# python src/flan_t5_attention_mod.py --model google/flan-t5-xl --output results/attention_experiments-equations/encoder_only/xl/inference_both.csv --num_scaling 1.5 --op_scaling 2.0 --modification "Please solve the following problem and only output the answer at the end with \"The answer is: \". " --model_part encoder

# Only modify decoder attention:
# python src/flan_t5_attention_mod.py --model google/flan-t5-xl --output results/attention_experiments-equations/decoder_only/xl/inference_both.csv --num_scaling 1.5 --op_scaling 2.0 --modification "Please solve the following problem and only output the answer at the end with \"The answer is: \". " --model_part decoder
