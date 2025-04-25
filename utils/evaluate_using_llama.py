import argparse
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
from openai import OpenAI

class MathAnswerEvaluator:
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-70B-Instruct", max_new_tokens=256, use_babel=False, use_explanation=False):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.use_babel = use_babel
        self.use_explanation = use_explanation

        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.client = None

        if self.use_babel:
            self._load_babel_client()
        else:
            self._load_model()

    def _load_model(self):
        print("Loading local Huggingface model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype="auto"
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens
        )
        print("Model loaded.")

    def _load_babel_client(self):
        print("Connecting to Babel server...")
        self.client = OpenAI(
            base_url="http://babel-6-21:8081/v1",
            api_key="EMPTY"
        )
        print("Babel client ready.")

    def build_prompt(self, model_response, ground_truth):
        prompt = f"""You are a rigorous mathematical reasoning evaluator.
Your task is to compare two solutions and decide if they are logically and mathematically equivalent.

Solution A:
{model_response}

Solution B:
{ground_truth}

Respond ONLY with "Yes" or "No". Do NOT add anything else.

Are these two solutions equivalent?"""
        return prompt

    def generate_response(self, prompt):
        if self.use_babel:
            try:
                response = self.client.chat.completions.create(
                    model="meta-llama/Llama-3.1-70B-Instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=10
                )
                time.sleep(0.5)
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[ERROR] Llama API call failed: {str(e)}")
                time.sleep(2)
                raise
        else:
            return self.pipeline(prompt)[0]["generated_text"]

    def evaluate_pair(self, model_response, ground_truth):
        prompt = self.build_prompt(model_response, ground_truth)
        response = self.generate_response(prompt)

        if "Yes" in response or "yes" in response:
            return True, response
        elif "No" in response or "no" in response:
            return False, response
        else:
            return None, response  # shouldn't happen with new prompt

class BatchMathEvaluator:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def evaluate_csv(self, input_path: str, output_path: str, batch_size: int = 8):
        df = pd.read_csv(input_path)
        df = df.head(10)
        print(f"Evaluating {len(df)} examples...")

        evaluation_explanations = []
        final_evaluations = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            # Choose ground truth based on --exp flag
            ground_truth = row["ground_truth_full"] if self.evaluator.use_explanation else row["ground_truth"]

            prompt = self.evaluator.build_prompt(row["model_response"], ground_truth)
            response = self.evaluator.generate_response(prompt)

            if "Yes" in response or "yes" in response:
                final_eval = "yes"
            elif "No" in response or "no" in response:
                final_eval = "no"
            else:
                final_eval = "unknown"

            evaluation_explanations.append(response.strip())
            final_evaluations.append(final_eval)

        # Add results to original dataframe
        df["evaluation_explanation"] = evaluation_explanations
        df["final_evaluation"] = final_evaluations

        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

        # Score calculation
        df["final_evaluation"] = df["final_evaluation"].str.lower()
        yes_count = (df["final_evaluation"] == "yes").sum()
        no_count = (df["final_evaluation"] == "no").sum()
        score = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0

        print(f"Evaluation Score (yes / (yes + no)): {score:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run customized evaluation on MMIQC dataset")
    parser.add_argument("--file", type=str, required=True, help="Input file name (CSV)")
    parser.add_argument("--babel", action="store_true", help="Use Babel server instead of local model")
    parser.add_argument("--exp", action="store_true", help="Use ground_truth_full instead of ground_truth (more explanation)")
    args = parser.parse_args()

    evaluator = MathAnswerEvaluator(use_babel=args.babel, use_explanation=args.exp)
    batch_evaluator = BatchMathEvaluator(evaluator)

    batch_evaluator.evaluate_csv(args.file, args.file + "_evaluated")
