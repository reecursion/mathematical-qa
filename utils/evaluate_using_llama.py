import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

class MathAnswerEvaluator:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", max_new_tokens=256):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        print("Loading model...")
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

    def build_prompt(self, model_response, ground_truth):
        return f"""You are a mathematical reasoning assistant.
Compare the following two mathematical solutions and determine if they are equivalent in logic and correctness.
Respond only with "Yes" or "No", followed by a short explanation (no more than two sentences).

Solution A:
{model_response}

Solution B:
{ground_truth}

Are these two solutions equivalent?"""

    def evaluate_pair(self, model_response, ground_truth):
        prompt = self.build_prompt(model_response, ground_truth)
        response = self.pipeline(prompt)[0]["generated_text"]
        index = response.rfind("Are these two solutions equivalent?")
        extracted_response = response[index + len("Are these two solutions equivalent?"):].strip()

        if "Yes" in extracted_response or "yes" in extracted_response:
            return True, extracted_response
        elif "No" in extracted_response or "no" in extracted_response:
            return False, extracted_response
        else:
            return None, extracted_response

class BatchMathEvaluator:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def evaluate_csv(self, input_path: str, output_path: str, batch_size: int = 8):
        df = pd.read_csv(input_path)
        df=df.head(10)
        print(f"Evaluating {len(df)} examples in batches of {batch_size}...")

        prompts = [
            self.evaluator.build_prompt(row["model_response"], row["ground_truth_full"])
            for _, row in df.iterrows()
        ]

        evaluation_explanations = []
        final_evaluations = []

        for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating"):
            batch_prompts = prompts[i:i+batch_size]
            batch_outputs = self.evaluator.pipeline(batch_prompts)

            for output in batch_outputs:
                response = output[0]["generated_text"]
                index = response.rfind("Are these two solutions equivalent?")
                extracted_response = response[index + len("Are these two solutions equivalent?"):].strip()


                if "Yes" in extracted_response or "yes" in extracted_response:
                    final_eval = "yes"
                elif "No" in extracted_response or "no" in extracted_response:
                    final_eval = "no"
                else:
                    final_eval = "unknown"

                evaluation_explanations.append(extracted_response.strip())
                final_evaluations.append(final_eval)

        # Add results to original dataframe
        df["evaluation_explanation"] = evaluation_explanations
        df["final_evaluation"] = final_evaluations

        df.to_csv(output_path, index=False)
        print(f"✅ Results saved to {output_path}")

        # Score calculation
        df["final_evaluation"] = df["final_evaluation"].str.lower()
        yes_count = (df["final_evaluation"] == "yes").sum()
        no_count = (df["final_evaluation"] == "no").sum()
        score = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0

        print(f"✅ Evaluation Score (yes / (yes + no)): {score:.2f}")

if __name__ == "__main__":
    evaluator = MathAnswerEvaluator()
    batch_evaluator = BatchMathEvaluator(evaluator)
    parser = argparse.ArgumentParser(description="Run customized Flan-T5 inference on MMIQC dataset")
    parser.add_argument("--file", type=str, default="", help="Enter the input file name")

    batch_evaluator.evaluate_csv(args.file, args.file+"_evaluated")
