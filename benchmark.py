import json
import matplotlib.pyplot as plt
import seaborn as sns
from valyu import Valyu
import ollama
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import dotenv
import tiktoken
import re
import argparse
import random
import logging
import datetime
import os
from typing import Literal, List
from llms.impl.simple_rag_llm import SimpleRagLlm
from llms.impl.rag_token_llm import RagTokenLlm
from llms.impl.no_rag_llm import NoRagLlm
from llms.impl.fine_tuned_llm import FineTunedLlm

Model = Literal["simple_rag_llm", "rag_token_llm", "no_rag_llm", "fine_tuned_llm"]

# Load environment variables
dotenv.load_dotenv()

# Initialize tokenizer for token counting
tokenizer = tiktoken.get_encoding("cl100k_base")

# Add a model mapping dictionary
MODEL_MAP = {
    "simple_rag_llm": SimpleRagLlm,
    "rag_token_llm": RagTokenLlm,
    "no_rag_llm": NoRagLlm,
    "fine_tuned_llm": FineTunedLlm,
}


def create_results_folder():
    """Create a results folder with timestamp subfolder and return the path."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path("results") / timestamp
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path


def setup_logging(log_file="benchmark.log", results_folder=None):
    """Set up logging to file."""
    if results_folder:
        log_file = results_folder / log_file

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger()


logger = setup_logging()


def load_benchmark_dataset(path):
    """Load the benchmark dataset from a JSON file."""
    logger.info(f"Loading benchmark dataset from {path}")
    with open(path, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded dataset with {len(data)} subject areas")
    return data


def save_results(results, results_folder=None):
    """Save detailed results and generate summary statistics and visualizations."""
    if results_folder is None:
        results_folder = create_results_folder()

    print(f"\nSaving results to {results_folder}...")
    logger.info(f"Saving benchmark results to {results_folder}")

    # Save raw results
    results_json = results_folder / "benchmark_results.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved raw results to {results_json}")

    # Extract model implementations used
    model_types = []
    model_names = []
    if results and "llm_results" in results[0]:
        for i, result in enumerate(results[0]["llm_results"]):
            model_key = f"model_{i}"
            model_types.append(model_key)
            # Extract the model implementation name from the result
            model_name = result.get("model_impl", f"model_{i}")
            model_names.append(model_name)

    # Convert to DataFrame
    data_rows = []
    for r in results:
        base_row = {
            "subject": r["subject"],
            "difficulty": r["difficulty"],
            "question": r["question"],
        }

        for i, result in enumerate(r["llm_results"]):
            model_key = f"model_{i}"
            if model_key not in model_types:
                model_types.append(model_key)
                model_name = result.get("model_impl", f"model_{i}")
                if model_name not in model_names:
                    model_names.append(model_name)

            row = base_row.copy()
            row[f"{model_key}_tokens"] = result["response_tokens"]
            row[f"{model_key}_search_queries"] = result.get("search_queries", 0)
            row[f"{model_key}_search_results"] = result.get("search_results", 0)
            data_rows.append(row)

    df = pd.DataFrame(data_rows)

    # Summary statistics
    logger.info("Generating summary statistics")

    # Group by subject and difficulty
    token_cols = [f"{model}_tokens" for model in model_types]
    summary = df.groupby(["subject", "difficulty"])[token_cols].mean().round(2)

    # Compute token reduction percentage if we have at least 2 models to compare
    if len(model_types) >= 2:
        # Assuming model_0 is the baseline (e.g., no_rag) and model_1 is the comparison (e.g., simple_rag)
        baseline_col = f"{model_types[0]}_tokens"
        for i in range(1, len(model_types)):
            comparison_col = f"{model_types[i]}_tokens"
            reduction_col = f"{model_names[i]}_vs_{model_names[0]}_reduction_percent"

            df[reduction_col] = (
                (df[baseline_col] - df[comparison_col]) / df[baseline_col] * 100
            ).replace([float("inf"), -float("inf")], 0)

            summary[reduction_col] = summary.apply(
                lambda row: (
                    (row[baseline_col] - row[comparison_col]) / row[baseline_col] * 100
                    if row[baseline_col] != 0
                    else 0
                ),
                axis=1,
            )

    summary_csv = results_folder / "benchmark_summary.csv"
    summary.to_csv(summary_csv)
    logger.info(f"Saved summary statistics to {summary_csv}")

    # Visualizations
    logger.info("Generating visualizations")
    plt.figure(figsize=(15, 10))

    # Plot 1: Token Usage by Model and Difficulty
    plt.subplot(2, 2, 1)
    token_data = pd.melt(
        df,
        value_vars=token_cols,
        id_vars=["difficulty"],
    )
    # Create a mapping for labels
    label_map = {
        f"{model_types[i]}_tokens": f"{model_names[i]}" for i in range(len(model_types))
    }
    token_data["variable"] = token_data["variable"].map(label_map)

    sns.boxplot(
        data=token_data,
        x="difficulty",
        y="value",
        hue="variable",
    )
    plt.title("Token Usage by Difficulty and Model")
    plt.xticks(rotation=45)
    plt.ylabel("Tokens")
    plt.legend(title="Model")

    # Plot 2: Token Reduction Percentage by Difficulty (if applicable)
    if len(model_types) >= 2:
        plt.subplot(2, 2, 2)
        reduction_cols = [
            f"{model_names[i]}_vs_{model_names[0]}_reduction_percent"
            for i in range(1, len(model_types))
        ]
        if reduction_cols:
            reduction_data = pd.melt(
                df,
                value_vars=reduction_cols,
                id_vars=["difficulty"],
            )
            sns.barplot(data=reduction_data, x="difficulty", y="value", hue="variable")
            plt.title("Token Reduction (%) by Difficulty")
            plt.xticks(rotation=45)
            plt.ylabel("Reduction %")
            plt.legend(title="Model Comparison")

    # Plot 3: Token Usage by Subject
    plt.subplot(2, 2, 3)
    subject_data = pd.melt(
        df,
        value_vars=token_cols,
        id_vars=["subject"],
    )
    subject_data["variable"] = subject_data["variable"].map(label_map)

    sns.barplot(data=subject_data, x="subject", y="value", hue="variable")
    plt.title("Token Usage by Subject")
    plt.xticks(rotation=45)
    plt.ylabel("Tokens")
    plt.legend(title="Model")

    # Plot 4: Token Reduction by Subject (if applicable)
    if len(model_types) >= 2:
        plt.subplot(2, 2, 4)
        if reduction_cols:
            subject_reduction = pd.melt(
                df.groupby("subject")[reduction_cols].mean().reset_index(),
                id_vars=["subject"],
                value_vars=reduction_cols,
            )
            sns.barplot(data=subject_reduction, x="subject", y="value", hue="variable")
            plt.title("Token Reduction (%) by Subject")
            plt.xticks(rotation=45)
            plt.ylabel("Reduction %")
            plt.legend(title="Model Comparison")

    plt.tight_layout()
    results_png = results_folder / "benchmark_results.png"
    plt.savefig(results_png)
    logger.info(f"Saved visualizations to {results_png}")

    print("\nResults saved to:")
    print(f"- {results_json} (detailed results)")
    print(f"- {summary_csv} (summary statistics)")
    print(f"- {results_png} (visualizations)")
    print(f"- {results_folder / 'benchmark.log'} (detailed execution log)")


# take list of models
def run_benchmark(
    model_name: str,
    model_impl: List[Model],
    dataset_path="benchmark.json",
    sample_size=None,
    sample_percent=None,
    difficulties=None,
    topics=None,
):
    """Run the benchmark comparing token usage with and without RAG.

    Args:
        model_name: Model name to use for the benchmark
        model_impl: Model implementations to use for the benchmark
        dataset_path: Path to the benchmark dataset JSON
        sample_size: Exact number of questions to sample (overrides sample_percent)
        sample_percent: Percentage of questions to sample (0-100)
        difficulties: List of difficulty levels to include (e.g., ["medium", "hard"])
        topics: List of subject areas to include (e.g., ["quantum_computing", "decentralized_finance"])
    """
    print("Starting benchmark run...")
    logger.info("Starting benchmark run")

    # Initialize the models using the mapping
    llms = [MODEL_MAP[model](model_name) for model in model_impl]
    print(llms)

    # Load benchmarking dataset
    dataset = load_benchmark_dataset(dataset_path)

    # Filter by topics if specified (e.g. ["quantum_computing", "decentralized_finance"])
    if topics:
        logger.info(f"Filtering by topics: {topics}")
        dataset = {topic: dataset[topic] for topic in topics if topic in dataset}

    # Collect all matching questions at the requested difficulty level
    all_questions = []
    for subject, difficulty_levels in dataset.items():
        for difficulty, questions in difficulty_levels.items():
            # Skip if not in requested difficulties
            if difficulties and difficulty not in difficulties:
                continue

            for question_dict in questions:
                all_questions.append(
                    {
                        "subject": subject,
                        "difficulty": difficulty,
                        "question_dict": question_dict,
                    }
                )

    logger.info(f"Collected {len(all_questions)} questions matching criteria")

    # Sample questions if requested
    if sample_size or sample_percent:
        total_questions = len(all_questions)
        if sample_size:
            sample_count = min(sample_size, total_questions)
        else:
            sample_count = int(total_questions * sample_percent / 100)

        print(f"Sampling {sample_count} questions from a total of {total_questions}")
        logger.info(
            f"Sampling {sample_count} questions from a total of {total_questions}"
        )
        all_questions = random.sample(all_questions, sample_count)

    results = []

    # Process the selected questions
    for i, question_info in enumerate(tqdm(all_questions, desc="Processing questions")):
        subject = question_info["subject"]
        difficulty = question_info["difficulty"]
        question_dict = question_info["question_dict"]
        question = question_dict["question"]

        logger.info(
            f"Processing question {i+1}/{len(all_questions)}: {subject}/{difficulty}"
        )
        logger.info(f"Question: {question}")

        llm_results = []

        for llm in llms:
            response = llm.generate_output(question)
            llm_results.append(response)

        results.append(
            {
                "subject": subject,
                "difficulty": difficulty,
                "question": question,
                "llm_results": llm_results,
            }
        )

    logger.info(f"Completed benchmark with {len(results)} questions")
    return results


if __name__ == "__main__":
    # This bnchmark pipeline can be configured by command line arguments to run with different:
    # - models (e.g. deepseek-r1:1.5b, ...)
    # - datasets (for benchmarking ttc)
    # - search implementations (no rag, simple rag, rag token, fine-tuned)
    # - dataset sizes (e.g. percent or exact number of questions)
    # - difficulty levels (e.g. medium, hard)
    # - topics (e.g. quantum computing, decentralized finance)

    parser = argparse.ArgumentParser(description="Run RAG thinking token benchmark")
    parser.add_argument(
        "--model",
        type=str,
        choices=["deepseek-r1:1.5b", "deepseek-r1:7b", "llama2:7b", "mistral:7b"],
        default="deepseek-r1:1.5b",
        help="Model name to use for the benchmark",
    )
    parser.add_argument(
        "--model-impl",
        type=str,
        nargs="+",
        choices=["simple_rag_llm", "rag_token_llm", "no_rag_llm", "fine_tuned_llm"],
        default=["simple_rag_llm", "no_rag_llm"],
        help="Model implementations to use for the benchmark (multiple can be specified)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmark.json",
        help="Path to benchmark dataset",
    )
    parser.add_argument("--sample-size", type=int, help="Number of questions to sample")
    parser.add_argument(
        "--sample-percent", type=float, help="Percentage of questions to sample"
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        choices=["medium", "hard"],
        help="Difficulty levels to include (medium, hard)",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        help="Subject areas to include (e.g., quantum_computing, decentralized_finance)",
    )
    parser.add_argument(
        "--log-file", type=str, default="benchmark.log", help="Path to log file"
    )

    args = parser.parse_args()

    # Create results folder
    results_folder = create_results_folder()

    # Setup logging with custom log file in results folder
    logger = setup_logging(args.log_file, results_folder)
    logger.info(f"Logging to: {results_folder / args.log_file}")
    logger.info(f"Command line arguments: {args}")

    # Run benchmark with the specified parameters
    results = run_benchmark(
        model_name=args.model,
        model_impl=args.model_impl,
        dataset_path=args.dataset,
        sample_size=args.sample_size,
        sample_percent=args.sample_percent,
        difficulties=args.difficulties,
        topics=args.topics,
    )
    save_results(results, results_folder)
