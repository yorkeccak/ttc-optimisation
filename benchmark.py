"""
TTC Optimization Benchmark Pipeline

This script benchmarks different LLM implementation strategies to evaluate how access to external
data during inference reduces Test-Time Compute (TTC) - specifically measuring reduction in thinking tokens.

Key features:
- Compares multiple model implementations (No RAG, Simple RAG, RAG Token, Fine-tuned)
- Measures thinking tokens (tokens within <think></think> tags)
- Evaluates across diverse subject areas and difficulty levels
- Uses Cohere Command model to judge response quality
- Generates comprehensive visualizations and detailed results

Usage:
  python benchmark.py --model deepseek-r1:1.5b --model-impl simple_rag_llm no_rag_llm --sample-size 5
"""

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
import cohere
import numpy as np
import sys
from typing import Literal, List, Dict, Any, Optional
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

# Regular expressions for thinking token counting
START_THINK = "<think>"
END_THINK = "</think>"


def count_thinking_tokens(text: str) -> int:
    """
    Count tokens within <think></think> tags.

    Args:
        text: Text containing thinking sections

    Returns:
        Total number of tokens in thinking sections
    """
    pattern = re.escape(START_THINK) + r"(.*?)" + re.escape(END_THINK)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return sum(len(tokenizer.encode(match.strip())) for match in matches)


def create_results_folder():
    """
    Create a timestamped results folder and return the path.

    Returns:
        Path object pointing to the created results folder
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path("results") / timestamp
    results_path.mkdir(parents=True, exist_ok=True)

    # Create subfolders
    (results_path / "plots").mkdir(exist_ok=True)
    (results_path / "data").mkdir(exist_ok=True)
    (results_path / "evaluations").mkdir(exist_ok=True)
    (results_path / "search_results").mkdir(exist_ok=True)

    return results_path


def setup_logging(log_file="benchmark.log", results_folder=None):
    """
    Set up logging to file with appropriate format.

    Args:
        log_file: Name of the log file
        results_folder: Optional path to store log file in

    Returns:
        Configured logger object
    """
    if results_folder:
        log_file = results_folder / log_file

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger()


def judge_responses(
    responses: List[Dict[str, Any]], question: str, reference_answer: str
) -> Dict[str, Any]:
    """
    Use Cohere Command model to evaluate responses for correctness and conciseness.

    Args:
        responses: List of response dictionaries containing model outputs
        question: The original question
        reference_answer: The reference answer for evaluation

    Returns:
        Dictionary containing evaluation results
    """
    # Get Cohere API key
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY environment variable is not set")

    co = cohere.Client(cohere_api_key)

    # Format all responses for the judge prompt
    response_sections = []
    for resp in responses:
        model_impl = resp.get("model_impl", "unknown")
        response_text = resp.get("response", "No response generated")
        response_sections.append(f"{model_impl} Model answer: {response_text}")

    # Construct the judge prompt
    judge_prompt = (
        "You are evaluating multiple AI-generated answers against a reference answer. "
        "Do not rely on your knowledge base to evaluate the answer, just use the reference answer as an oracle.\n\n"
        f"Question: {question}\n\n"
        f"Reference answer: {reference_answer}\n\n"
        f"{chr(10).join(response_sections)}\n\n"
        "For each model, assess:\n"
        "1. Correctness: Is the answer factually correct compared to the reference? Output T (true) or F (false).\n"
        "2. Conciseness: Rate the conciseness as a percentage (0-100), where 100% means the answer contains "
        "only necessary information with no extraneous details.\n\n"
        "Format your evaluation for each model exactly as follows:\n"
    )

    # Add output format instructions for each model
    for resp in responses:
        model_impl = resp.get("model_impl", "unknown")
        judge_prompt += f"<{model_impl}><Correctness>T/F</Correctness><Conciseness>0-100</Conciseness></{model_impl}>\n"

    # Call Cohere API
    try:
        print(f"\r⚖️  Judging responses with Cohere Command model...", end="")
        response = co.chat(
            model="command-a-03-2025", message=judge_prompt, temperature=0.0
        )
        print(f"\r✅ Responses judged by Cohere Command model                ")

        print(response.text)
        evaluation_text = response.text.strip()
        print(f'\n\nEvaluation text: \n{evaluation_text}' )


        # Extract evaluations
        evaluations = {}
        for resp in responses:
            model_impl = resp.get("model_impl", "unknown")
            print(f'\n\nModel implementation: {model_impl}')

            # Extract correctness
            correctness_pattern = f"<{model_impl}><Correctness>(.*?)</Correctness>"
            correctness_match = re.search(
                correctness_pattern, evaluation_text, re.DOTALL
            )
            correctness = (
                correctness_match.group(1).strip() if correctness_match else "Unknown"
            )
            print(f'Correctness: {correctness}')

            # Extract conciseness
            conciseness_pattern = f"<{model_impl}>.*?<Conciseness>(.*?)</Conciseness>"
            conciseness_match = re.search(
                conciseness_pattern, evaluation_text, re.DOTALL
            )
            conciseness = (
                conciseness_match.group(1).strip() if conciseness_match else "Unknown"
            )
            print(f'Conciseness: {conciseness}')

            evaluations[model_impl] = {
                "correctness": correctness == "T",
                "conciseness": int(conciseness) if conciseness.isdigit() else 0,
                "raw_evaluation": evaluation_text,
            }

        return evaluations

    except Exception as e:
        print(f"\r❌ Error during response evaluation: {e}                    ")
        logging.error(f"Error during response evaluation: {e}")
        return {
            resp.get("model_impl", "unknown"): {
                "correctness": False,
                "conciseness": 0,
                "raw_evaluation": f"Error: {str(e)}",
            }
            for resp in responses
        }


def load_benchmark_dataset(path):
    """
    Load the benchmark dataset from a JSON file.

    Args:
        path: Path to the JSON benchmark file

    Returns:
        Dictionary containing benchmark questions
    """
    print(f"📚 Loading benchmark dataset from {path}...")
    logger.info(f"Loading benchmark dataset from {path}")
    try:
        with open(path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded dataset with {len(data)} subject areas")
        print(f"✅ Loaded {len(data)} subject areas from benchmark dataset")
        return data
    except FileNotFoundError:
        print(f"❌ Error: Benchmark dataset not found at {path}")
        logger.error(f"Benchmark dataset not found at {path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in benchmark dataset at {path}")
        logger.error(f"Invalid JSON in benchmark dataset at {path}")
        sys.exit(1)


def save_results(results, results_folder=None, model_name=""):
    """
    Save detailed benchmark results and generate visualization plots.

    Args:
        results: List of result dictionaries
        results_folder: Path to save results in
        model_name: Base model name used in benchmark
    """
    if results_folder is None:
        results_folder = create_results_folder()

    print(f"\n💾 Saving results to {results_folder}...")
    logger.info(f"Saving benchmark results to {results_folder}")

    # Save raw results
    results_json = results_folder / "data" / "benchmark_results.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved raw results to {results_json}")

    # Save search results separately
    for i, result in enumerate(results):
        if "llm_results" in result:
            for j, llm_result in enumerate(result["llm_results"]):
                if "filtered_search_results" in llm_result:
                    search_file = (
                        results_folder / "search_results" / f"q{i+1}_model{j+1}.txt"
                    )
                    with open(search_file, "w") as f:
                        f.write(llm_result["filtered_search_results"])

    # Extract model implementations used
    model_types = {}
    model_names = []

    # For each benchmark result, get unique model implementations
    if results and "llm_results" in results[0]:
        for i, result in enumerate(results[0]["llm_results"]):
            model_impl = result.get("model_impl", f"model_{i}")
            model_key = f"model_{i}"
            model_types[model_key] = model_impl
            model_names.append(model_impl)

    # Convert results to DataFrame for analysis
    data_rows = []

    for r in results:
        base_row = {
            "subject": r["subject"],
            "difficulty": r["difficulty"],
            "question": r["question"],
        }

        # Add evaluations if available
        if "evaluations" in r:
            for model_impl, eval_data in r["evaluations"].items():
                base_row[f"{model_impl}_correct"] = eval_data.get("correctness", False)
                base_row[f"{model_impl}_conciseness"] = eval_data.get("conciseness", 0)

        for i, result in enumerate(r["llm_results"]):
            model_key = f"model_{i}"
            model_impl = result.get("model_impl", model_key)

            if model_key not in model_types:
                model_types[model_key] = model_impl
                if model_impl not in model_names:
                    model_names.append(model_impl)

            row = base_row.copy()
            # Store response metrics
            row[f"{model_impl}_total_tokens"] = result.get("response_tokens", 0)
            row[f"{model_impl}_thinking_tokens"] = result.get("thinking_tokens", 0)
            row[f"{model_impl}_search_queries"] = result.get("search_queries", 0)
            row[f"{model_impl}_search_results"] = result.get("search_results", 0)
            data_rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(data_rows)

    # Save complete DataFrame
    df_csv = results_folder / "data" / "benchmark_results_full.csv"
    df.to_csv(df_csv, index=False)
    print(f"📊 Full results saved to {df_csv}")

    # Save just the performance metrics
    performance_cols = []
    for model in model_names:
        if f"{model}_correct" in df.columns:
            performance_cols.extend([f"{model}_correct", f"{model}_conciseness"])

    if performance_cols:
        perf_df = df[["subject", "difficulty", "question"] + performance_cols]
        perf_csv = results_folder / "data" / "performance_metrics.csv"
        perf_df.to_csv(perf_csv, index=False)
        print(f"🏆 Performance metrics saved to {perf_csv}")

    # Summary statistics
    logger.info("Generating summary statistics")
    print("📈 Generating summary statistics...")

    # Calculate metrics by model implementation
    summary_metrics = {model: {} for model in model_names}

    for model in model_names:
        # Calculate basic stats
        summary_metrics[model]["avg_total_tokens"] = df[f"{model}_total_tokens"].mean()
        summary_metrics[model]["avg_thinking_tokens"] = (
            df[f"{model}_thinking_tokens"].mean()
            if f"{model}_thinking_tokens" in df
            else 0
        )
        summary_metrics[model]["avg_search_queries"] = (
            df[f"{model}_search_queries"].mean()
            if f"{model}_search_queries" in df
            else 0
        )

        # Calculate average correctness and conciseness if available
        if f"{model}_correct" in df.columns:
            summary_metrics[model]["avg_correctness"] = (
                df[f"{model}_correct"].mean() * 100
            )
        if f"{model}_conciseness" in df.columns:
            summary_metrics[model]["avg_conciseness"] = df[
                f"{model}_conciseness"
            ].mean()

    # Find baseline model (assuming NoRagLlm is the baseline)
    baseline_model = next(
        (m for m in model_names if "no_rag" in m.lower()), model_names[0]
    )

    # Calculate reduction percentages compared to baseline
    for model in model_names:
        if model != baseline_model:
            baseline_thinking = summary_metrics[baseline_model]["avg_thinking_tokens"]
            model_thinking = summary_metrics[model]["avg_thinking_tokens"]

            if baseline_thinking > 0:
                reduction_pct = (
                    (baseline_thinking - model_thinking) / baseline_thinking
                ) * 100
                summary_metrics[model]["thinking_reduction_pct"] = reduction_pct
            else:
                summary_metrics[model]["thinking_reduction_pct"] = 0

    # Convert summary metrics to DataFrame
    summary_df = pd.DataFrame(summary_metrics).transpose()
    summary_df_csv = results_folder / "data" / "benchmark_summary.csv"
    summary_df.to_csv(summary_df_csv)
    print(f"📊 Summary statistics saved to {summary_df_csv}")

    # Group by subject and difficulty
    thinking_cols = [
        f"{model}_thinking_tokens"
        for model in model_names
        if f"{model}_thinking_tokens" in df.columns
    ]
    subject_difficulty_summary = (
        df.groupby(["subject", "difficulty"])[thinking_cols].mean().round(2)
    )
    subject_difficulty_csv = results_folder / "data" / "subject_difficulty_summary.csv"
    subject_difficulty_summary.to_csv(subject_difficulty_csv)

    # Generate visualizations
    print("🎨 Generating visualizations...")
    generate_visualizations(df, model_names, results_folder, model_name, baseline_model)

    # Print summary
    print("\n📋 Benchmark Results Summary:")
    print(f"Base Model: {model_name}")
    print(f"Model Implementations: {', '.join(model_names)}")

    print("\n🧠 Average Thinking Tokens:")
    for model in model_names:
        print(f"  {model}: {summary_metrics[model]['avg_thinking_tokens']:.1f}")

    if len(model_names) > 1:
        print("\n📉 Thinking Token Reduction vs Baseline:")
        for model in model_names:
            if (
                model != baseline_model
                and "thinking_reduction_pct" in summary_metrics[model]
            ):
                print(
                    f"  {model} vs {baseline_model}: {summary_metrics[model]['thinking_reduction_pct']:.1f}%"
                )

    # Print performance metrics if available
    has_performance = any(
        "avg_correctness" in summary_metrics[model] for model in model_names
    )
    if has_performance:
        print("\n🎯 Response Quality Metrics:")
        for model in model_names:
            if "avg_correctness" in summary_metrics[model]:
                print(f"  {model}:")
                print(
                    f"    Correctness: {summary_metrics[model]['avg_correctness']:.1f}%"
                )
                print(
                    f"    Conciseness: {summary_metrics[model]['avg_conciseness']:.1f}/100"
                )

    print("\n📁 Results saved to:")
    print(f"- {results_json} (detailed results)")
    print(f"- {summary_df_csv} (summary statistics)")
    print(f"- {results_folder / 'plots'} (visualizations)")
    print(f"- {results_folder / 'search_results'} (filtered search results)")
    print(f"- {results_folder / 'benchmark.log'} (detailed execution log)")


def generate_visualizations(
    df, model_names, results_folder, model_name, baseline_model
):
    """
    Generate visualizations for benchmark results.

    Args:
        df: DataFrame containing benchmark results
        model_names: List of model implementation names
        results_folder: Path to save visualizations
        model_name: Base model name used in benchmark
        baseline_model: Name of the baseline model implementation
    """
    plots_folder = results_folder / "plots"
    logger.info("Generating visualizations")

    # Set plot style
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 12})

    # 1. Thinking tokens by model implementation
    plt.figure(figsize=(12, 8))
    thinking_cols = [
        f"{model}_thinking_tokens"
        for model in model_names
        if f"{model}_thinking_tokens" in df.columns
    ]

    if thinking_cols:
        thinking_data = pd.melt(
            df,
            value_vars=thinking_cols,
            id_vars=["difficulty"],
        )

        # Clean up variable names for display
        thinking_data["variable"] = thinking_data["variable"].apply(
            lambda x: x.split("_thinking_tokens")[0]
        )

        sns.boxplot(
            data=thinking_data,
            x="variable",
            y="value",
            hue="difficulty",
        )
        plt.title(f"Thinking Tokens by Model Implementation ({model_name})")
        plt.xlabel("Model Implementation")
        plt.ylabel("Thinking Tokens")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_folder / "thinking_tokens_by_model.png", dpi=300)
        plt.close()

    # 2. Thinking tokens by difficulty
    plt.figure(figsize=(12, 8))
    if thinking_cols:
        difficulty_data = thinking_data.copy()
        sns.boxplot(
            data=difficulty_data,
            x="difficulty",
            y="value",
            hue="variable",
        )
        plt.title(f"Thinking Tokens by Difficulty ({model_name})")
        plt.xlabel("Difficulty")
        plt.ylabel("Thinking Tokens")
        plt.legend(title="Model Implementation")
        plt.tight_layout()
        plt.savefig(plots_folder / "thinking_tokens_by_difficulty.png", dpi=300)
        plt.close()

    # 3. Thinking token reduction percentage (if applicable)
    if len(model_names) > 1:
        plt.figure(figsize=(12, 8))

        # Calculate reduction percentage for each row
        baseline_col = f"{baseline_model}_thinking_tokens"

        if baseline_col in df.columns:
            reduction_data = []

            for model in model_names:
                if model != baseline_model:
                    model_col = f"{model}_thinking_tokens"

                    if model_col in df.columns:
                        # Calculate reduction percentage row by row
                        df[f"{model}_reduction_pct"] = (
                            (df[baseline_col] - df[model_col])
                            / df[baseline_col].replace(0, np.nan)
                        ) * 100

                        reduction_rows = df[
                            ["difficulty", f"{model}_reduction_pct"]
                        ].copy()
                        reduction_rows["model"] = model
                        reduction_rows = reduction_rows.rename(
                            columns={f"{model}_reduction_pct": "reduction_pct"}
                        )
                        reduction_data.append(reduction_rows)

            if reduction_data:
                reduction_df = pd.concat(reduction_data, ignore_index=True)

                sns.barplot(
                    data=reduction_df,
                    x="model",
                    y="reduction_pct",
                    hue="difficulty",
                )
                plt.title(
                    f"Thinking Token Reduction vs {baseline_model} ({model_name})"
                )
                plt.xlabel("Model Implementation")
                plt.ylabel("Reduction Percentage (%)")
                plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(plots_folder / "thinking_token_reduction.png", dpi=300)
                plt.close()

    # 4. Subject area performance
    plt.figure(figsize=(15, 10))
    if thinking_cols:
        subject_data = pd.melt(
            df,
            value_vars=thinking_cols,
            id_vars=["subject"],
        )

        # Clean up variable names for display
        subject_data["variable"] = subject_data["variable"].apply(
            lambda x: x.split("_thinking_tokens")[0]
        )

        # Get unique subjects and handle if there are many
        subjects = df["subject"].unique()

        if len(subjects) > 8:
            # For many subjects, create a separate plot
            plt.figure(figsize=(max(15, len(subjects)), 10))

        sns.barplot(
            data=subject_data,
            x="subject",
            y="value",
            hue="variable",
        )
        plt.title(f"Thinking Tokens by Subject Area ({model_name})")
        plt.xlabel("Subject Area")
        plt.ylabel("Thinking Tokens")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Model Implementation")
        plt.tight_layout()
        plt.savefig(plots_folder / "thinking_tokens_by_subject.png", dpi=300)
        plt.close()

    # 5. Correctness and conciseness (if available)
    correctness_cols = [
        f"{model}_correct" for model in model_names if f"{model}_correct" in df.columns
    ]
    conciseness_cols = [
        f"{model}_conciseness"
        for model in model_names
        if f"{model}_conciseness" in df.columns
    ]

    if correctness_cols and conciseness_cols:
        plt.figure(figsize=(15, 8))

        # Prepare data
        correctness_data = []
        conciseness_data = []

        for model in model_names:
            if (
                f"{model}_correct" in df.columns
                and f"{model}_conciseness" in df.columns
            ):
                # Correctness
                correct_pct = df[f"{model}_correct"].mean() * 100
                correctness_data.append({"model": model, "correctness": correct_pct})

                # Conciseness
                conciseness_avg = df[f"{model}_conciseness"].mean()
                conciseness_data.append(
                    {"model": model, "conciseness": conciseness_avg}
                )

        correctness_df = pd.DataFrame(correctness_data)
        conciseness_df = pd.DataFrame(conciseness_data)

        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot correctness
        sns.barplot(data=correctness_df, x="model", y="correctness", ax=ax1)
        ax1.set_title("Answer Correctness by Model")
        ax1.set_xlabel("Model Implementation")
        ax1.set_ylabel("Correctness (%)")
        ax1.set_ylim(0, 100)

        # Plot conciseness
        sns.barplot(data=conciseness_df, x="model", y="conciseness", ax=ax2)
        ax2.set_title("Answer Conciseness by Model")
        ax2.set_xlabel("Model Implementation")
        ax2.set_ylabel("Conciseness Score (0-100)")
        ax2.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(plots_folder / "correctness_conciseness.png", dpi=300)
        plt.close()

    # 6. Combined metric visualization (if all metrics available)
    if thinking_cols and correctness_cols and conciseness_cols:
        plt.figure(figsize=(10, 8))

        # Calculate aggregated metrics by model
        metrics_data = []

        for model in model_names:
            if all(
                col in df.columns
                for col in [
                    f"{model}_thinking_tokens",
                    f"{model}_correct",
                    f"{model}_conciseness",
                ]
            ):
                # Get average metrics
                thinking_avg = df[f"{model}_thinking_tokens"].mean()
                correct_pct = df[f"{model}_correct"].mean() * 100
                conciseness_avg = df[f"{model}_conciseness"].mean()

                # Calculate reduction from baseline if applicable
                reduction_pct = 0
                if (
                    model != baseline_model
                    and f"{baseline_model}_thinking_tokens" in df.columns
                ):
                    baseline_avg = df[f"{baseline_model}_thinking_tokens"].mean()
                    if baseline_avg > 0:
                        reduction_pct = (
                            (baseline_avg - thinking_avg) / baseline_avg
                        ) * 100

                # Add to data
                metrics_data.append(
                    {
                        "model": model,
                        "thinking_tokens": thinking_avg,
                        "correctness": correct_pct,
                        "conciseness": conciseness_avg,
                        "reduction_pct": reduction_pct,
                    }
                )

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)

            # Create radar chart
            categories = [
                "Thinking Tokens (Lower is Better)",
                "Correctness (%)",
                "Conciseness",
                "Reduction vs Baseline (%)",
            ]

            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

            # Normalize values for radar chart
            max_thinking = metrics_df["thinking_tokens"].max()

            # Plot each model
            for _, row in metrics_df.iterrows():
                model = row["model"]

                # Prepare values (normalize and invert thinking tokens)
                values = [
                    (
                        1 - (row["thinking_tokens"] / max_thinking)
                        if max_thinking > 0
                        else 0
                    ),  # Invert so lower is better
                    row["correctness"] / 100,  # Normalize to 0-1
                    row["conciseness"] / 100,  # Normalize to 0-1
                    max(
                        0, row["reduction_pct"] / 100
                    ),  # Normalize and ensure non-negative
                ]
                values += values[:1]  # Close the loop

                # Plot values
                ax.plot(angles, values, linewidth=2, label=model)
                ax.fill(angles, values, alpha=0.1)

            # Set category labels
            plt.xticks(angles[:-1], categories)

            # Add legend
            plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
            plt.title(f"Model Performance Comparison ({model_name})")

            plt.tight_layout()
            plt.savefig(plots_folder / "model_performance_radar.png", dpi=300)
            plt.close()

    # 7. Create a comprehensive summary plot
    plt.figure(figsize=(15, 10))

    # Combine key metrics in one visualization
    metrics_summary = []

    for model in model_names:
        model_metrics = {}

        # Add thinking tokens
        if f"{model}_thinking_tokens" in df.columns:
            model_metrics["thinking_tokens"] = df[f"{model}_thinking_tokens"].mean()

        # Add correctness
        if f"{model}_correct" in df.columns:
            model_metrics["correctness"] = df[f"{model}_correct"].mean() * 100

        # Add conciseness
        if f"{model}_conciseness" in df.columns:
            model_metrics["conciseness"] = df[f"{model}_conciseness"].mean()

        # Add search queries
        if f"{model}_search_queries" in df.columns:
            model_metrics["search_queries"] = df[f"{model}_search_queries"].mean()

        if model_metrics:
            for metric, value in model_metrics.items():
                metrics_summary.append(
                    {"model": model, "metric": metric, "value": value}
                )

    if metrics_summary:
        metrics_df = pd.DataFrame(metrics_summary)

        # Create a grouped bar chart
        plt.figure(figsize=(15, 8))
        g = sns.catplot(
            data=metrics_df,
            kind="bar",
            x="model",
            y="value",
            hue="metric",
            height=6,
            aspect=2,
        )
        g.set_xticklabels(rotation=45)
        g.set(xlabel="Model Implementation", ylabel="Value")
        plt.title(f"Model Performance Metrics Summary ({model_name})")
        plt.tight_layout()
        plt.savefig(plots_folder / "metrics_summary.png", dpi=300)
        plt.close()

    # 8. Create a comprehensive PDF report with all visualizations
    try:
        from fpdf import FPDF

        # Create PDF
        pdf = FPDF()

        # Title page
        pdf.add_page()
        pdf.set_font("Arial", "B", 24)
        pdf.cell(0, 20, f"TTC Benchmark Results", ln=True, align="C")
        pdf.set_font("Arial", "", 14)
        pdf.cell(0, 10, f"Base Model: {model_name}", ln=True, align="C")
        pdf.cell(
            0,
            10,
            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ln=True,
            align="C",
        )
        pdf.cell(
            0,
            10,
            f"Model Implementations: {', '.join(model_names)}",
            ln=True,
            align="C",
        )

        # Add all plots
        for plot_file in plots_folder.glob("*.png"):
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            title = " ".join(plot_file.stem.split("_")).title()
            pdf.cell(0, 10, title, ln=True, align="C")
            pdf.image(str(plot_file), x=10, y=30, w=190)

        # Save PDF
        pdf_path = results_folder / "benchmark_report.pdf"
        pdf.output(str(pdf_path))
        logger.info(f"Created PDF report at {pdf_path}")
        print(f"📑 Created PDF report: {pdf_path}")
    except ImportError:
        logger.info("FPDF not installed, skipping PDF report generation")
        print("📝 Note: Install FPDF package to generate PDF reports")


def run_benchmark(
    model_name: str,
    model_impl: List[Model],
    dataset_path="benchmark.json",
    sample_size=None,
    sample_percent=None,
    difficulties=None,
    topics=None,
    use_judge=True,
):
    """
    Run the benchmark comparing token usage with different model implementations.

    Args:
        model_name: Model name to use for the benchmark
        model_impl: Model implementations to use for the benchmark
        dataset_path: Path to the benchmark dataset JSON
        sample_size: Exact number of questions to sample (overrides sample_percent)
        sample_percent: Percentage of questions to sample (0-100)
        difficulties: List of difficulty levels to include
        topics: List of subject areas to include
        use_judge: Whether to use Cohere model to judge response quality

    Returns:
        List of result dictionaries
    """
    print("🚀 Starting benchmark run...")
    logger.info("Starting benchmark run")
    logger.info(f"Model: {model_name}")
    logger.info(f"Model implementations: {model_impl}")

    # Initialize the models using the mapping
    llms = []
    for impl in model_impl:
        try:
            print(f"\r🔧 Initializing {impl} with model {model_name}...", end="")
            llms.append(MODEL_MAP[impl](model_name))
            logger.info(f"Initialized {impl} with model {model_name}")
            print(f"\r✅ Initialized {impl} with model {model_name}             ")
        except Exception as e:
            print(f"\r❌ Failed to initialize {impl}: {str(e)}                  ")
            logger.error(f"Failed to initialize {impl}: {str(e)}")
            raise

    # Load benchmarking dataset
    dataset = load_benchmark_dataset(dataset_path)

    # Filter by topics if specified
    if topics:
        print(f"🔍 Filtering by topics: {', '.join(topics)}")
        logger.info(f"Filtering by topics: {topics}")
        dataset = {topic: dataset[topic] for topic in topics if topic in dataset}

    # Collect all matching questions at the requested difficulty level
    all_questions = []
    # for subject, difficulty_levels in dataset.items():
    #     for difficulty, questions in difficulty_levels.items():
    #         # Skip if not in requested difficulties
    #         if difficulties and difficulty not in difficulties:
    #             continue

    #         for question_dict in questions:
    #             all_questions.append(
    #                 {
    #                     "subject": subject,
    #                     "difficulty": difficulty,
    #                     "question_dict": question_dict,
    #                 }
    #             )
    count = 0
    for qobj in dataset:
        question_dict = {
            "question": qobj["Question"],
            "answer": qobj["Answer"]
        }
        all_questions.append(
            {
                "subject": "math",
                "difficulty": "easy",
                "question_dict": question_dict,
            }
        )
        count+=1
        if count >= 200:
            break

    logger.info(f"Collected {len(all_questions)} questions matching criteria")
    print(f"📋 Collected {len(all_questions)} questions matching criteria")

    # Sample questions if requested
    if sample_size or sample_percent:
        total_questions = len(all_questions)
        if sample_size:
            sample_count = min(sample_size, total_questions)
        else:
            sample_count = int(total_questions * sample_percent / 100)

        print(f"🎲 Sampling {sample_count} questions from a total of {total_questions}")
        logger.info(
            f"Sampling {sample_count} questions from a total of {total_questions}"
        )
        all_questions = random.sample(all_questions, sample_count)

    results = []

    # Create progress bar
    pbar = tqdm(all_questions, desc="Processing questions", unit="question")

    # Process the selected questions
    for i, question_info in enumerate(pbar):
        subject = question_info["subject"]
        difficulty = question_info["difficulty"]
        question_dict = question_info["question_dict"]
        question = question_dict["question"]
        reference_answer = question_dict.get("answer", "No reference answer provided")

        pbar.set_description(
            f"Question {i+1}/{len(all_questions)}: {subject}/{difficulty}"
        )

        logger.info(
            f"Processing question {i+1}/{len(all_questions)}: {subject}/{difficulty}"
        )
        logger.info(f"Question: {question}")

        llm_results = []

        # Get responses from all model implementations
        for model_idx, llm in enumerate(llms):

            impl_name = model_impl[model_idx]
            pbar.set_postfix({"model": impl_name})
            logger.info(f"Generating response with {impl_name}")

            try:
                # Generate response
                response_dict = llm.generate_output(question)

                # Add model implementation name
                response_dict["model_impl"] = impl_name

                # Count thinking tokens
                thinking_tokens = count_thinking_tokens(
                    response_dict.get("response", "")
                )
                response_dict["thinking_tokens"] = thinking_tokens

                llm_results.append(response_dict)

                logger.info(f"{impl_name} thinking tokens: {thinking_tokens}")
            except Exception as e:
                logger.error(f"Error with {impl_name}: {str(e)}")
                # Add empty response to maintain ordering
                llm_results.append(
                    {
                        "model_impl": impl_name,
                        "response": f"Error: {str(e)}",
                        "response_tokens": 0,
                        "thinking_tokens": 0,
                        "search_queries": 0,
                        "search_results": 0,
                    }
                )

        # Judge responses if requested
        evaluations = None
        if use_judge:
            try:
                logger.info("Evaluating responses with Cohere judge")
                evaluations = judge_responses(llm_results, question, reference_answer)
                logger.info(f"Evaluation results: {evaluations}")
            except Exception as e:
                logger.error(f"Error during evaluation: {str(e)}")

        # Compile result
        result = {
            "subject": subject,
            "difficulty": difficulty,
            "question": question,
            "reference_answer": reference_answer,
            "llm_results": llm_results,
        }

        if evaluations:
            result["evaluations"] = evaluations

        results.append(result)

    logger.info(f"Completed benchmark with {len(results)} questions")
    print(f"\n✅ Completed benchmark with {len(results)} questions")
    return results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="TTC Optimization Benchmark Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "deepseek-r1:1.5b",
            "deepseek-r1:7b",
            "deepseek-r1:14b",
            "llama2:7b",
            "mistral:7b",
        ],
        default="deepseek-r1:1.5b",
        help="Model name to use for the benchmark",
    )
    parser.add_argument(
        "--model-impl",
        type=str,
        nargs="+",
        choices=["simple_rag_llm", "rag_token_llm", "no_rag_llm", "fine_tuned_llm"],
        default=["simple_rag_llm", "no_rag_llm", "rag_token_llm"],
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
        choices=["easy", "medium", "hard"],
        help="Difficulty levels to include",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        help="Subject areas to include (e.g., quantum_computing, decentralized_finance)",
    )
    parser.add_argument(
        "--log-file", type=str, default="benchmark.log", help="Path to log file"
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        default=False,  # Changed default to False to enable judging by default
        help="Disable response quality judging with Cohere",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Custom directory name for results (timestamp will be appended)",
    )

    args = parser.parse_args()

    # Create results folder
    if args.results_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_folder = Path("results") / f"{args.results_dir}_{timestamp}"
    else:
        results_folder = create_results_folder()

    # Create necessary subfolders
    (results_folder / "plots").mkdir(exist_ok=True)
    (results_folder / "data").mkdir(exist_ok=True)
    (results_folder / "evaluations").mkdir(exist_ok=True)
    (results_folder / "search_results").mkdir(exist_ok=True)

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
        use_judge=not args.no_judge,
    )

    # Save results
    save_results(results, results_folder, args.model)
