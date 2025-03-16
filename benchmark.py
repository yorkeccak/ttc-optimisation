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

# Load environment variables
dotenv.load_dotenv()

# Initialize tokenizer for token counting
tokenizer = tiktoken.get_encoding("cl100k_base")

def create_results_folder():
    """Create a results folder with timestamp subfolder and return the path."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path("results") / timestamp
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path

def setup_logging(log_file='benchmark.log', results_folder=None):
    """Set up logging to file."""
    if results_folder:
        log_file = results_folder / log_file
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger()

logger = setup_logging()

def load_benchmark_dataset(path):
    """Load the benchmark dataset from a JSON file."""
    logger.info(f"Loading benchmark dataset from {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded dataset with {len(data)} subject areas")
    return data

def stream_and_get_response(messages):
    """Stream the model response and return the full response with token count."""
    logger.info("Sending request to model")
    response = ollama.chat(
        model='deepseek-r1:1.5b',
        messages=messages,
        stream=True
    )
    
    full_response = ""
    for chunk in response:
        full_response += chunk['message']['content']
    
    # Count tokens in the entire response
    response_tokens = len(tokenizer.encode(full_response))
    logger.info(f"Received response with {response_tokens} tokens")
    
    return {
        'response_tokens': response_tokens,
        'full_response': full_response
    }

def generate_prompt(query, context=None):
    """Generate a simple prompt with optional context."""
    base_prompt = f"""
You are a knowledgeable research assistant. Answer the following question:
"{query}"
"""
    if context:
        base_prompt += f"\nYou can use this context to inform your answer:\n{context}"
    return base_prompt

def run_benchmark(dataset_path='datasets/benchmark_dataset_advanced.json', 
                 sample_size=None, sample_percent=None, 
                 difficulties=None, topics=None):
    """Run the benchmark comparing token usage with and without RAG.
    
    Args:
        dataset_path: Path to the benchmark dataset JSON
        sample_size: Exact number of questions to sample (overrides sample_percent)
        sample_percent: Percentage of questions to sample (0-100)
        difficulties: List of difficulty levels to include (e.g., ["medium", "hard"])
        topics: List of subject areas to include (e.g., ["quantum_computing", "decentralized_finance"])
    """
    print("Starting benchmark run...")
    logger.info("Starting benchmark run")
    valyu = Valyu()
    dataset = load_benchmark_dataset(dataset_path)
    
    # Filter by topics if specified
    if topics:
        logger.info(f"Filtering by topics: {topics}")
        dataset = {topic: dataset[topic] for topic in topics if topic in dataset}
    
    # Collect all matching questions
    all_questions = []
    for subject, difficulty_levels in dataset.items():
        for difficulty, questions in difficulty_levels.items():
            # Skip if not in requested difficulties
            if difficulties and difficulty not in difficulties:
                continue
            
            for question_dict in questions:
                all_questions.append({
                    'subject': subject,
                    'difficulty': difficulty,
                    'question_dict': question_dict
                })
    
    logger.info(f"Collected {len(all_questions)} questions matching criteria")
    
    # Sample questions if requested
    if sample_size or sample_percent:
        total_questions = len(all_questions)
        if sample_size:
            sample_count = min(sample_size, total_questions)
        else:
            sample_count = int(total_questions * sample_percent / 100)
        
        print(f"Sampling {sample_count} questions from a total of {total_questions}")
        logger.info(f"Sampling {sample_count} questions from a total of {total_questions}")
        all_questions = random.sample(all_questions, sample_count)
    
    results = []
    
    # Process the selected questions
    for i, question_info in enumerate(tqdm(all_questions, desc="Processing questions")):
        subject = question_info['subject']
        difficulty = question_info['difficulty']
        question_dict = question_info['question_dict']
        question = question_dict['question']
        
        logger.info(f"Processing question {i+1}/{len(all_questions)}: {subject}/{difficulty}")
        logger.info(f"Question: {question}")
        
        # Without RAG
        logger.info("Generating response without RAG")
        prompt_without_rag = generate_prompt(question)
        without_rag_metrics = stream_and_get_response([
            {'role': 'user', 'content': prompt_without_rag}
        ])
        
        # With RAG
        logger.info("Fetching context with Valyu")
        try:
            context_response = valyu.context(
                query=question,
                search_type="all",
                max_num_results=10,
                max_price=100,
                similarity_threshold=0.4
            )
            logger.info(f"Received {len(context_response.results)} context results")
            
            context_text = '\n'.join([f"Source {i+1}:\n{result.content}" 
                                     for i, result in enumerate(context_response.results)])
        except Exception as e:
            logger.error(f"Error retrieving context from Valyu: {str(e)}")
            logger.info("Continuing without RAG context")
            context_text = "No relevant context found."
        
        prompt_with_rag = generate_prompt(question, context_text)
        
        logger.info("Generating response with RAG")
        with_rag_metrics = stream_and_get_response([
            {'role': 'user', 'content': prompt_with_rag}
        ])
        
        # Store results
        results.append({
            'subject': subject,
            'difficulty': difficulty,
            'question': question,
            'without_rag': {
                'response_tokens': without_rag_metrics['response_tokens'],
                'full_response': without_rag_metrics['full_response']
            },
            'with_rag': {
                'response_tokens': with_rag_metrics['response_tokens'],
                'full_response': with_rag_metrics['full_response'],
                'context': context_text
            }
        })
        
        logger.info(f"Tokens without RAG: {without_rag_metrics['response_tokens']}, "
                   f"with RAG: {with_rag_metrics['response_tokens']}")
    
    logger.info(f"Completed benchmark with {len(results)} questions")
    return results

def save_results(results, results_folder=None):
    """Save detailed results and generate summary statistics and visualizations."""
    if results_folder is None:
        results_folder = create_results_folder()
    
    print(f"\nSaving results to {results_folder}...")
    logger.info(f"Saving benchmark results to {results_folder}")
    
    # Save raw results
    results_json = results_folder / 'benchmark_results.json'
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved raw results to {results_json}")
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'subject': r['subject'],
            'difficulty': r['difficulty'],
            'without_rag_tokens': r['without_rag']['response_tokens'],
            'with_rag_tokens': r['with_rag']['response_tokens']
        }
        for r in results
    ])
    
    # Summary statistics
    logger.info("Generating summary statistics")
    summary = df.groupby(['subject', 'difficulty']).agg({
        'without_rag_tokens': 'mean',
        'with_rag_tokens': 'mean'
    }).round(2)
    
    # Compute token reduction percentage
    df['token_reduction_percent'] = ((df['without_rag_tokens'] - df['with_rag_tokens']) / 
                                    df['without_rag_tokens'] * 100).replace([float('inf'), -float('inf')], 0)
    summary['token_reduction_percent'] = summary.apply(
        lambda row: ((row['without_rag_tokens'] - row['with_rag_tokens']) / 
                     row['without_rag_tokens'] * 100) if row['without_rag_tokens'] != 0 else 0,
        axis=1
    )
    summary_csv = results_folder / 'benchmark_summary.csv'
    summary.to_csv(summary_csv)
    logger.info(f"Saved summary statistics to {summary_csv}")
    
    # Visualizations
    logger.info("Generating visualizations")
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Thinking Tokens by Difficulty
    plt.subplot(2, 2, 1)
    sns.boxplot(data=pd.melt(df, 
                            value_vars=['without_rag_tokens', 'with_rag_tokens'],
                            id_vars=['difficulty']),
                x='difficulty', y='value', hue='variable')
    plt.title('Thinking Tokens by Difficulty')
    plt.xticks(rotation=45)
    
    # Plot 2: Token Reduction Percentage by Difficulty
    plt.subplot(2, 2, 2)
    sns.barplot(data=df, x='difficulty', y='token_reduction_percent')
    plt.title('Token Reduction (%) by Difficulty')
    plt.xticks(rotation=45)
    
    # Plot 3: Token Reduction Percentage by Subject
    plt.subplot(2, 2, 3)
    subject_reduction = df.groupby('subject')['token_reduction_percent'].mean().reset_index()
    sns.barplot(data=subject_reduction, x='subject', y='token_reduction_percent')
    plt.title('Token Reduction (%) by Subject')
    plt.xticks(rotation=45)
    
    # Plot 4: Token Usage Comparison
    plt.subplot(2, 2, 4)
    token_comparison = df.groupby('subject')[['without_rag_tokens', 'with_rag_tokens']].mean().reset_index()
    token_comparison_melted = pd.melt(token_comparison, 
                                     id_vars=['subject'],
                                     value_vars=['without_rag_tokens', 'with_rag_tokens'])
    sns.barplot(data=token_comparison_melted, x='subject', y='value', hue='variable')
    plt.title('Average Token Usage by Subject')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    results_png = results_folder / 'benchmark_results.png'
    plt.savefig(results_png)
    logger.info(f"Saved visualizations to {results_png}")
    
    print("\nResults saved to:")
    print(f"- {results_json} (detailed results)")
    print(f"- {summary_csv} (summary statistics)")
    print(f"- {results_png} (visualizations)")
    print(f"- {results_folder / 'benchmark.log'} (detailed execution log)")

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Run RAG thinking token benchmark')
    parser.add_argument('--dataset', type=str, default='datasets/benchmark_dataset_advanced.json',
                        help='Path to benchmark dataset')
    parser.add_argument('--sample-size', type=int, help='Number of questions to sample')
    parser.add_argument('--sample-percent', type=float, help='Percentage of questions to sample')
    parser.add_argument('--difficulties', nargs='+', choices=['medium', 'hard'],
                        help='Difficulty levels to include (medium, hard)')
    parser.add_argument('--topics', nargs='+',
                        help='Subject areas to include (e.g., quantum_computing, decentralized_finance)')
    parser.add_argument('--log-file', type=str, default='benchmark.log',
                        help='Path to log file')
    
    args = parser.parse_args()
    
    # Create results folder
    results_folder = create_results_folder()
    
    # Setup logging with custom log file in results folder
    logger = setup_logging(args.log_file, results_folder)
    logger.info(f"Logging to: {results_folder / args.log_file}")
    logger.info(f"Command line arguments: {args}")
    
    # Run benchmark with the specified parameters
    results = run_benchmark(
        dataset_path=args.dataset,
        sample_size=args.sample_size,
        sample_percent=args.sample_percent,
        difficulties=args.difficulties,
        topics=args.topics
    )
    save_results(results, results_folder)