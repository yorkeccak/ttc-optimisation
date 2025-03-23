# TTC Optimization Benchmarking Tool

A comprehensive benchmarking framework for measuring and optimizing Test-Time Compute (TTC) in large language models, specifically focusing on reducing thinking tokens through data access during inference.

## Overview

This repository provides tools to benchmark how different model configurations and retrieval strategies affect the "thinking token" usage in LLMs. The core hypothesis is that providing models with access to external data during inference (through RAG and other methods) reduces the computational burden of reasoning from scratch, measured by tokens generated between `<think>` and `</think>` tags.

### What is Test-Time Compute (TTC)?

Test-Time Compute refers to the computational resources used by a model during inference. In reasoning-focused models, a significant portion of this computation involves internal "thinking" steps. Our framework measures this through thinking tokens - the tokens generated within `<think></think>` tags in the model's response.

### Key Features

- Benchmarks multiple model implementations (No RAG, Simple RAG, RAG Token, Fine-tuned)
- Measures thinking token usage across diverse subject domains and difficulty levels
- Evaluates response quality (correctness and conciseness) using Cohere's Command model
- Generates comprehensive visualizations and analysis reports
- Supports customization through command-line parameters

## Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/your-username/ttc-optimization.git
   cd ttc-optimization
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file in the root directory with your API keys:
   ```
   VALYU_API_KEY=your_valyu_api_key_here
   COHERE_API_KEY=your_cohere_api_key_here
   ```

4. **Install Ollama**
   Follow the instructions at [ollama.ai](https://ollama.ai) to install Ollama for running local models.

5. **Pull required models**
   ```bash
   ollama pull deepseek-r1:1.5b
   # Add other models as needed
   ```

## Usage

### Basic Benchmark Run

Run a basic benchmark comparing No-RAG vs Simple-RAG implementations:

```bash
python benchmark.py --model deepseek-r1:1.5b --model-impl no_rag_llm simple_rag_llm --sample-size 5
```

### Comprehensive Benchmark

For a comprehensive analysis across multiple model implementations:

```bash
python benchmark.py \
  --model deepseek-r1:7b \
  --model-impl no_rag_llm simple_rag_llm rag_token_llm fine_tuned_llm \
  --difficulties medium hard \
  --sample-percent 20 \
  --results-dir comprehensive_benchmark
```

### Command-line Arguments

| Argument | Description |
|----------|-------------|
| `--model` | Base model to use (e.g., deepseek-r1:1.5b, deepseek-r1:7b) |
| `--model-impl` | Model implementations to benchmark (can specify multiple) |
| `--dataset` | Path to benchmark dataset JSON |
| `--sample-size` | Number of questions to sample |
| `--sample-percent` | Percentage of questions to sample (0-100) |
| `--difficulties` | Difficulty levels to include (easy, medium, hard) |
| `--topics` | Subject areas to include |
| `--no-judge` | Disable response quality judging with Cohere |
| `--results-dir` | Custom directory name for results |

## Model Implementations

The framework supports four model implementations:

1. **No RAG (no_rag_llm)**: Baseline model without access to external data
2. **Simple RAG (simple_rag_llm)**: Standard RAG approach where search happens before model inference
3. **RAG Token (rag_token_llm)**: Agentic approach where the model can issue search queries during inference
4. **Fine-tuned (fine_tuned_llm)**: Fine-tuned model with specialized retrieval capabilities

## Benchmark Dataset

The benchmark dataset contains questions across 20 subject areas, each with three difficulty levels:
- **Easy**: Foundational knowledge questions
- **Medium**: Moderately complex questions
- **Hard**: Challenging questions requiring deep understanding

Subject areas include Mathematics, Computer Science, Physics, Economics, Medicine, and more.

## Results and Analysis

Benchmark results include:

1. **Raw Metrics**: 
   - Thinking token usage
   - Response token usage
   - Number of search queries
   - Search results utilized

2. **Performance Metrics**:
   - Thinking token reduction percentage
   - Response correctness
   - Response conciseness

3. **Visualizations**:
   - Model comparison plots
   - Subject and difficulty analysis
   - Performance radar charts
   - Comprehensive PDF report

Results are saved to a timestamped directory with the following structure:
```
results/[timestamp]/
├── data/
│   ├── benchmark_results.json
│   ├── benchmark_results_full.csv
│   ├── benchmark_summary.csv
│   └── subject_difficulty_summary.csv
├── plots/
│   ├── thinking_tokens_by_model.png
│   ├── thinking_tokens_by_difficulty.png
│   ├── thinking_token_reduction.png
│   └── ...
├── evaluations/
├── benchmark.log
└── benchmark_report.pdf
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Open a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this benchmarking tool in your research, please cite:

```
@software{ttc-optimization,
  author = {Your Name},
  title = {TTC Optimization: Benchmarking Framework for Test-Time Compute Reduction},
  year = {2024},
  url = {https://github.com/your-username/ttc-optimization}
}
```

---
