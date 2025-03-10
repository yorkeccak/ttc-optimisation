# Benchmark Questions Repository

Welcome to the Benchmark Questions Repository! This repository features a comprehensive benchmark designed to test and expand knowledge across a wide range of subjects, with a focus on evaluating large language model performance.

> **UCL > ICL 💩💩💩💩💩💩💩**  
> *(Universal Truth)*

## Overview

The benchmark is structured into **20 Topics**, each divided into **3 Subcategories** based on difficulty:

- **Easy:** 10 introductory knowledge-based questions.
- **Medium:** 10 questions with moderate complexity.
- **Hard:** 10 challenging questions requiring in-depth understanding.

## Setup

1. Clone this repository
2. Install the required packages: `pip install -r requirements.txt`
3. Create a `.env` file in the root directory with your Valyu API key:
   ```
   VALYU_API_KEY=your_api_key_here
   ```

## Scripts

- **benchmark.py**: The main script that runs comprehensive benchmarks comparing token usage with and without RAG (Retrieval-Augmented Generation).
  ```bash
  python benchmark.py --sample-size 5 --difficulties medium hard
  ```

- **deepseek.py**: A simple test script to demonstrate how to use the Ollama API with the DeepSeek model.

- **main.py**: An example script showing how to use the Valyu API for contextual retrieval.

## Topics

1. **Mathematics**
2. **Geography**
3. **Quantitative Finance**
4. **Computer Science**
5. **Physics**
6. **Chemistry**
7. **Biology**
8. **Economics**
9. **Philosophy**
10. **World History**
11. **Art**
12. **Music**
13. **Sports**
14. **Medicine**
15. **Engineering**
16. **Astronomy**
17. **Psychology**
18. **Law**
19. **Religion**
20. **Culinary Arts**

## Structure

Each topic contains:

- **Easy:** 10 foundational questions.
- **Medium:** 10 moderately challenging questions.
- **Hard:** 10 advanced, in-depth questions.

This consistent structure allows you to quickly navigate through topics and focus on the level of challenge.

## How to Use

- **For LLMs:** Integrate these benchmarks with models to test their performance in terms of output tokens (ttc).
- **With RAG:** The benchmark compares performance with and without retrieval-augmented generation.

## Contributing

Contributions are welcome! If you have suggestions, improvements, or additional questions, please feel free to open an issue or submit a pull request.

## License

This repository is licensed under the [MIT License](LICENSE).

---
