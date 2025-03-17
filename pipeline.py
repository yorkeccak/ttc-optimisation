import dotenv
import json
from llms.impl.no_rag_llm import NoRagLlm
from llms.impl.simple_rag_llm import SimpleRagLlm
from llms.impl.rag_token_llm import RagTokenLlm
from llms.impl.fine_tuned_llm import FineTunedLlm
import re
import tiktoken
import cohere
import os
from collections import Counter


START_THINK = "<think>"
END_THINK = "</think>"
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_think_tokens(output: str) -> int:
    pattern = re.escape(START_THINK) + r"(.*?)" + re.escape(END_THINK)
    matches = re.findall(pattern, output, flags=re.DOTALL)
    return sum(len(TOKENIZER.encode(match.strip())) for match in matches)


def judge_all_answers(norag_answer: str, simple_answer: str, ragtoken_answer: str, finetuned_answer: str, question: str, reference_answer: str) -> str:
    """
    Uses the Cohere API to evaluate all three model answers against a reference answer in a single prompt.
    Returns a formatted assessment with correctness (T/F) and conciseness (percentage) for each model.
    """
    judge_prompt = (
        "You are evaluating three AI-generated answers against a reference answer. Do not rely on your knowledge base to evaluate the answer, just use the reference answer as an oracle."
        "For each model, assess:\n\n"
        "1. Correctness: Is the answer factually correct? Output T (true) or F (false).\n"
        "2. Conciseness: Rate the conciseness as a percentage (0-100), where 100% means the answer contains "
        "only necessary information with no extraneous details, and 0% means no relevant information.\n\n"
        f"Reference answer: {reference_answer}\n\n"
        f"NoRag Model answer: {norag_answer}\n\n"
        f"SimpleRag Model answer: {simple_answer}\n\n" 
        f"RagToken Model answer: {ragtoken_answer}\n\n"
        f"FineTuned Model answer: {ragtoken_answer}\n\n"
        "Format your evaluation exactly as follows (replace values with your assessment):\n"
        "<ModelNoRag><Correctness>T/F</Correctness><Conciseness>0-100</Conciseness></ModelNoRag>\n"
        "<ModelSimpleRag><Correctness>T/F</Correctness><Conciseness>0-100</Conciseness></ModelSimpleRag>\n"
        "<ModelRagToken><Correctness>T/F</Correctness><Conciseness>0-100</Conciseness></ModelRagToken>"
        "<ModelFinetuned><Correctness>T/F</Correctness><Conciseness>0-100</Conciseness></ModelFinetuned>\n"
    )
    
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY environment variable is not set.")
    
    co = cohere.Client(cohere_api_key)
    
    response = co.chat(
        # command-r-plus-04-2024
        model="command-a-03-2025",
        message=judge_prompt,
        temperature=0.0,
        stop_sequences=[]
    )

    return response.text.strip()

def main() -> None:
    with open("./benchmark/benchmark.json", "r") as file:
        benchmark = json.load(file)

    model = "deepseek-r1:14b"

    think_tokens = Counter()
    
    for category, difficulties in benchmark.items():
        print(f"\n{'='*80}\nCategory: {category}\n{'='*80}")
        
        for difficulty, questions in difficulties.items():
            print(f"\n{'-'*60}\nDifficulty: {difficulty}\n{'-'*60}")
            
            for i, question_data in enumerate(questions):
                question = question_data["question"]
                reference_answer = question_data["answer"]

                question = 'What was the unemployment rate in the USA in December 2024?'
                reference_answer = 'The unemployment rate in the USA in December 2024 was 4.1%.'
                print(f"\nQuestion {i+1}: {question}")
                print(f"Reference answer: {reference_answer}")
                
                # Generate answers from all three models
                answer1 = NoRagLlm(model).generate_output(question)
                print(f"\nNoRagLlm Answer:\n{answer1}")
                think_tokens['NoRagLlm'] += count_think_tokens(answer1)
                
                answer2 = SimpleRagLlm(model).generate_output(question)
                print(f"\nSimpleRagLlm Answer:\n{answer2}")
                think_tokens['SimpleRagLlm'] += count_think_tokens(answer2)
                
                answer3 = RagTokenLlm(model).generate_output(question)
                print(f"\nRagTokenLlm Answer:\n{answer3}")
                think_tokens['RagTokenLlm'] += count_think_tokens(answer3)

                answer4 = FineTunedLlm(model).generate_output(question)
                print(f"\nFineTunedLlm Answer:\n{answer4}")
                think_tokens['FineTunedLlm'] += count_think_tokens(answer4)

                evaluations = judge_all_answers(answer1, answer2, answer3, answer4, question, reference_answer)
                print(f"\nEvaluations:\n{evaluations}")
                break
            break
        break

    # Print summary of think tokens
    print("\n\nSummary of think tokens:")
    for model_name, token_count in think_tokens.items():
        print(f"{model_name}: {token_count} tokens")


if __name__ == "__main__":
    dotenv.load_dotenv()
    main()
