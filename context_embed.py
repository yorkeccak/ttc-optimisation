import ollama
import re
import json

from valyu import Valyu
import dotenv

dotenv.load_dotenv()
valyu = Valyu()

# Constants defining special tags
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

# Function to extract text between special tokens
def extract_between(text, start_tag, end_tag):
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches[-1].strip() if matches else None


def run_external_search(q):
    response = valyu.context(
        query=q,
        search_type="proprietary",
        max_num_results=1,
        max_price=5,
    )
    return response.results[0].content

# Generate response from DeepSeek using Ollama
def generate_ollama(prompt, model="deepseek:latest", stop_tokens=None):
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={"stop": stop_tokens or []} if stop_tokens else {}

    )
    return response['response']

# Main loop implementing iterative retrieval and reasoning
def rag_agent(query, model_name='deepseek-r1:1.5b', max_turns=5):
    prompt = f"""
Answer the following question step-by-step. 
When you encounter information you're uncertain about, place your search query between these tokens:
<|begin_search_query|>[your search query here]<|end_search_query|>
Use this format whenever you need to verify facts, obtain up-to-date information, or fill knowledge gaps.
This helps identify exactly what information you need to search for.
\nQuestion: {query}
\nAnswer
    """

    for turn in range(max_turns):
        print(f"\n--- Turn {turn+1} ---")
        print(prompt)
        response = generate_ollama(prompt, model_name)
        print(f"Model Response:\n{response}")

        # Extract search query if any
        search_query = extract_between(response, "<|begin_search_query|>", "<|end_search_query|>")
        if search_query:
            print(f"\n[🔍 Searching for: '{search_query}']")
            res = run_external_search(prompt)

            embedded_context = f"\n{BEGIN_SEARCH_RESULT}\n{res}\n{END_SEARCH_RESULT}\n"
            prompt += f"\n{response}\n{embedded_context}\n"
            print(f"\n[✅ Search Results embedded into prompt.]\n")
            continue

        else:
            print("\n[✅ No further searches required.]")
            print("\nFinal Response:\n")
            print(response)
            break

if __name__ == "__main__":
    question = "What is the current inflation rate as of 2025 January in the US? I want exact figures not predictions output the tokens if not sure."
    rag_agent(question)