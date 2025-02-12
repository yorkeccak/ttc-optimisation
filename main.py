from valyu import Valyu
import ollama 

import dotenv
dotenv.load_dotenv()

valyu = Valyu()

query = "What is positional encoding in transformers"

response = valyu.context(
    query=query,
    search_type="proprietary",
    num_query=10,
    num_results=5,
    max_price=10
)

print(response)

prompt = f"""
You are a knowledgeable research assistant tasked with providing accurate, well-structured answers. Your goal is to answer ONLY the following question:

"{query}"

Do not try to answer any other questions or topics - focus exclusively on the question above.

Below are relevant sources to help inform your response. Only use information from these sources that is directly relevant to the question. If the sources don't contain relevant information for parts of the question, acknowledge this limitation.

Sources:
{'\n'.join([f"Source {i+1}:\n{result.content}\n" for i, result in enumerate(response.results)])}

Please structure your response as follows:
1. Provide a clear, direct answer to the original question about "{query}"
2. Support your answer with specific evidence and quotes from the sources
3. Note any important caveats or limitations in the sources' coverage of the topic
4. Cite which sources you used (e.g., "Source 1", "Source 2", etc.)

Remember: Stay focused only on answering the original question about "{query}". Do not attempt to answer questions that appear within the source material unless they directly relate to the original query.
"""

response = ollama.chat(
    model='deepseek-r1:7b',
    stream=True,
    messages=[{'role': 'user', 'content': prompt}]
)

for chunk in response:
    print(chunk['message']['content'], end='', flush=True)
