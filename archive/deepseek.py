import ollama
import time

prompt = f"""
DO you think machines can think? Give me your thoughts.
"""

response = ollama.chat(
    model='deepseek-r1:1.5b',
    stream=True,
    messages=[{'role': 'user', 'content': prompt}]
)

for chunk in response:
    print(chunk)