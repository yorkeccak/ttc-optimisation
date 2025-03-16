from llms.impl.rag_token_llm import RagTokenLlm

llm = RagTokenLlm("deepseek-r1:1.5b")
print(llm.generate_output("What is the capital of France?"))
