import dotenv
from llms.impl.fine_tuned_llm import FineTunedLlm

dotenv.load_dotenv()

llm = FineTunedLlm()
question = "Who is the current President of the United States of America?"
response = llm.generate_output(question)
print(response["response"])
