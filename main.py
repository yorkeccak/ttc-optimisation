from valyu import Valyu
from dotenv import load_dotenv
load_dotenv()

# Requires .env file with VALYU_API_KEY
valyu = Valyu()

response = valyu.context(
    query="What is positional encoding in transformers",
    search_type="proprietary",
    num_query=5,
    num_results=5,
    max_price=10
)

print(response)