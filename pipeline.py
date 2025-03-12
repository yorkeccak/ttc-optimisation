import dotenv
import json
from llms.impl.no_rag_llm import NoRagLlm
from llms.impl.simple_rag_llm import SimpleRagLlm
from llms.impl.rag_token_llm import RagTokenLlm
from llms.impl.fine_tuned_llm import FineTunedLlm
import re
import tiktoken

START_THINK = "<think>"
END_THINK = "</think>"
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_think_tokens(output: str) -> int:
    pattern = re.escape(START_THINK) + r"(.*?)" + re.escape(END_THINK)
    matches = re.findall(pattern, output, flags=re.DOTALL)
    return sum(len(TOKENIZER.encode(match.strip())) for match in matches)


def main() -> None:
    with open("./benchmark.json", "r") as file:
        benchmark = json.load(file)

    model = "deepseek-r1:7b"
    question = benchmark["advanced_biology"]["medium"][0]["question"]

    # answer1 = NoRagLlm(model).generate_output(question)
    # print(f"\n\nNoRagLlm answer:\n{answer1}")
    # print(f"\n\nNoRagLlm think tokens: {count_think_tokens(answer1)}")

    answer2 = SimpleRagLlm(model).generate_output(question)
    print(f"\n\nSimpleRagLlm answer:\n{answer2}")
    print(f"\n\nSimpleRagLlm think tokens: {count_think_tokens(answer2)}")

    # answer3 = RagTokenLlm(model).generate_output(question)
    # print(f"\n\nRagTokenLlm answer:\n{answer3}")
    # print(f"\n\nRagTokenLlm think tokens: {count_think_tokens(answer3)}")


if __name__ == "__main__":
    dotenv.load_dotenv()
    main()
