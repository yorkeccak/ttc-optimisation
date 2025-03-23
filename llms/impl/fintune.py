from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ===================
# = LOAD BASE MODEL =
# ===================

max_seq_length = 2048
base_model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B"
tokenizer_name = "canertugrul/DeepSeek-R1-Distill-Llama-8B-Tool-Use-Tokenizer"

# ============================
# = LOAD TOKENIZER =
# ============================

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Define special tokens with proper escaping
special_tokens = {"<SEARCH>", "</SEARCH>", "<RESULT>", "</RESULT>"}

# Check for existing tokens and add new ones if needed
new_tokens = list(special_tokens - set(tokenizer.vocab.keys()))

if new_tokens:
    print("Adding new tokens... ")
    # Add new tokens to tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    print(f"Added {len(new_tokens)} new tokens to the tokenizer!")
else:
    print("No new tokens added.")

# ============================
# = LOAD MODEL WITH MPS =
# ============================

# Check if MPS is available (Mac with Apple Silicon)
use_mps = torch.backends.mps.is_available()
print(f"Using MPS (Apple Silicon): {use_mps}")

# Set the appropriate device
if use_mps:
    device = torch.device("mps")
    print("Using MPS device for inference")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device for inference")
else:
    device = torch.device("cpu")
    print("Using CPU for inference")

# Load model with appropriate settings for MPS
print(f"Loading model {base_model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16 if use_mps else torch.float32,  # Use float16 for MPS
    device_map="auto",  # This will automatically use MPS if available
)

print(f"Model loaded successfully on {model.device}")

# ========================
# = INFERENCE FUNCTION =
# ========================

inference_system_prompt = """Below is an instruction that describes a task, paired with an input that provides further context.

Write a response that appropriately completes the request.

Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a scientific expert with advanced knowledge in analytical reasoning, problem-solving, and quantitative analysis across various disciplines, including mathematics, physics, biology, chemistry, and engineering. Please answer the following question.

### Question:
{}

### Response:
<think>
{}
"""

question = "Who is the current president of the United States? What did they mention in their latest speech?"

# Format the prompt
prompt = inference_system_prompt.format(question, "")
formatted_prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
)

# Prepare the input for model generation
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

print("Generating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
    )

# Decode the response
full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Remove the prompt part to get only the generated response
response = full_response.replace(formatted_prompt, "")

print("Response generated successfully!")
print(response)
