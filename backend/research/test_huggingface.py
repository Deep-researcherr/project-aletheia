import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
    provider="auto",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-4B-Thinking-2507",
    messages=[
        {
            "role": "user",
            "content": "Explain what a binary search tree is in 2 sentences.",
        }
    ],
)

print(response.choices[0].message.content)