from logger import log_to_file
from openai import OpenAI

client = OpenAI()

prompt = "Explain quantum computing"
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

answer = resp.choices[0].message.content

log_to_file(
    prompt,
    answer,
    metadata={
        "model": "gpt-4o-mini",
        "tokens_used": resp.usage.total_tokens
    }
)