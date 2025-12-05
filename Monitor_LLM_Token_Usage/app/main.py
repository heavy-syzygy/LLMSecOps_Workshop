# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse

from prometheus_client import Counter, generate_latest

from dotenv import load_dotenv
from openai import OpenAI
import os

# Load .env and get API key
load_dotenv()
##openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI app
app = FastAPI()

# Prometheus metrics
token_counter_total = Counter("llm_tokens_total", "Total tokens used", ["model"])
token_counter_prompt = Counter("llm_tokens_prompt", "Prompt tokens used", ["model"])
token_counter_completion = Counter("llm_tokens_completion", "Completion tokens used", ["model"])


class PromptRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o-mini"  # modern default model

##@app.post("/chat")
##async def chat(request: PromptRequest):
##    response = openai.ChatCompletion.create(
##        model=request.model,
##        messages=[{"role": "user", "content": request.prompt}],
##    )
##
##    usage = response["usage"]
##    token_counter_prompt.labels(model=request.model).inc(usage["prompt_tokens"])
##    token_counter_completion.labels(model=request.model).inc(usage["completion_tokens"])
##    token_counter_total.labels(model=request.model).inc(usage["total_tokens"])
##
##    return {
##        "response": response["choices"][0]["message"]["content"],
##        "usage": usage
##    }
    
@app.post("/chat")
async def chat(request: PromptRequest):
    # Call the OpenAI chat completion API (new syntax)
    completion = client.chat.completions.create(
        model=request.model,
        messages=[
            {"role": "user", "content": request.prompt}
        ],
    )

    usage = completion.usage  # usage object

    # Update Prometheus counters
    token_counter_prompt.labels(model=request.model).inc(usage.prompt_tokens)
    token_counter_completion.labels(model=request.model).inc(usage.completion_tokens)
    token_counter_total.labels(model=request.model).inc(usage.total_tokens)

    # Return response + usage structure
    return {
        "response": completion.choices[0].message.content,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()
