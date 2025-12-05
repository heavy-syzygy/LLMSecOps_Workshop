##from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

##llm = Ollama(model="llama2")

llm = OllamaLLM(model="llama2")
response = llm.invoke("tell me about partial functions in python")
print(response)
