# Detecting Toxicity, Bias \&

# Hallucination

NUS-ISS IDAS Workshop





LlamaCpp does not work for me as it requires installation of VS 2022 C++ Build tools and langchain-openai has some installation issue so I replaced them with just using OpenAI directly with DeepEval



Removed:

from langchain\_community.llms import LlamaCpp

from langchain\_core.callbacks import Callbacks

from langchain\_core.caches import BaseCache

LlamaCpp.model\_rebuild()

The LlamaCpp(...) instantiation with the GGUF model.



Added:

from langchain\_openai import ChatOpenAI

llm = ChatOpenAI(...) and response = llm.invoke(prompt); actual\_output = response.content

