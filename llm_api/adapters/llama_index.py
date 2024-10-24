import os

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


def setup_llama_index(model: str = 'gpt-4o-mini', embedd_model: str = 'text-embedding-3-small') -> None:
    # defining a LLM for Llama-index, here we'll use OpenAI
    llm = OpenAI(
        temperature=0.9,
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # setting the default llm as the llm defined above
    Settings.llm = llm

    # setting a default embedding model.
    Settings.embed_model = OpenAIEmbedding(
        model=embedd_model,
        api_key=os.getenv("OPENAI_API_KEY")
    )
