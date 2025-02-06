from llama_index.core import Settings
from llama_index.llms.litellm import LiteLLM
from llama_index.embeddings.litellm import LiteLLMEmbedding


def setup_llama_index(model: str = 'gpt-4o-mini', embedd_model: str = 'text-embedding-3-small') -> None:
    # defining a LLM for Llama-index, here we'll use OpenAI
    print(f'\n\n\n {model} \n\n\n')

    llm = LiteLLM(
        temperature=0.9,
        model=model,
    )

    # setting the default llm as the llm defined above
    Settings.llm = llm

    # setting a default embedding model.
    Settings.embed_model = LiteLLMEmbedding(
        model_name=embedd_model,
    )
