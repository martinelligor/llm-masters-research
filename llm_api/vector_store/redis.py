import os
import json
import tiktoken

from uuid import uuid4
from redis import Redis
from typing import List
from fastapi import HTTPException
from redisvl.schema import IndexSchema
from redis.exceptions import ResponseError
from llm_api.utils.prompt import SYSTEM_PROMPT

from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document, Settings, VectorStoreIndex

from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache


class RedisVS:
    def __init__(self, vs_name: str, model: str = 'gpt-4o-mini', decode_responses: bool = False) -> None:

        if not vs_name:
            raise ValueError("Vector store name is required")

        # setting up vector store settings
        self.index_name = vs_name
        self.vector_dimensions = 1536
        self.model = model

        # building connection
        self.client = Redis(
            host=os.getenv("REDIS_HOST"),
            port=os.getenv("REDIS_PORT"),
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=decode_responses
        )

        # building vector store
        self.vector_store = self.__build_vector_store

    @property
    def __make_custom_schema(self) -> IndexSchema:
        custom_schema = IndexSchema.from_dict({
            "index": {"name": self.index_name, "prefix": self.index_name},
            # customize fields that are indexed
            "fields": [
                # required fields for llamaindex
                {"type": "tag", "name": "id"},
                {"type": "tag", "name": "doc_id"},
                {"type": "text", "name": "text"},
                # custom vector field for bge-small-en-v1.5 embeddings
                {
                    "type": "vector",
                    "name": "vector",
                    "attrs": {
                        "dims": self.vector_dimensions,
                        "algorithm": "hnsw",
                        "distance_metric": "cosine",
                    },
                },
            ],
        })

        return custom_schema

    @property
    def __build_vector_store(self) -> RedisVectorStore:
        vector_store = RedisVectorStore(
            redis_client=self.client,
            schema=self.__make_custom_schema
        )

        return vector_store

    @property
    def __build_docstore(self) -> RedisDocumentStore:
        docstore = RedisDocumentStore(
            redis_kvstore=RedisCache.from_redis_client(redis_client=self.client),
            namespace=f"{self.index_name}_document_store"
        )

        return docstore

    @property
    def __build_ingestion_cache(self) -> RedisCache:
        ingestion_cache = IngestionCache(
            cache=RedisCache.from_redis_client(redis_client=self.client),
            collection=f"{self.index_name}_cache"
        )

        return ingestion_cache

    def __build_chatstore(self, ttl: int) -> RedisChatStore:
        chatstore = RedisChatStore(
            redis_client=self.client,
            ttl=ttl
        )

        return chatstore

    def __get_chat_memory_buffer(
            self,
            thread_id: str | None,
            token_limit=10000,
            ttl: int = 3600 * 1 * 1):  # 2 hours

        chatstore = self.__build_chatstore(ttl)
        thread_id = str(uuid4()) if not thread_id else thread_id
        tokenizer = tiktoken.encoding_for_model(self.model).encode

        chat_memory_buffer = ChatMemoryBuffer.from_defaults(
            chat_store=chatstore,
            tokenizer_fn=tokenizer,
            token_limit=token_limit,
            chat_store_key=thread_id
        )

        return chat_memory_buffer

    @property
    def __get_vector_store(self) -> VectorStoreIndex:
        index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            embed_model=Settings.embed_model
        )

        return index

    def get_vector_store_nodes(self) -> list[str]:
        vs_documents = self.client.hgetall(f'{self.index_name}_document_store/doc')

        if isinstance(vs_documents, dict):
            nodes = []
            for document in vs_documents:
                json_document = json.loads(vs_documents[document].decode())
                nodes.append(json_document['__data__']['metadata'])

            return nodes
        return []

    def remove_database(self):
        try:
            redis_keys = self.client.keys()
            if isinstance(redis_keys, list):
                for key in redis_keys:
                    if key.decode().startswith(self.index_name):
                        self.client.delete(key)

            self.client.ft(self.index_name).dropindex()
        except ResponseError:
            print(f"index {self.index_name} does not exist")

    def chat(
            self,
            thread_id: str | None = None,
            chat_mode: ChatMode = ChatMode.CONTEXT) -> tuple[RedisVectorStore, str]:

        vector_store = self.__get_vector_store
        thread = self.__get_chat_memory_buffer(thread_id=thread_id)

        chat_engine = vector_store.as_chat_engine(
            chat_mode=chat_mode,
            memory=thread,
            system_prompt=SYSTEM_PROMPT
        )

        return chat_engine, thread.chat_store_key

    def run_pipeline_with_debug(self, pipeline: IngestionPipeline, documents: list[Document]):
        vs_nodes = self.get_vector_store_nodes()

        vs_files: List[str] = []
        for node in vs_nodes:
            if isinstance(node, dict):
                vs_files.append(node.get("file name", ""))

        for i, document in enumerate(documents):
            document.text = document.text.encode('utf-8').decode('utf-8').replace('\"', '\'')
            filename = documents[0].metadata['file name']
            if filename in vs_files:
                print(f"{filename} already exists in knowledge base {self.index_name}")

            try:
                nodes = pipeline.run(documents=[document])
                return nodes

            except Exception as e:
                print(f"Failed to ingest {filename} into knowledge base {self.index_name}\n\n {e}")
                continue

    def insert_documents(self, documents: list[Document], debug: bool = False):
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512),
                Settings.embed_model
            ],
            docstore=self.__build_docstore,
            vector_store=self.vector_store,
            cache=self.__build_ingestion_cache,
            docstore_strategy=DocstoreStrategy.UPSERTS,
        )

        try:
            if debug:
                nodes = self.run_pipeline_with_debug(pipeline=pipeline, documents=documents)
            else:
                for document in documents:
                    document.text = document.text.encode('utf-8').decode('utf-8').replace('\"', '\'')

                nodes = pipeline.run(documents=documents)

            return nodes

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM RAG API: Error inserting KB ({str(e)})")
