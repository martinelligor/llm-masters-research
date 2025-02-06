import logging

from http import HTTPStatus
from typing import Optional
from typing import List, Union
from pydantic import BaseModel, Field

from fastapi import FastAPI, Request, HTTPException

from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from llm_api.vector_store.redis_vs import RedisVS
from llm_api.readers.reader import GlobalReader
from llm_api.utils.healthcheck import healthcheck
from llm_api.adapters.llama_index import setup_llama_index


app = FastAPI()
logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)


@app.post("/headers")
async def read_headers(request: Request):
    return request.headers


@app.get("/health")
async def healthcheck_handler():
    return healthcheck()


class GetAnswerInput(BaseModel):
    question: str
    top_k: Optional[int] = 5
    model: Optional[str] = None
    kb_id: Union[str, List[str]]
    thread_id: Optional[str] = None
    get_references: Optional[bool] = False
    filters: Optional[List[MetadataFilter]] = None


def process_metadata_filters(filters):
    metadata_filters = []
    for f in filters:
        metadata_filters.append(
            MetadataFilter(
                key=f.key,
                value=f.value,
                operator=f.operator,
            )
        )
    return MetadataFilters(filters=metadata_filters, condition="and")


@app.post("/get_answer")
def get_answer(answer_input: GetAnswerInput, request: Request):
    top_k = answer_input.top_k if answer_input.top_k else 7
    model = answer_input.model if answer_input.model else "gpt-4o-mini"
    kb_ids = answer_input.kb_id if isinstance(answer_input.kb_id, list) else [answer_input.kb_id]

    # setting up llm
    setup_llama_index(model=model)

    # getting filters
    filters = process_metadata_filters(answer_input.filters) if answer_input.filters else None

    # loading vector_stores
    redis_vs = RedisVS(vs_name=kb_ids[0])
    chat_engine, thread_id = redis_vs.chat(
        filters=filters,
        similarity_top_k=top_k,
        thread_id=answer_input.thread_id
    )

    try:
        answer = chat_engine.chat(answer_input.question)
    except Exception as e:
        logger.error("Error in getting answer: {}".format(str(e)))
        raise HTTPException(status_code=500, detail=f"LLM RAG API: Error in getting answer ({str(e)})")

    if answer_input.get_references:
        references = [
            {
                "filename": node.metadata.get("file_name", "-"),
                "score": node.score,
                "node_start": node.metadata.get("node_start", "-"),
                "node_end": node.metadata.get("node_end", "-"),
                "ref_lines": node.metadata.get("ref_lines", "-"),
            }
            for node in answer.source_nodes
        ]
        return {
            "answer": answer.response,
            "thread_id": thread_id,
            "references": references,
        }
    else:
        return {"answer": answer.response, "thread_id": thread_id}


class InsertKBInput(BaseModel):
    kb_id: str


@app.post("/insert_kb")
def insert_kb(insert_kb_input: InsertKBInput, request: Request):
    setup_llama_index(model='gpt-4o-mini')

    # Make sure credentials.json file exists in the current directory (data_connectors)
    reader = GlobalReader()
    # getting documents
    llama_index_documents = reader.parse()

    redis_vs = RedisVS(vs_name=insert_kb_input.kb_id)
    redis_vs.insert_documents(documents=llama_index_documents)

    return {"status": "The documents have been successfully inserted into the vector store."}


class RemoveKBInput(BaseModel):
    kb_id: str


@app.post("/remove_kb")
def remove_kb(remove_kb_input: RemoveKBInput, request: Request):
    setup_llama_index(model='gpt-4o-mini')

    redis_vs = RedisVS(vs_name=remove_kb_input.kb_id)
    redis_vs.remove_database()

    return {"status": "The documents have been successfully removed from vector store."}


class NodeMetadataConfig():
    ref_name: str = Field(description="Reference name of the node")
    knowledge_base_id: str = Field(description="Knowledge base ID of the node")
    filename: Optional[str] = Field("", description="Filename of the node")
    extension: Optional[str] = Field("", description="Filename`s extension")


class InsertNodesInput(BaseModel):
    kb_id: str


def _update_node_text_references(nodes: List[TextNode]) -> List[TextNode]:
    ref_line_per_doc = {}
    for node in nodes:
        current_doc = node.metadata.get("ref_name")
        if current_doc not in ref_line_per_doc:
            ref_line_per_doc[current_doc] = 1

        count = node.text.count("\n")
        words = node.text.split()

        text_len = None
        if node.end_char_idx and node.start_char_idx:
            text_len = node.end_char_idx - node.start_char_idx + 1

        node.metadata.update(
            {
                "ref_lines": f"{ref_line_per_doc[current_doc]}-{ref_line_per_doc[current_doc]+count}",
                "text_len": f"{text_len}",
                "node_start": f"[...] {' '.join(words[:10])} [...]",
                "node_end": f"[...] {' '.join(words[-10:])} [...]",
            }
        )

        ref_line_per_doc[current_doc] += count + 1 * (node.text[-1] == "\n")

    return nodes


def assign_pii_metadata(nodes: List[TextNode]):
    pass


@app.post("/insert_nodes")
def insert_nodes(insert_nodes_input: InsertNodesInput, request: Request):
    headers = request.headers
    kb_id = insert_nodes_input.kb_id
    embedd_model = 'text-embedding-3-small'
    setup_llama_index(embedd_model=embedd_model)

    # Semantic splitter definition
    semantic_splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=Settings.embed_model
    )

    # Make sure credentials.json file exists in the current directory (data_connectors)
    reader = GlobalReader()
    # getting documents
    llama_index_documents = reader.parse()
    # getting documents nodes
    nodes = semantic_splitter.get_nodes_from_documents(llama_index_documents)
    nodes = _update_node_text_references(nodes)

    assign_pii_metadata(nodes)

    text_nodes = []
    try:
        for node in nodes:
            text_nodes.append(TextNode().from_dict(node))
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Error in inserting nodes",
        )

    redis_vs = RedisVS(vs_name=kb_id)
    redis_vs.insert_nodes(text_nodes)

    return HTTPStatus.OK
