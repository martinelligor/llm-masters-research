import logging

from typing import Optional
from typing import List, Union
from pydantic import BaseModel

from fastapi import FastAPI, Request, HTTPException

from llm_api.vector_store.redis import RedisVS
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
    model: Optional[str] = None
    kb_id: Union[str, List[str]]
    thread_id: Optional[str] = None
    get_vs_used_files: Optional[bool] = False


@app.post("/get_answer")
def get_answer(answer_input: GetAnswerInput, request: Request):
    model = 'gpt-4o-mini'
    headers = request.headers

    kb_ids = answer_input.kb_id if isinstance(answer_input.kb_id, list) else [answer_input.kb_id]

    # setting up llm
    setup_llama_index(model=model)

    # loading vector_stores
    redis_vs = RedisVS(vs_name=kb_ids[0], model=model)
    chat_engine, thread_id = redis_vs.chat(answer_input.thread_id)

    try:
        answer = chat_engine.chat(answer_input.question)
        vs_used_files = [node.metadata.get("file_name") for node in answer.source_nodes]
    except Exception as e:
        logger.error("Error in getting answer: {}".format(str(e)))
        raise HTTPException(status_code=500, detail=f"LLM RAG API: Error in getting answer ({str(e)})")

    if answer_input.get_vs_used_files:
        return {"answer": answer.response, "thread_id": thread_id, "vs_used_files": vs_used_files}
    else:
        return {"answer": answer.response, "thread_id": thread_id}


class InsertKBInput(BaseModel):
    kb_id: str


@app.post("/insert_kb")
def insert_kb(insert_kb_input: InsertKBInput, request: Request):
    setup_llama_index()

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
    setup_llama_index()

    redis_vs = RedisVS(vs_name=remove_kb_input.kb_id)
    redis_vs.remove_database()

    return {"status": "The documents have been successfully removed from vector store."}
