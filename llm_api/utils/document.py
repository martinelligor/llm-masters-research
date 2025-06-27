from typing import List
from pydantic import BaseModel


class Document(BaseModel):
    """Initializes the Document object.

    Args:
        id (str): The document id.
        text (str): The document content.
        metadata (dict, optional): The document metadata. Defaults to None.
        embedding (list, optional): The document embedding. Defaults to None.
    """
    id: str
    text: str | None
    metadata: dict | None
    embedding: List[float] | None
