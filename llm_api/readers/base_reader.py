from abc import ABC, abstractmethod
from llama_index.core.schema import Document


class BaseReader(ABC):
    @abstractmethod
    def parse(self) -> list[Document]:
        """Parse the input data and return a Document"""
        raise NotImplementedError
