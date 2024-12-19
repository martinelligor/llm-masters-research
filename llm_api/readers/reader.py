import os

from llm_api.readers.base_reader import BaseReader
from llama_index.readers.file import (
    DocxReader,
    HWPReader,
    PDFReader,
    EpubReader,
    FlatReader,
    HTMLTagReader,
    IPYNBReader,
    MarkdownReader,
    PandasCSVReader,
    UnstructuredReader,
    XMLReader,
    PagedCSVReader,
    CSVReader,
)
from llama_index.core.schema import Document
from llama_index.core.readers import SimpleDirectoryReader


class GlobalReader(BaseReader):
    """Global reader"""
    def __init__(self) -> None:
        super().__init__()

        self.parsers = self.__get_parsers()

    def __get_parsers(self):
        return {
            "docx": DocxReader(),
            "hwp": HWPReader(),
            "pdf": PDFReader(),
            "epub": EpubReader(),
            "flat": FlatReader(),
            "html": HTMLTagReader(),
            "ipynb": IPYNBReader(),
            "markdown": MarkdownReader(),
            "pandas_csv": PandasCSVReader(),
            "unstructured": UnstructuredReader(),
            "xml": XMLReader(),
            "paged_csv": PagedCSVReader(),
            "csv": CSVReader(),
        }

    def parse(self) -> list[Document]:
        """Parse JSON document into Python dict"""

        cwd = os.getcwd()
        loader = SimpleDirectoryReader(f'{cwd}/data/', file_extractor=self.parsers)
        llama_index_documents = loader.load_data()

        for doc in llama_index_documents:
            doc.id_ = doc.metadata.get("file id", doc.id_)

        return llama_index_documents
