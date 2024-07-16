import os
from llm_api.readers.base_reader import BaseReader
from llm_api.adapters.google_drive import _download_file as my_download_file

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
from llama_index.readers.json import JSONReader
from llm_api.readers.google import GoogleDriveReader

GoogleDriveReader._download_file = my_download_file


class GlobalReader(BaseReader):
    """Global reader"""
    def __init__(self, folder_id) -> None:
        super().__init__()

        self.folder_id = folder_id
        self.parsers = self.__get_parsers()

    @property
    def __get_service_account_key(self) -> dict:
        return {
            "type": "service_account",
            "project_id": "llm-research-429518",
            "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
            "private_key": os.getenv("GOOGLE_PRIVATE_KEY"),
            "client_email": "llm-research@llm-research-429518.iam.gserviceaccount.com",
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_CERT_URL"),
            "universe_domain": "googleapis.com"
        }

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
            "json": JSONReader()
        }

    def parse(self) -> list[Document]:
        """Parse JSON document into Python dict"""

        llama_index_documents = GoogleDriveReader(
            service_account_key=self.__get_service_account_key,
            file_extractor=self.parsers).load_data(folder_id=self.folder_id)

        return llama_index_documents
