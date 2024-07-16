import logging

LOGGER = logging.getLogger(__name__)


def _download_file(self, fileid: str, filename: str) -> str:
    """Download the file with fileid and filename
    Args:
        fileid: file id of the file in google drive
        filename: filename with which it will be downloaded
    Returns:
        The downloaded filename, which which may have a new extension.
    """
    from io import BytesIO

    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    try:
        # Get file details
        service = build("drive", "v3", credentials=self._creds)
        file = service.files().get(fileId=fileid, supportsAllDrives=True).execute()

        if file["mimeType"] in self._mimetypes:
            download_mimetype = self._mimetypes[file["mimeType"]]["mimetype"]
            download_extension = self._mimetypes[file["mimeType"]]["extension"]
            new_file_name = filename + download_extension

            # Download and convert file
            request = service.files().export_media(
                fileId=fileid, mimeType=download_mimetype
            )
        else:
            try:
                extension = file.get('name').split(".")[-1].lower()
                new_file_name = filename + "." + extension
            except Exception as e:
                LOGGER.error(e)
                new_file_name = filename

            # Download file without conversion
            request = service.files().get_media(fileId=fileid)

        # Download file data
        file_data = BytesIO()
        downloader = MediaIoBaseDownload(file_data, request)
        done = False

        while not done:
            status, done = downloader.next_chunk()

        # Save the downloaded file
        with open(new_file_name, "wb") as f:
            f.write(file_data.getvalue())

        return new_file_name

    except Exception as e:
        LOGGER.error(
            f"An error occurred while downloading file: {e}", exc_info=True
        )

        return ""
