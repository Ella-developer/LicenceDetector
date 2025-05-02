# app/utils.py

import os
from datetime import datetime
from fastapi import UploadFile

def generate_timestamped_filename(filename: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{filename}"

async def save_uploaded_file(upload_file: UploadFile, destination: str):
    with open(destination, "wb") as buffer:
        content = await upload_file.read()
        buffer.write(content)
