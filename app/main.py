# app/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
from app.detector import YOLODetector
from app.config import MODEL_PATH, VIDEO_DIR
from app.utils import generate_timestamped_filename, save_uploaded_file
import traceback


app = FastAPI()

os.makedirs(VIDEO_DIR, exist_ok=True)
detector = YOLODetector(MODEL_PATH)

@app.post("/detect/")
async def detect_helmet_violations(video: UploadFile = File(...)):
    try:
        filename = generate_timestamped_filename(video.filename)
        video_path = os.path.join(VIDEO_DIR, filename)
        await save_uploaded_file(video, video_path)
        plate_numbers = detector.process_video(video_path)
        return JSONResponse(content={
            "video_saved_as": filename,
            "plate_numbers": plate_numbers
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
