# 🪖 Helmet Violation Detection API 🎥🚔

This project is a REST API built with FastAPI that analyzes motorcycle videos using a trained YOLOv8 model and PaddleOCR to detect riders not wearing helmets and extract their license plate numbers.

---

## 🚀 Features

- 🎥 Accepts video file uploads via POST request
- 🧠 Detects helmet violations using YOLOv8
- 🔍 Extracts license plates using PaddleOCR (supports Persian plates)
- 💾 Saves uploaded videos for future reference
- 📦 Returns results as a structured JSON list

---

## 📁 Project Structure

```text
helmet-violation-api/
├── app/
│   ├── main.py             # FastAPI app entrypoint
│   ├── detector.py         # YOLODetector class to process video frames
│   ├── config.py           # Config paths for model, videos, and outputs
│   └── utils.py            # Helper functions (e.g. file saving)
├── models/
│   └── best.pt             # Trained YOLOv8 model
├── videos/                 # Uploaded videos storage
├── results/
│   ├── annotated_videos/   # Future annotated videos
│   └── output_json/        # Future saved detection results
├── Pipfile
├── Pipfile.lock
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### 1. Create Virtual Environment with Pipenv

```bash
pipenv install
pipenv shell
```

> You can also install from `requirements.txt` if you're not using pipenv.

---

### 2. Run the FastAPI Server

```bash
uvicorn app.main:app --reload
```

Then open in your browser:

```
http://127.0.0.1:8000/docs
```

---

## 📤 API Endpoint

### `POST /detect/`

- Accepts `.mp4` video file
- Returns:
  - `video_saved_as`: saved filename
  - `plate_numbers`: list of detected plate numbers

#### Example using `curl`

```bash
curl -X 'POST' http://127.0.0.1:8000/detect/ \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'video=@1.mp4;type=video/mp4'
```

---

## 📦 Dependencies

Main packages include:

- `fastapi==0.95.2`
- `uvicorn==0.22.0`
- `torch==2.0.1+cpu`
- `paddleocr`
- `paddlepaddle` (CPU version)
- `opencv-python-headless`
- `python-multipart`

To install manually:

```bash
pip install -r requirements.txt
```

---

## 📌 Notes

- Place trained YOLO model at: `models/best.pt`
- Required class IDs:
  - `0`: license plate
  - `2`: rider without helmet
- Currently optimized for CPU-based execution

---

## 🌱 Future Work

- Save JSON results under `/results/output_json/`
- Save annotated videos in `/results/annotated_videos/`
- Add authentication and dashboard (e.g. Streamlit or React)

---

## 👤 Author

🎓 Ella K. — Focused on practical applications of machine learning and computer vision.
