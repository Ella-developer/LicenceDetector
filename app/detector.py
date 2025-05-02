# app/detector.py

import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def preprocess(self, image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (640, 640))
        return image / 255.0

    def predict_frame(self, frame):
        original_image = frame.copy()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preprocessed_image = self.preprocess(image)
        image = Image.fromarray((preprocessed_image * 255).astype(np.uint8))
        results = self.model(image)

        if not results or len(results[0].boxes) == 0:
            return original_image, []

        detections = results[0].boxes.xyxy
        class_ids = results[0].boxes.cls
        confidences = results[0].boxes.conf

        plate_numbers = []
        without_helmet_detected = any(cls == 2 for cls in class_ids)  # Assuming 2 = NoHelmet

        for box, class_id, confidence in zip(detections, class_ids, confidences):
            plate_number = None
            if class_id == 0 and without_helmet_detected:
                scale_x = original_image.shape[1] / 640
                scale_y = original_image.shape[0] / 640
                x1, y1, x2, y2 = (box.cpu().numpy() * [scale_x, scale_y, scale_x, scale_y]).round().astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(original_image.shape[1], x2), min(original_image.shape[0], y2)
                plate_region = original_image[y1:y2, x1:x2]
                if plate_region.size == 0:
                    continue

                try:
                    ocr_results = self.ocr.ocr(plate_region, cls=True)
                    text = ''.join([line[1][0] for line in ocr_results[0]])
                    plate_number = text.strip()
                    if plate_number:
                        plate_numbers.append(plate_number)
                        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(original_image, plate_number, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    print(f"OCR error: {e}")
        return original_image, plate_numbers

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        all_plate_numbers = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame, plates = self.predict_frame(frame)
            if plates:
                all_plate_numbers.update(plates)
        cap.release()
        return list(all_plate_numbers)
