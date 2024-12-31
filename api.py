from sort import *
import uvicorn, asyncio, fastapi
import cv2, numpy as np, pandas as pd
from Yolov8.yolov8 import YOLOv8_det  # Import YOLOv8 object detector
import time, pytz, os, random
from loguru import logger
from PIL import Image
from io import BytesIO
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError

# Set of class IDs for vehicles
vehicles = set([2, 3, 5, 7])  # Update these class IDs as needed

# Object tracking
tracker = Sort()

# Model path for object detection
model_path = './models/yolov8n.onnx'
my_yolo = YOLOv8_det(model_path)

# Set up the CORS middleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# TODO: exception handling for file upload
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: fastapi.Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({
            "returncode": {
                "code": 422,
                "message": "Request json has syntax error"
            },
            "output": {"result": [], "timing": 0.0}
        }),
    )

# Convert image to numpy array
def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

@app.post("/detections")
async def get_body(file: UploadFile = File(...)):
    st = time.time()
    bbyte = await file.read()
    frame = load_image_into_numpy_array(bbyte)

    # Phát hiện đối tượng
    detections = my_yolo.detect_objects(frame)
    boxes, scores, class_ids = detections

    # Lọc các đối tượng là vehicles
    detections_ = []
    for i in range(len(boxes)):
        box = boxes[i].tolist()
        score = scores[i].tolist()
        class_id = class_ids[i].tolist()
        if int(class_id) in vehicles:
            detections_.append(box + [score, class_id])

    # Theo dõi đối tượng
    results = []
    if detections_:
        track_ids = tracker.update(np.asarray(detections_))

        for i, track_id in enumerate(track_ids):
            print(dir(track_id))
            x3, y3, x4, y4, id = map(int, track_id)
            score = scores[i].tolist()  # Lấy điểm tin cậy tương ứng

            output = {
                "type": "ObjectDetectionPrediction",
                "predictions": {
                    "boxes": [x3, y3, x4, y4],
                    "tracking_ids": id,
                    "confidence": np.round(score, 2)
                }
            }
            results.append(output)
    et = time.time()
    timing = et - st
    logger.info(f"Processing time: {timing:.2f} seconds")
    return JSONResponse(content=jsonable_encoder(results))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=6000)
