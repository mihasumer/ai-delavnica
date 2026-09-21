import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.detector import Detector, DetectionError
from app.schemas import HealthResponse, InfoResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.main")

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolo26n.pt")
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.25"))
YOLO_MAX_DETECTIONS = int(os.environ.get("YOLO_MAX_DETECTIONS", "100"))

detector: Detector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    detector = Detector(
        model_path=YOLO_MODEL,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        max_detections=YOLO_MAX_DETECTIONS,
    )
    logger.info("Confidence threshold: %s", YOLO_CONF)
    logger.info("Inference image size: %s", YOLO_IMGSZ)
    yield


app = FastAPI(title="CPU-Only YOLO Web Detection Demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/api/info", response_model=InfoResponse)
def info() -> InfoResponse:
    if detector is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return InfoResponse(
        model=detector.model_path,
        backend=detector.backend,
        device=detector.device,
        imgsz=detector.imgsz,
    )


@app.post("/api/detect")
async def detect(image: UploadFile = File(...)):
    if detector is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    body = await image.read()
    if len(body) > MAX_UPLOAD_BYTES:
        return JSONResponse(status_code=422, content={"error": "image exceeds maximum upload size"})
    if not body:
        return JSONResponse(status_code=422, content={"error": "empty image upload"})

    try:
        result = detector.detect(body)
    except DetectionError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})

    return result
