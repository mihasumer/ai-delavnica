from pydantic import BaseModel


class ImageInfo(BaseModel):
    width: int
    height: int


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    box: Box


class DetectResponse(BaseModel):
    image: ImageInfo
    model: str
    device: str
    inference_ms: float
    detections: list[Detection]


class InfoResponse(BaseModel):
    model: str
    backend: str
    device: str
    imgsz: int


class HealthResponse(BaseModel):
    status: str
