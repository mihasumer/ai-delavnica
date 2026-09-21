import io
import logging
import threading
import time

from PIL import Image, UnidentifiedImageError

from app.schemas import Box, Detection, DetectResponse, ImageInfo

logger = logging.getLogger("app.detector")


class DetectionError(Exception):
    """Raised when an uploaded image cannot be decoded or detected."""


class Detector:
    """Loads a single Ultralytics YOLO model instance and serializes CPU inference."""

    def __init__(self, model_path: str, imgsz: int, conf: float, max_detections: int):
        from ultralytics import YOLO

        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.max_detections = max_detections
        self.device = "cpu"
        self._lock = threading.Lock()

        logger.info("Loading Ultralytics model %s ...", model_path)
        self.model = YOLO(model_path)
        self.model.to(self.device)

        backend = "openvino" if "openvino" in model_path.lower() else "pytorch"
        self.backend = backend

        logger.info("Inference device: CPU")
        logger.info("Model: %s", model_path)
        logger.info("Backend: %s CPU", backend)

    def detect(self, image_bytes: bytes) -> DetectResponse:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
            image = image.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise DetectionError(f"invalid image data: {exc}") from exc

        width, height = image.size

        with self._lock:
            start = time.perf_counter()
            results = self.model.predict(
                source=image,
                imgsz=self.imgsz,
                conf=self.conf,
                device=self.device,
                max_det=self.max_detections,
                verbose=False,
            )
            inference_ms = (time.perf_counter() - start) * 1000.0

        detections: list[Detection] = []
        if results:
            result = results[0]
            names = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                x1 = max(0.0, min(x1, width))
                y1 = max(0.0, min(y1, height))
                x2 = max(0.0, min(x2, width))
                y2 = max(0.0, min(y2, height))
                if x2 <= x1 or y2 <= y1:
                    continue
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=str(names[class_id]),
                        confidence=confidence,
                        box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )

        return DetectResponse(
            image=ImageInfo(width=width, height=height),
            model=self.model_path,
            device=self.device,
            inference_ms=round(inference_ms, 2),
            detections=detections,
        )
