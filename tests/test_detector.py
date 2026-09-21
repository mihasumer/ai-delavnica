import pytest

from app.detector import Detector, DetectionError


@pytest.fixture(scope="module")
def detector():
    return Detector(model_path="yolo26n.pt", imgsz=640, conf=0.25, max_detections=100)


def test_detector_reports_cpu_device(detector):
    assert detector.device == "cpu"


def test_detector_raises_on_invalid_bytes(detector):
    with pytest.raises(DetectionError):
        detector.detect(b"not an image")


def test_detector_returns_detect_response(detector, sample_jpeg_bytes):
    result = detector.detect(sample_jpeg_bytes)
    assert result.device == "cpu"
    assert result.image.width == 320
    assert result.image.height == 240
    assert result.inference_ms >= 0
