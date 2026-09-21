from fastapi.testclient import TestClient

from app.main import app


def test_info_reports_cpu_device():
    with TestClient(app) as client:
        response = client.get("/api/info")
    assert response.status_code == 200
    body = response.json()
    assert body["device"] == "cpu"
    assert body["model"]


def test_detect_rejects_invalid_image():
    with TestClient(app) as client:
        response = client.post(
            "/api/detect",
            files={"image": ("not_an_image.jpg", b"this is not image data", "image/jpeg")},
        )
    assert response.status_code == 422
    assert "error" in response.json()


def test_detect_returns_valid_contract(sample_jpeg_bytes):
    with TestClient(app) as client:
        response = client.post(
            "/api/detect",
            files={"image": ("sample.jpg", sample_jpeg_bytes, "image/jpeg")},
        )
    assert response.status_code == 200
    body = response.json()

    assert body["device"] == "cpu"
    assert "detections" in body
    assert isinstance(body["detections"], list)
    assert body["inference_ms"] >= 0

    for det in body["detections"]:
        box = det["box"]
        assert 0 <= box["x1"] < box["x2"] <= body["image"]["width"]
        assert 0 <= box["y1"] < box["y2"] <= body["image"]["height"]
        assert 0.0 <= det["confidence"] <= 1.0
