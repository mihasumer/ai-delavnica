const CAPTURE_INTERVAL_MS = 500;
const JPEG_QUALITY = 0.7;

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const fileInput = document.getElementById("fileInput");
const statusEl = document.getElementById("status");
const timingEl = document.getElementById("timing");

let stream = null;
let running = false;
let requestInFlight = false;
let loopTimer = null;
const captureCanvas = document.createElement("canvas");

function setStatus(text) {
  statusEl.textContent = text;
}

function drawDetections(detections, imageWidth, imageHeight) {
  overlay.width = overlay.clientWidth;
  overlay.height = overlay.clientHeight;
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const scaleX = overlay.width / imageWidth;
  const scaleY = overlay.height / imageHeight;

  ctx.lineWidth = 2;
  ctx.strokeStyle = "#00ff88";
  ctx.font = "14px monospace";
  ctx.fillStyle = "#00ff88";

  for (const det of detections) {
    const { x1, y1, x2, y2 } = det.box;
    const rx = x1 * scaleX;
    const ry = y1 * scaleY;
    const rw = (x2 - x1) * scaleX;
    const rh = (y2 - y1) * scaleY;
    ctx.strokeRect(rx, ry, rw, rh);
    const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
    ctx.fillText(label, rx + 2, Math.max(12, ry - 4));
  }
}

async function sendFrame(blob) {
  if (requestInFlight) return;
  requestInFlight = true;
  const t0 = performance.now();
  try {
    const form = new FormData();
    form.append("image", blob, "frame.jpg");
    const response = await fetch("/api/detect", { method: "POST", body: form });
    const requestMs = performance.now() - t0;

    if (!response.ok) {
      const err = await response.json().catch(() => ({ error: "unknown error" }));
      setStatus(`Error: ${err.error || response.status}`);
      return;
    }

    const data = await response.json();
    drawDetections(data.detections, data.image.width, data.image.height);
    setStatus(`Detections: ${data.detections.length}`);
    timingEl.textContent = `inference: ${data.inference_ms.toFixed(1)} ms | request: ${requestMs.toFixed(1)} ms`;
  } catch (e) {
    setStatus(`Request failed: ${e.message}`);
  } finally {
    requestInFlight = false;
  }
}

function captureAndSend() {
  if (!running) return;
  if (video.videoWidth === 0) {
    loopTimer = setTimeout(captureAndSend, CAPTURE_INTERVAL_MS);
    return;
  }
  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  const cctx = captureCanvas.getContext("2d");
  cctx.drawImage(video, 0, 0);
  captureCanvas.toBlob(
    async (blob) => {
      if (blob) await sendFrame(blob);
      if (running) loopTimer = setTimeout(captureAndSend, CAPTURE_INTERVAL_MS);
    },
    "image/jpeg",
    JPEG_QUALITY
  );
}

async function start() {
  if (running) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
  } catch (e) {
    setStatus(`Camera permission denied or unavailable: ${e.message}`);
    return;
  }
  video.srcObject = stream;
  running = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  setStatus("Running.");
  captureAndSend();
}

function stop() {
  running = false;
  if (loopTimer) clearTimeout(loopTimer);
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("Stopped.");
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  const img = new Image();
  img.onload = async () => {
    overlay.width = img.width;
    overlay.height = img.height;
    await sendFrame(file);
  };
  img.src = URL.createObjectURL(file);
});

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
