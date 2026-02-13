from __future__ import annotations

import io
import os
from typing import Any

from fastapi import FastAPI, File, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

MODEL_NAME = "MobileNetV3-Small (ImageNet1K)"
ALLOWED_EXTENSIONS = (".jpg", ".jpeg")
ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

app = FastAPI(title="JPG Image Classifier API")

# Model state is kept globally so startup loads once per app process.
MODEL_STATE: dict[str, Any] = {
    "model": None,
    "preprocess": None,
    "categories": [],
    "loaded": False,
    "error": None,
}


def filename_is_jpg(filename: str | None) -> bool:
    if not filename:
        return False
    lowered = filename.lower()
    return lowered.endswith(ALLOWED_EXTENSIONS)


def detect_image_format(data: bytes) -> str | None:
    try:
        with Image.open(io.BytesIO(data)) as img:
            return (img.format or "").upper() or None
    except UnidentifiedImageError:
        return None
    except Exception:
        return None


def build_model() -> None:
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    model.eval()

    MODEL_STATE["model"] = model
    MODEL_STATE["preprocess"] = weights.transforms()
    MODEL_STATE["categories"] = list(weights.meta.get("categories", []))
    MODEL_STATE["loaded"] = True
    MODEL_STATE["error"] = None


def classify_jpeg_bytes(data: bytes) -> list[dict[str, Any]]:
    import torch

    if not MODEL_STATE["loaded"]:
        raise RuntimeError("Model not loaded")

    with Image.open(io.BytesIO(data)) as img:
        rgb = img.convert("RGB")

    image_tensor = MODEL_STATE["preprocess"](rgb).unsqueeze(0)
    with torch.inference_mode():
        logits = MODEL_STATE["model"](image_tensor)[0]
        probs = torch.nn.functional.softmax(logits, dim=0)
        top_probs, top_indices = torch.topk(probs, k=5)

    categories = MODEL_STATE["categories"]
    top5: list[dict[str, Any]] = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        label = categories[idx] if idx < len(categories) else f"class_{idx}"
        top5.append(
            {
                "label": label,
                "confidence": round(float(prob), 6),
            }
        )
    return top5


def not_jpg_response(
    *,
    filename: str | None,
    content_type: str | None,
    detected_format: str | None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        content={
            "ok": False,
            "error": "Only JPG/JPEG files are accepted",
            "filename": filename,
            "uploaded_content_type": content_type,
            "detected_format": detected_format,
        },
    )


@app.on_event("startup")
def startup_event() -> None:
    if os.getenv("SKIP_MODEL_LOAD") == "1":
        MODEL_STATE["loaded"] = False
        MODEL_STATE["error"] = "Model loading skipped by SKIP_MODEL_LOAD=1"
        return

    try:
        build_model()
    except Exception as exc:
        MODEL_STATE["loaded"] = False
        MODEL_STATE["error"] = str(exc)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>JPG Image Classifier</title>
  <style>
    :root { color-scheme: light; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f8fb;
      color: #17202a;
      min-height: 100vh;
      display: grid;
      place-items: center;
    }
    .card {
      width: min(720px, 90vw);
      background: #ffffff;
      border-radius: 14px;
      box-shadow: 0 10px 28px rgba(23, 32, 42, 0.08);
      padding: 24px;
    }
    h1 { margin-top: 0; }
    .small { color: #52606d; font-size: 14px; }
    form { display: flex; gap: 12px; margin-top: 14px; flex-wrap: wrap; }
    button {
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      background: #0f6fff;
      color: #fff;
      cursor: pointer;
      font-weight: 600;
    }
    pre {
      margin-top: 18px;
      background: #0b1220;
      color: #dbe7ff;
      border-radius: 10px;
      padding: 12px;
      overflow-x: auto;
    }
  </style>
</head>
<body>
  <main class="card">
    <h1>JPG Image Classifier</h1>
    <p class="small">Upload a JPG/JPEG image and get the top ImageNet predictions.</p>
    <form id="upload-form">
      <input id="file-input" type="file" accept=".jpg,.jpeg,image/jpeg" name="file" required />
      <button type="submit">Upload & Classify</button>
    </form>
    <pre id="result">Waiting for upload...</pre>
  </main>

  <script>
    const form = document.getElementById("upload-form");
    const fileInput = document.getElementById("file-input");
    const result = document.getElementById("result");

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const file = fileInput.files[0];
      if (!file) {
        result.textContent = JSON.stringify({ ok: false, error: "No file selected" }, null, 2);
        return;
      }
      const formData = new FormData();
      formData.append("file", file);
      result.textContent = "Uploading and running inference...";
      try {
        const response = await fetch("/predict", { method: "POST", body: formData });
        const data = await response.json();
        result.textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        result.textContent = JSON.stringify({ ok: false, error: String(err) }, null, 2);
      }
    });
  </script>
</body>
</html>
"""


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_loaded": MODEL_STATE["loaded"],
        "error": MODEL_STATE["error"],
        "max_upload_mb": MAX_UPLOAD_MB,
    }


@app.post("/predict")
async def predict(file: UploadFile | None = File(default=None)) -> JSONResponse:
    if file is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"ok": False, "error": "No file uploaded"},
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={
                "ok": False,
                "error": f"File too large. Max size is {MAX_UPLOAD_MB}MB",
            },
        )

    filename = file.filename or ""
    content_type = file.content_type

    extension_ok = filename_is_jpg(filename)
    mime_ok = content_type in ALLOWED_MIME_TYPES
    detected_format = detect_image_format(data)
    decode_failed = detected_format is None
    detected_is_jpeg = detected_format == "JPEG"

    if extension_ok and mime_ok and decode_failed:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "ok": False,
                "error": "Invalid image file",
                "filename": filename,
                "uploaded_content_type": content_type,
            },
        )

    if not extension_ok or not mime_ok or not detected_is_jpeg:
        return not_jpg_response(
            filename=filename,
            content_type=content_type,
            detected_format=detected_format,
        )

    if not MODEL_STATE["loaded"]:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "ok": False,
                "error": "Model not loaded",
                "details": MODEL_STATE["error"],
            },
        )

    try:
        top5 = classify_jpeg_bytes(data)
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"ok": False, "error": "Invalid image file"},
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "ok": True,
            "filename": filename,
            "content_type": content_type,
            "model": MODEL_NAME,
            "top1": top5[0],
            "top5": top5,
        },
    )
