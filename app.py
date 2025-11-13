import base64
import io
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from ultralytics import YOLO

# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "plate_detector_v1" / "weights"
DEFAULT_MODEL_PATH = Path(
    os.getenv("MODEL_WEIGHTS_PATH", WEIGHTS_DIR / "best.pt")
)
SUMMARY_PATH = BASE_DIR / "plate_detector_v1_summary.json"

GITHUB_USER = "sidnei-almeida"
GITHUB_REPO = "brazilian-license-plate-recognition"
GITHUB_BRANCH = "main"
GITHUB_IMAGES_BASE = (
    f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/images/"
)
EXAMPLE_IMAGES = [
    "DCAM0015_JPG_jpg.rf.72c86340f8f15c0a24c50bde98fa8f57.jpg",
    "DCAM0019_JPG_jpg.rf.4fe1c21ca9db3bf51ecb2eca2dfa2924.jpg",
    "DCAM0019_JPG_jpg.rf.9b2a03f1db093f23eebaab9ae0c24d0c.jpg",
    "DCAM0019_jpg.rf.b83d52425fc18b9861a453d0555be5dc.jpg",
    "DCAM0026_JPG_jpg.rf.f04431ad830e8af87618e14df2ede13a.jpg",
    "DCAM0027_JPG_jpg.rf.75c8a42daa4ee11e52e33f9f81524440.jpg",
    "DCAM0037_JPG_jpg.rf.da0ac338a913572b8246466136be098d.jpg",
    "DCAM0040_JPG_jpg.rf.f0319334d8ed56b1102db20b11f6f138.jpg",
    "DCAM0046_JPG_jpg.rf.650333eab92ea5ae034cc4d8ea43273b.jpg",
    "DCAM0046_JPG_jpg.rf.9a074131c18947bc622fee6b31df3602.jpg",
]


# ---------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------
class BoundingBox(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    width: int
    height: int
    xmin_norm: float = Field(..., ge=0.0, le=1.0)
    ymin_norm: float = Field(..., ge=0.0, le=1.0)
    xmax_norm: float = Field(..., ge=0.0, le=1.0)
    ymax_norm: float = Field(..., ge=0.0, le=1.0)


class Detection(BaseModel):
    id: int
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    box: BoundingBox


class ImageInfo(BaseModel):
    width: int
    height: int
    mode: str


class PerformanceInfo(BaseModel):
    inference_time_ms: float
    model_name: str
    framework_version: str


class DetectionResponse(BaseModel):
    detections: List[Detection]
    image: ImageInfo
    performance: PerformanceInfo
    annotated_image_base64: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_path: Optional[str]
    weights_available: bool
    detections_ready: bool


class ModelMetrics(BaseModel):
    model_name: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    training_settings: Dict[str, str] = Field(default_factory=dict)


class SamplesResponse(BaseModel):
    images: List[str]


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------
def _prepare_model_path() -> Path:
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH
    available = sorted(WEIGHTS_DIR.glob("*.pt"))
    if available:
        return available[0]
    raise FileNotFoundError(
        f"Model weights not found. Expected at {DEFAULT_MODEL_PATH} or any *.pt file under {WEIGHTS_DIR}"
    )


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    model_path = _prepare_model_path()
    model = YOLO(str(model_path))
    model.to("cpu")
    return model


@lru_cache(maxsize=1)
def get_model_metadata() -> ModelMetrics:
    if not SUMMARY_PATH.exists():
        return ModelMetrics()
    with SUMMARY_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    metrics = data.get("best_model_metrics", {})
    meta = data.get("training_details", {})
    model_name = data.get("model_name") or data.get("model", "YOLOv8")
    return ModelMetrics(
        model_name=str(model_name),
        metrics={
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        },
        training_settings={
            "epochs": str(meta.get("epochs") or data.get("epochs", "")),
            "imgsz": str(meta.get("imgsz") or data.get("imgsz", "")),
            "batch": str(meta.get("batch") or data.get("batch", "")),
        },
    )


def _load_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        return image.convert("RGB")
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc


def _draw_boxes(image_np: np.ndarray, boxes_xyxy: np.ndarray, confs: np.ndarray) -> np.ndarray:
    annotated = image_np.copy()
    color = (53, 107, 255)
    for idx in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[idx].astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"Plate {confs[idx]:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - text_height - 8)),
            (x1 + text_width + 8, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _encode_image(image_np: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image_np).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _collect_sample_images() -> List[str]:
    return [GITHUB_IMAGES_BASE + name for name in EXAMPLE_IMAGES]


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------
app = FastAPI(
    title="Brazilian License Plate Recognition API",
    version="1.0.0",
    description=(
        "REST API que expõe um modelo YOLOv8 treinado para detectar placas Mercosul "
        "brasileiras. Pronto para implantação em plataformas de container como Render."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/", response_model=Dict[str, str])
def root() -> Dict[str, str]:
    return {
        "message": "Brazilian License Plate Recognition API",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health_check() -> HealthResponse:
    weights_available = DEFAULT_MODEL_PATH.exists() or (WEIGHTS_DIR / "last.pt").exists()
    status = "ok" if weights_available else "missing-model"
    model_path = str(_prepare_model_path()) if weights_available else None
    return HealthResponse(
        status=status,
        model_path=model_path,
        weights_available=weights_available,
        detections_ready=weights_available,
    )


@app.get("/model/info", response_model=ModelMetrics, tags=["Model"])
def model_info() -> ModelMetrics:
    return get_model_metadata()


@app.get("/samples", response_model=SamplesResponse, tags=["Utility"])
def sample_images() -> SamplesResponse:
    return SamplesResponse(images=_collect_sample_images())


@app.post(
    "/v1/detect",
    response_model=DetectionResponse,
    tags=["Detection"],
    summary="Detect license plates in an image",
)
async def detect_license_plate(
    file: UploadFile = File(..., description="Image file (PNG or JPEG)"),
    confidence: float = Query(
        0.25,
        ge=0.01,
        le=0.99,
        description="Confidence threshold passed to YOLO.",
    ),
    iou: float = Query(
        0.5,
        ge=0.05,
        le=0.95,
        description="IoU threshold used during non-maximum suppression.",
    ),
    image_size: int = Query(
        768,
        ge=320,
        le=1280,
        description="Square image size used by YOLO during inference.",
    ),
    return_image: bool = Query(
        False,
        description="If true, returns an annotated PNG encoded as base64.",
    ),
) -> DetectionResponse:
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=415, detail="Unsupported file type.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file received.")

    pil_image = _load_image(image_bytes)
    width, height = pil_image.size

    try:
        weights_path = _prepare_model_path()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model weights not available.") from exc
    model = get_model()
    start_time = time.perf_counter()

    try:
        results = model.predict(
            np.array(pil_image),
            conf=confidence,
            iou=iou,
            imgsz=image_size,
            verbose=False,
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail="Model inference failed.") from exc

    inference_time_ms = (time.perf_counter() - start_time) * 1000.0

    detections: List[Detection] = []
    annotated_image_base64: Optional[str] = None

    if results:
        first_result = results[0]
        names = first_result.names
        boxes = first_result.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            for idx, (bbox, conf_val, cls_val) in enumerate(zip(xyxy, confs, classes)):
                x1, y1, x2, y2 = bbox.astype(float)
                box_width = max(0, x2 - x1)
                box_height = max(0, y2 - y1)
                detections.append(
                    Detection(
                        id=idx,
                        class_id=int(cls_val),
                        class_name=str(names.get(int(cls_val), "plate")),
                        confidence=float(conf_val),
                        box=BoundingBox(
                            xmin=int(round(x1)),
                            ymin=int(round(y1)),
                            xmax=int(round(x2)),
                            ymax=int(round(y2)),
                            width=int(round(box_width)),
                            height=int(round(box_height)),
                            xmin_norm=float(np.clip(x1 / width, 0.0, 1.0)),
                            ymin_norm=float(np.clip(y1 / height, 0.0, 1.0)),
                            xmax_norm=float(np.clip(x2 / width, 0.0, 1.0)),
                            ymax_norm=float(np.clip(y2 / height, 0.0, 1.0)),
                        ),
                    )
                )

            if return_image:
                annotated = _draw_boxes(
                    np.array(pil_image),
                    xyxy,
                    confs,
                )
                annotated_image_base64 = _encode_image(annotated)

    return DetectionResponse(
        detections=detections,
        image=ImageInfo(width=width, height=height, mode=pil_image.mode),
        performance=PerformanceInfo(
            inference_time_ms=inference_time_ms,
            model_name=weights_path.name,
            framework_version=str(getattr(model, "version", "unknown")),
        ),
        annotated_image_base64=annotated_image_base64,
    )

