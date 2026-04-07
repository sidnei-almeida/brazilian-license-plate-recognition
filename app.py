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
ALPR_WEIGHTS_DIR = BASE_DIR / "license_plate_alpr" / "weights"
DEFAULT_MODEL_PATH = Path(
    os.getenv("MODEL_WEIGHTS_PATH", WEIGHTS_DIR / "best.pt")
)
DEFAULT_ALPR_MODEL_PATH = Path(
    os.getenv("ALPR_WEIGHTS_PATH", ALPR_WEIGHTS_DIR / "best.pt")
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
    plate_text: Optional[str] = None
    plate_text_confidence: Optional[float] = Field(
        default=None,
        description="Média das confianças dos caracteres detectados pelo modelo ALPR.",
    )


class ImageInfo(BaseModel):
    width: int
    height: int
    mode: str


class PerformanceInfo(BaseModel):
    inference_time_ms: float
    model_name: str
    framework_version: str
    detector_inference_time_ms: Optional[float] = None
    alpr_inference_time_ms: Optional[float] = None
    alpr_model_name: Optional[str] = None


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
    alpr_model_path: Optional[str] = None
    alpr_weights_available: bool = False
    alpr_ready: bool = False


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


def _prepare_alpr_model_path() -> Path:
    if DEFAULT_ALPR_MODEL_PATH.exists():
        return DEFAULT_ALPR_MODEL_PATH
    available = sorted(ALPR_WEIGHTS_DIR.glob("*.pt"))
    if available:
        return available[0]
    raise FileNotFoundError(
        f"ALPR weights not found. Expected at {DEFAULT_ALPR_MODEL_PATH} "
        f"or any *.pt file under {ALPR_WEIGHTS_DIR}"
    )


@lru_cache(maxsize=1)
def get_alpr_model() -> YOLO:
    model_path = _prepare_alpr_model_path()
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


def _expand_plate_box_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    pad_ratio: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio
    nx1 = max(0.0, x1 - pad_x)
    ny1 = max(0.0, y1 - pad_y)
    nx2 = min(float(img_w), x2 + pad_x)
    ny2 = min(float(img_h), y2 + pad_y)
    return nx1, ny1, nx2, ny2


def _assign_chars_to_plates_exclusive(
    char_xyxy: np.ndarray,
    char_cls: np.ndarray,
    char_conf: np.ndarray,
    names: Dict[int, str],
    plate_xyxy: np.ndarray,
    img_w: int,
    img_h: int,
    box_padding: float,
) -> List[tuple[str, float]]:
    """
    Cada caractere (ALPR na imagem inteira) vai para no máximo uma placa.
    Usa caixas de placa expandidas; em sobreposição, escolhe a placa cujo centro
    está mais próximo do centro do caractere. Ordena por x dentro de cada placa.
    """
    n_plates = len(plate_xyxy)
    if n_plates == 0:
        return []
    n_chars = len(char_xyxy) if char_xyxy.size else 0
    if n_chars == 0:
        return [("", 0.0)] * n_plates

    pcx = (plate_xyxy[:, 0] + plate_xyxy[:, 2]) / 2.0
    pcy = (plate_xyxy[:, 1] + plate_xyxy[:, 3]) / 2.0
    expanded: List[tuple[float, float, float, float]] = []
    for j in range(n_plates):
        x1, y1, x2, y2 = plate_xyxy[j].astype(float)
        expanded.append(_expand_plate_box_xyxy(x1, y1, x2, y2, box_padding, img_w, img_h))

    ccx = (char_xyxy[:, 0] + char_xyxy[:, 2]) / 2.0
    ccy = (char_xyxy[:, 1] + char_xyxy[:, 3]) / 2.0
    assignment = np.full(n_chars, -1, dtype=int)

    for i in range(n_chars):
        cx, cy = float(ccx[i]), float(ccy[i])
        best: Optional[tuple[float, int]] = None
        for j in range(n_plates):
            ex1, ey1, ex2, ey2 = expanded[j]
            if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
                dist_sq = (cx - pcx[j]) ** 2 + (cy - pcy[j]) ** 2
                cand = (dist_sq, j)
                if best is None or cand < best:
                    best = cand
        if best is not None:
            assignment[i] = best[1]

    results: List[tuple[str, float]] = []
    for j in range(n_plates):
        idxs = np.flatnonzero(assignment == j)
        if idxs.size == 0:
            results.append(("", 0.0))
            continue
        order = idxs[np.argsort(ccx[idxs])]
        chars: List[str] = []
        confs: List[float] = []
        for ii in order:
            cid = int(char_cls[ii])
            chars.append(str(names.get(cid, "")))
            confs.append(float(char_conf[ii]))
        text = "".join(chars)
        avg_conf = float(np.mean(confs)) if confs else 0.0
        results.append((text, avg_conf))
    return results


def _pil_crop_xyxy(
    pil_image: Image.Image,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    pad_ratio: float,
) -> Image.Image:
    w, h = pil_image.size
    bw = max(0.0, float(x2 - x1))
    bh = max(0.0, float(y2 - y1))
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio
    nx1 = max(0, int(round(x1 - pad_x)))
    ny1 = max(0, int(round(y1 - pad_y)))
    nx2 = min(w, int(round(x2 + pad_x)))
    ny2 = min(h, int(round(y2 + pad_y)))
    if nx2 <= nx1 or ny2 <= ny1:
        return pil_image.crop((0, 0, min(w, 1), min(h, 1)))
    return pil_image.crop((nx1, ny1, nx2, ny2))


def _upscale_crop_min_side(crop: Image.Image, min_side: int = 160) -> Image.Image:
    w, h = crop.size
    if min(w, h) <= 0:
        return crop
    if min(w, h) >= min_side:
        return crop
    scale = min_side / float(min(w, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return crop.resize((nw, nh), Image.Resampling.LANCZOS)


def _alpr_infer_on_crop(
    alpr_model: YOLO,
    crop: Image.Image,
    conf: float,
    iou: float,
    imgsz: int,
) -> tuple[str, float]:
    crop = _upscale_crop_min_side(crop.convert("RGB"))
    arr = np.array(crop)
    if arr.size == 0:
        return "", 0.0
    try:
        results = alpr_model.predict(
            arr,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
        )
    except Exception:  # pylint: disable=broad-except
        return "", 0.0
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return "", 0.0
    first = results[0]
    xyxy = first.boxes.xyxy.cpu().numpy()
    confs = first.boxes.conf.cpu().numpy()
    classes = first.boxes.cls.cpu().numpy().astype(int)
    names = first.names
    centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    order = np.argsort(centers_x)
    chars: List[str] = []
    char_confs: List[float] = []
    for k in order:
        cid = int(classes[k])
        chars.append(str(names.get(cid, "")))
        char_confs.append(float(confs[k]))
    return "".join(chars), float(np.mean(char_confs)) if char_confs else 0.0


def _draw_boxes(
    image_np: np.ndarray,
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    plate_texts: Optional[List[Optional[str]]] = None,
) -> np.ndarray:
    annotated = image_np.copy()
    color = (53, 107, 255)
    for idx in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[idx].astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        read = None
        if plate_texts and idx < len(plate_texts):
            read = plate_texts[idx]
        if read:
            label = f"{read} ({confs[idx]:.2f})"
        else:
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
    alpr_weights_available = DEFAULT_ALPR_MODEL_PATH.exists() or (ALPR_WEIGHTS_DIR / "last.pt").exists()
    both_ok = weights_available and alpr_weights_available
    status = "ok" if both_ok else "missing-model"
    model_path = str(_prepare_model_path()) if weights_available else None
    alpr_model_path = None
    if alpr_weights_available:
        try:
            alpr_model_path = str(_prepare_alpr_model_path())
        except FileNotFoundError:
            alpr_weights_available = False
    return HealthResponse(
        status=status,
        model_path=model_path,
        weights_available=weights_available,
        detections_ready=weights_available,
        alpr_model_path=alpr_model_path,
        alpr_weights_available=alpr_weights_available,
        alpr_ready=bool(alpr_weights_available),
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
    read_plate: bool = Query(
        True,
        description="Se true, aplica o modelo ALPR (caracteres) em cada placa detectada.",
    ),
    alpr_confidence: float = Query(
        0.25,
        ge=0.01,
        le=0.99,
        description="Limite de confiança do YOLO de leitura (ALPR).",
    ),
    alpr_iou: float = Query(
        0.5,
        ge=0.05,
        le=0.95,
        description="IoU do NMS do modelo ALPR.",
    ),
    alpr_image_size: int = Query(
        640,
        ge=320,
        le=1280,
        description="Tamanho de entrada do YOLO ALPR no recorte da placa.",
    ),
    alpr_box_padding: float = Query(
        0.12,
        ge=0.0,
        le=0.45,
        description=(
            "Expansão relativa da caixa da placa ao associar caracteres da imagem inteira "
            "(ajuda múltiplas placas e bordas)."
        ),
    ),
    alpr_crop_fallback: bool = Query(
        True,
        description=(
            "Se true, quando não houver texto pela associação na imagem inteira, "
            "roda o ALPR de novo só no recorte da placa (com ampliação mínima)."
        ),
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

    alpr_weights_path: Optional[Path] = None
    if read_plate:
        try:
            alpr_weights_path = _prepare_alpr_model_path()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail="ALPR weights not available.") from exc

    model = get_model()
    t0 = time.perf_counter()

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

    detector_time_ms = (time.perf_counter() - t0) * 1000.0
    alpr_time_ms = 0.0
    alpr_model: Optional[YOLO] = None
    if read_plate:
        alpr_model = get_alpr_model()

    detections: List[Detection] = []
    annotated_image_base64: Optional[str] = None
    plate_texts_for_draw: Optional[List[Optional[str]]] = None

    if results:
        first_result = results[0]
        names = first_result.names
        boxes = first_result.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            char_xyxy = np.empty((0, 4), dtype=np.float64)
            char_cls = np.empty((0,), dtype=int)
            char_conf = np.empty((0,), dtype=np.float64)
            char_names_map: Dict[int, str] = {}

            if read_plate and alpr_model is not None:
                t_alpr = time.perf_counter()
                try:
                    alpr_results = alpr_model.predict(
                        np.array(pil_image),
                        conf=alpr_confidence,
                        iou=alpr_iou,
                        imgsz=alpr_image_size,
                        verbose=False,
                    )
                except Exception:  # pylint: disable=broad-except
                    alpr_results = None
                alpr_time_ms = (time.perf_counter() - t_alpr) * 1000.0
                if (
                    alpr_results
                    and alpr_results[0].boxes is not None
                    and len(alpr_results[0].boxes) > 0
                ):
                    abox = alpr_results[0].boxes
                    char_xyxy = abox.xyxy.cpu().numpy()
                    char_conf = abox.conf.cpu().numpy()
                    char_cls = abox.cls.cpu().numpy().astype(int)
                    raw_nm = alpr_results[0].names
                    char_names_map = {int(k): str(v) for k, v in dict(raw_nm).items()}

            n_det = len(xyxy)
            plate_readings: List[tuple[str, float]] = [("", 0.0)] * n_det
            if read_plate and alpr_model is not None and n_det > 0 and char_xyxy.size > 0:
                plate_readings = _assign_chars_to_plates_exclusive(
                    char_xyxy,
                    char_cls,
                    char_conf,
                    char_names_map,
                    xyxy,
                    width,
                    height,
                    alpr_box_padding,
                )

            plate_texts_for_draw = []
            for idx, (bbox, conf_val, cls_val) in enumerate(zip(xyxy, confs, classes)):
                x1, y1, x2, y2 = bbox.astype(float)
                box_width = max(0, x2 - x1)
                box_height = max(0, y2 - y1)
                plate_text: Optional[str] = None
                text_conf: Optional[float] = None
                if read_plate and alpr_model is not None:
                    raw_text, avg_c = plate_readings[idx]
                    if raw_text:
                        plate_text = raw_text
                        text_conf = avg_c
                    if alpr_crop_fallback and not raw_text:
                        t_fb = time.perf_counter()
                        crop = _pil_crop_xyxy(pil_image, x1, y1, x2, y2, 0.08)
                        fb_text, fb_c = _alpr_infer_on_crop(
                            alpr_model,
                            crop,
                            alpr_confidence,
                            alpr_iou,
                            alpr_image_size,
                        )
                        alpr_time_ms += (time.perf_counter() - t_fb) * 1000.0
                        if fb_text:
                            plate_text = fb_text
                            text_conf = fb_c
                plate_texts_for_draw.append(plate_text)

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
                        plate_text=plate_text,
                        plate_text_confidence=text_conf,
                    )
                )

            if return_image:
                annotated = _draw_boxes(
                    np.array(pil_image),
                    xyxy,
                    confs,
                    plate_texts_for_draw,
                )
                annotated_image_base64 = _encode_image(annotated)

    total_time_ms = detector_time_ms + alpr_time_ms
    perf = PerformanceInfo(
        inference_time_ms=total_time_ms,
        model_name=weights_path.name,
        framework_version=str(getattr(model, "version", "unknown")),
        detector_inference_time_ms=detector_time_ms,
        alpr_inference_time_ms=alpr_time_ms if read_plate else None,
        alpr_model_name=(
            str(alpr_weights_path.relative_to(BASE_DIR)) if alpr_weights_path else None
        ),
    )

    return DetectionResponse(
        detections=detections,
        image=ImageInfo(width=width, height=height, mode=pil_image.mode),
        performance=perf,
        annotated_image_base64=annotated_image_base64,
    )

