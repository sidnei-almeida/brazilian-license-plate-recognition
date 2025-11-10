---
title: Brazilian License Plate Recognition API
emoji: 🚗
colorFrom: blue
colorTo: yellow
sdk: docker
pinned: false
license: mit
---

# Brazilian License Plate Recognition API

REST API for detecting Brazilian Mercosul license plates using a fine-tuned YOLOv8 model.  
This repository targets Hugging Face Spaces (Docker runtime) and serves as the backend for a custom front-end experience.

## Features
- YOLOv8 small model trained for Brazilian Mercosul plates.
- Single endpoint (`POST /v1/detect`) to run inference on user-supplied images.
- Optional return of annotated images (PNG, base64 encoded).
- Health, metadata, and sample utilities for front-end integration.
- Containerized deployment flow tailored for Hugging Face Spaces.

## Quick Start
```bash
git clone https://github.com/sidnei-almeida/brazilian-license-plate-recognition.git
cd brazilian-license-plate-recognition
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python setup.py  # installs dependencies if needed and starts uvicorn
```

The API listens on `http://127.0.0.1:7860` and exposes interactive docs at `http://127.0.0.1:7860/docs`.

## Project Structure
```
.
├── app.py                     # FastAPI application entrypoint
├── Dockerfile                 # Hugging Face Space definition (docker runtime)
├── requirements.txt           # Python dependencies (CPU-friendly)
├── packages.txt               # System dependency manifest (mirrors Dockerfile apt installs)
├── setup.py                   # Helper script to validate environment and run server
├── plate_detector_v1/         # Model assets
│   ├── weights/
│   │   ├── best.pt            # Primary YOLO weights (required)
│   │   └── last.pt            # Fallback weights
│   ├── args.yaml              # Training configuration snapshot
│   ├── results.csv            # Training metrics per epoch
│   └── *.png                  # Training visualizations
├── plate_detector_v1_summary.json
├── images/                    # Sample Mercosul plate photographs
└── notebooks/                 # Training notebooks (reference only)
```

## API Overview

| Method | Path          | Description                                           |
|--------|---------------|-------------------------------------------------------|
| GET    | `/`           | Basic welcome payload with links to docs and health. |
| GET    | `/health`     | Checks for model availability and readiness.         |
| GET    | `/model/info` | Returns metrics found in `plate_detector_v1_summary.json`. |
| GET    | `/samples`    | Lists sample image URLs hosted on GitHub.            |
| POST   | `/v1/detect`  | Runs inference on an uploaded image.                 |

### Detection Request
- **Content-Type**: `multipart/form-data`
- **File field**: `file` (PNG or JPEG)
- **Query parameters** (optional):
  - `confidence` (`float`, default `0.25`, range `0.01-0.99`)
  - `iou` (`float`, default `0.5`, range `0.05-0.95`)
  - `image_size` (`int`, default `768`, range `320-1280`)
  - `return_image` (`bool`, default `false`)

### Detection Response (excerpt)
```json
{
  "detections": [
    {
      "id": 0,
      "class_id": 0,
      "class_name": "plate",
      "confidence": 0.93,
      "box": {
        "xmin": 215,
        "ymin": 142,
        "xmax": 398,
        "ymax": 220,
        "width": 183,
        "height": 78,
        "xmin_norm": 0.34,
        "ymin_norm": 0.28,
        "xmax_norm": 0.63,
        "ymax_norm": 0.43
      }
    }
  ],
  "image": {"width": 640, "height": 480, "mode": "RGB"},
  "performance": {"inference_time_ms": 74.2, "model_name": "best.pt", "framework_version": "8.2.0"},
  "annotated_image_base64": null
}
```
If `return_image=true`, `annotated_image_base64` contains a base64-encoded PNG with bounding boxes and scores.

## Local Development

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure model weights**
   Place `best.pt` inside `plate_detector_v1/weights/` (already included in this repo). `last.pt` can act as fallback.

3. **Run tests manually**
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 7860
   ```

4. **Use the API**
   ```bash
   curl -X POST "http://127.0.0.1:7860/v1/detect" \
     -F "file=@images/DCAM0015_JPG_jpg.rf.72c86340f8f15c0a24c50bde98fa8f57.jpg"
   ```

## Docker & Hugging Face Deployment

Hugging Face Spaces (Docker runtime) automatically builds the image using the provided `Dockerfile`.

### Dockerfile Highlights
- Based on `python:3.11-slim`.
- Installs system packages required by OpenCV (`packages.txt` content).
- Installs dependencies from `requirements.txt`.
- Sets `PORT=7860` and launches `uvicorn app:app`.

### Deploying on Hugging Face
1. Create a new Space with **SDK = Docker**.
2. Push this repository (including weights) to the Space.
3. Hugging Face builds and runs the container automatically.
4. The API will be available at `https://<space-name>.hf.space/v1/detect`.

## Model Summary

- **Architecture**: YOLOv8s (small)
- **Dataset**: Vehicles with Brazilian Mercosul plates
- **Epochs**: 300 (early stopping around epoch 170)
- **Metrics** (from `plate_detector_v1_summary.json`):
  - Precision: ~99.7%
  - Recall: ~99.2%
  - mAP@50: ~99.5%
  - mAP@50-95: ~95.6%

## Front-End Integration Tips
- Use `GET /samples` to bootstrap a gallery of test images.
- Use normalized box coordinates (`xmin_norm`, `ymin_norm`, `xmax_norm`, `ymax_norm`) to draw overlays independent of resizing.
- Latency is measured per request and returned as `inference_time_ms`.
- When requesting the annotated image, decode the base64 string into a PNG and display or store it directly.

## License
Distributed under the MIT License. See `LICENSE` for details.

## Acknowledgements
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8.
- Brazilian plate dataset contributors.

## Support
- Create an [issue](https://github.com/sidnei-almeida/brazilian-license-plate-recognition/issues) for bugs or questions.
- Connect on [LinkedIn](https://www.linkedin.com/in/saaelmeida93/).
