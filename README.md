---
title: Brazilian License Plate Recognition API
emoji: 🚗
colorFrom: blue
colorTo: orange
sdk: docker
pinned: false
license: mit
---

# Brazilian License Plate Recognition API

REST API for detecting Brazilian Mercosul license plates using a fine-tuned YOLOv8 model.  
This repository is optimized for container-based deployments (Render by default) and can serve as the backend for any custom front end.

## Features
- YOLOv8 small model trained for Brazilian Mercosul plates.
- Single endpoint (`POST /v1/detect`) to run inference on user-supplied images.
- Optional return of annotated images (PNG, base64 encoded).
- Health, metadata, and sample utilities for front-end integration.
- Containerized deployment flow set up for Render Web Services.

## Quick Start
```bash
git clone https://github.com/sidnei-almeida/brazilian-license-plate-recognition.git
cd brazilian-license-plate-recognition
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python setup.py  # installs dependencies if needed and starts uvicorn
```

The API listens on `http://127.0.0.1:8000` (configurable via `PORT`) and exposes interactive docs at `http://127.0.0.1:8000/docs`.

## Project Structure
```
.
├── app.py                     # FastAPI application entrypoint
├── Dockerfile                 # Container definition
├── render.yaml                # Render web service configuration
├── requirements.txt           # Python dependencies (CPU-friendly)
├── packages.txt               # System dependency manifest (mirrors Dockerfile apt installs)
├── setup.py                   # Helper script to validate environment and run server
├── plate_detector_v1/         # Model assets
│   ├── weights/
│   │   └── best.pt            # Primary YOLO weights (required)
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
   Place `best.pt` inside `plate_detector_v1/weights/` (already included in this repo).

3. **Run tests manually**
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Use the API**
   ```bash
   curl -X POST "http://127.0.0.1:8000/v1/detect" \
     -F "file=@images/DCAM0015_JPG_jpg.rf.72c86340f8f15c0a24c50bde98fa8f57.jpg"
   ```

## Docker Deployment

Use the provided `Dockerfile` to build locally or on any container platform; Render consumes this file automatically.

### Dockerfile Highlights
- Based on `python:3.11-slim`.
- Installs system packages required by OpenCV (`packages.txt` content).
- Installs dependencies from `requirements.txt`.
- Sets `PORT=8000` (overridable) and launches `uvicorn app:app`.

### Deploying on Render
1. Fork or push this repository to your GitHub account.
2. Create a new **Web Service** on Render and connect it to the repository.
3. Choose **Docker** as the environment; Render will detect `render.yaml` automatically.
4. Keep the default build command (Render builds using the `Dockerfile`).
5. The service will boot with the `PORT` injected by Render (defaults to `8000` locally).
6. After deployment, the API will be available at `https://<service-name>.onrender.com/v1/detect`.

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
