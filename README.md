# 🚗 Brazilian License Plate Recognition System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[![GitHub](https://img.shields.io/badge/GitHub-sidnei--almeida-181717?logo=github)](https://github.com/sidnei-almeida)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-saaelmeida93-0A66C2?logo=linkedin)](https://www.linkedin.com/in/saaelmeida93/)

</div>

## 📋 Project Description

Advanced **Automatic License Plate Recognition (ALPR)** system developed specifically for Brazilian license plates, including the Mercosul standard. Uses a custom-trained YOLOv8 model with high precision to detect plates in vehicle images.

## ✨ Key Features

- 🔍 **Plate Detection**: YOLOv8 model optimized for Brazilian plates
- 🚗 **Mercosul Standard**: Full support for the new Brazilian plate format
- 📊 **Interactive Interface**: Streamlit application with advanced visualizations
- 📈 **Performance Analysis**: Detailed metrics and interactive charts
- 🧪 **Real-Time Testing**: Interface to test the model with your own images
- 📱 **Back Camera Input**: Specifically use your back camera for real-time detection
- 📚 **Complete Documentation**: Detailed usage and development guides

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Interface     │    │   Modelo YOLOv8  │    │   Processamento │
│   Streamlit     │───▶│   Treinado       │───▶│   de Imagens    │
│                 │    │                  │    │                 │
│ - Navigation    │    │ - Detection      │    │ - Bounding      │
│ - Visualizations│    │ - Classification │    │   Boxes         │
│ - Testing       │    │ - Confidence     │    │ - Confidence    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📊 Model Performance

### Training Metrics
- **Precision**: 99.69%
- **Recall**: 99.19%
- **mAP@50**: 99.49%
- **mAP@50-95**: 95.56%
- **Best Epoch**: 170/300

### Resources Used
- **Base Model**: YOLOv8s (Small)
- **Dataset**: Specialized for Brazilian plates
- **Training Epochs**: 300 (with early stopping)
- **Batch Size**: 16
- **Image Size**: 640x640

## 🚀 Installation and Execution

### Prerequisites

- **Python 3.11+** (including 3.13)
- **pip** (package manager)
- **🌐 Streamlit Cloud** (recommended) or local environment

### ⚡ Optimized Performance

This system has been specially optimized for **Streamlit Cloud**:

- ✅ **CPU versions** of libraries (smaller size)
- ✅ **No GPU required** for operation
- ✅ **Direct deployment** on Streamlit Cloud
- ✅ **Adequate performance** even with limited resources

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sidnei-almeida/brazilian-license-plate-recognition.git
   cd brazilian-license-plate-recognition
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Access in browser:**
   Open `http://localhost:8501` to view the application.

### 🚀 Deploy to Streamlit Cloud (Recommended)

To deploy for free on Streamlit Cloud:

1. **Fork** this repository on GitHub
2. **Visit** [share.streamlit.io](https://share.streamlit.io)
3. **Connect** your GitHub repository
4. **Configure**:
   - **Main file path**: `app.py`
   - **Python version**: 3.13
   - **Requirements**: already included in `requirements.txt`
   - **System packages**: already included in `packages.txt`
5. **Deploy!** - The system will work perfectly in the cloud

> 💡 **Important notes**:
> - ✅ Test images are automatically loaded from GitHub
> - ✅ The `packages.txt` installs system dependencies required for OpenCV
> - ✅ `opencv-python-headless` is used to avoid conflicts on Streamlit Cloud
> - ✅ Uses `streamlit-back-camera-input` for specific access to the back camera
> - ✅ The back camera is specifically used for better license plate detection quality
> - ✅ Correct syntax: `back_camera_input(key="key_name")` without additional parameters

## ⚡ Performance

### Optimized for Streamlit Cloud

| Environment | Inference Time | Resources | Status |
|-------------|----------------|-----------|--------|
| **Streamlit Cloud** | ~3-8 seconds | Shared CPU | ✅ **Optimized** |
| **Local Development** | ~2-5 seconds | Local CPU | ✅ Supported |
| **Local GPU** | ~0.5-2 seconds | NVIDIA GPU | ⚠️ Optional |

### Applied Optimizations

- ✅ **CPU Optimized**: Lightweight library versions
- ✅ **Memory Efficient**: Optimized RAM usage
- ✅ **Streamlit Cloud Ready**: Direct deployment without configurations
- ✅ **Smart Caching**: Pre-loaded model to reduce latency
- ✅ **Batch Processing**: Efficient processing for limited resources

## 🔧 Troubleshooting

### Common Issues

**❌ "Model not found"**
- Make sure the `plate_detector_v1/weights/` folder exists
- Check if the `best.pt` file is present

**❌ "Import error for torch/ultralytics"**
```bash
# Reinstall dependencies:
pip uninstall torch torchvision torchaudio ultralytics
pip install -r requirements.txt
```

**❌ "Insufficient memory on Streamlit Cloud"**
- The system is optimized to work with limited resources
- If necessary, use smaller images (the model accepts up to 640x640)

**❌ "Slow processing"**
- On Streamlit Cloud: ~3-8 seconds per image (normal)
- Locally: ~2-5 seconds per image (CPU)
- To speed up: consider using local GPU (optional)

### CPU Performance

The system works perfectly with CPU:
- **Streamlit Cloud**: 3-8 seconds per image
- **Local development**: 2-5 seconds per image
- **RAM**: ~2-4GB required

### Logs and Debug

To enable detailed logs in the code:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 📁 Project Structure

```
brazilian-license-plate-recognition/
│
├── 📁 images/                          # Test images
│   ├── DCAM0015_JPG_jpg.rf.72c8...jpg
│   ├── DCAM0019_JPG_jpg.rf.4fe1...jpg
│   └── ...
│
├── 📁 plate_detector_v1/               # Trained model
│   ├── weights/
│   │   ├── best.pt                     # Best model
│   │   └── last.pt                     # Last model
│   ├── args.yaml                       # Hyperparameters
│   ├── results.csv                     # Metrics per epoch
│   └── results.png                     # Results chart
│
├── 📁 notebooks/                       # Training notebooks
│   └── 1_YOLOv8_Training_Brazilian_Plates.ipynb
│
├── 📄 app.py                           # Main Streamlit application
├── 📄 requirements.txt                 # Python dependencies
├── 📄 README.md                        # This file
└── 📄 plate_detector_v1_summary.json   # Training summary
```

## 🎯 How to Use

### 1. Home Page
- System overview
- Key features
- Results chart do treinamento

### 2. Model Testing
- **Visual Selector**: Choose images using `streamlit-image-select`
- **Detection**: Click "Detect Plates" to process
- **Results**: View bounding boxes and confidence levels

### 3. Results Analysis
- **Metrics**: Cards with main performance indicators
- **Interactive Charts**: Metric evolution during training
- **Loss Analysis**: Detailed training curves

### 4. About the Model
- **Architecture**: Technical details of YOLOv8
- **Hyperparameters**: Settings used in training
- **Process**: Step-by-step explanation of detection

### 5. About the Data
- **Dataset**: Training set characteristics
- **Plate Types**: Examples of different formats
- **Test Images**: Gallery of available images

## 🛠️ Development

### YOLOv8 Model Architecture

The model uses the YOLOv8 architecture with the following features:

- **Backbone**: Modified CSPDarknet53
- **Neck**: PAN (Path Aggregation Network)
- **Head**: YOLOv8 detection head
- **Size**: "small" variant (YOLOv8s)

### Training Process

1. **Data Preparation**: Dataset formatted in YOLO standard
2. **Configuration**: Hyperparameter definition
3. **Training**: 300 epochs with early stopping
4. **Validation**: Evaluation on validation set
5. **Optimization**: Best model selection

### Metrics Used

- **Precision**: Fraction of correct detections
- **Recall**: Fraction of real plates detected
- **mAP@50**: Mean Average Precision (IoU ≥ 0.5)
- **mAP@50-95**: Mean Average Precision (average IoU 0.5-0.95)

## 🔧 Customization

### Test Images

**✅ Test images are automatically loaded from GitHub!**

- The system loads images directly from the repository
- No need to have images locally
- Works perfectly on Streamlit Cloud
- Automatic cache for better performance

### Add New Images

To add your own test images:

1. Upload via interface **"Upload"** in the Detector tab
2. Or, to add permanently:
   - Place your images in the folder `images/`
    - Add file names to the list `EXAMPLE_IMAGES` in `app.py`
   - Commit to GitHub
3. Images appear automatically in the selector

### Use Back Camera for Detection

To specifically use your **back camera** to detect plates:

1. **Access the "Camera" tab** in the Detector section
2. **Allow camera access** when requested by the browser
3. **The application automatically uses your back camera** (ideal for plates)
4. **Point the camera** at a Brazilian license plate and take a photo
5. **Click "Detect Plates"** to analyze the captured image
6. **View the results** with bounding boxes and detection details

> 💡 **Note**: Detection quality depends on lighting and plate angle. The back camera is perfect for capturing plates at a distance.

### Adjust Model Parameters

To modify the minimum confidence or other parameters:

```python
# In app.py, line 54
results = model(image, conf=0.5)  # Adjust the threshold here
```

## 🔧 Troubleshooting

### Error: `ImportError: libGL.so.1: cannot open shared object file`

This error occurs when OpenCV cannot find the system graphics libraries. **Solution:**

1. **No Streamlit Cloud**: The `packages.txt` file is already configured to install the necessary dependencies
2. **Locally (Linux)**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
   ```
3. **Locally (Mac)**: Not needed, already works natively
4. **Locally (Windows)**: Not needed, already works natively

### Error: Conflict between `opencv-python` and `opencv-python-headless`

**Solution:** The `requirements.txt` is already configured to install `opencv-python-headless` before `ultralytics`, avoiding conflicts.

### Error: `back_camera_input() got an unexpected keyword argument 'help'`

**Cause:** The `streamlit-back-camera-input` component does not accept the `help` parameter.

**Solution:** Use only the `key` parameter:
```python
camera_image = back_camera_input(key="back_camera_input")
```

### Deploy hanging on Streamlit Cloud

**Possible causes:**
- Model size too large
- Lack of memory during installation

**Solution:** The repository is already optimized with CPU versions of libraries, which are smaller and faster to install.

## 📈 Future Improvements

- [ ] OCR integration for character reading
- [ ] Real-time video support
- [ ] REST API for integration with other systems
- [ ] Complementary mobile app
- [ ] Additional model optimization for edge devices

## 🤝 Contributing

Contributions are welcome! Follow these steps:

1. Fork the project
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is under the MIT license. See the `LICENSE` file for more details.

## 👨‍💻 Author

<div align="center">

**Sidnei Almeida**

[![GitHub](https://img.shields.io/badge/GitHub-sidnei--almeida-181717?style=for-the-badge&logo=github)](https://github.com/sidnei-almeida)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-saaelmeida93-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/saaelmeida93/)

Developer specialized in Machine Learning and Computer Vision

</div>

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8 developers
- **Streamlit**: Framework for creating the interface
- **Python Community**: For exceptional libraries and tools

## 📞 Support

For support and questions:

- 💬 Open an [Issue](https://github.com/sidnei-almeida/brazilian-license-plate-recognition/issues)
- 💼 Contact via [LinkedIn](https://www.linkedin.com/in/saaelmeida93/)
- 📧 Discussions on [GitHub Discussions](https://github.com/sidnei-almeida/brazilian-license-plate-recognition/discussions)

---

⭐ **If this project was useful to you, consider giving it a star!** ⭐
