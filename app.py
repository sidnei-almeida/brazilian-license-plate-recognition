import os
import io
import glob
import yaml
import time
import platform
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from ultralytics import __version__ as yolo_version
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import cv2
from streamlit_image_select import image_select
import torch
import json

# Base do app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "plate_detector_v1", "weights")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

st.set_page_config(
    page_title="Brazilian License Plate Recognition • ALPR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo premium dark - Tons de cinza com acentos quentes
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
  --primary: #ff6b35;
  --accent: #f7931e;
  --success: #52b788;
  --danger: #e63946;
  --warning: #ffa500;
  --dark-bg: #18181b;
  --card-bg: #27272a;
  --sidebar-bg: #1f1f23;
  --text: #e4e4e7;
  --text-secondary: #a1a1aa;
  --muted: #71717a;
  --border: #3f3f46;
  --shadow: 0 4px 16px rgba(0,0,0,0.25);
}
.stApp { 
  background: var(--dark-bg); 
  color: var(--text); 
}
.main .block-container { 
  max-width: none !important; 
  padding-left: 1.5rem; 
  padding-right: 1.5rem; 
}

h1, h2, h3, h4, h5 { 
  font-family: 'Inter', sans-serif; 
  color: var(--text);
  font-weight: 600;
}

h1 { font-size: 1.75rem; }
h2 { font-size: 1.5rem; }
h3 { font-size: 1.25rem; }
h4 { font-size: 1rem; }

/* Hero Title */
.main-hero { 
  font-size: 2rem; 
  font-weight: 700; 
  margin: 0.75rem 0 1rem; 
  display: flex;
  align-items: center;
  gap: 0.875rem;
}
.title-gradient { 
  background: linear-gradient(135deg, var(--primary), var(--accent)); 
  -webkit-background-clip: text; 
  -webkit-text-fill-color: transparent; 
}
.subtitle { 
  color: var(--text-secondary); 
  margin-bottom: 1.5rem; 
  font-size: 0.938rem;
}

/* Cards */
.card { 
  background: var(--card-bg); 
  border: 1px solid var(--border); 
  border-radius: 8px; 
  padding: 1rem; 
  box-shadow: var(--shadow); 
  margin-bottom: 0.875rem;
  font-size: 0.875rem;
}
.metric-card { 
  background: linear-gradient(135deg, rgba(255,107,53,0.05), rgba(247,147,30,0.05)); 
  border: 1px solid rgba(255,107,53,0.15); 
  border-radius: 8px; 
  padding: 0.875rem; 
  text-align: center;
}
.metric-value {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--primary);
  margin: 0.375rem 0;
}
.metric-label {
  font-size: 0.688rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--muted);
  font-weight: 600;
}

/* Badge */
.badge { 
  display: inline-block; 
  padding: 0.25rem 0.625rem; 
  border-radius: 4px; 
  font-size: 0.75rem; 
  font-weight: 600;
  border: 1px solid;
}
.badge-primary {
  background: rgba(255,107,53,0.12); 
  border-color: rgba(255,107,53,0.25); 
  color: var(--primary);
}
.badge-success {
  background: rgba(82,183,136,0.12); 
  border-color: rgba(82,183,136,0.25); 
  color: var(--success);
}
.badge-danger {
  background: rgba(230,57,70,0.12); 
  border-color: rgba(230,57,70,0.25); 
  color: var(--danger);
}

/* Plate icon */
.plate-icon {
  width: 54px;
  height: 36px;
  background: linear-gradient(135deg, #f4f4f5 0%, #e4e4e7 100%);
  border: 2px solid #52525b;
  border-radius: 3px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.688rem;
  font-weight: 700;
  color: #27272a;
  box-shadow: 0 2px 6px rgba(0,0,0,0.3);
}

hr { 
  border-top: 1px solid var(--border); 
  margin: 1.5rem 0;
}

/* Custom alerts */
.stAlert {
  background: var(--card-bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 0.875rem !important;
  font-size: 0.875rem !important;
}

.stAlert > div {
  gap: 0.5rem !important;
}

/* Success alert */
div[data-baseweb="notification"][kind="success"] {
  background: rgba(82,183,136,0.08) !important;
  border-left: 3px solid var(--success) !important;
  color: var(--text) !important;
}

/* Error alert */
div[data-baseweb="notification"][kind="error"] {
  background: rgba(230,57,70,0.08) !important;
  border-left: 3px solid var(--danger) !important;
  color: var(--text) !important;
}

/* Warning alert */
div[data-baseweb="notification"][kind="warning"] {
  background: rgba(255,165,0,0.08) !important;
  border-left: 3px solid var(--warning) !important;
  color: var(--text) !important;
}

/* Info alert */
div[data-baseweb="notification"][kind="info"] {
  background: rgba(255,107,53,0.08) !important;
  border-left: 3px solid var(--primary) !important;
  color: var(--text) !important;
}

/* Streamlit buttons */
.stButton > button {
  background: linear-gradient(135deg, rgba(255,107,53,0.12), rgba(247,147,30,0.12)) !important;
  border: 1px solid rgba(255,107,53,0.25) !important;
  border-radius: 6px !important;
  color: var(--text) !important;
  font-size: 0.875rem !important;
  font-weight: 600 !important;
  padding: 0.5rem 1rem !important;
  transition: all 0.2s ease !important;
}

.stButton > button:hover {
  background: linear-gradient(135deg, rgba(255,107,53,0.2), rgba(247,147,30,0.2)) !important;
  border-color: rgba(255,107,53,0.4) !important;
  transform: translateY(-1px);
}

.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
  border: 1px solid var(--primary) !important;
  color: white !important;
}

.stButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, var(--accent), var(--primary)) !important;
  box-shadow: 0 4px 12px rgba(255,107,53,0.3);
}

/* Sliders */
.stSlider {
  padding: 0.5rem 0;
}

.stSlider > div > div > div {
  background: var(--border) !important;
}

.stSlider > div > div > div > div {
  background: var(--primary) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.5rem;
  background: transparent;
}

.stTabs [data-baseweb="tab"] {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 6px 6px 0 0;
  color: var(--text-secondary);
  font-size: 0.875rem;
  font-weight: 600;
  padding: 0.625rem 1.25rem;
}

.stTabs [data-baseweb="tab"]:hover {
  background: var(--sidebar-bg);
  color: var(--text);
}

.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(255,107,53,0.12), rgba(247,147,30,0.12));
  border-bottom-color: var(--primary);
  color: var(--primary);
}

/* Download button */
.stDownloadButton > button {
  background: linear-gradient(135deg, var(--success), rgba(82,183,136,0.8)) !important;
  border: 1px solid var(--success) !important;
  color: white !important;
}

.stDownloadButton > button:hover {
  background: linear-gradient(135deg, rgba(82,183,136,0.8), var(--success)) !important;
  box-shadow: 0 4px 12px rgba(82,183,136,0.3);
}

/* File uploader */
.stFileUploader {
  background: var(--card-bg);
  border: 1px dashed var(--border);
  border-radius: 8px;
  padding: 1rem;
}

.stFileUploader:hover {
  border-color: var(--primary);
  background: rgba(255,107,53,0.03);
}

/* Image selector do streamlit-image-select */
/* Container principal do image_select */
div[data-testid="column"] > div > div {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 0.75rem !important;
  justify-content: space-between !important;
  width: 100% !important;
}

/* Cada imagem individual no image_select - 5 por linha com espaço distribuído */
div[data-testid="column"] > div > div > div {
  flex: 1 1 calc(20% - 0.6rem) !important;
  max-width: calc(20% - 0.6rem) !important;
  min-width: 150px !important;
  margin: 0 !important;
}

/* Imagens dentro do image_select - altura proporcional */
div[data-testid="column"] > div > div > div img {
  width: 100% !important;
  height: auto !important;
  aspect-ratio: 4/3 !important;
  object-fit: cover !important;
  border: 2px solid var(--border) !important;
  border-radius: 8px !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
}

/* Hover nas imagens do image_select */
div[data-testid="column"] > div > div > div img:hover {
  transform: translateY(-2px) !important;
  border-color: var(--primary) !important;
  box-shadow: 0 6px 16px rgba(255,107,53,0.25) !important;
}

/* Legenda das imagens */
div[data-testid="column"] > div > div > div p {
  font-size: 0.75rem !important;
  color: var(--text-secondary) !important;
  text-align: center !important;
  margin-top: 0.5rem !important;
}

/* Imagens normais do Streamlit (Analytics, etc) - manter tamanho normal */
div[data-testid="stImage"]:not(div[data-testid="column"] div[data-testid="stImage"]) img {
  border: 2px solid var(--border);
  border-radius: 8px;
  transition: all 0.2s ease;
}

div[data-testid="stImage"]:not(div[data-testid="column"] div[data-testid="stImage"]) img:hover {
  transform: translateY(-2px);
  border-color: var(--primary);
  box-shadow: 0 6px 16px rgba(255,107,53,0.25);
}

/* Responsive */
@media (max-width: 768px) {
  .main-hero { font-size: 1.5rem; }
  .card { padding: 0.875rem; }
  .metric-card { padding: 0.75rem; }
  
  /* Image select mobile - 2 por linha */
  div[data-testid="column"] > div > div > div {
    flex: 1 1 calc(50% - 0.375rem) !important;
    max-width: calc(50% - 0.375rem) !important;
    min-width: 120px !important;
  }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model() -> YOLO | None:
    """Carrega o modelo YOLO de detecção de placas"""
    model_path = os.path.join(WEIGHTS_DIR, "best.pt")
    if not os.path.exists(model_path):
        alt_path = os.path.join(WEIGHTS_DIR, "last.pt")
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            st.error(f"Modelo não encontrado em {WEIGHTS_DIR}")
            return None
    
    try:
        model = YOLO(model_path)
        # Força CPU se não houver GPU
        if not torch.cuda.is_available():
            model.to('cpu')
        return model
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {str(e)}")
        return None


@st.cache_data(show_spinner=False)
def load_training_data():
    """Carrega dados de treinamento"""
    summary_path = os.path.join(BASE_DIR, "plate_detector_v1_summary.json")
    results_csv = os.path.join(BASE_DIR, "plate_detector_v1", "results.csv")
    args_yaml = os.path.join(BASE_DIR, "plate_detector_v1", "args.yaml")
    
    summary = None
    results_df = None
    args = None
    
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
    
    if os.path.exists(results_csv):
        results_df = pd.read_csv(results_csv)
        results_df.columns = results_df.columns.str.strip()
    
    if os.path.exists(args_yaml):
        with open(args_yaml, "r") as f:
            args = yaml.safe_load(f)
    
    # Imagens de treino
    images = {
        "results": os.path.join(BASE_DIR, "plate_detector_v1", "results.png"),
        "confusion": os.path.join(BASE_DIR, "plate_detector_v1", "confusion_matrix.png"),
        "confusion_norm": os.path.join(BASE_DIR, "plate_detector_v1", "confusion_matrix_normalized.png"),
        "labels": os.path.join(BASE_DIR, "plate_detector_v1", "labels.jpg"),
        "pr_curve": os.path.join(BASE_DIR, "plate_detector_v1", "BoxPR_curve.png"),
        "f1_curve": os.path.join(BASE_DIR, "plate_detector_v1", "BoxF1_curve.png"),
    }
    
    return summary, results_df, args, images


def get_env_status():
    """Retorna informações do ambiente"""
    gpu = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu else platform.processor() or "CPU"
    torch_ver = torch.__version__
    cuda_ver = torch.version.cuda if gpu else "N/A"
    
    return {
        "device": "GPU" if gpu else "CPU",
        "device_name": device_name,
        "torch": torch_ver,
        "cuda": cuda_ver,
        "ultralytics": yolo_version,
        "python": platform.python_version(),
    }


def _gather_test_images():
    """Coleta imagens de teste"""
    if not os.path.exists(IMAGES_DIR):
        return []
    
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
    
    return sorted(images)[:12]


def _draw_boxes(image_np: np.ndarray, boxes_xyxy: np.ndarray, confs: np.ndarray) -> np.ndarray:
    """Desenha bounding boxes nas placas detectadas"""
    out = image_np.copy()
    color = (53, 107, 255)  # Primary color (laranja) em BGR: #ff6b35 -> BGR(53, 107, 255)
    
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)  # Espessura reduzida de 3 para 2
        
        label = f"Plate {confs[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  # Fonte menor
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return out


def yolo_predict(model: YOLO, image: Image.Image, conf: float, iou: float, imgsz: int):
    """Realiza predição com YOLO"""
    img_np = np.array(image.convert("RGB"))
    results = model.predict(img_np, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    return results


def page_home(model, summary, results_df):
    """Página inicial"""
    st.markdown(
        '<div class="main-hero">\
          <div class="plate-icon">ABC1D23</div>\
          <span class="title-gradient">Brazilian License Plate Recognition</span>\
        </div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="subtitle">Sistema ALPR com YOLOv8 para detecção de placas Mercosul</div>', unsafe_allow_html=True)
    
    # Status cards
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "Carregado" if model else "Erro"
        badge_class = "badge-success" if model else "badge-danger"
        st.markdown(f"""
<div class="metric-card">
  <p class="metric-label">Modelo YOLO</p>
  <span class="badge {badge_class}">{status}</span>
</div>
""", unsafe_allow_html=True)
    
    with col2:
        num_images = len(_gather_test_images())
        st.markdown(f"""
<div class="metric-card">
  <p class="metric-label">Imagens de Teste</p>
  <div class="metric-value">{num_images}</div>
</div>
""", unsafe_allow_html=True)
    
    with col3:
        if summary and 'best_model_metrics' in summary:
            map_val = summary['best_model_metrics'].get('mAP50-95(B)', 0)
            map_text = f"{map_val:.1%}"
        else:
            map_text = "N/A"
        st.markdown(f"""
<div class="metric-card">
  <p class="metric-label">mAP@50-95</p>
  <div class="metric-value">{map_text}</div>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Destaques
    if summary and 'best_model_metrics' in summary:
        st.markdown('<h3 style="color: var(--text); font-size: 1.125rem; margin: 1.5rem 0 0.75rem 0;">Model Performance</h3>', unsafe_allow_html=True)
        metrics = summary['best_model_metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        metrics_display = [
            (col1, "Precision", metrics.get('precision(B)', 0), "--primary"),
            (col2, "Recall", metrics.get('recall(B)', 0), "--success"),
            (col3, "mAP@50", metrics.get('mAP50(B)', 0), "--primary"),
            (col4, "mAP@50-95", metrics.get('mAP50-95(B)', 0), "--accent"),
        ]
        
        for col, label, value, color in metrics_display:
            with col:
                st.markdown(f"""
<div class="card" style="text-align: center;">
  <p style="font-size: 0.75rem; color: var(--muted); margin: 0;">{label}</p>
  <p style="font-size: 1.75rem; font-weight: 700; color: var({color}); margin: 0.5rem 0;">{value:.1%}</p>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualizações
    st.markdown('<h3 style="color: var(--text); font-size: 1.125rem; margin: 1.5rem 0 0.75rem 0;">Training Results</h3>', unsafe_allow_html=True)
    _, _, _, images = load_training_data()
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(images["results"]):
            st.image(images["results"], caption="Training Evolution", use_container_width=True)
    with col2:
        if os.path.exists(images["confusion_norm"]):
            st.image(images["confusion_norm"], caption="Confusion Matrix", use_container_width=True)


def page_detect(model):
    """Página de detecção"""
    st.markdown('<h2 style="color: var(--primary); font-size: 1.5rem; margin-bottom: 0.5rem;">License Plate Detector</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: var(--text-secondary); font-size: 0.938rem; margin-bottom: 1.5rem;">Upload an image or select a test example to detect Brazilian license plates</p>', unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not loaded. Please check the weights directory.")
        return
    
    # Presets
    st.markdown('<h3 style="color: var(--text); font-size: 1.125rem; margin-bottom: 0.75rem;">Detection Settings</h3>', unsafe_allow_html=True)
    cols_p = st.columns(3)
    preset = st.session_state.get("preset", "Balanced")
    with cols_p[0]:
        if st.button("Fast", use_container_width=True):
            preset = "Fast"
    with cols_p[1]:
        if st.button("Balanced", use_container_width=True):
            preset = "Balanced"
    with cols_p[2]:
        if st.button("Precise", use_container_width=True):
            preset = "Precise"
    st.session_state["preset"] = preset
    
    # Valores por preset
    if preset == "Fast":
        conf_default, iou_default, size_default = 0.30, 0.45, 640
    elif preset == "Precise":
        conf_default, iou_default, size_default = 0.15, 0.55, 960
    else:
        conf_default, iou_default, size_default = 0.25, 0.50, 768
    
    col1, col2, col3 = st.columns(3)
    with col1:
        conf = st.slider("Confidence", 0.05, 0.95, conf_default, 0.05)
    with col2:
        iou = st.slider("IoU", 0.1, 0.9, iou_default, 0.05)
    with col3:
        imgsz = st.select_slider("Image Size", options=[640, 768, 896, 960, 1024], value=size_default)
    
    # Tabs
    tab_upload, tab_examples = st.tabs(["Upload", "Examples"])
    
    def run_detection(pil_img: Image.Image, key_prefix: str = "single"):
        """Executa detecção e mostra resultados"""
        start = time.time()
        results = yolo_predict(model, pil_img, conf, iou, imgsz)
        latency = (time.time() - start) * 1000
        
        if results:
            r0 = results[0]
            xyxy = r0.boxes.xyxy.cpu().numpy() if r0.boxes is not None else np.empty((0, 4))
            confs = r0.boxes.conf.cpu().numpy() if r0.boxes is not None else np.empty((0,))
            
            img_np = np.array(pil_img.convert("RGB"))
            annotated = _draw_boxes(img_np, xyxy, confs)
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.image(pil_img, caption="Original Image", use_container_width=True)
            with col_res2:
                st.image(annotated, caption=f"Detection Results ({latency:.1f} ms)", use_container_width=True)
            
            # Detalhes
            st.markdown('<h3 style="color: var(--text); font-size: 1.125rem; margin: 1.5rem 0 0.75rem 0;">Detection Details</h3>', unsafe_allow_html=True)
            if len(xyxy) > 0:
                st.markdown(f"""
<div style="background: rgba(82,183,136,0.08); border-left: 3px solid var(--success); padding: 0.875rem; border-radius: 6px; margin-bottom: 1rem;">
  <span style="color: var(--success); font-weight: 600;">✓ {len(xyxy)} plate(s) detected</span>
</div>
""", unsafe_allow_html=True)
                for i, (box, conf_val) in enumerate(zip(xyxy, confs)):
                    x1, y1, x2, y2 = box.astype(int)
                    st.markdown(f"""
<div class="card">
  <b>Plate {i+1}</b><br>
  Confidence: <span class="badge badge-primary">{conf_val:.1%}</span><br>
  Position: ({x1}, {y1}) → ({x2}, {y2})
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown("""
<div style="background: rgba(255,165,0,0.08); border-left: 3px solid var(--warning); padding: 0.875rem; border-radius: 6px;">
  <span style="color: var(--warning); font-weight: 600;">⚠ No plates detected in this image</span>
</div>
""", unsafe_allow_html=True)
            
            # Download
            buf = io.BytesIO()
            Image.fromarray(annotated).save(buf, format="PNG")
            st.download_button(
                "Download Annotated Image",
                data=buf.getvalue(),
                file_name=f"detection_{key_prefix}.png",
                mime="image/png",
                use_container_width=True
            )
    
    with tab_upload:
        uploaded = st.file_uploader("Select an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            image = Image.open(uploaded)
            run_detection(image, key_prefix="upload")
    
    with tab_examples:
        examples = _gather_test_images()
        if examples:
            captions = [f"Image {i+1}" for i in range(len(examples))]
            
            # Container com scroll horizontal
            st.markdown('<p style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0.75rem;">Choose a test image:</p>', unsafe_allow_html=True)
            
            selected_idx = image_select(
                "",
                images=[Image.open(p) for p in examples],
                captions=captions,
                use_container_width=False,
                return_value="index"
            )
            
            if selected_idx is not None:
                if st.button("Detect Plates", type="primary", use_container_width=True):
                    img = Image.open(examples[selected_idx])
                    run_detection(img, key_prefix="example")
        else:
            st.markdown("""
<div style="background: rgba(255,107,53,0.08); border-left: 3px solid var(--primary); padding: 0.875rem; border-radius: 6px;">
  <span style="color: var(--primary); font-weight: 600;">ℹ No test images found in the images/ directory</span>
</div>
""", unsafe_allow_html=True)


def page_training():
    """Página de análise de treinamento"""
    st.markdown('<h2 style="color: var(--primary); font-size: 1.5rem; margin-bottom: 1rem;">Training Analytics</h2>', unsafe_allow_html=True)
    
    summary, results_df, args, images = load_training_data()
    
    # Matriz de confusão e gráficos
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(images["confusion"]):
            st.image(images["confusion"], caption="Confusion Matrix", use_container_width=True)
    with col2:
        if os.path.exists(images["pr_curve"]):
            st.image(images["pr_curve"], caption="Precision-Recall Curve", use_container_width=True)
    
    st.markdown("---")
    
    # Gráficos de evolução
    if results_df is not None:
        st.markdown('<h3 style="color: var(--text); font-size: 1.125rem; margin: 1.5rem 0 0.75rem 0;">Training Evolution</h3>', unsafe_allow_html=True)
        
        metric_cols = [
            c for c in results_df.columns
            if any(k in c for k in ["precision", "recall", "mAP50", "box_loss", "cls_loss"])
        ]
        
        if metric_cols:
            # Gráfico de métricas
            fig = go.Figure()
            epochs = results_df.get("epoch", pd.Series(range(len(results_df))))
            
            for c in metric_cols:
                if "metrics" in c:
                    fig.add_trace(go.Scatter(
                        x=epochs, 
                        y=results_df[c], 
                        mode="lines+markers", 
                        name=c.replace("metrics/", "").replace("(B)", "")
                    ))
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e4e4e7',
                xaxis_title="Epoch",
                yaxis_title="Value",
                legend_title="Metric",
                height=500,
                colorway=['#ff6b35', '#52b788', '#f7931e', '#e63946']  # Paleta quente
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
<div style="background: rgba(255,165,0,0.08); border-left: 3px solid var(--warning); padding: 0.875rem; border-radius: 6px;">
  <span style="color: var(--warning); font-weight: 600;">⚠ Training results not found</span>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Hiperparâmetros
    if args:
        st.markdown('<h3 style="color: var(--text); font-size: 1.125rem; margin-bottom: 0.75rem;">Hyperparameters</h3>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
<div class="card">
  <h4>Basic</h4>
  <p>Epochs: <span class="badge badge-primary">{}</span></p>
  <p>Batch: <span class="badge badge-primary">{}</span></p>
  <p>Image Size: <span class="badge badge-primary">{}</span></p>
</div>
""".format(args.get('epochs', 'N/A'), args.get('batch', 'N/A'), args.get('imgsz', 'N/A')), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
<div class="card">
  <h4>Optimization</h4>
  <p>LR: <span class="badge badge-success">{}</span></p>
  <p>Momentum: <span class="badge badge-success">{}</span></p>
  <p>Weight Decay: <span class="badge badge-success">{}</span></p>
</div>
""".format(args.get('lr0', 'N/A'), args.get('momentum', 'N/A'), args.get('weight_decay', 'N/A')), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
<div class="card">
  <h4>Augmentation</h4>
  <p>Flip: <span class="badge badge-primary">{}</span></p>
  <p>Mosaic: <span class="badge badge-primary">{}</span></p>
  <p>MixUp: <span class="badge badge-primary">{}</span></p>
</div>
""".format(args.get('fliplr', 'N/A'), args.get('mosaic', 'N/A'), args.get('mixup', 'N/A')), unsafe_allow_html=True)


def page_about():
    """Página sobre"""
    st.markdown('<h2 style="color: var(--primary); font-size: 1.5rem; margin-bottom: 1rem;">About</h2>', unsafe_allow_html=True)
    
    st.markdown("""
<div class="card">
<h3>Brazilian License Plate Recognition System</h3>
<p>Professional ALPR system using YOLOv8 for detecting Brazilian Mercosul standard license plates.</p>

<h4>Features</h4>
<ul>
  <li>Real-time license plate detection</li>
  <li>Support for Mercosul standard plates</li>
  <li>Adjustable confidence and IoU thresholds</li>
  <li>Multiple detection presets (Fast, Balanced, Precise)</li>
  <li>Training analytics and performance metrics</li>
  <li>Export annotated images</li>
</ul>

<h4>Technical Stack</h4>
<ul>
  <li><b>Model:</b> YOLOv8s</li>
  <li><b>Framework:</b> PyTorch + Ultralytics</li>
  <li><b>Interface:</b> Streamlit</li>
  <li><b>Visualization:</b> Plotly</li>
</ul>

<p><b>Author:</b> <a href="https://github.com/sidnei-almeida" target="_blank">Sidnei Almeida</a></p>
</div>
""", unsafe_allow_html=True)


def main():
    """Função principal"""
    with st.sidebar:
        st.markdown("<h3 style='color:#ff6b35; margin-bottom: 0.875rem; font-size: 1.125rem;'>Navigation</h3>", unsafe_allow_html=True)
        
        selected = option_menu(
            menu_title=None,
            options=["Home", "Detector", "Analytics", "About"],
            icons=["house", "search", "graph-up", "info-circle"],
            default_index=0,
            styles={
                "container": {"padding": "0", "background": "transparent"},
                "icon": {"color": "#ff6b35", "font-size": "14px"},
                "nav-link": {
                    "color": "#e4e4e7",
                    "font-size": "0.875rem",
                    "padding": "0.625rem 0.875rem",
                    "border-radius": "6px",
                    "margin": "0.2rem 0"
                },
                "nav-link-selected": {
                    "background": "linear-gradient(135deg, rgba(255,107,53,0.12), rgba(247,147,30,0.12))",
                    "color": "#ff6b35",
                    "border-left": "3px solid #ff6b35",
                    "font-weight": "600"
                },
            },
        )
        
        # Status do ambiente
        st.markdown("---")
        st.markdown("<h4 style='margin-bottom:0.625rem; font-size: 0.938rem;'>System Info</h4>", unsafe_allow_html=True)
        env = get_env_status()
        
        device_badge = "badge-success" if env['device'] == "GPU" else "badge-primary"
        st.markdown(f"""
<div class="card" style="padding: 0.875rem;">
  <p style="margin: 0.2rem 0; font-size: 0.813rem;"><b>Device:</b> <span class="badge {device_badge}">{env['device']}</span></p>
  <p style="margin: 0.2rem 0; font-size: 0.688rem; color: var(--muted);">{env['device_name'][:30]}</p>
  <hr style="margin: 0.625rem 0;">
  <p style="margin: 0.2rem 0; font-size: 0.75rem;"><b>Python:</b> <span style="color: var(--text-secondary);">{env['python']}</span></p>
  <p style="margin: 0.2rem 0; font-size: 0.75rem;"><b>PyTorch:</b> <span style="color: var(--text-secondary);">{env['torch']}</span></p>
  <p style="margin: 0.2rem 0; font-size: 0.75rem;"><b>YOLO:</b> <span style="color: var(--text-secondary);">{env['ultralytics']}</span></p>
</div>
""", unsafe_allow_html=True)
    
    # Carrega recursos
    model = load_model()
    summary, results_df, _, _ = load_training_data()
    
    # Roteamento de páginas
    if selected == "Home":
        page_home(model, summary, results_df)
    elif selected == "Detector":
        page_detect(model)
    elif selected == "Analytics":
        page_training()
    else:
        page_about()


if __name__ == "__main__":
    main()
