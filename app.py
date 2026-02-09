from pathlib import Path
import sys
import numpy as np
import streamlit as st
import torch

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.append(str(ROOT))

from src.config import METADATA_CSV
from src.data.loader import load_mat_signal, list_mat_keys
from src.data.metadata import load_metadata, material_label_map
from src.models.cnn1d import CNN1D
from src.processing.preprocess import preprocess_signal
from src.processing.music import music_spectrum

st.set_page_config(page_title="TWI Hidden Target Detection", layout="wide")

st.title("Through‑Wall Imaging: Hidden Target Detection")

st.sidebar.header("Model")
model_path = st.sidebar.text_input("Model path", value=str(ROOT / "outputs" / "models" / "best_model_nb.pt"))
model_upload = st.sidebar.file_uploader("Upload model (.pt)", type=["pt"])

st.sidebar.header("Input")
uploaded = st.sidebar.file_uploader("Upload .mat file", type=["mat"])

st.sidebar.header("Signal Settings")
key_override = st.sidebar.text_input("MAT key (optional)", value="")
num_sources = st.sidebar.slider("MUSIC sources", 1, 3, 1)
fft_bins = st.sidebar.selectbox("MUSIC FFT bins", [128, 256, 512], index=1)

@st.cache_resource
def load_model_from_bytes(model_bytes: bytes, in_channels: int, num_classes: int):
    model = CNN1D(in_channels=in_channels, num_classes=num_classes)
    state = torch.load(model_bytes, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource
def load_model_from_path(path: str, in_channels: int, num_classes: int):
    model = CNN1D(in_channels=in_channels, num_classes=num_classes)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

if uploaded is None:
    st.info("Upload a .mat file to visualize and detect targets.")
    st.stop()

mat_bytes = uploaded.getvalue()
tmp_path = ROOT / "outputs" / "logs" / "_uploaded.mat"
tmp_path.parent.mkdir(parents=True, exist_ok=True)
tmp_path.write_bytes(mat_bytes)

keys = list_mat_keys(tmp_path)
if key_override:
    mat_key = key_override
else:
    mat_key = "dataMeasured1" if "dataMeasured1" in keys else keys[0]

signal = load_mat_signal(tmp_path, key=mat_key)
raw = signal.real if np.iscomplexobj(signal) else signal
clean = preprocess_signal(raw)

st.subheader("Raw Signal")
st.line_chart(raw)

st.subheader("Preprocessed Signal")
st.line_chart(clean)

spectrum = music_spectrum(clean, num_sources=num_sources, n_fft=fft_bins)
peak_idx = int(np.argmax(spectrum))

st.subheader("MUSIC Spectrum (Hidden Target Localization)")
st.line_chart(spectrum)
st.caption(f"Peak bin: {peak_idx}")

metadata = load_metadata(METADATA_CSV)
label_map = material_label_map(metadata[metadata["material"].notna()])
inv_label = {v: k for k, v in label_map.items()}

in_channels = 2 if np.iscomplexobj(clean) else 1
model = None
try:
    if model_upload is not None:
        model = load_model_from_bytes(model_upload.getvalue(), in_channels, num_classes=len(label_map))
    else:
        model = load_model_from_path(model_path, in_channels, num_classes=len(label_map))
except FileNotFoundError:
    st.warning("Model file not found. Upload a .pt model or update the model path.")
except Exception as exc:
    st.error(f"Failed to load model: {exc}")

if model is not None:
    if np.iscomplexobj(clean):
        x = np.stack([clean.real, clean.imag], axis=0)
    else:
        x = np.expand_dims(clean, axis=0)
    logits = model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
    probs = torch.softmax(logits, dim=1).detach().numpy().squeeze()
    pred = inv_label[int(np.argmax(probs))]

    st.subheader("Predicted Target")
    st.write(pred)
    st.bar_chart({inv_label[i]: float(probs[i]) for i in range(len(probs))})
