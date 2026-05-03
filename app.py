"""Streamlit demo GUI for Pokemon classification (dark Pokemon theme)."""

import os
import sys
from pathlib import Path

import streamlit as st
from PIL import Image
import torch

# Add src/ to path so imports work
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import get_val_transform, load_class_names
from model import build_model


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="PokéClassifier", page_icon="🔴", layout="centered")

# ── Custom CSS (dark Pokemon theme) ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@700;900&family=Space+Mono&display=swap');

.stApp { background-color: #1a1a2e; color: #e8f0fe; font-family: 'Nunito', sans-serif; }

/* Model selector radio → tab style */
div[data-testid="stRadio"] > div { display: flex; gap: 8px; flex-wrap: wrap; }
div[data-testid="stRadio"] label {
    background: #16213e;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px; padding: 8px 16px; cursor: pointer;
    font-weight: 800; font-size: 13px; transition: all .2s;
    color: #e8f0fe;
}
div[data-testid="stRadio"] label:has(input:checked) {
    border-color: #F7C948; color: #F7C948;
    background: rgba(247,201,72,0.08);
    border-top: 2px solid #F7C948;
}

/* Upload zone */
div[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(255,255,255,0.2) !important;
    border-radius: 14px; background: rgba(255,255,255,0.02); padding: 8px;
}

/* Classify button */
div[data-testid="stButton"] button {
    background: #E3350D !important; color: white !important;
    border: none; border-radius: 12px; padding: 12px;
    font-family: 'Nunito', sans-serif; font-weight: 900;
    font-size: 15px; width: 100%; transition: all .2s;
}
div[data-testid="stButton"] button:hover { background: #c52a0a !important; }

/* Progress bar */
div[data-testid="stProgress"] > div {
    background: rgba(255,255,255,0.08); border-radius: 20px;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.08); }

/* Caption */
.stCaption { color: #8899bb !important; }

/* Metric */
[data-testid="metric-container"] { background: #16213e; border-radius: 12px; padding: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Model map ────────────────────────────────────────────────────────────────
MODEL_MAP = {
    "ResNet-50 (head only)":  ("EXP-01", "resnet50",       "head_only",    True),
    "ResNet-50 (full FT)":    ("EXP-02", "resnet50",       "full",         True),
    "EfficientNet-B0":        ("EXP-03", "efficientnet_b0","last2_blocks",  True),
    "ViT-B/16":               ("EXP-04", "vit_b16",        "full",         True),
    "ResNet-50 (scratch)":    ("EXP-05", "resnet50",       "full",         False),
}


@st.cache_resource(show_spinner=False)
def load_model_cached(model_label: str, num_classes: int = 150):
    """Load model weights (cached after first call)."""
    exp_id, backbone, finetune_scope, pretrained = MODEL_MAP[model_label]
    ckpt_path = f"models/{exp_id}_{backbone}.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(backbone, num_classes=num_classes,
                        pretrained=False, finetune_scope="full")

    if not os.path.exists(ckpt_path):
        return None, device, f"모델 파일을 찾을 수 없습니다: `{ckpt_path}`\n먼저 `python src/experiment.py`로 학습을 실행해주세요."

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.eval().to(device), device, None


def predict(model, img: Image.Image, device, class_names, topk: int = 5):
    """Return Top-k (name, probability) list."""
    transform = get_val_transform()
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    topk_probs, topk_ids = probs.topk(topk)
    return [(class_names[i.item()], p.item()) for i, p in zip(topk_ids, topk_probs)]


# ── Header ───────────────────────────────────────────────────────────────────
col_icon, col_title = st.columns([1, 6])
with col_icon:
    pokeball_path = "assets/pokeball.png"
    if os.path.exists(pokeball_path):
        st.image(pokeball_path, width=52)
    else:
        st.markdown("<span style='font-size:36px'>🔴</span>", unsafe_allow_html=True)
with col_title:
    st.markdown(
        "<h2 style='color:#e8f0fe;font-family:Nunito;margin-bottom:0'>PokéClassifier</h2>",
        unsafe_allow_html=True,
    )
    st.caption("Transfer Learning Demo · 150 classes")

st.divider()

# ── Model selector ───────────────────────────────────────────────────────────
model_label = st.radio(
    "모델 선택",
    list(MODEL_MAP.keys()),
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()

# ── Image upload ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "포켓몬 이미지 업로드 (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    col_prev, _ = st.columns([1, 3])
    with col_prev:
        st.image(img, width=160, caption="업로드된 이미지")

st.divider()

# ── Classify button ───────────────────────────────────────────────────────────
if st.button("🔴  분류 시작", use_container_width=True):
    if uploaded is None:
        st.warning("이미지를 먼저 업로드해주세요.")
    else:
        with st.spinner("모델 로딩 및 분석 중..."):
            # Load class names
            data_dir = "data/PokemonData"
            if os.path.exists(data_dir):
                class_names = load_class_names(data_dir)
            else:
                st.error(f"데이터 폴더를 찾을 수 없습니다: `{data_dir}`")
                st.stop()

            model, device, err = load_model_cached(model_label, num_classes=len(class_names))
            if err:
                st.error(err)
                st.stop()

            img = Image.open(uploaded).convert("RGB")
            top5 = predict(model, img, device, class_names, topk=5)

        # ── Results ──────────────────────────────────────────────────────────
        st.divider()
        winner_name, winner_conf = top5[0]

        st.markdown(
            f"<h2 style='color:#F7C948;font-family:Nunito;margin:0'>{winner_name}</h2>"
            f"<p style='color:#8899bb;font-family:Space Mono;font-size:12px'>"
            f"{winner_conf * 100:.1f}% confidence · 모델: {model_label}</p>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Top-5 bar chart
        for i, (name, prob) in enumerate(top5):
            col_rank, col_name, col_bar, col_pct = st.columns([0.4, 1.8, 4, 1])
            col_rank.caption(str(i + 1))
            if i == 0:
                col_name.markdown(
                    f"<span style='color:#F7C948;font-weight:800'>{name}</span>",
                    unsafe_allow_html=True,
                )
            else:
                col_name.markdown(name)
            col_bar.progress(float(prob))
            col_pct.caption(f"{prob * 100:.1f}%")
