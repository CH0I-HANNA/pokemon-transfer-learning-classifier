"""Streamlit demo GUI for Pokemon classification."""

import os
import sys
from pathlib import Path

import streamlit as st
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import get_val_transform, load_class_names
from model import build_model


st.set_page_config(page_title="PokéClassifier", page_icon="🔴", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700;900&family=Space+Mono:wght@400;700&display=swap');

/* ── 베이스: 포켓몬 옐로우 배경 ── */
.stApp {
    background: #FFF8DC;
    color: #1a1a2e;
    font-family: 'Nunito', sans-serif;
}

/* ── 헤더: 포켓볼 레드 ── */
.pk-header {
    background: linear-gradient(135deg, #E3350D 0%, #cc2200 50%, #aa1a00 100%);
    border-radius: 20px 20px 0 0;
    padding: 28px 36px;
    margin-bottom: 0px;
    display: flex;
    align-items: center;
    gap: 20px;
    border: 2px solid #1a1a2e;
    border-bottom: 6px solid #1a1a2e;
}
.pk-title {
    font-size: 36px;
    font-weight: 900;
    letter-spacing: -1px;
    color: #F7C948;
    text-shadow: 2px 2px 0px rgba(0,0,0,0.3);
    line-height: 1.1;
}
.pk-subtitle {
    color: rgba(255,255,255,0.75);
    font-size: 13px;
    font-family: 'Space Mono', monospace;
    margin-top: 6px;
}

/* ── 모델 선택 바: 흰색 (포켓볼 하단) ── */
.model-bar {
    background: #ffffff;
    border-radius: 0 0 16px 16px;
    padding: 16px 28px;
    margin-bottom: 24px;
    border: 2px solid #1a1a2e;
    border-top: none;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    box-shadow: 4px 4px 0px #1a1a2e;
}
.model-bar-label {
    font-size: 12px;
    font-weight: 900;
    color: #E3350D;
    letter-spacing: 1px;
    text-transform: uppercase;
    white-space: nowrap;
}

/* ── 라디오 버튼: 포켓몬 타입 뱃지 ── */
div[data-testid="stRadio"] > div {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
div[data-testid="stRadio"] label {
    background: #fff;
    border: 2px solid #1a1a2e;
    border-radius: 20px;
    padding: 6px 16px;
    cursor: pointer;
    font-weight: 800;
    font-size: 12px;
    transition: all .15s;
    color: #1a1a2e;
    box-shadow: 2px 2px 0px #1a1a2e;
}
div[data-testid="stRadio"] label:hover {
    background: #F7C948;
    transform: translate(-1px, -1px);
    box-shadow: 3px 3px 0px #1a1a2e;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: #E3350D;
    color: white;
    border-color: #1a1a2e;
    box-shadow: 2px 2px 0px #1a1a2e;
}

/* ── 카드 ── */
.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 16px;
    border: 2px solid #1a1a2e;
    box-shadow: 4px 4px 0px #1a1a2e;
}
.card-label {
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 2px;
    color: #3B5FA0;
    text-transform: uppercase;
    margin-bottom: 14px;
}

/* ── 업로드 존: 파란 점선 ── */
div[data-testid="stFileUploader"] {
    border: 2px dashed #3B5FA0 !important;
    border-radius: 12px;
    background: #EFF6FF;
    padding: 8px;
}

/* ── 분류 버튼 ── */
div[data-testid="stButton"] button {
    background: #E3350D !important;
    color: white !important;
    border: 2px solid #1a1a2e !important;
    border-radius: 12px;
    padding: 14px;
    font-family: 'Nunito', sans-serif;
    font-weight: 900;
    font-size: 15px;
    width: 100%;
    transition: all .15s;
    box-shadow: 4px 4px 0px #1a1a2e;
    letter-spacing: 0.5px;
}
div[data-testid="stButton"] button:hover {
    transform: translate(-2px, -2px);
    box-shadow: 6px 6px 0px #1a1a2e;
}
div[data-testid="stButton"] button:active {
    transform: translate(2px, 2px);
    box-shadow: 2px 2px 0px #1a1a2e;
}

/* ── 결과 winner 카드: 옐로우 ── */
.winner-card {
    background: linear-gradient(135deg, #FFF8DC, #FFF3A3);
    border: 2px solid #1a1a2e;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 16px;
    box-shadow: 4px 4px 0px #1a1a2e;
}
.winner-name {
    font-size: 32px;
    font-weight: 900;
    color: #1a1a2e;
    letter-spacing: -1px;
}
.winner-conf {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: #E3350D;
    margin-top: 4px;
    font-weight: 700;
}

/* ── Top-5 바 ── */
.top5-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 6px;
    background: #FFF8DC;
    border: 1.5px solid #1a1a2e;
}
.rank-num {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #3B5FA0;
    width: 20px;
    flex-shrink: 0;
    font-weight: 700;
}
.poke-name {
    font-weight: 800;
    font-size: 14px;
    flex: 1;
    color: #1a1a2e;
}
.bar-track {
    flex: 2;
    height: 8px;
    background: #e2e8f0;
    border-radius: 999px;
    overflow: hidden;
    border: 1px solid #cbd5e1;
}
.bar-fill {
    height: 100%;
    border-radius: 999px;
}
.conf-pct {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    width: 48px;
    text-align: right;
    flex-shrink: 0;
    font-weight: 700;
}

/* ── 안내 박스 ── */
.guide-box {
    text-align: center;
    padding: 60px 20px;
}

hr { border-color: #1a1a2e; }
.stCaption { color: #3B5FA0 !important; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


MODEL_MAP = {
    "ResNet-50 (Head only)":  ("EXP-01", "resnet50",        "head_only",    True,  67.7),
    "ResNet-50 (Full FT)":    ("EXP-02", "resnet50",        "full",         True,  96.2),
    "EfficientNet-B0":        ("EXP-03", "efficientnet_b0", "last2_blocks", True,  84.3),
    "ViT-B/16":               ("EXP-04", "vit_b16",         "full",         True,  93.8),
    "ResNet-50 (Scratch)":    ("EXP-05", "resnet50",        "full",         False, 28.7),
}

RANK_ICONS = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]


def bar_color(prob: float) -> str:
    if prob >= 0.7:  return "#E3350D"
    if prob >= 0.4:  return "#F7C948"
    if prob >= 0.15: return "#3B5FA0"
    return "#94a3b8"


def conf_color(prob: float) -> str:
    if prob >= 0.7:  return "#E3350D"
    if prob >= 0.4:  return "#c49a00"
    if prob >= 0.15: return "#3B5FA0"
    return "#94a3b8"


@st.cache_resource(show_spinner=False)
def load_model_cached(model_label: str, num_classes: int = 150):
    exp_id, backbone, finetune_scope, pretrained, _ = MODEL_MAP[model_label]
    ckpt_path = f"models/{exp_id}_{backbone}.pth"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = build_model(backbone, num_classes=num_classes,
                        pretrained=False, finetune_scope="full")

    if not os.path.exists(ckpt_path):
        return None, device, f"모델 파일 없음: `{ckpt_path}`"

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.eval().to(device), device, None


def predict(model, img: Image.Image, device, class_names, topk: int = 5):
    transform = get_val_transform()
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    topk_probs, topk_ids = probs.topk(topk)
    return [(class_names[i.item()], p.item()) for i, p in zip(topk_ids, topk_probs)]


# ── 헤더 ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pk-header">
  <span style="font-size:48px;line-height:1"></span>
  <div>
    <div class="pk-title">PokéClassifier</div>
    <div class="pk-subtitle">포켓몬 이미지 분류기 · Transfer Learning · 150 Classes · 5 Models</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── 모델 선택 (헤더 바로 아래) ────────────────────────────────────────────────
st.markdown('<div class="model-bar"><span class="model-bar-label">모델 선택 ▶</span>', unsafe_allow_html=True)
model_label = st.radio(
    "model",
    list(MODEL_MAP.keys()),
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)

exp_id, backbone, ft_scope, pretrained, acc = MODEL_MAP[model_label]
pretrained_str = "✅ ImageNet 사전학습" if pretrained else "❌ 사전학습 없음"
st.caption(f"선택된 모델 · Test Accuracy: **{acc}%** · {pretrained_str}")

st.markdown("<br>", unsafe_allow_html=True)

# ── 두 컬럼 레이아웃 ───────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:

    img_placeholder = st.empty()

    uploaded = st.file_uploader(
        "이미지 업로드",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded:
        img_preview = Image.open(uploaded).convert("RGB")
        img_placeholder.image(img_preview, use_container_width=True)

    classify = st.button("분류 시작", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    if classify:
        if uploaded is None:
            st.warning("⚠️ 이미지를 먼저 업로드해주세요!")
        else:
            with st.spinner("분석 중..."):
                data_dir = "data/PokemonData"
                if not os.path.exists(data_dir):
                    st.error(f"데이터 폴더를 찾을 수 없습니다: `{data_dir}`")
                    st.stop()

                class_names = load_class_names(data_dir)
                model, device, err = load_model_cached(model_label, num_classes=len(class_names))
                if err:
                    st.error(err)
                    st.stop()

                img = Image.open(uploaded).convert("RGB")
                top5 = predict(model, img, device, class_names, topk=5)

            winner_name, winner_conf = top5[0]

            st.markdown(f"""
            <div class="winner-card">
              <div style="font-size:11px;letter-spacing:2px;color:#3B5FA0;font-weight:900;margin-bottom:8px">🎯 분류 결과</div>
              <div class="winner-name">🥇 {winner_name}</div>
              <div class="winner-conf">{winner_conf * 100:.1f}% confidence · {model_label}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="card"><div class="card-label">📊 Top-5 예측 결과</div>', unsafe_allow_html=True)

            for i, (name, prob) in enumerate(top5):
                pct = prob * 100
                color = conf_color(prob)
                bg = bar_color(prob)
                bar_w = max(int(prob * 100), 2)
                name_style = f"color:{color};font-weight:900" if i == 0 else "color:#1a1a2e"
                st.markdown(f"""
                <div class="top5-row">
                  <div class="rank-num">{RANK_ICONS[i]}</div>
                  <div class="poke-name" style="{name_style}">{name}</div>
                  <div class="bar-track">
                    <div class="bar-fill" style="width:{bar_w}%;background:{bg}"></div>
                  </div>
                  <div class="conf-pct" style="color:{color}">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
            st.caption(f"디바이스: {str(device).upper()} · 클래스 수: {len(class_names)}")

    else:
        st.markdown("""
        <div class="card" style="min-height:460px;display:flex;align-items:center;justify-content:center">
          <div class="guide-box">
            <div style="font-size:52px;margin-bottom:16px">🎯</div>
            <div style="color:#1a1a2e;font-size:16px;font-weight:900;margin-bottom:16px">사용 방법</div>
            <div style="color:#3B5FA0;line-height:2.4;font-size:14px">
              위에서 <strong style="color:#1a1a2e">모델을 선택</strong>하세요<br>
              포켓몬 <strong style="color:#1a1a2e">이미지를 업로드</strong>하세요<br>
              <strong style="color:#E3350D">분류 시작</strong> 버튼을 누르세요
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)