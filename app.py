import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import io
import tensorflow as tf
from PIL import Image, ImageOps, ImageCms


st.set_page_config(
    page_title="Skinlite",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

MODEL_CANDIDATES = [
    ARTIFACTS_DIR / "model_full.keras",
    ARTIFACTS_DIR / "model_full.h5",
    CHECKPOINTS_DIR / "last_full_model.keras",
]
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
PREPROCESSING_CONFIG_PATH = ARTIFACTS_DIR / "preprocessing_config.json"
EVAL_METRICS_PATH = ARTIFACTS_DIR / "eval_metrics.json"

DEFAULT_CLASS_DISPLAY = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}

ACCENT_COLORS = {
    "primary": "#111111",
    "surface": "#f6f6f6",
    "surface_alt": "#ffffff",
    "ink": "#111111",
    "muted": "#3f3f3f",
    "success": "#2d8a3d",
    "border": "#d9d9d9",
}


def inject_custom_css():
    st.markdown(
        f"""
        <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        h1, h2, h3 {{
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }}
        .hero-card {{
            margin-bottom: 1rem;
        }}
        .hero-title {{
            font-size: 1.8rem;
            font-weight: 600;
            margin: 0.5rem 0;
        }}
        .section-card {{
            margin-bottom: 0.5rem;
        }}
        .status-pill {{
            display: inline-block;
            padding: 0.3rem 0.6rem;
            background: #e8f5e9;
            color: {ACCENT_COLORS["success"]};
            font-size: 0.85rem;
            margin-bottom: 0.5rem;
        }}
        .compact-note {{
            color: {ACCENT_COLORS["muted"]};
            font-size: 0.85rem;
            margin: 0.3rem 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero-card">
          <div class="hero-title">HAM10000 Skin Lesion Classifier</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def open_card():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)


def close_card():
    st.markdown("</div>", unsafe_allow_html=True)


def find_first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def enhance_image(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2RGB)


def normalize_map(explanation_map: np.ndarray) -> np.ndarray:
    explanation_map = np.nan_to_num(
        explanation_map.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    explanation_map -= explanation_map.min()
    max_value = explanation_map.max()
    if max_value <= 0:
        return np.zeros_like(explanation_map, dtype=np.float32)
    return explanation_map / (max_value + 1e-8)


def overlay_explanation(display_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = cv2.resize(
        heatmap.astype(np.float32),
        (display_image.shape[1], display_image.shape[0]),
    )
    colored = plt.get_cmap("inferno")(normalize_map(heatmap))[..., :3]
    return np.clip((1.0 - alpha) * display_image + alpha * colored, 0.0, 1.0)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_class_payload(class_payload) -> Tuple[List[str], Dict[str, str]]:
    if isinstance(class_payload, dict):
        class_names = class_payload.get("class_names", [])
        display_names = class_payload.get("display_names", DEFAULT_CLASS_DISPLAY)
    else:
        class_names = class_payload
        display_names = DEFAULT_CLASS_DISPLAY
    return class_names, display_names


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model_path = find_first_existing(MODEL_CANDIDATES)
    if model_path is None:
        searched = "\n".join(str(path) for path in MODEL_CANDIDATES)
        raise FileNotFoundError(f"No model file found. Searched:\n{searched}")
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Missing class names file: {CLASS_NAMES_PATH}")
    if not PREPROCESSING_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing preprocessing config file: {PREPROCESSING_CONFIG_PATH}")

    model = tf.keras.models.load_model(model_path, compile=False)
    class_payload = load_json(CLASS_NAMES_PATH)
    preprocessing_config = load_json(PREPROCESSING_CONFIG_PATH)
    class_names, display_names = parse_class_payload(class_payload)

    if not class_names:
        raise ValueError("`class_names.json` did not contain any classes.")

    return model, class_names, display_names, preprocessing_config, model_path


def get_backbone(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
            return layer
    raise ValueError("Could not locate EfficientNet backbone in the loaded model.")


@st.cache_resource(show_spinner=False)
def build_gradcam_components():
    model, _, _, _, _ = load_artifacts()
    backbone = get_backbone(model)
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
    )

    last_conv_name = None
    for layer in reversed(backbone.layers):
        if isinstance(layer, conv_types):
            last_conv_name = layer.name
            break

    if last_conv_name is None:
        raise ValueError("No convolutional layer found for Grad-CAM.")

    conv_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=backbone.get_layer(last_conv_name).output,
        name="streamlit_gradcam_extractor",
    )
    post_layers = [
        layer
        for layer in model.layers
        if layer.name != backbone.name and not isinstance(layer, tf.keras.layers.InputLayer)
    ]
    return conv_model, post_layers, last_conv_name


def preprocess_uploaded_image(image: Image.Image, config: dict):
    image_size = int(config.get("image_size", 224))
    mean = np.array(config.get("normalization_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.array(config.get("normalization_std", [0.229, 0.224, 0.225]), dtype=np.float32)

    rgb_image = np.array(image.convert("RGB"))
    enhanced = enhance_image(rgb_image)
    resized = cv2.resize(enhanced, (image_size, image_size))

    display_image = resized.astype(np.float32) / 255.0
    normalized = (display_image - mean) / std

    return display_image, normalized.astype(np.float32)


def predict_image(image_tensor: np.ndarray):
    model, class_names, display_names, _, _ = load_artifacts()
    probs = model.predict(np.expand_dims(image_tensor, axis=0), verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_code = class_names[pred_idx]
    pred_name = display_names.get(pred_code, pred_code)
    return probs, pred_idx, pred_code, pred_name


def compute_gradcam(image_tensor: np.ndarray, class_index: Optional[int] = None):
    conv_model, post_layers, last_conv_name = build_gradcam_components()
    inputs = tf.convert_to_tensor(np.expand_dims(image_tensor, axis=0), dtype=tf.float32)
    conv_output = conv_model(inputs, training=False)
    conv_variable = tf.Variable(conv_output, trainable=True, dtype=tf.float32)

    with tf.GradientTape() as tape:
        x = conv_variable
        for layer in post_layers:
            x = layer(x, training=False)
        probs_tensor = x
        if class_index is None:
            class_index = int(tf.argmax(probs_tensor[0]).numpy())
        target_score = probs_tensor[:, class_index]

    gradients = tape.gradient(target_score, conv_variable)
    if gradients is None:
        raise ValueError(f"Grad-CAM gradients are None for layer {last_conv_name}.")

    weights = tf.reduce_mean(gradients, axis=(1, 2))
    cam = tf.reduce_sum(conv_variable[0] * weights[0][None, None, :], axis=-1)
    cam = tf.nn.relu(cam).numpy()
    return normalize_map(cam), last_conv_name


def make_probability_table(
    probs: np.ndarray,
    class_names: List[str],
    display_names: Dict[str, str],
) -> pd.DataFrame:
    rows = []
    for idx, class_code in enumerate(class_names):
        rows.append(
            {
                "class_code": class_code,
                "class_name": display_names.get(class_code, class_code),
                "probability": float(probs[idx]),
                "confidence_pct": round(float(probs[idx]) * 100.0, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)


def render_artifact_status():
    status_rows = [
        {"artifact": "artifacts/model_full.keras", "present": (ARTIFACTS_DIR / "model_full.keras").exists()},
        {"artifact": "artifacts/model_full.h5", "present": (ARTIFACTS_DIR / "model_full.h5").exists()},
        {"artifact": "checkpoints/last_full_model.keras", "present": (CHECKPOINTS_DIR / "last_full_model.keras").exists()},
        {"artifact": "artifacts/class_names.json", "present": CLASS_NAMES_PATH.exists()},
        {"artifact": "artifacts/preprocessing_config.json", "present": PREPROCESSING_CONFIG_PATH.exists()},
        {"artifact": "artifacts/eval_metrics.json", "present": EVAL_METRICS_PATH.exists()},
    ]
    st.dataframe(pd.DataFrame(status_rows), hide_index=True)


def render_metrics_summary():
    if not EVAL_METRICS_PATH.exists():
        return

    try:
        metrics_payload = load_json(EVAL_METRICS_PATH)
    except Exception:
        return

    metric_labels = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1-score"),
    ]
    available = [(key, label) for key, label in metric_labels if key in metrics_payload]
    if not available:
        return

    st.subheader("Saved Test Metrics")
    columns = st.columns(len(available))
    for column, (key, label) in zip(columns, available):
        value = metrics_payload[key]
        try:
            column.metric(label, f"{float(value) * 100.0:.2f}%")
        except (TypeError, ValueError):
            column.metric(label, str(value))


def main():
    inject_custom_css()
    render_hero()

    with st.sidebar:
        st.header("Artifacts")
        st.write("Model search order:")
        for model_path in MODEL_CANDIDATES:
            st.write(f"- `{model_path.relative_to(BASE_DIR)}`")
        st.caption(f"Classes: `{CLASS_NAMES_PATH.relative_to(BASE_DIR)}`")
        st.caption(f"Preprocessing: `{PREPROCESSING_CONFIG_PATH.relative_to(BASE_DIR)}`")
        show_gradcam = st.toggle("Show Grad-CAM explanation", value=True)

    # open_card()
    # st.subheader("Runtime Check")
    # # render_artifact_status()
    # close_card()

    try:
        _, class_names, display_names, preprocessing_config, model_path = load_artifacts()
    except Exception as exc:
        st.error("Failed to load model artifacts.")
        st.exception(exc)
        st.info(
            "Expected structure:\n"
            "- `artifacts/model_full.keras` or `artifacts/model_full.h5`\n"
            "- `artifacts/class_names.json`\n"
            "- `artifacts/preprocessing_config.json`"
        )
        return

    open_card()
    st.markdown('<div class="status-pill">Model Ready</div>', unsafe_allow_html=True)
    st.success(f"Loaded model: `{model_path.relative_to(BASE_DIR)}`")
    render_metrics_summary()
    close_card()

    uploaded_file = st.file_uploader(
        "Upload a dermatoscopic image",
        type=["jpg", "jpeg", "png"],
        help="The app applies the same CLAHE + resize + normalization pipeline used in your notebook.",
    )

    if uploaded_file is None:
        st.info("Upload an image to run a prediction.")
        return

    try:
        input_image = Image.open(uploaded_file)
        
        # Convert color profile to sRGB if it exists (e.g., Display P3 from iPhones)
        icc = input_image.info.get('icc_profile')
        if icc:
            try:
                io_handle = io.BytesIO(icc)
                src_profile = ImageCms.ImageCmsProfile(io_handle)
                dst_profile = ImageCms.createProfile('sRGB')
                input_image = ImageCms.profileToProfile(input_image, src_profile, dst_profile)
            except Exception:
                pass
                
        input_image = ImageOps.exif_transpose(input_image)
        display_image, preprocessed_image = preprocess_uploaded_image(input_image, preprocessing_config)
        probs, pred_idx, pred_code, pred_name = predict_image(preprocessed_image)
    except Exception as exc:
        st.error("Prediction failed.")
        st.exception(exc)
        return

    prob_table = make_probability_table(probs, class_names, display_names)
    top_confidence = probs[pred_idx] * 100.0

    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric("Predicted class", pred_name)
    with summary_cols[1]:
        st.metric("Class code", pred_code)
    with summary_cols[2]:
        st.metric("Confidence", f"{top_confidence:.2f}%")

    left_col, right_col = st.columns([1.05, 1.15], gap="medium")

    with left_col:
        open_card()
        st.subheader("Input")
        st.image(input_image, caption="Uploaded image")
        close_card()

    with right_col:
        open_card()
        st.subheader("Prediction")
        st.markdown('<div class="compact-note">Ranked probabilities across all seven classes.</div>', unsafe_allow_html=True)
        st.dataframe(
            prob_table[["class_name", "class_code", "confidence_pct"]],
            hide_index=True,
            use_container_width=True,
        )
        st.bar_chart(prob_table.set_index("class_code")["probability"])
        close_card()

    if show_gradcam:
        try:
            with st.spinner("Generating Grad-CAM..."):
                heatmap, gradcam_layer = compute_gradcam(preprocessed_image, class_index=pred_idx)
                overlay = overlay_explanation(display_image, heatmap)
        except Exception as exc:
            st.warning("Prediction succeeded, but Grad-CAM failed.")
            st.exception(exc)
        else:
            open_card()
            st.subheader("Grad-CAM Explanation")
            st.caption(f"Last convolutional layer: `{gradcam_layer}`")

            cam_col1, cam_col2, cam_col3 = st.columns(3, gap="large")

            with cam_col1:
                st.image(display_image, caption="Preprocessed display image", clamp=True)

            with cam_col2:
                fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
                hm = ax.imshow(
                    cv2.resize(heatmap, (display_image.shape[1], display_image.shape[0])),
                    cmap="inferno",
                    vmin=0.0,
                    vmax=1.0,
                )
                ax.axis("off")
                ax.set_title("Grad-CAM Heatmap")
                fig.colorbar(hm, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig)
                plt.close(fig)

            with cam_col3:
                st.image(
                    overlay,
                    caption=f"Pred: {pred_name} ({top_confidence:.2f}%)",
                    clamp=True,
                )
            close_card()

    with st.expander("Debug Details"):
        st.code(
            "\n".join(
                [
                    f"Python file: {__file__}",
                    f"Artifacts dir: {ARTIFACTS_DIR}",
                    f"Checkpoints dir: {CHECKPOINTS_DIR}",
                    f"Loaded model: {model_path}",
                    f"Classes path: {CLASS_NAMES_PATH}",
                    f"Preprocessing path: {PREPROCESSING_CONFIG_PATH}",
                ]
            )
        )

    st.markdown("---")
    st.markdown(
        ""
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        st.error("The app hit an unexpected error.")
        st.exception(exc)
        st.code(traceback.format_exc())
