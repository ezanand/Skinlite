# Standalone EfficientNet-B0 Training & Deployment Script
# HAM10000 Skin Lesion Classification

# Requirements should be installed using:
# pip install -r requirements.txt

# EfficientNet-B0 Skin Lesion Classification
# Cleaned training and deployment pipeline for HAM10000

# -*- coding: utf-8 -*-

import os, zipfile, cv2, warnings, time, sys, gc, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

print(f'TensorFlow version : {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPU available      : {gpus if gpus else "None - switch to GPU runtime!"}')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print('GPU memory growth   : enabled')

# Kaggle dataset setup
import shutil

os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

if os.path.exists("kaggle.json"):
    shutil.copy(
        "kaggle.json",
        os.path.expanduser("~/.kaggle/kaggle.json")
    )

os.system(
    "kaggle datasets download -d kmader/skin-cancer-mnist-ham10000"
)

print('\nDownloading HAM10000...')
print('\nExtracting...')
with zipfile.ZipFile('skin-cancer-mnist-ham10000.zip', 'r') as z:
    z.extractall('ham10000')

print('\nContents:')
for item in os.listdir('ham10000'):
    print(f'  {item}')

# Replaced torch.manual_seed() with tf.random.set_seed().

# ─── CONFIGURATION ───────────────────────────────────────────
IMG_SIZE         = 224
BATCH_SIZE       = 32
EPOCHS           = 50
PATIENCE         = 10
LEARN_RATE       = 1e-4
NUM_CLASSES      = 7
TARGET_PER_CLASS = 5700
SEED             = 42

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

tf.random.set_seed(SEED)   # FIX: was torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── PATHS ────────────────────────────────────────────────────
BASE_DIR     = 'ham10000'
METADATA_CSV = os.path.join(BASE_DIR, 'HAM10000_metadata.csv')

IMG_DIRS = [d for d in [
    os.path.join(BASE_DIR, 'HAM10000_images_part1'),
    os.path.join(BASE_DIR, 'HAM10000_images_part2'),
    os.path.join(BASE_DIR, 'ham10000_images_part_1'),
    os.path.join(BASE_DIR, 'ham10000_images_part_2'),
] if os.path.isdir(d)]

MASK_DIR = '/content/ham10000_seg/masks'
MASK_DIR = MASK_DIR if os.path.isdir(MASK_DIR) else None

def find_image(image_id):
    for d in IMG_DIRS:
        p = os.path.join(d, image_id + '.jpg')
        if os.path.exists(p): return p
    return None

print(f'Metadata CSV : {METADATA_CSV}  exists={os.path.exists(METADATA_CSV)}')
print(f'Image dirs   : {IMG_DIRS}')
print(f'Mask dir     : {MASK_DIR}')

df = pd.read_csv(METADATA_CSV)
label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
df['label'] = df['dx'].map(label_map)

print(f'Total images : {len(df)}')
print(f'Classes      : {NUM_CLASSES}')
print('\nClass distribution:')
print(df['dx'].value_counts())

fig, ax = plt.subplots(figsize=(8, 4))
counts = df['dx'].value_counts()
ax.bar(counts.index, counts.values, color=plt.cm.Set2.colors)
ax.set_title('HAM10000 — Class Distribution (Unbalanced)', fontsize=13)
ax.set_xlabel('Lesion Type'); ax.set_ylabel('Count')
plt.tight_layout(); plt.show()

def enhance_image(image):
    

# 70/15/15 split
train_df_raw, test_df = train_test_split(
    df, test_size=0.15, random_state=SEED, stratify=df['dx']
)
train_df_raw, val_df = train_test_split(
    train_df_raw, test_size=(0.15/0.85), random_state=SEED, stratify=train_df_raw['dx']
)

print(f'Raw train : {len(train_df_raw)}')

balanced_df = []
for class_name in CLASS_NAMES:
    class_df = train_df_raw[train_df_raw['dx'] == class_name]
    if len(class_df) < TARGET_PER_CLASS:
        class_df = class_df.sample(TARGET_PER_CLASS, replace=True, random_state=SEED)
    balanced_df.append(class_df)

train_df_bal = pd.concat(balanced_df).sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f'Balanced train : {len(train_df_bal)}')
print(f'Val            : {len(val_df)}')
print(f'Test           : {len(test_df)}')
print('\nClass distribution (balanced):')
print(train_df_bal['dx'].value_counts())

# (~200 bytes), NOT the memory used by the numpy arrays inside it.
# Replaced with the correct formula: n_images × H × W × C × 4 bytes (float32).

print('Preloading images into RAM...')
print('(This takes ~3-5 mins but saves hours of training time)')

image_cache = {}
missing = 0

all_ids = df['image_id'].unique()
for idx, image_id in enumerate(all_ids):
    img = load_image(image_id)
    if img is not None:
        image_cache[image_id] = img
    else:
        missing += 1
    if (idx + 1) % 1000 == 0:
        print(f'  {idx+1}/{len(all_ids)} images cached...')

cache_size_gb = len(image_cache) * IMG_SIZE * IMG_SIZE * 3 * 4 / (1024**3)
print(f'\nCached  : {len(image_cache):,} images')
print(f'Missing : {missing}')
print(f'RAM used: ~{cache_size_gb:.2f} GB')
print('Image cache ready ✓')

def mixup_batch(images, labels, alpha=0.2):
    

def build_efficientnet():
    

DRIVE_ROOT = '/content/drive/MyDrive/skinlite_outputs'
RUN_NAME = 'efficientnet_b0_training_false'
RUN_DIR = os.path.join(DRIVE_ROOT, RUN_NAME)
CHECKPOINT_DIR = os.path.join(RUN_DIR, 'checkpoints')
FIGURES_DIR = os.path.join(RUN_DIR, 'figures')
XAI_OUTPUT_ROOT = os.path.join(RUN_DIR, 'xai_outputs')
ARTIFACTS_DIR = os.path.join(RUN_DIR, 'artifacts')

for directory in [RUN_DIR, CHECKPOINT_DIR, FIGURES_DIR, XAI_OUTPUT_ROOT, ARTIFACTS_DIR]:
    os.makedirs(directory, exist_ok=True)

BEST_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, 'efficientnet_best.weights.h5')
FULL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'last_full_model.keras')

if os.path.exists(FULL_MODEL_PATH):
    print('Resuming from deployment-friendly checkpoint...')
    effnet = load_model(FULL_MODEL_PATH)
else:
    print('Starting fresh training with training=False backbone forward pass...')
    effnet = build_efficientnet()

effnet.compile(
    optimizer=optimizers.Adam(LEARN_RATE),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3')
    ]
)

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                      min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_loss', patience=PATIENCE,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        BEST_WEIGHTS_PATH,
        monitor='val_accuracy', save_best_only=True,
        save_weights_only=True, verbose=1
    ),
    ModelCheckpoint(
        FULL_MODEL_PATH,
        save_best_only=False, save_weights_only=False,
        save_freq='epoch', verbose=0
    )
]

print('Training EfficientNet-B0...')
print(f'  Backbone      : EfficientNetB0 (ImageNet pretrained)')
print(f'  Forward mode  : training=False inside backbone call')
print(f'  Fine-tune     : ALL layers trainable')
print(f'  MixUp         : enabled (alpha=0.2)')
print(f'  ImageNet norm : enabled')
print(f'  Oversampling  : {TARGET_PER_CLASS} per class')
print(f'  Run dir       : {RUN_DIR}')
print(f'  Checkpoints   : {CHECKPOINT_DIR}')
print(f'  Figures       : {FIGURES_DIR}')
print(f'  XAI outputs   : {XAI_OUTPUT_ROOT}')
print(f'  Artifacts     : {ARTIFACTS_DIR}')

effnet_history = effnet.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print(f'\nBest Val Accuracy : {max(effnet_history.history["val_accuracy"]):.4f}')
print('EfficientNet-B0 training complete.')

print('\nFreeing image_cache RAM...')
image_cache.clear()
gc.collect()
print('RAM freed.')

# (it was an orphan cell between the training cell and the backup-listing cell).
# It has been moved to AFTER training so effnet_history is defined.

os.makedirs(FIGURES_DIR, exist_ok=True)
training_curve_path = os.path.join(FIGURES_DIR, 'training_curve_effnet.png')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(effnet_history.history['accuracy'],     label='Train')
axes[0].plot(effnet_history.history['val_accuracy'], label='Val')
axes[0].set_title('EfficientNet-B0 - Accuracy')
axes[0].legend(); axes[0].set_xlabel('Epoch')

axes[1].plot(effnet_history.history['loss'],     label='Train')
axes[1].plot(effnet_history.history['val_loss'], label='Val')
axes[1].set_title('EfficientNet-B0 - Loss')
axes[1].legend(); axes[1].set_xlabel('Epoch')
plt.tight_layout()
plt.savefig(training_curve_path, dpi=300, bbox_inches='tight')
plt.show()
print(f'Saved: {training_curve_path}')

test_gen = SkinDataGenerator(
    test_df,
    batch_size=BATCH_SIZE,
    augment=False,
    use_mixup=False
)

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

os.makedirs(FIGURES_DIR, exist_ok=True)
confusion_matrix_path = os.path.join(FIGURES_DIR, 'cm_efficientnet.png')

print('Evaluating on test set...')

results = effnet.evaluate(test_gen, verbose=1)
metrics = dict(zip(effnet.metrics_names, results))

y_true, y_pred, y_prob = [], [], []
test_records = []

for i in range(len(test_gen)):
    x_batch, y_batch = test_gen[i]
    preds = effnet.predict(x_batch, verbose=0)
    batch_true = np.argmax(y_batch, axis=1)
    batch_pred = np.argmax(preds, axis=1)
    batch_rows = test_gen.df.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

    y_true.extend(batch_true.tolist())
    y_pred.extend(batch_pred.tolist())
    y_prob.extend(preds.tolist())

    for row, true_idx, pred_idx, prob_vector in zip(batch_rows.itertuples(index=False), batch_true, batch_pred, preds):
        test_records.append({
            'image_id': row.image_id,
            'true_idx': int(true_idx),
            'pred_idx': int(pred_idx),
            'true_label': CLASS_NAMES[int(true_idx)],
            'pred_label': CLASS_NAMES[int(pred_idx)],
            'confidence': float(prob_vector[pred_idx]),
            'correct': bool(pred_idx == true_idx)
        })

y_true = np.asarray(y_true, dtype=np.int32)
y_pred = np.asarray(y_pred, dtype=np.int32)
test_probabilities = np.asarray(y_prob, dtype=np.float32)
test_prediction_df = pd.DataFrame(test_records)

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

confusion_mat = confusion_matrix(y_true, y_pred)
acc = float(results[1])

eval_metrics = {
    'accuracy': acc,
    'precision': float(report['weighted avg']['precision']),
    'recall': float(report['weighted avg']['recall']),
    'f1_score': float(report['weighted avg']['f1-score'])
}

if 'top2' in metrics:
    eval_metrics['top2_accuracy'] = float(metrics['top2'])
if 'top3' in metrics:
    eval_metrics['top3_accuracy'] = float(metrics['top3'])

print(f'\n{"=" * 55}')
print('  EfficientNet-B0 - TEST SET EVALUATION')
print(f'{"=" * 55}')
print(f'  Accuracy  : {eval_metrics["accuracy"]:.4f}')
print(f'  Precision : {eval_metrics["precision"]:.4f}')
print(f'  Recall    : {eval_metrics["recall"]:.4f}')
print(f'  F1-Score  : {eval_metrics["f1_score"]:.4f}')

print(f'\n{"Class":<10} {"Precision":>10} {"Recall":>10} {"F1-Score":>10}')
print('-' * 45)
for cls in CLASS_NAMES:
    r = report[cls]
    print(f'{cls:<10} {r["precision"]:>10.2f} {r["recall"]:>10.2f} {r["f1-score"]:>10.2f}')

plt.figure(figsize=(7, 6), dpi=300)
sns.heatmap(
    confusion_mat,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('EfficientNet-B0 Confusion Matrix')
plt.tight_layout()
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
plt.show()

print(f'Saved: {confusion_matrix_path}')

effnet_test_acc = acc

# Replaced with on-demand load_image() calls.

os.makedirs(FIGURES_DIR, exist_ok=True)
mean_dn = np.array([0.485, 0.456, 0.406])
std_dn  = np.array([0.229, 0.224, 0.225])

def save_prediction(image_id, true_label, filename):
    img = load_image(image_id)
    if img is None:
        return False
    pred_idx = np.argmax(effnet.predict(np.expand_dims(img, 0), verbose=0))
    display_img = np.clip(img * std_dn + mean_dn, 0, 1)
    plt.figure(figsize=(3, 3))
    plt.imshow(display_img)
    plt.title(f"{true_label} -> {CLASS_NAMES[pred_idx]}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return True

for cls in CLASS_NAMES:
    print(f'Processing class: {cls}')
    found = False
    for _, row in test_df[test_df['dx'] == cls].iterrows():
        img = load_image(row['image_id'])
        if img is None:
            continue
        pred = np.argmax(effnet.predict(np.expand_dims(img, 0), verbose=0))
        if CLASS_NAMES[pred] == cls:
            prediction_path = os.path.join(FIGURES_DIR, f'pred_{cls}.png')
            save_prediction(row['image_id'], cls, prediction_path)
            print(f'  Saved: {prediction_path}')
            found = True
            break

    if not found:
        print(f'  No correct prediction found for {cls}')

import json
import shap

for directory in [XAI_OUTPUT_ROOT]:
    os.makedirs(directory, exist_ok=True)

XAI_DIRS = {
    'gradcam': os.path.join(XAI_OUTPUT_ROOT, 'gradcam'),
    'integrated_gradients': os.path.join(XAI_OUTPUT_ROOT, 'integrated_gradients'),
    'shap': os.path.join(XAI_OUTPUT_ROOT, 'shap'),
    'occlusion': os.path.join(XAI_OUTPUT_ROOT, 'occlusion')
}

for directory in XAI_DIRS.values():
    os.makedirs(directory, exist_ok=True)

CLASS_DISPLAY_NAMES = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesion',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevus',
    'vasc': 'Vascular lesion'
}

def pretty_label(class_name):
    return CLASS_DISPLAY_NAMES.get(class_name, class_name)

def denormalize_image(img_normalized):
    return np.clip(img_normalized * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)

def normalize_map(explanation_map):
    explanation_map = np.nan_to_num(
        np.asarray(explanation_map, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )
    explanation_map -= explanation_map.min()
    max_value = explanation_map.max()
    if max_value <= 0:
        return np.zeros_like(explanation_map, dtype=np.float32)
    return explanation_map / (max_value + 1e-8)

def resize_map(explanation_map):
    return cv2.resize(
        np.asarray(explanation_map, dtype=np.float32),
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_CUBIC
    )

def apply_colormap(explanation_map, cmap_name='inferno'):
    cmap = plt.get_cmap(cmap_name)
    return cmap(normalize_map(explanation_map))[..., :3]

def overlay_explanation(img_normalized, explanation_map, alpha=0.45, cmap_name='inferno'):
    base_image = denormalize_image(img_normalized)
    colored_map = apply_colormap(resize_map(explanation_map), cmap_name=cmap_name)
    return np.clip((1.0 - alpha) * base_image + alpha * colored_map, 0.0, 1.0)

def predict_probabilities(model, image):
    probs = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return probs, pred_idx

def make_explanation_record(method_name, image_id, img_normalized, explanation_map, probs, pred_idx, true_idx):
    return {
        'method_name': method_name,
        'image_id': image_id,
        'img': img_normalized,
        'map': normalize_map(explanation_map),
        'probs': np.asarray(probs, dtype=np.float32),
        'pred_idx': int(pred_idx),
        'true_idx': int(true_idx)
    }

def save_four_panel_figure(record, out_path, cmap_name='inferno', colorbar_label='Importance'):
    display_image = denormalize_image(record['img'])
    explanation_map = resize_map(record['map'])
    overlay = overlay_explanation(record['img'], record['map'], cmap_name=cmap_name)

    fig, axes = plt.subplots(1, 4, figsize=(12, 6), dpi=300)

    axes[0].imshow(display_image)
    axes[0].set_title(
        f'Original\nTrue: {pretty_label(CLASS_NAMES[record["true_idx"]])}',
        fontsize=11
    )
    axes[0].axis('off')

    heat_artist = axes[1].imshow(explanation_map, cmap=cmap_name, vmin=0.0, vmax=1.0)
    axes[1].set_title(f'{record["method_name"]} Heatmap', fontsize=11)
    axes[1].axis('off')
    fig.colorbar(heat_artist, ax=axes[1], fraction=0.046, pad=0.04, label=colorbar_label)

    axes[2].imshow(overlay)
    axes[2].set_title(
        f'Pred: {pretty_label(CLASS_NAMES[record["pred_idx"]])} '
        f'({record["probs"][record["pred_idx"]] * 100:.1f}%)',
        fontsize=11
    )
    axes[2].axis('off')

    bar_colors = []
    for idx in range(NUM_CLASSES):
        if idx == record['pred_idx'] and idx == record['true_idx']:
            bar_colors.append('#2ca02c')
        elif idx == record['pred_idx']:
            bar_colors.append('#d62728')
        elif idx == record['true_idx']:
            bar_colors.append('#1f77b4')
        else:
            bar_colors.append('#b0b0b0')

    axes[3].bar(CLASS_NAMES, record['probs'] * 100.0, color=bar_colors)
    axes[3].set_ylim(0, 100)
    axes[3].set_ylabel('Confidence (%)')
    axes[3].set_title('Confidence', fontsize=11)
    axes[3].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def save_overlay_summary(records, title, out_path, cmap_name='inferno', max_cols=4):
    if not records:
        print(f'No records available for {title}')
        return

    n_records = len(records)
    cols = min(max_cols, n_records)
    rows = int(np.ceil(n_records / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(6, rows * 3)), dpi=300)
    axes = np.atleast_2d(axes)

    for ax in axes.flat:
        ax.axis('off')

    for ax, record in zip(axes.flat, records):
        ax.imshow(overlay_explanation(record['img'], record['map'], cmap_name=cmap_name))
        true_name = pretty_label(CLASS_NAMES[record['true_idx']])
        pred_name = pretty_label(CLASS_NAMES[record['pred_idx']])
        confidence = record['probs'][record['pred_idx']] * 100.0
        title_color = 'green' if record['pred_idx'] == record['true_idx'] else 'red'
        ax.set_title(
            f'{true_name}\nPred: {pred_name} ({confidence:.1f}%)',
            fontsize=9,
            color=title_color
        )
        ax.axis('off')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def select_class_examples(prediction_df, prefer_correct=True):
    selected = {}
    for class_name in CLASS_NAMES:
        class_rows = prediction_df[prediction_df['true_label'] == class_name]
        if len(class_rows) == 0:
            continue
        if prefer_correct:
            preferred_rows = class_rows[class_rows['correct']]
            candidate_rows = preferred_rows if len(preferred_rows) else class_rows
        else:
            candidate_rows = class_rows
        selected[class_name] = candidate_rows.sort_values(
            'confidence',
            ascending=False
        ).iloc[0].to_dict()
    return selected

def prepare_records_from_rows(rows, method_name, explainer_fn):
    records = []
    for row in rows:
        image_id = row['image_id']
        image = load_image(image_id)
        if image is None:
            continue
        explanation_map, probs, pred_idx = explainer_fn(image)
        records.append(
            make_explanation_record(
                method_name=method_name,
                image_id=image_id,
                img_normalized=image,
                explanation_map=explanation_map,
                probs=probs,
                pred_idx=pred_idx,
                true_idx=int(row['true_idx'])
            )
        )
    return records

print('XAI output directories:')
for name, directory in XAI_DIRS.items():
    print(f'  {name:<22} -> {directory}')

def get_efficientnet_backbone(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and 'efficientnet' in layer.name.lower():
            return layer
    raise ValueError('Could not locate the EfficientNet backbone inside the model.')

def auto_detect_last_conv_layer(model):
    backbone = get_efficientnet_backbone(model)
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D
    )
    for layer in reversed(backbone.layers):
        if isinstance(layer, conv_types):
            return layer.name
    raise ValueError('No convolutional layer found for Grad-CAM.')

GRADCAM_BACKBONE = get_efficientnet_backbone(effnet)
LAST_CONV_LAYER = auto_detect_last_conv_layer(effnet)
GRADCAM_CONV_MODEL = tf.keras.Model(
    inputs=GRADCAM_BACKBONE.input,
    outputs=GRADCAM_BACKBONE.get_layer(LAST_CONV_LAYER).output,
    name='gradcam_conv_extractor'
)
GRADCAM_POST_LAYERS = [
    layer for layer in effnet.layers
    if layer.name != GRADCAM_BACKBONE.name and not isinstance(layer, tf.keras.layers.InputLayer)
]

def compute_gradcam_map(model, image, class_index=None):
    inputs = tf.convert_to_tensor(np.expand_dims(image, axis=0), dtype=tf.float32)
    conv_output = GRADCAM_CONV_MODEL(inputs, training=False)
    conv_variable = tf.Variable(conv_output, trainable=True, dtype=tf.float32)

    with tf.GradientTape() as tape:
        x = conv_variable
        for layer in GRADCAM_POST_LAYERS:
            x = layer(x, training=False)
        probs_tensor = x
        if class_index is None:
            class_index = int(tf.argmax(probs_tensor[0]).numpy())
        target_score = probs_tensor[:, class_index]

    gradients = tape.gradient(target_score, conv_variable)
    if gradients is None:
        raise ValueError(f'Gradients are None for layer {LAST_CONV_LAYER}.')

    channel_weights = tf.reduce_mean(gradients, axis=(1, 2))
    cam = tf.reduce_sum(conv_variable[0] * channel_weights[0][None, None, :], axis=-1)
    cam = tf.nn.relu(cam)
    probs = model(inputs, training=False).numpy()[0]
    return normalize_map(cam.numpy()), probs, int(class_index)

print(f'Grad-CAM last convolutional layer: {LAST_CONV_LAYER}')

gradcam_sample_row = test_prediction_df.iloc[0].to_dict()
gradcam_sample_records = prepare_records_from_rows(
    [gradcam_sample_row],
    method_name='Grad-CAM',
    explainer_fn=lambda image: compute_gradcam_map(effnet, image)
)
save_four_panel_figure(
    gradcam_sample_records[0],
    os.path.join(XAI_DIRS['gradcam'], 'sample_gradcam.png'),
    cmap_name='inferno',
    colorbar_label='Activation'
)

gradcam_class_rows = list(select_class_examples(test_prediction_df, prefer_correct=True).values())
gradcam_class_records = prepare_records_from_rows(
    gradcam_class_rows,
    method_name='Grad-CAM',
    explainer_fn=lambda image: compute_gradcam_map(effnet, image)
)

for record in gradcam_class_records:
    class_name = CLASS_NAMES[record['true_idx']]
    save_four_panel_figure(
        record,
        os.path.join(XAI_DIRS['gradcam'], f'per_class_{class_name}.png'),
        cmap_name='inferno',
        colorbar_label='Activation'
    )

save_overlay_summary(
    gradcam_class_records,
    'Grad-CAM Per-Class Summary',
    os.path.join(XAI_DIRS['gradcam'], 'summary_panel.png'),
    cmap_name='inferno',
    max_cols=4
)

gradcam_mis_rows = (
    test_prediction_df[~test_prediction_df['correct']]
    .sort_values('confidence', ascending=False)
    .head(6)
    .to_dict('records')
)
gradcam_mis_records = prepare_records_from_rows(
    gradcam_mis_rows,
    method_name='Grad-CAM',
    explainer_fn=lambda image: compute_gradcam_map(effnet, image)
)

for index, record in enumerate(gradcam_mis_records, start=1):
    save_four_panel_figure(
        record,
        os.path.join(XAI_DIRS['gradcam'], f'misclassified_{index:02d}.png'),
        cmap_name='inferno',
        colorbar_label='Activation'
    )

save_overlay_summary(
    gradcam_mis_records,
    'Grad-CAM Misclassified Samples',
    os.path.join(XAI_DIRS['gradcam'], 'misclassified_summary.png'),
    cmap_name='inferno',
    max_cols=3
)

print('Grad-CAM outputs saved to:', XAI_DIRS['gradcam'])

IG_STEPS = 50
IG_BATCH_SIZE = 10

def compute_integrated_gradients(model, image, class_index=None, steps=IG_STEPS, batch_size=IG_BATCH_SIZE):
    image_tensor = tf.convert_to_tensor(np.expand_dims(image, axis=0), dtype=tf.float32)
    baseline_tensor = tf.zeros_like(image_tensor)

    base_probs = model(image_tensor, training=False).numpy()[0]
    if class_index is None:
        class_index = int(np.argmax(base_probs))

    alphas = tf.linspace(0.0, 1.0, steps + 1)
    total_gradients = tf.zeros_like(image_tensor)

    for start in range(0, steps + 1, batch_size):
        alpha_batch = alphas[start:start + batch_size]
        alpha_batch = tf.reshape(alpha_batch, (-1, 1, 1, 1, 1))
        interpolated = baseline_tensor + alpha_batch * (image_tensor - baseline_tensor)
        interpolated = tf.reshape(interpolated, (-1, IMG_SIZE, IMG_SIZE, 3))

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred_batch = model(interpolated, training=False)
            target_scores = pred_batch[:, class_index]

        gradients = tape.gradient(target_scores, interpolated)
        gradients = tf.reshape(gradients, (-1, 1, IMG_SIZE, IMG_SIZE, 3))
        total_gradients += tf.reduce_sum(gradients, axis=0)

    avg_gradients = total_gradients / tf.cast(steps + 1, tf.float32)
    integrated_grads = (image_tensor - baseline_tensor) * avg_gradients
    attribution_map = tf.reduce_sum(tf.abs(integrated_grads), axis=-1)[0].numpy().astype(np.float32)
    attribution_map = cv2.GaussianBlur(attribution_map, (0, 0), sigmaX=2.0)
    return normalize_map(attribution_map), base_probs, int(class_index)

ig_sample_records = prepare_records_from_rows(
    [gradcam_sample_row],
    method_name='Integrated Gradients',
    explainer_fn=lambda image: compute_integrated_gradients(effnet, image)
)
save_four_panel_figure(
    ig_sample_records[0],
    os.path.join(XAI_DIRS['integrated_gradients'], 'sample_integrated_gradients.png'),
    cmap_name='inferno',
    colorbar_label='Attribution'
)

ig_class_records = prepare_records_from_rows(
    gradcam_class_rows,
    method_name='Integrated Gradients',
    explainer_fn=lambda image: compute_integrated_gradients(effnet, image)
)

for record in ig_class_records:
    class_name = CLASS_NAMES[record['true_idx']]
    save_four_panel_figure(
        record,
        os.path.join(XAI_DIRS['integrated_gradients'], f'per_class_{class_name}.png'),
        cmap_name='inferno',
        colorbar_label='Attribution'
    )

save_overlay_summary(
    ig_class_records,
    'Integrated Gradients Per-Class Summary',
    os.path.join(XAI_DIRS['integrated_gradients'], 'summary_panel.png'),
    cmap_name='inferno',
    max_cols=4
)

print('Integrated Gradients outputs saved to:', XAI_DIRS['integrated_gradients'])

SHAP_BACKGROUND_SIZE = min(8, len(val_df))
SHAP_EXPLAIN_LIMIT = min(10, len(test_prediction_df))

def collect_loaded_images(rows, limit=None):
    images = []
    valid_rows = []
    for row in rows:
        image = load_image(row['image_id'])
        if image is None:
            continue
        images.append(image)
        valid_rows.append(row)
        if limit is not None and len(images) >= limit:
            break
    if not images:
        return np.empty((0, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32), []
    return np.asarray(images, dtype=np.float32), valid_rows

def extract_shap_map(shap_values, sample_index, class_index):
    if isinstance(shap_values, list):
        sample_values = np.asarray(shap_values[class_index][sample_index])
    else:
        values = np.asarray(shap_values)
        if values.ndim == 5 and values.shape[-1] == NUM_CLASSES:
            sample_values = values[sample_index, ..., class_index]
        elif values.ndim == 5 and values.shape[0] == NUM_CLASSES:
            sample_values = values[class_index, sample_index]
        elif values.ndim == 4:
            sample_values = values[sample_index]
        else:
            raise ValueError(f'Unexpected SHAP output shape: {values.shape}')

    if sample_values.ndim == 3:
        sample_values = np.mean(np.abs(sample_values), axis=-1)
    return normalize_map(sample_values)

background_rows = val_df[['image_id', 'label', 'dx']].head(SHAP_BACKGROUND_SIZE).to_dict('records')
background_images, _ = collect_loaded_images(
    [{'image_id': row['image_id']} for row in background_rows],
    limit=SHAP_BACKGROUND_SIZE
)

shap_candidate_rows = gradcam_class_rows.copy()
if len(shap_candidate_rows) < SHAP_EXPLAIN_LIMIT:
    existing_ids = {row['image_id'] for row in shap_candidate_rows}
    extra_rows = test_prediction_df.sort_values('confidence', ascending=False).to_dict('records')
    for row in extra_rows:
        if row['image_id'] in existing_ids:
            continue
        shap_candidate_rows.append(row)
        existing_ids.add(row['image_id'])
        if len(shap_candidate_rows) >= SHAP_EXPLAIN_LIMIT:
            break

shap_images, shap_valid_rows = collect_loaded_images(shap_candidate_rows, limit=SHAP_EXPLAIN_LIMIT)

if len(background_images) == 0 or len(shap_images) == 0:
    raise ValueError('SHAP requires non-empty background and explain image subsets.')

print(
    f'SHAP subset: {len(background_images)} background images, '
    f'{len(shap_images)} explained test images'
)

shap_explainer = shap.GradientExplainer(effnet, background_images)
shap_values = shap_explainer.shap_values(shap_images)

shap_records = []
for sample_index, row in enumerate(shap_valid_rows):
    image = shap_images[sample_index]
    probs, pred_idx = predict_probabilities(effnet, image)
    shap_map = extract_shap_map(shap_values, sample_index, pred_idx)
    record = make_explanation_record(
        method_name='SHAP',
        image_id=row['image_id'],
        img_normalized=image,
        explanation_map=shap_map,
        probs=probs,
        pred_idx=pred_idx,
        true_idx=int(row['true_idx'])
    )
    shap_records.append(record)
    save_four_panel_figure(
        record,
        os.path.join(XAI_DIRS['shap'], f'shap_{sample_index + 1:02d}_{row["image_id"]}.png'),
        cmap_name='inferno',
        colorbar_label='SHAP magnitude'
    )

save_overlay_summary(
    shap_records,
    'SHAP Overlay Summary',
    os.path.join(XAI_DIRS['shap'], 'summary_panel.png'),
    cmap_name='inferno',
    max_cols=4
)

print('SHAP outputs saved to:', XAI_DIRS['shap'])

OCCLUSION_PATCH_SIZE = 32
OCCLUSION_STRIDE = 32

def compute_occlusion_sensitivity(
    model,
    image,
    class_index=None,
    patch_size=OCCLUSION_PATCH_SIZE,
    stride=OCCLUSION_STRIDE,
    fill_value=0.0
):
    probs, pred_idx = predict_probabilities(model, image)
    if class_index is None:
        class_index = pred_idx

    baseline_score = float(probs[class_index])
    sensitivity = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    counts = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    for top in range(0, IMG_SIZE, stride):
        for left in range(0, IMG_SIZE, stride):
            bottom = min(top + patch_size, IMG_SIZE)
            right = min(left + patch_size, IMG_SIZE)
            occluded = image.copy()
            occluded[top:bottom, left:right, :] = fill_value
            occluded_probs = model.predict(np.expand_dims(occluded, axis=0), verbose=0)[0]
            score_drop = baseline_score - float(occluded_probs[class_index])
            sensitivity[top:bottom, left:right] += score_drop
            counts[top:bottom, left:right] += 1.0

    sensitivity = sensitivity / np.maximum(counts, 1.0)
    sensitivity = np.maximum(sensitivity, 0.0)
    sensitivity = cv2.GaussianBlur(sensitivity, (0, 0), sigmaX=3.0)
    return normalize_map(sensitivity), probs, int(class_index)

occlusion_sample_records = prepare_records_from_rows(
    [gradcam_sample_row],
    method_name='Occlusion',
    explainer_fn=lambda image: compute_occlusion_sensitivity(effnet, image)
)
save_four_panel_figure(
    occlusion_sample_records[0],
    os.path.join(XAI_DIRS['occlusion'], 'sample_occlusion.png'),
    cmap_name='inferno',
    colorbar_label='Sensitivity'
)

occlusion_class_records = prepare_records_from_rows(
    gradcam_class_rows,
    method_name='Occlusion',
    explainer_fn=lambda image: compute_occlusion_sensitivity(effnet, image)
)

for record in occlusion_class_records:
    class_name = CLASS_NAMES[record['true_idx']]
    save_four_panel_figure(
        record,
        os.path.join(XAI_DIRS['occlusion'], f'per_class_{class_name}.png'),
        cmap_name='inferno',
        colorbar_label='Sensitivity'
    )

save_overlay_summary(
    occlusion_class_records,
    'Occlusion Sensitivity Per-Class Summary',
    os.path.join(XAI_DIRS['occlusion'], 'summary_panel.png'),
    cmap_name='inferno',
    max_cols=4
)

print('Occlusion outputs saved to:', XAI_DIRS['occlusion'])

MODEL_FULL_KERAS_PATH = os.path.join(ARTIFACTS_DIR, 'model_full.keras')
MODEL_FULL_H5_PATH = os.path.join(ARTIFACTS_DIR, 'model_full.h5')
MODEL_WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, 'model.weights.h5')
CLASS_NAMES_PATH = os.path.join(ARTIFACTS_DIR, 'class_names.json')
PREPROCESSING_CONFIG_PATH = os.path.join(ARTIFACTS_DIR, 'preprocessing_config.json')
EVAL_METRICS_PATH = os.path.join(ARTIFACTS_DIR, 'eval_metrics.json')

effnet.save(MODEL_FULL_KERAS_PATH)
effnet.save(MODEL_FULL_H5_PATH)
effnet.save_weights(MODEL_WEIGHTS_PATH)

class_names_payload = {
    'class_names': CLASS_NAMES,
    'display_names': CLASS_DISPLAY_NAMES
}

preprocessing_config = {
    'image_size': IMG_SIZE,
    'normalization_mean': IMAGENET_MEAN.tolist(),
    'normalization_std': IMAGENET_STD.tolist(),
    'clahe': {
        'clip_limit': 2.0,
        'tile_grid_size': [8, 8]
    },
    'mask_directory': MASK_DIR,
    'num_classes': NUM_CLASSES,
    'run_dir': RUN_DIR
}

with open(CLASS_NAMES_PATH, 'w', encoding='utf-8') as f:
    json.dump(class_names_payload, f, indent=2)

with open(PREPROCESSING_CONFIG_PATH, 'w', encoding='utf-8') as f:
    json.dump(preprocessing_config, f, indent=2)

with open(EVAL_METRICS_PATH, 'w', encoding='utf-8') as f:
    json.dump(eval_metrics, f, indent=2)

print('Saved deployment artifacts to Drive:')
print(f'  full model (.keras) -> {MODEL_FULL_KERAS_PATH}')
print(f'  full model (.h5)    -> {MODEL_FULL_H5_PATH}')
print(f'  model weights       -> {MODEL_WEIGHTS_PATH}')
print(f'  class names         -> {CLASS_NAMES_PATH}')
print(f'  preprocessing config-> {PREPROCESSING_CONFIG_PATH}')
print(f'  eval metrics        -> {EVAL_METRICS_PATH}')
print(f'  figures dir         -> {FIGURES_DIR}')
print(f'  xai outputs dir     -> {XAI_OUTPUT_ROOT}')
