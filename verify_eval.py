import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import gc

# 1. Setup preprocessing functions (same as in the notebook)
def apply_clahe_and_resize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    img_clahe = cv2.merge((cl,cl,cl))
    resized = cv2.resize(img_clahe, (128, 128))
    return resized

from mtcnn import MTCNN
detector = MTCNN()

def detect_and_preprocess_face(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    if not results:
        return None
    x, y, w, h = results[0]['box']
    x, y = max(0, x), max(0, y)
    face = img[y:y+h, x:x+w]
    if face.size == 0:
        return None
    return apply_clahe_and_resize(face)

# 2. Load dataset
def load_dataset(base_folder):
    images = []
    labels = []
    classes = ['live', 'spoof']
    print(f"[INFO] Loading from {base_folder}...")
    for class_name in classes:
        folder_path = os.path.join(base_folder, class_name)
        if not os.path.isdir(folder_path):
            continue
        files = os.listdir(folder_path)
        used = 0
        # Load up to 30 samples to verify quickly without taking too long
        for i, name in enumerate(files[:30]):
            if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(folder_path, name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            try:
                processed_img = detect_and_preprocess_face(img)
            except Exception as e:
                continue
            if processed_img is not None:
                images.append(processed_img)
                labels.append(class_name)
                used += 1
        print(f"[OK] {class_name}: {used} images loaded.")
    return np.array(images, dtype="float32"), np.array(labels)

x_test, y_test_labels = load_dataset('Dataset/Dataset_liveness/test')
x_test = x_test / 255.0

# 3. Label encoding
le = LabelEncoder()
y_test = le.fit_transform(y_test_labels)

# 4. Load model
model_file = 'liveness_model.keras'
if not os.path.exists(model_file):
    model_file = 'best_model.keras'
print(f"[INFO] Loading model from {model_file}...")
model = load_model(model_file)

# 5. Run evaluation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    use_seaborn = True
except ImportError:
    use_seaborn = False

y_pred = model.predict(x_test)
y_pred_classes = (y_pred >= 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes)

print("\n=== METRIK EVALUASI ===")
print(f"Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score  : {f1:.4f} ({f1*100:.2f}%)")
print("=======================\n")

print("Classification Report:")
target_names = le.classes_ # ['live', 'spoof']
print(classification_report(y_test, y_pred_classes, target_names=target_names))

cm = confusion_matrix(y_test, y_pred_classes)

if use_seaborn:
    plt.figure(figsize=(6, 5))
    sns.set_theme(style="white")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names,
                annot_kws={"size": 14, "weight": "bold"}, cbar=True)
    plt.title('Confusion Matrix - Liveness Detection Model', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('Actual Label', fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig('confusion_matrix_test.png', dpi=150, bbox_inches='tight')
else:
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#F5F5F5')
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=target_names, yticklabels=target_names)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=11)
    plt.setp(ax.get_yticklabels(), rotation=90, va="center", fontsize=11)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix - Liveness Detection Model', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
    ax.set_ylabel('Actual Label', fontsize=12, labelpad=10)
    fig.tight_layout()
    plt.savefig('confusion_matrix_test.png', dpi=150, bbox_inches='tight')

print("[OK] confusion_matrix_test.png saved successfully!")
