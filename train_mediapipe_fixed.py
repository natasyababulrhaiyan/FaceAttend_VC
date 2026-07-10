"""
Training script fixed untuk MediaPipe + PointNet-style CNN.

Cara pakai:
    python train_mediapipe_fixed.py

Output:
    model_cnn_last.keras
    label_encoder_last.pickle
    mean_landmark.npy
    std_landmark.npy
    face_centroids.npy
    face_thresholds.npy
"""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

from mediapipe_face_utils import (
    extract_aligned_landmarks,
    build_model,
    compute_centroids_and_thresholds,
)


DATASET_PATH = "Dataset/Dataset_wajah"
AUGMENT = True
NUM_AUG = 2
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 200
BATCH_SIZE = 32


def main():
    # Reproducibility
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    print("[INFO] Inisialisasi MediaPipe FaceMesh...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    )

    print(f"[INFO] Memuat dataset dari: {DATASET_PATH}")
    X_raw, Y = [], []
    total_fail = 0

    for person_name in sorted(os.listdir(DATASET_PATH)):
        person_dir = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(person_dir):
            continue

        count = 0
        for image_name in sorted(os.listdir(person_dir)):
            image_path = os.path.join(person_dir, image_name)
            lms_list, fails = extract_aligned_landmarks(
                image_path, face_mesh,
                augment=AUGMENT, num_aug=NUM_AUG
            )
            total_fail += fails
            for lms in lms_list:
                X_raw.append(lms)
                Y.append(person_name)
                count += 1

        print(f"  {person_name}: {count} sampel")

    X = np.array(X_raw, dtype=np.float32)
    Y = np.array(Y)
    print(f"\n[INFO] Total sampel: {len(X)}")
    print(f"[INFO] Total gagal deteksi: {total_fail}")
    print(f"[INFO] Shape X: {X.shape}")

    # Split
    print("\n[INFO] Splitting data (80/20)...")
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=Y
    )

    # Normalisasi Z-score dari TRAINING saja
    print("[INFO] Normalisasi Z-score...")
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-8
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    np.save("mean_landmark.npy", mean)
    np.save("std_landmark.npy", std)

    # Label encoding
    print("[INFO] Label encoding...")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    num_classes = len(le.classes_)

    y_train_cat = keras.utils.to_categorical(y_train_enc, num_classes=num_classes)
    y_test_cat = keras.utils.to_categorical(y_test_enc, num_classes=num_classes)

    with open("label_encoder_last.pickle", "wb") as f:
        pickle.dump(le, f)

    print(f"[INFO] Jumlah kelas: {num_classes}")
    for idx, name in enumerate(le.classes_):
        print(f"  {idx}: {name}")

    # Build model
    print("\n[INFO] Building model...")
    model = build_model(input_shape=(468, 3), num_classes=num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True, verbose=1
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1
    )

    # Train
    print("\n[INFO] Training...")
    history = model.fit(
        x_train, y_train_cat,
        validation_data=(x_test, y_test_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
    )

    # Save model
    model.save("model_cnn_last.keras")
    print("[INFO] Model disimpan: model_cnn_last.keras")

    # Embedding model
    embedding_model = keras.models.Model(
        inputs=model.input,
        outputs=model.get_layer("embedding").output,
    )

    # Compute centroids & thresholds
    print("[INFO] Menghitung centroid embedding untuk unknown detection...")
    train_emb = embedding_model.predict(x_train, verbose=0)
    test_emb = embedding_model.predict(x_test, verbose=0)

    centroids, thresholds = compute_centroids_and_thresholds(
        train_emb, y_train_enc, num_classes,
        percentile=95, slack=1.3,
        val_embeddings=test_emb, val_labels=y_test_enc
    )
    np.save("face_centroids.npy", centroids)
    np.save("face_thresholds.npy", thresholds)

    print("[INFO] Threshold per kelas:")
    for i, name in enumerate(le.classes_):
        print(f"  {name:25s}: {thresholds[i]:.4f}")

    # Evaluation
    print("\n[INFO] Evaluasi...")
    loss, acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"[HASIL] Test accuracy: {acc*100:.2f}%")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Akurasi")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    print("[INFO] Plot training disimpan: training_history.png")

    # Confusion matrix
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test_enc, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("[INFO] Confusion matrix disimpan: confusion_matrix.png")

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test_enc, y_pred_classes, target_names=le.classes_))

    face_mesh.close()
    print("[INFO] Selesai.")


if __name__ == "__main__":
    main()
