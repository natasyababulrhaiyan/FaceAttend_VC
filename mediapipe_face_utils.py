"""
Utils bersama untuk pipeline MediaPipe Face Recognition (fixed).
"""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import random
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers


# Indeks landmark referensi MediaPipe (468 landmark)
NOSE_TIP = 1
LEFT_EYE = 33
RIGHT_EYE = 263


def apply_clahe(img):
    """CLAHE enhancement untuk meningkatkan deteksi landmark."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def align_landmarks(landmarks):
    """
    Normalisasi geometris landmark wajah.
    Input/output shape: (468, 3)

    - Translate nose tip ke origin
    - Scale oleh jarak antar mata (scale invariant)
    - Rotate agar garis antar mata horizontal
    """
    lm = np.array(landmarks, dtype=np.float32)

    # translate
    lm -= lm[NOSE_TIP]

    # scale
    eye_dist = np.linalg.norm(lm[LEFT_EYE] - lm[RIGHT_EYE])
    if eye_dist > 1e-6:
        lm /= eye_dist

    # rotate
    left = lm[LEFT_EYE].copy()
    right = lm[RIGHT_EYE].copy()
    angle = np.arctan2(right[1] - left[1], right[0] - left[0])
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    lm = lm @ R.T

    return lm


def augment_image(img):
    """Augmentasi sederhana: brightness/contrast + rotasi + scale."""
    h, w = img.shape[:2]

    # brightness / contrast
    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-30, 30)
    aug = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # rotasi & scale
    angle = random.uniform(-12, 12)
    scale = random.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    aug = cv2.warpAffine(aug, M, (w, h), borderValue=(128, 128, 128))

    return aug


def extract_aligned_landmarks(image_path, face_mesh, augment=False, num_aug=2):
    """
    Load gambar, apply CLAHE, ekstrak landmark MediaPipe, lalu align.

    Returns:
        list[np.ndarray]: list landmark (468, 3). Jika augment=True,
                          mengembalikan [original, aug1, aug2, ...].
        int: jumlah gagal deteksi.
    """
    img = cv2.imread(image_path)
    if img is None:
        return [], 1

    variants = [img]
    if augment:
        variants += [augment_image(img) for _ in range(num_aug)]

    landmarks_list = []
    fails = 0
    for variant in variants:
        enhanced = apply_clahe(variant)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lms = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark],
                dtype=np.float32
            )
            landmarks_list.append(align_landmarks(lms))
        else:
            fails += 1

    return landmarks_list, fails


def build_model(input_shape=(468, 3), num_classes=20):
    """PointNet-style Conv1D + GlobalMaxPooling untuk landmark wajah."""
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, 1, activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 1, activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, 1, activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalMaxPooling1D()(x)

    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)

    embedding = layers.Dense(128, activation='relu', name='embedding')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='softmax')(embedding)

    return models.Model(inputs, outputs, name='mediapipe_pointnet')


def compute_centroids_and_thresholds(embeddings, labels, num_classes,
                                     percentile=95):
    """
    Hitung centroid embedding tiap kelas dan threshold rejection.

    Returns:
        centroids: (num_classes, emb_dim)
        thresholds: (num_classes,)
    """
    emb_dim = embeddings.shape[1]
    centroids = np.zeros((num_classes, emb_dim), dtype=np.float32)
    thresholds = np.zeros(num_classes, dtype=np.float32)

    for i in range(num_classes):
        mask = labels == i
        cls_emb = embeddings[mask]
        centroids[i] = cls_emb.mean(axis=0)
        dists = np.linalg.norm(cls_emb - centroids[i], axis=1)
        thresholds[i] = np.percentile(dists, percentile)

    return centroids, thresholds
