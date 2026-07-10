"""
Inference webcam fixed untuk MediaPipe + PointNet-style CNN.

Cara pakai:
    python inference_mediapipe_fixed.py

Tekan 'q' untuk keluar.
"""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import pickle
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

from mediapipe_face_utils import apply_clahe, align_landmarks


EMB_BUFFER_SIZE = 5      # smoothing embedding
SOFT_CONF = 0.55         # fallback softmax confidence
SOFT_MARGIN = 0.12       # fallback margin top1 - top2


def main():
    print("[INFO] Loading model...")
    model = tf.keras.models.load_model("model_cnn_last.keras")

    with open("label_encoder_last.pickle", "rb") as f:
        le = pickle.load(f)

    mean = np.load("mean_landmark.npy")
    std = np.load("std_landmark.npy")
    centroids = np.load("face_centroids.npy")
    thresholds = np.load("face_thresholds.npy")

    embedding_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=model.get_layer("embedding").output,
    )

    print("[INFO] Inisialisasi webcam & MediaPipe...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam tidak bisa dibuka.")
        return

    emb_buffer = deque(maxlen=EMB_BUFFER_SIZE)

    print("[INFO] Running. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # SAMAKAN PREPROCESSING DENGAN TRAINING
        enhanced = apply_clahe(frame)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        display_name = "No Face"
        conf = 0.0
        dist = 0.0
        nearest_name = "-"

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            lms = np.array(
                [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark],
                dtype=np.float32,
            )
            aligned = align_landmarks(lms)
            norm = (aligned - mean) / std
            x_input = np.expand_dims(norm, axis=0)

            pred = model.predict(x_input, verbose=0)[0]
            emb = embedding_model.predict(x_input, verbose=0)[0]

            emb_buffer.append(emb)

            # Smoothing embedding dari N frame terakhir
            smoothed_emb = np.mean(emb_buffer, axis=0)

            # Nearest centroid
            dists = np.linalg.norm(centroids - smoothed_emb, axis=1)
            cls_idx = int(np.argmin(dists))
            dist = float(dists[cls_idx])
            nearest_name = le.classes_[cls_idx]

            # Fallback pakai softmax confidence + margin
            sorted_pred = np.sort(pred)
            top1 = sorted_pred[-1]
            top2 = sorted_pred[-2]
            conf = float(pred[cls_idx])

            if dist <= thresholds[cls_idx]:
                display_name = le.classes_[cls_idx]
            elif top1 >= SOFT_CONF and (top1 - top2) >= SOFT_MARGIN:
                display_name = le.classes_[int(np.argmax(pred))]
            else:
                display_name = "Unknown"

            # Bounding box dari landmark
            h, w, _ = frame.shape
            xs = [lm.x * w for lm in face_landmarks.landmark]
            ys = [lm.y * h for lm in face_landmarks.landmark]
            x1 = int(min(xs))
            y1 = int(min(ys))
            x2 = int(max(xs))
            y2 = int(max(ys))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{display_name} ({conf*100:.1f}%, d={dist:.2f})"
            cv2.putText(
                frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            # Debug: tampilkan keputusan AKHIR, bukan hanya nearest class
            print(
                f"[DEBUG] display={display_name} | nearest={nearest_name} | "
                f"dist={dist:.3f} | threshold={thresholds[cls_idx]:.3f} | "
                f"softmax={top1:.3f} (margin={top1-top2:.3f})"
            )
        else:
            # Reset buffer kalau wajah hilang
            emb_buffer.clear()

        cv2.imshow("MediaPipe + Embedding Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("[INFO] Selesai.")


if __name__ == "__main__":
    main()
